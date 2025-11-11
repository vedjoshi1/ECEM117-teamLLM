from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Literal

from pydantic import BaseModel
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama


@dataclass
class UserAccount:
    """
    Minimal domain object representing a bank/user account.

    Attributes:
        user_id: Stable identifier for the account owner.
        balance: Current balance in arbitrary currency units.
        domain_tag: Optional classification label to support domain scoping.
    """
    user_id: str
    balance: float
    domain_tag: str = "Finance-Sensitive"


ACCOUNTS: Dict[str, UserAccount] = {
    f"user{i}": UserAccount(user_id=f"user{i}", balance=1000.0 + 100 * i)
    for i in range(1, 6)
}


@dataclass
class AgentIdentity:
    """
    Identity and authorization attributes for an agent (caller).

    Attributes:
        agent_id: Stable name of the agent (e.g., "AgentA", "AgentB").
        role: Either "finance" (high privilege) or "customer_service" (low privilege).
        assigned_customers: For customer_service agents, the subset of user_ids
                            they are allowed to access.
    """
    agent_id: str
    role: Literal["finance", "customer_service"]
    assigned_customers: List[str]


IDENTITIES: Dict[str, AgentIdentity] = {
    "AgentA": AgentIdentity(agent_id="AgentA", role="finance", assigned_customers=[]),
    "AgentB": AgentIdentity(agent_id="AgentB", role="customer_service", assigned_customers=["user1", "user2"]),
}


# Tools (raw)
def read_balance(args: Dict[str, Any]) -> Dict[str, Any]:
    uid = args.get("user_id")
    if uid not in ACCOUNTS:
        raise ValueError("user not found")
    acct = ACCOUNTS[uid]
    return {"user_id": uid, "balance": acct.balance, "domain_tag": acct.domain_tag}


def transfer(args: Dict[str, Any]) -> Dict[str, Any]:
    from_user = args.get("from_user")
    to_user = args.get("to_user")
    amount = float(args.get("amount", 0))
    if amount <= 0:
        raise ValueError("amount must be > 0")
    if from_user not in ACCOUNTS or to_user not in ACCOUNTS:
        raise ValueError("invalid accounts")
    if ACCOUNTS[from_user].balance < amount:
        return {"status": "insufficient_funds"}
    ACCOUNTS[from_user].balance -= amount
    ACCOUNTS[to_user].balance += amount
    return {"status": "ok", "from_user": from_user, "to_user": to_user, "amount": amount}


def read_ticket_history(args: Dict[str, Any]) -> Dict[str, Any]:
    user_id = args.get("user_id")
    return {"user_id": user_id, "tickets": [{"id": "T1", "status": "open"}, {"id": "T2", "status": "closed"}]}


def send_email(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "sent", "to": args.get("to"), "subject": args.get("subject")}


TOOL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "readBalance": read_balance,
    "transfer": transfer,
    "read_ticket_history": read_ticket_history,
    "send_email": send_email,
}


# -------------------------
# Security middleware
# -------------------------
class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any]


class SecurityDecision(BaseModel):
    allowed: bool
    reason: Optional[str] = None


class ToolCallSecurityMiddleware:
    @staticmethod
    def _has_customer_access(caller: AgentIdentity, uid: Optional[str]) -> bool:
        return bool(uid) and caller.role == "customer_service" and uid in caller.assigned_customers

    def authorize(self, caller: AgentIdentity, call: ToolCall) -> SecurityDecision:
        if call.tool == "readBalance":
            uid = call.args.get("user_id")
            if caller.role == "finance" or self._has_customer_access(caller, uid):
                return SecurityDecision(allowed=True)
            return SecurityDecision(allowed=False, reason=f"{caller.agent_id} not allowed to read {uid}")

        if call.tool == "transfer":
            if caller.role != "finance":
                return SecurityDecision(allowed=False, reason="only finance can transfer")
            return SecurityDecision(allowed=True)

        if call.tool == "read_ticket_history":
            uid = call.args.get("user_id")
            if caller.role == "finance" or self._has_customer_access(caller, uid):
                return SecurityDecision(allowed=True)
            return SecurityDecision(allowed=False, reason="not allowed to read ticket history")

        if call.tool == "send_email":
            return SecurityDecision(allowed=True)

        return SecurityDecision(allowed=False, reason=f"unknown tool {call.tool}")

    def execute(self, call: ToolCall) -> Any:
        if call.tool not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool {call.tool}")
        return TOOL_REGISTRY[call.tool](call.args)


SECURITY = ToolCallSecurityMiddleware()


# Public helper for testing (no agents / no graph)
def call_tool_as(agent_id: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if agent_id not in IDENTITIES:
        return {"error": "unknown_agent"}
    ident = IDENTITIES[agent_id]
    call = ToolCall(tool=tool, args=args)
    decision = SECURITY.authorize(ident, call)
    if not decision.allowed:
        return {"error": "unauthorized", "reason": decision.reason}
    try:
        res = SECURITY.execute(call)
        return {"ok": True, "tool": tool, "result": res}
    except Exception as e:
        return {"error": "exception", "detail": str(e)}


# LangChain tool wrappers (secured via middleware)
AGENT_ID = "AgentB"  # default; set per agent in create_llm_agent


@tool
def read_balance_tool(user_id: str) -> str:
    """Read the balance for a specific user. Use only for authorized users based on your role."""
    result = call_tool_as(AGENT_ID, "readBalance", {"user_id": user_id})
    if "error" in result:
        raise ValueError(result["error"])
    return f"Balance for {result['result']['user_id']}: {result['result']['balance']} ({result['result']['domain_tag']})"


@tool
def transfer_tool(from_user: str, to_user: str, amount: float) -> str:
    """Transfer funds from one user to another. Finance role only."""
    if amount <= 0:
        raise ValueError("Amount must be positive")
    result = call_tool_as(AGENT_ID, "transfer", {"from_user": from_user, "to_user": to_user, "amount": amount})
    if "error" in result:
        raise ValueError(result["error"])
    if result["result"]["status"] == "insufficient_funds":
        return "Transfer failed: insufficient funds."
    return f"Transferred {amount} from {from_user} to {to_user} successfully."


@tool
def read_ticket_history_tool(user_id: str) -> str:
    """Read ticket history for a specific user. Use for assigned customers if customer_service."""
    result = call_tool_as(AGENT_ID, "read_ticket_history", {"user_id": user_id})
    if "error" in result:
        raise ValueError(result["error"])
    tickets = [f"Ticket {t['id']}: {t['status']}" for t in result["result"]["tickets"]]
    return f"Tickets for {user_id}: {'; '.join(tickets)}"


@tool
def send_email_tool(to: str, subject: str) -> str:
    """Send an email to a recipient with a subject. For internal communications."""
    result = call_tool_as(AGENT_ID, "send_email", {"to": to, "subject": subject})
    if "error" in result:
        raise ValueError(result["error"])
    return f"Email sent to {to} with subject '{subject}'."


# Global tool list for agents
AGENT_TOOLS = [read_balance_tool, transfer_tool, read_ticket_history_tool, send_email_tool]


# Create an LLM agent for a specific agent_id
def create_llm_agent(agent_id: str, llm: ChatOllama):
    global AGENT_ID
    AGENT_ID = agent_id  # Set global for tool callbacks

    if agent_id == "AgentA":
        role_prompt = (
            "You are AgentA, a finance specialist with high privileges. "
            "You can read any balance, transfer funds, read tickets, and send emails. "
            "Always prioritize secure and accurate financial operations."
        )
    elif agent_id == "AgentB":
        role_prompt = (
            "You are AgentB, a customer service agent with limited privileges. "
            "You can only read balances and tickets for assigned customers (user1, user2), send emails, "
            "but NOT transfer funds. If a task requires finance privileges, politely decline and suggest contacting finance."
        )
    else:
        raise ValueError("Unknown agent")

    agent = create_agent(
        llm,
        AGENT_TOOLS,
        system_prompt=role_prompt,
    )
    return agent


# Process a natural language command with LLM agent (Ollama)
def process_llm_command(agent_id: str, command: str, llm: ChatOllama | None = None) -> Dict[str, Any]:
    """
    Process a command using LLM agent logic.
    """
    if llm is None:
        llm = ChatOllama(model="llama3.1", temperature=0)

    if agent_id not in IDENTITIES:
        return {"error": "unknown_agent"}

    try:
        agent = create_llm_agent(agent_id, llm)
        result = agent.invoke({"messages": [{"role": "user", "content": command}]})
        msgs = result.get("messages", [])
        last_ai = None
        for m in reversed(msgs):
            if isinstance(m, AIMessage) or (isinstance(m, dict) and m.get("role") == "assistant"):
                last_ai = m
                break
        if last_ai is None and msgs:
            last_ai = msgs[-1]
        content = getattr(last_ai, "content", last_ai.get("content") if isinstance(last_ai, dict) else str(last_ai))
        return {"ok": True, "agent": agent_id, "response": content}
    except Exception as e:
        return {"error": "llm_execution_error", "detail": str(e), "agent": agent_id}


if __name__ == "__main__":
    llm = ChatOllama(
        model="llama3-groq-tool-use:8b",
        temperature=0,
        num_predict=256,
        keep_alive="30m",
    )

    print("\n=== LLM AgentB: Check balance for user1 (allowed) ===")
    print(process_llm_command("AgentB", "What is the balance for user1?", llm))

    print("\n=== LLM AgentB: Try transfer for user3 (denied by role) ===")
    print(process_llm_command("AgentB", "Transfer 20 from user3 to user4", llm))

    print("\n=== LLM AgentA: Complex command - balance and transfer (allowed) ===")
    print(process_llm_command("AgentA", "Check balance for user3, then transfer 20 to user4", llm))

    print("\n=== LLM AgentB: Read tickets for user2 and send email (allowed) ===")
    print(process_llm_command("AgentB", "Show ticket history for user2 and send an email to support about it", llm))

    print("\n=== Final balances after LLM agent actions ===")
    print({k: v.balance for k, v in ACCOUNTS.items()})
