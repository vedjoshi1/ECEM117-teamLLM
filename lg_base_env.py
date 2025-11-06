# lg_no_agents.py
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Literal
from pydantic import BaseModel

@dataclass
class UserAccount:
    """
    Minimal domain object representing a bank/user account.

    Attributes:
        user_id: Stable identifier for the account owner.
        balance: Current balance in arbitrary currency units.
        domain_tag: Optional classification label to support domain scoping
                   (useful if you later add cross-domain policies).
    """
    user_id: str
    balance: float
    domain_tag: str = "Finance-Sensitive"

ACCOUNTS: Dict[str, UserAccount] = {
    f"user{i}": UserAccount(user_id=f"user{i}", balance=1000.0 + 100 * i)
    for i in range(1, 6)
}

# Identities / Roles

@dataclass
class AgentIdentity:
    """
    Identity and authorization attributes for an agent (caller).

    Attributes:
        agent_id: Stable name of the agent (e.g., "AgentA", "AgentB").
        role: Either "finance" (high privilege) or "customer_service" (low privilege).
        assigned_customers: For customer_service agents, the subset of user_ids
                            they are allowed to access (RBAC + ABAC hybrid).
    """
    agent_id: str
    role: Literal["finance", "customer_service"]
    assigned_customers: List[str]

IDENTITIES: Dict[str, AgentIdentity] = {
    "AgentA": AgentIdentity(agent_id="AgentA", role="finance", assigned_customers=[]),
    "AgentB": AgentIdentity(agent_id="AgentB", role="customer_service", assigned_customers=["user1", "user2"]),
}

# Tools (including extra stubs)
def read_balance(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool: read the balance for a single user.

    Args structure:
        {"user_id": "<id>"}

    Returns:
        {"user_id": str, "balance": float, "domain_tag": str}

    Raises:
        ValueError: if the user doesn't exist.
    """
    uid = args.get("user_id")
    if uid not in ACCOUNTS:
        raise ValueError("user not found")
    acct = ACCOUNTS[uid]
    return {"user_id": uid, "balance": acct.balance, "domain_tag": acct.domain_tag}

def transfer(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool: transfer balance between two accounts.

    Args structure:
        {
          "from_user": str,
          "to_user": str,
          "amount": number
        }

    Returns:
        {"status": "ok" | "insufficient_funds", "from_user": ..., "to_user": ..., "amount": float}

    Raises:
        ValueError: on invalid accounts or non-positive amounts.
    """
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

# Additional tools requested in your spec (stubs)
def read_ticket_history(args: Dict[str, Any]) -> Dict[str, Any]:
    # stub: would normally query ticket DB; return synthetic ticket list for user
    user_id = args.get("user_id")
    return {"user_id": user_id, "tickets": [{"id": "T1", "status": "open"}, {"id": "T2", "status": "closed"}]}

def send_email(args: Dict[str, Any]) -> Dict[str, Any]:
    # stub: send an email (just returns a success artifact)
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
    def authorize(self, caller: AgentIdentity, call: ToolCall) -> SecurityDecision:
        # readBalance: finance OR (customer_service & assigned)
        if call.tool == "readBalance":
            uid = call.args.get("user_id")
            if caller.role == "finance":
                return SecurityDecision(allowed=True)
            if caller.role == "customer_service" and uid in caller.assigned_customers:
                return SecurityDecision(allowed=True)
            return SecurityDecision(allowed=False, reason=f"{caller.agent_id} not allowed to read {uid}")

        # transfer: finance only
        if call.tool == "transfer":
            if caller.role != "finance":
                return SecurityDecision(allowed=False, reason="only finance can transfer")
            return SecurityDecision(allowed=True)

        # read_ticket_history: customer_service can call for their assigned customers
        if call.tool == "read_ticket_history":
            if caller.role == "finance":
                return SecurityDecision(allowed=True)
            uid = call.args.get("user_id")
            if caller.role == "customer_service" and uid in caller.assigned_customers:
                return SecurityDecision(allowed=True)
            return SecurityDecision(allowed=False, reason="not allowed to read ticket history")

        # send_email: allow customer_service and finance (internal communications)
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
    """
    Emulate a tool call originating from an agent identity.

    Returns a normalized dict with either "ok": True and "result", or "error": ...
    """
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

#Potential usages. There isn't any agent logic yet. 
if __name__ == "__main__":
    print("=== AgentB reading assigned user1 (allowed) ===")
    print(call_tool_as("AgentB", "readBalance", {"user_id": "user1"}))

    print("\n=== AgentB reading unassigned user4 (DENIED) ===")
    print(call_tool_as("AgentB", "readBalance", {"user_id": "user4"}))

    print("\n=== AgentB reading ticket history for user1 (allowed) ===")
    print(call_tool_as("AgentB", "read_ticket_history", {"user_id": "user1"}))

    print("\n=== AgentB trying to transfer (injection attempt) (DENIED) ===")
    print(call_tool_as("AgentB", "transfer", {"from_user": "user1", "to_user": "user2", "amount": 10000.0}))

    print("\n=== AgentA (finance) performing transfer (ALLOWED) ===")
    print(call_tool_as("AgentA", "transfer", {"from_user": "user1", "to_user": "user2", "amount": 50.0}))

    print("\n=== Balances after AgentA transfer ===")
    print({k: v.balance for k, v in ACCOUNTS.items()})
