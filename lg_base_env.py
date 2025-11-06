# lg_base_env.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict, Literal
from typing_extensions import NotRequired
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

# =========================
# Domain models (base only)
# =========================
@dataclass
class UserAccount:
    user_id: str
    balance: float
    domain_tag: str = "Finance-Sensitive"

# In-memory store (placeholder data)
ACCOUNTS: Dict[str, UserAccount] = {
    f"user{i}": UserAccount(user_id=f"user{i}", balance=1000.0 + 100 * i)
    for i in range(1, 6)
}

# =========================
# Tool registry (skeleton)
# =========================
class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any]

# Registry type: name -> callable(args)->result
ToolFn = Callable[[Dict[str, Any]], Any]
TOOL_REGISTRY: Dict[str, ToolFn] = {}

# Example stubs (you can wire real functions later)
def _read_balance_stub(args: Dict[str, Any]) -> Dict[str, Any]:
    """Stub: replace with real implementation later."""
    uid = args.get("user_id")
    if uid not in ACCOUNTS:
        raise ValueError("user not found")
    return {"user_id": uid, "balance": ACCOUNTS[uid].balance}

def _transfer_stub(args: Dict[str, Any]) -> Dict[str, Any]:
    """Stub: replace with real implementation later."""
    return {"status": "noop", "note": "transfer not implemented in base model"}

# Register stubs (can be replaced when you implement real tools)
TOOL_REGISTRY["readBalance"] = _read_balance_stub
TOOL_REGISTRY["transfer"] = _transfer_stub

# =========================
# Security middleware stub
# =========================
class SecurityDecision(BaseModel):
    allowed: bool
    reason: NotRequired[str]

class ToolCallSecurityMiddleware:
    """
    Drop-in place to enforce your authorization rules.
    - Map caller identity -> role/privs
    - Enforce per-tool policies
    """
    def authorize(self, caller_token: str, call: ToolCall) -> SecurityDecision:
        # TODO: lookup caller_token -> role/privileges
        # For now, allow ONLY readBalance; deny transfer by default.
        if call.tool == "readBalance":
            return SecurityDecision(allowed=True)
        return SecurityDecision(allowed=False, reason="Not permitted in base model")

    def execute(self, call: ToolCall) -> Any:
        if call.tool not in TOOL_REGISTRY:
            raise ValueError(f"Unknown tool: {call.tool}")
        return TOOL_REGISTRY[call.tool](call.args)

SECURITY = ToolCallSecurityMiddleware()

# =========================
# LangGraph State & Nodes
# =========================
class LGState(TypedDict, total=False):
    """
    Base state for the graph. Agents are NOT included yet.
    Extend later with:
      - caller metadata (role, assigned customers)
      - conversation messages
      - audit trail
    """
    caller_token: str
    pending_call: Optional[ToolCall]
    last_result: Optional[Dict[str, Any]]
    log: List[str]

def _init_state(caller_token: str, pending_call: Optional[ToolCall] = None) -> LGState:
    return {
        "caller_token": caller_token,
        "pending_call": pending_call,
        "last_result": None,
        "log": [],
    }

# Node: router (in a larger system this could choose which agent to run)
def node_router(state: LGState) -> LGState:
    log = state.get("log", [])
    log.append("router: start")
    state["log"] = log
    # In the base model, just go to tool_executor if a call exists; otherwise end.
    return state

# Node: tool_executor (runs through middleware)
def node_tool_executor(state: LGState) -> LGState:
    log = state.get("log", [])
    call: Optional[ToolCall] = state.get("pending_call")
    if not call:
        log.append("tool_executor: no pending call")
        state["log"] = log
        return state

    # Authorize
    decision = SECURITY.authorize(state.get("caller_token", ""), call)
    if not decision.allowed:
        log.append(f"tool_executor: DENY {call.tool} - {decision.reason}")
        state["last_result"] = {"error": "unauthorized", "reason": decision.reason}
        state["log"] = log
        return state

    # Execute
    try:
        result = SECURITY.execute(call)
        log.append(f"tool_executor: OK {call.tool}")
        state["last_result"] = {"ok": True, "tool": call.tool, "result": result}
    except Exception as e:
        log.append(f"tool_executor: ERROR {call.tool} - {e}")
        state["last_result"] = {"error": "exception", "detail": str(e)}
    state["log"] = log
    return state

# Graph wiring
def build_graph():
    g = StateGraph(LGState)
    g.add_node("router", node_router)
    g.add_node("tool_executor", node_tool_executor)

    # Flow: router -> tool_executor -> END
    g.set_entry_point("router")
    g.add_edge("router", "tool_executor")
    g.add_edge("tool_executor", END)
    return g.compile()

# =========================
# Simple "get started" run
# =========================
if __name__ == "__main__":
    app = build_graph()

    # Example 1: allowed base call (readBalance is allowed by stub policy)
    s1 = _init_state(
        caller_token="token-dev",  # TODO: map to real identities later
        pending_call=ToolCall(tool="readBalance", args={"user_id": "user2"}),
    )
    out1 = app.invoke(s1)
    print("=== RUN 1 (readBalance) ===")
    print("last_result:", out1.get("last_result"))
    print("log:", out1.get("log"))

    # Example 2: denied base call (transfer is denied by stub policy)
    s2 = _init_state(
        caller_token="token-dev",
        pending_call=ToolCall(tool="transfer", args={"from_user": "user1", "to_user": "user2", "amount": 100}),
    )
    out2 = app.invoke(s2)
    print("\n=== RUN 2 (transfer) ===")
    print("last_result:", out2.get("last_result"))
    print("log:", out2.get("log"))
