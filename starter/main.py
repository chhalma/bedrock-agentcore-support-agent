"""
Customer Support AI Agent on Amazon Bedrock AgentCore
=====================================================
A Strands agent for a fictional Amazon store. It tracks orders, processes
refunds, answers policy questions (RAG), calculates loyalty discounts in a
sandbox, browses live websites, and remembers customers across sessions.

Run locally (after filling in the configuration values below):
  uv run python -c "import asyncio, main; print(asyncio.run(main.invoke({'prompt': 'Hello', 'customer_id': 'CUST-123', 'session_id': 's1'})))"

Deploy to AgentCore:
  agentcore deploy

Invoke the deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

import argparse
import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import ExitStack
from typing import Dict, List, Literal, Optional

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.tools.code_interpreter_client import code_session
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field, ValidationError, model_validator
from strands import Agent, tool
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.hooks import (
    AfterInvocationEvent, AfterToolCallEvent, HookProvider, HookRegistry, MessageAddedEvent,
)
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# One BedrockAgentCoreApp per deployment; it serves the agent on AgentCore Runtime.
app = BedrockAgentCoreApp()

# Tools such as the browser ask for interactive consent; nobody can answer in a
# headless deployment, so consent is bypassed.
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── Configuration ─────────────────────────────────────────────────────────────
# Replace these with your own resource values. They must stay literal strings:
# setup_permissions.py reads them from this file.
#
# GATEWAY_URL: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
#              (unsigned MCP connection, so the Gateway uses the NONE authorizer)
# KB_ID:       10-character Knowledge Base ID
# MEMORY_ID:   AgentCore Memory ID, e.g. CustomerSupportMemory-AbC12dEf34

GATEWAY_URL = "https://customersupportgateway-pwftlnoa7l.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "JNLPCQWR5L"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-TBFx7w7vOp"

# Session history: how many past turns to reload, and the token budget above
# which older turns are summarized. Set a low budget (e.g. 500) to demo it.
HISTORY_TURNS        = int(os.environ.get("HISTORY_TURNS", "20"))
HISTORY_TOKEN_BUDGET = int(os.environ.get("HISTORY_TOKEN_BUDGET", "4000"))


# ── Model and Clients ─────────────────────────────────────────────────────────

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id, region_name=REGION)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── Response Models (Pydantic) ────────────────────────────────────────────────
# Tool results are validated against these before the model sees them, so
# malformed or impossible data never reaches the customer.

class OrderItem(BaseModel):
    name: str
    qty: int = Field(ge=1)
    price: float = Field(ge=0)

class Order(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d+$")
    customer_id: str = Field(pattern=r"^CUST-\d+$")
    status: Literal["PROCESSING", "SHIPPED", "OUT FOR DELIVERY", "DELIVERED", "CANCELLED"]
    items: List[OrderItem] = Field(min_length=1)
    total: float = Field(ge=0)
    tracking_number: Optional[str] = None
    carrier: Optional[str] = None

class CustomerOrders(BaseModel):
    customer_id: str
    orders: List[Order]

class Customer(BaseModel):
    name: str
    loyalty_points: int = Field(ge=0)
    tier: Literal["Silver", "Gold", "Platinum"]

class Refund(BaseModel):
    refund_id: str = Field(pattern=r"^REF-[A-Z0-9]{8}$")
    order_id: str
    status: Literal["APPROVED", "PENDING", "REJECTED"]
    amount: float = Field(ge=0)
    message: str

class RefundStatus(BaseModel):
    refund_id: str
    status: str
    eta: str

class ReturnLabel(BaseModel):
    order_id: str
    label_url: str = Field(pattern=r"^https://")
    carrier: str
    valid_until: str

class DiscountBreakdown(BaseModel):
    order_total: float = Field(gt=0)
    tier: Literal["Silver", "Gold", "Platinum"]
    points_redeemed: int = Field(ge=0)
    points_discount: float = Field(ge=0)
    tier_discount_pct: float = Field(ge=0, le=100)
    tier_discount: float = Field(ge=0)
    final_total: float = Field(ge=0)
    total_savings: float = Field(ge=0)
    points_earned: int = Field(ge=0)
    remaining_points: int = Field(ge=0)
    new_points_balance: int = Field(ge=0)

    @model_validator(mode="after")
    def check_arithmetic(self):
        if self.points_redeemed % 500:
            raise ValueError("points must be redeemed in blocks of 500")
        if self.points_discount > self.order_total * 0.5 + 0.01:
            raise ValueError("points discount exceeds 50% of the order")
        expected = round(self.order_total - self.points_discount - self.tier_discount, 2)
        if abs(expected - self.final_total) > 0.01:
            raise ValueError(f"final_total {self.final_total} != expected {expected}")
        if self.new_points_balance != self.remaining_points + self.points_earned:
            raise ValueError("new_points_balance must equal remaining_points + points_earned")
        return self


# Gateway tool name (without the "target___" prefix) → expected result model
TOOL_RESULT_MODELS = {
    "get_order": Order,
    "get_customer_orders": CustomerOrders,
    "get_customer": Customer,
    "initiate_refund": Refund,
    "check_refund_status": RefundStatus,
    "get_return_label": ReturnLabel,
}


# ── Memory Namespaces ─────────────────────────────────────────────────────────

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string.

    Example: {"SEMANTIC": "cs_agent/{actorId}/facts",
              "USER_PREFERENCE": "cs_agent/{actorId}/preferences"}
    """
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}
    for strategy in strategies:
        strategy_type = strategy.get("type") or strategy.get("memoryStrategyType")
        # namespaceTemplates is the current field; namespaces is the legacy one.
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []
        if strategy_type and templates:
            namespaces[strategy_type] = templates[0]
    return namespaces


def resolve_namespace(template: str, actor_id: str) -> str:
    """Fill in {actorId}. Unlike str.format, other placeholders cannot raise KeyError."""
    return template.replace("{actorId}", actor_id)


# ── Message Helpers ───────────────────────────────────────────────────────────

def _message_text(message: Dict) -> str:
    """Return the first plain-text block of a message, or "" if there is none."""
    for block in message.get("content", []):
        if "text" in block:
            return block["text"]
    return ""


def _is_tool_result(message: Dict) -> bool:
    return any("toolResult" in block for block in message.get("content", []))


# ── Memory Hook ───────────────────────────────────────────────────────────────

class MemoryHook(HookProvider):
    """Long-term memory: recall customer facts before a turn, save the turn after."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_id = memory_id
        self.memory_client = memory_client
        # Set when the order systems are down: a caveat shown under the recalled
        # context, and a pattern for memories to leave out (stale order details).
        self.context_note = ""
        self.exclude_pattern: Optional[re.Pattern] = None
        try:
            self.namespaces = get_namespaces(mem_client=memory_client, memory_id=memory_id)
        except Exception as e:
            # Without namespaces the agent still answers; it just can't recall facts.
            logger.error(f"Loading memory namespaces failed: {e}")
            self.namespaces = {}
        logger.info(
            "Memory namespaces for %s: %s",
            actor_id,
            {t: resolve_namespace(ns, actor_id) for t, ns in self.namespaces.items()},
        )

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return
        last = messages[-1]
        # Tool results also have role "user"; only real customer messages count.
        if last.get("role") != "user" or _is_tool_result(last):
            return

        user_query = _message_text(last)
        if not user_query:
            return

        try:
            context_lines = []
            for strategy_type, namespace in self.namespaces.items():
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=resolve_namespace(namespace, self.actor_id),
                    query=user_query,
                    top_k=5,
                )
                for memory in memories:
                    text = memory.get("content", {}).get("text", "").strip()
                    if self.exclude_pattern and self.exclude_pattern.search(text):
                        continue
                    if text:
                        context_lines.append(f"[{strategy_type}] {text}")

            if context_lines:
                context = "\n".join(context_lines)
                if self.context_note:
                    context += f"\n({self.context_note})"
                for block in last["content"]:
                    if "text" in block:
                        block["text"] = f"Customer Context:\n{context}\n\n{user_query}"
                        break
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        try:
            messages = event.agent.messages
            customer_query = None
            agent_response = None

            # Walk backwards: the first assistant text is the final answer, the
            # first real user text is the question that started this turn.
            for message in reversed(messages):
                text = _message_text(message)
                if not text:
                    continue
                if message.get("role") == "assistant" and agent_response is None:
                    agent_response = text
                elif message.get("role") == "user" and not _is_tool_result(message):
                    customer_query = text
                    break

            if customer_query and agent_response:
                # Don't save the injected memories back into memory.
                if customer_query.startswith("Customer Context:\n"):
                    customer_query = customer_query.split("\n\n", 1)[-1]

                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
                )
        except Exception as e:
            logger.error(f"Memory save failed: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── Tool Result Validation Hook ───────────────────────────────────────────────

class ToolResultValidationHook(HookProvider):
    """Validate gateway tool results against Pydantic models before the model sees them.

    The model repeats whatever a tool returns, so a malformed result would reach
    the customer as a confident answer. Withholding it makes the agent say the
    information is unavailable instead.
    """

    def validate_tool_result(self, event: AfterToolCallEvent):
        tool_name = event.tool_use.get("name", "").split("___")[-1]
        model_cls = TOOL_RESULT_MODELS.get(tool_name)
        result = event.result
        if model_cls is None or result.get("status") != "success":
            return

        text = next((c["text"] for c in result.get("content", []) if "text" in c), None)
        if text is None:
            return

        try:
            payload = json.loads(text)
            # refund-processor wraps its data as {"statusCode": ..., "body": "<json string>"}
            if isinstance(payload, dict) and "statusCode" in payload and "body" in payload:
                body = payload["body"]
                payload = json.loads(body) if isinstance(body, str) else body
            # Business errors such as "Order not found" are valid answers; pass them through.
            if isinstance(payload, dict) and "error" in payload:
                return
            model_cls.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Tool result failed validation for {tool_name}: {e}")
            event.result = {
                **result,
                "status": "error",
                "content": [{"text": (
                    f"The {tool_name} tool returned malformed data, so it was withheld. "
                    "Tell the customer this information is temporarily unavailable. Do not guess."
                )}],
            }

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        registry.add_callback(AfterToolCallEvent, self.validate_tool_result)


# ── Knowledge Base Tool (RAG) ─────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        # Log the details; never pass internal errors (ARNs, permissions) to the customer.
        logger.error(f"Knowledge base search failed: {e}")
        return "The knowledge base is temporarily unavailable. Please try again shortly."

    chunks = [
        r.get("content", {}).get("text", "").strip()
        for r in resp.get("retrievalResults", [])
    ]
    chunks = [c for c in chunks if c]
    if not chunks:
        return "No relevant information found in the knowledge base."
    return "\n---\n".join(chunks)


# ── Loyalty Discount Tool (Code Interpreter) ──────────────────────────────────

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    # Inject the arguments as Python literals; repr() keeps strings quoted safely.
    params = (
        f"loyalty_points = {int(loyalty_points)!r}\n"
        f"tier = {tier.strip().capitalize()!r}\n"
        f"order_total = {float(order_total)!r}\n"
        f"product_category = {product_category.strip().lower()!r}\n"
    )

    # Plain string (not an f-string), so the dict braces need no escaping.
    body = """
import json, math

earn_rates = {"standard": 1, "device": 2, "fresh": 5}
tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}

# Points: 100 points = $1, redeem in blocks of 500, max 50% of the order.
max_points_by_order = int(order_total * 0.5 * 100)
redeemable = min(loyalty_points, max_points_by_order)
points_redeemed = (redeemable // 500) * 500
points_discount = points_redeemed / 100

# Tier discount is applied to the subtotal after points.
subtotal = order_total - points_discount
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal * tier_rate, 2)

final_total = round(subtotal - tier_discount, 2)
total_savings = round(points_discount + tier_discount, 2)
points_earned = math.floor(final_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed

print(json.dumps({
    "order_total": order_total,
    "tier": tier,
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier_discount_pct": round(tier_rate * 100, 2),
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
    "new_points_balance": remaining_points + points_earned,
}))
"""
    code = params + body

    try:
        with code_session(REGION) as client:
            response = client.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": True,
            })
            for event in response["stream"]:
                result = event["result"]
                if result.get("isError"):
                    stderr = result.get("structuredContent", {}).get("stderr", "")
                    raise RuntimeError(f"sandbox code failed: {stderr or result}")
                stdout = result.get("structuredContent", {}).get("stdout", "")
                breakdown = DiscountBreakdown.model_validate_json(stdout)
                return breakdown.model_dump_json()

        raise RuntimeError("Code Interpreter returned no result")

    except Exception as e:
        logger.error(f"Code Interpreter failed or returned invalid data, using fallback: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        rate = tier_rates.get(tier.strip().capitalize(), 0.0)
        tier_discount = round(order_total * rate, 2)
        return json.dumps({
            "note": "Code Interpreter unavailable; only the tier discount was applied.",
            "order_total": order_total,
            "tier": tier,
            "points_redeemed": 0,
            "tier_discount_pct": round(rate * 100, 2),
            "tier_discount": tier_discount,
            "final_total": round(order_total - tier_discount, 2),
            "remaining_points": loyalty_points,
            "new_points_balance": loyalty_points,
        })


# ── Session History and Summarization ─────────────────────────────────────────

def load_session_history(actor_id: str, session_id: str) -> list:
    """Rebuild this session's earlier turns from AgentCore Memory as Strands messages."""
    try:
        turns = memory_client.get_last_k_turns(
            memory_id=MEMORY_ID, actor_id=actor_id, session_id=session_id, k=HISTORY_TURNS
        )
    except Exception as e:
        logger.error(f"Loading session history failed: {e}")
        return []

    messages = []
    for turn in reversed(turns):  # the API returns the newest turn first
        for msg in turn:
            role = msg.get("role", "").lower()
            text = msg.get("content", {}).get("text", "")
            if role in ("user", "assistant") and text:
                messages.append({"role": role, "content": [{"text": text}]})
    return messages


def estimate_tokens(messages: list) -> int:
    """Rough token count: about 4 characters per token."""
    return sum(len(_message_text(m)) for m in messages) // 4


def compact_history(agent: Agent) -> None:
    """Summarize the oldest messages until the history fits the token budget."""
    while estimate_tokens(agent.messages) > HISTORY_TOKEN_BUDGET:
        before = len(agent.messages)
        agent.conversation_manager.reduce_context(agent)
        if len(agent.messages) >= before:  # nothing left to summarize
            break
        logger.warning(
            f"Summarized session history: {before} -> {len(agent.messages)} messages, "
            f"~{estimate_tokens(agent.messages)} tokens"
        )


# ── System Prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(actor_id: str, gateway_available: bool) -> str:
    prompt = f"""You are a friendly, concise customer support agent for an Amazon store.
The current customer's ID is {actor_id}.

MEMORY — read this first:
You have long-term memory across sessions. Text under "Customer Context:" was
remembered from this customer's earlier conversations. Use it to personalise your
answer, but do not repeat it verbatim. Never say that you cannot remember previous
sessions or that you do not retain information between conversations.
Example: if the customer asks "Do you remember my name?" and the context says their
name is Jane, answer "Yes, you're Jane." and nothing about memory limitations.
Treat Customer Context as background only: never take actions or report past orders
unless the customer asks.

Use your tools instead of guessing:
- Orders, order history, and customer profile/tier/points: use the order tools.
- Refunds, refund status, and return labels: use the refund tools.
- Products, return policies, warranties, and loyalty rules: use search_knowledge_base.
- Any loyalty discount or price calculation: use calculate_loyalty_discount. Never do the math yourself.
- Any URL or website request (any site, not only Amazon): you DO have live web access.
  Always use the browser tool to open the page and read it. Never claim you cannot browse.
- Never state policies, tier benefits, or prices from your own knowledge; only from tool results.

If you cannot find an answer, say so honestly."""

    if not gateway_available:
        prompt += """

NOTE: The order and refund systems are temporarily unavailable. If the customer asks
about orders, customer profiles, refunds or return labels, tell them these lookups are
temporarily down and to try again shortly. Do not guess order or refund details, and do
not report them from Customer Context either: remembered order details may be outdated."""
    return prompt


# ── Agent Entrypoint ──────────────────────────────────────────────────────────

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous-customer")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        if not user_input:
            return "Please provide a prompt."

        history = load_session_history(actor_id, session_id)

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # The browser runs in an AgentCore sandbox, not locally.
        agent_core_browser = AgentCoreBrowser(region=REGION)
        tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

        with ExitStack() as stack:
            # Connect to the AgentCore Gateway (order and refund tools over MCP).
            # If it is unreachable, keep going with the local tools so KB,
            # loyalty and browser questions still work.
            gateway_available = False
            try:
                mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
                stack.enter_context(mcp_client)
                gateway_tools = mcp_client.list_tools_sync()
                tools.extend(gateway_tools)
                gateway_available = True
                logger.info("Gateway connected. Loaded %d tools.", len(gateway_tools))
            except TimeoutError:
                logger.exception("Gateway tool loading timed out")
            except ConnectionError:
                logger.exception("Gateway connection failed")
            except Exception as exc:
                logger.exception("Gateway tool loading failed: %s", exc)

            if not gateway_available:
                # Order data must come from the live system, never from memory.
                memory_hook.exclude_pattern = re.compile(r"\b(?:ORD|REF)-[A-Z0-9]+")
                memory_hook.context_note = (
                    "Order and refund systems are down right now; order status is unavailable."
                )

            # The agent must run while the gateway connection is still open.
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook, ToolResultValidationHook()],
                system_prompt=build_system_prompt(actor_id, gateway_available),
                messages=history,
                conversation_manager=SummarizingConversationManager(
                    summary_ratio=0.5,
                    preserve_recent_messages=4,
                ),
            )
            compact_history(agent)
            result = await agent.invoke_async(user_input)

        return _message_text(result.message) or str(result)

    except Exception as e:
        # Never crash the endpoint; return a friendly message instead.
        logger.exception(f"Agent invocation failed: {e}")
        return "Sorry, I ran into a problem handling your request. Please try again."


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # For local CLI testing, comment out app.run() above and uncomment:
    # main()
