"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent, AfterToolCallEvent
)
from strands.agent.conversation_manager import SummarizingConversationManager

from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Literal, Optional

import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()
# Create the BedrockAgentCoreApp instance

app = BedrockAgentCoreApp(  )  # Replace this line


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in the infrastructure setup section of the project instructions.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# This starter uses an unsigned MCP connection and therefore assumes the
# project Gateway is configured with the NONE authorizer.
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-pwftlnoa7l.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "JNLPCQWR5L"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-TBFx7w7vOp"        #  Replace with your Memory ID
HISTORY_TURNS        = 20     # how many past turns of this session to reload
HISTORY_TOKEN_BUDGET = 500   # summarize older turns once history exceeds this, # low for the demo; production would be much higher


# ── 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"

#  Create the BedrockModel instance
model = BedrockModel(model_id = model_id, region_name = REGION) # Replace this line

#  Create the MemoryClient instance
memory_client = MemoryClient(region_name = REGION) # Replace this line

# Create the boto3 bedrock-agent-runtime client
_bedrock_runtime =  boto3.client("bedrock-agent-runtime", region_name = REGION) # Replace this line

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
    tier_discount_rate: str
    tier_discount: float = Field(ge=0)
    final_total: float = Field(ge=0)
    total_savings: float = Field(ge=0)
    points_earned: int = Field(ge=0)
    remaining_points: int = Field(ge=0)

    @model_validator(mode="after")
    def check_arithmetic(self):
        if self.points_redeemed % 500:
            raise ValueError("points must be redeemed in blocks of 500")
        if self.points_discount > self.order_total * 0.5 + 0.01:
            raise ValueError("points discount exceeds 50% of the order")
        expected = round(self.order_total - self.points_discount - self.tier_discount, 2)
        if abs(expected - self.final_total) > 0.01:
            raise ValueError(f"final_total {self.final_total} != expected {expected}")
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


# ──  4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Read the namespace from strategy["namespaceTemplates"][0].
#      For compatibility with older AgentCore responses, fall back to
#      strategy["namespaces"][0] when namespaceTemplates is absent.
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    #  Implement this function
    strategies =  mem_client.get_memory_strategies(memory_id)
    namespaces = {}
    for strategy in strategies:
        strategy_type = strategy.get("type") or strategy.get("memoryStrategyType")
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces")
        if strategy_type and templates:
            namespaces[strategy_type] =  templates[0]
    return namespaces       



# ── 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

def _message_text(message: Dict) -> str:
    """Return the first plain-text block of a message, or "" if there is none."""
    for block in message.get("content", []):
        if "text" in block:
            return block["text"]
    return ""


def _is_tool_result(message: Dict) -> bool:
    return any("toolResult" in block for block in message.get("content", []))

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        # Store actor_id, session_id, memory_id, memory_client as attributes
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_id =  memory_id
        self.memory_client = memory_client
        # Call get_namespaces() and store the result as self.namespaces
        self.namespaces = get_namespaces(mem_client=memory_client, memory_id=memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        # Implement memory retrieval
        # Steps:
        #   1. Get the last message from event.agent.messages
        #   2. Check it is a user message and not a tool result
        #   3. Extract the user query text
        #   4. For each namespace in self.namespaces, call retrieve_memories()
        #   5. Collect non-empty memory texts with strategy type tags
        #   6. If any found, prepend them to the user message
        messages = event.agent.messages
        if not messages:
            return
        last = messages[-1]
        if last.get("role") !=  "user" or _is_tool_result(last):
            return

        user_query = _message_text(last)
        if not user_query:
            return
        try:
            context_lines = []
            for strategy_type, namespace in self.namespaces.items():
                memories =  self.memory_client.retrieve_memories(
                    memory_id = self.memory_id,
                    namespace = namespace.format(actorId=self.actor_id),
                    query=user_query,
                    top_k=5,
                )
                for memory in memories:
                    text = memory.get("content", {}).get("text","").strip()
                    if text:
                        context_lines.append(f"[{strategy_type}]{text}")
            if context_lines:
                context =  "\n".join(context_lines)
                for block in last["content"]:
                    if "text" in block:
                        block["text"] = f"Customer Context:\n{context}\n\n{user_query}"
                        break
        except Exception as e:
            logger.error(f"Memory retriveal failed:{e}") 

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        # Implement memory saving
        # Steps:
        #   1. Get messages from event.agent.messages
        #   2. Walk backwards to find the last user query (plain text)
        #      and the last assistant response
        #   3. Call memory_client.create_event() with both messages
        try:
            messages = event.agent.messages
            customer_query = None
            agent_response = None

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
        """Register both memory callbacks."""

        # Register retrieve_customer_context on MessageAddedEvent
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
                
        # Register save_support_interaction on AfterInvocationEvent
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)

class ToolResultValidationHook(HookProvider):
    """Validate gateway tool results against Pydantic models before the model sees them."""

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


# ── 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

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
    # Implement the Knowledge Base search
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."

        return "\n---\n".join(r["content"]["text"] for r in results)
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return f"Knowledge base search failed: {e}"


# ── 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

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
    #  Build the code string (use an f-string to inject the arguments)
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
    "tier_discount_rate": f"{int(tier_rate * 100)}%",
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
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
                stdout = event["result"].get("structuredContent", {}).get("stdout", "")
                breakdown = DiscountBreakdown.model_validate_json(stdout)
                return breakdown.model_dump_json()

        return json.dumps({"error": "Code Interpreter returned no result."})

    except Exception as e:
        logger.error(f"Code Interpreter failed, using fallback: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        rate = tier_rates.get(tier.strip().capitalize(), 0.0)
        tier_discount = round(order_total * rate, 2)
        return json.dumps({
            "note": "Code Interpreter unavailable; only the tier discount was applied.",
            "order_total": order_total,
            "tier": tier,
            "tier_discount_rate": f"{int(rate * 100)}%",
            "tier_discount": tier_discount,
            "final_total": round(order_total - tier_discount, 2),
        })
 


# ── 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

# ── Session history and summarization ────────────────────────────────────────

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

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    # Implement the agent invocation
    try:
        # 1. Read the request
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous-customer")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        if not user_input:
            return "Please provide a prompt."

        history = load_session_history(actor_id, session_id)

        # 2. Long-term memory for this customer and session
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # 3. Browser tool (runs in an AgentCore sandbox, not locally)
        agent_core_browser = AgentCoreBrowser(region=REGION)

        # 4. Local tools
        tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

        system_prompt = f"""You are a friendly, concise customer support agent for an Amazon store.
The current customer's ID is {actor_id}.

Use your tools instead of guessing:
- Orders, order history, and customer profile/tier/points: use the order tools.
- Refunds, refund status, and return labels: use the refund tools.
- Products, return policies, warranties, and loyalty rules: use search_knowledge_base.
- Any loyalty discount or price calculation: use calculate_loyalty_discount. Never do the math yourself.
- Any URL or website request (any site, not only Amazon): you DO have live web access.
  Always use the browser tool to open the page and read it. Never claim you cannot browse.
- Never state policies, tier benefits, or prices from your own knowledge; only from tool results.


If a message starts with "Customer Context:", use it to personalise your answer,
but do not repeat it back verbatim. If you cannot find an answer, say so honestly."""

        # 5. Connect to the AgentCore Gateway (order and refund tools over MCP)
        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))

        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()
            tools.extend(gateway_tools)

            # 6. Create and run the agent while the gateway connection is open
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook,ToolResultValidationHook()],
                system_prompt=system_prompt,
                messages=history,
                conversation_manager=SummarizingConversationManager(
                    summary_ratio=0.5,
                    preserve_recent_messages=4,
                ),
            )
            compact_history(agent)
            result = await agent.invoke_async(user_input)


        # 7. Return the text of the final answer
        return _message_text(result.message) or str(result)

    except Exception as e:
        # 8. Never crash the endpoint; return a friendly message instead
        logger.exception(f"Agent invocation failed: {e}")
        return "Sorry, I ran into a problem handling your request. Please try again."

    


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
