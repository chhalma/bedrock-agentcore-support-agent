"""Demonstrate the Pydantic tool-result validation in main.py.

Feeds good and deliberately malformed tool results through the same
ToolResultValidationHook and DiscountBreakdown model the agent uses.
No AWS calls are made.

Run from starter/:
  uv run test_validation.py
"""
import json
import logging
from types import SimpleNamespace

from pydantic import ValidationError

import main

logging.getLogger("CSAI_Agent").setLevel(logging.ERROR)  # keep the demo output clean

hook = main.ToolResultValidationHook()

GOOD_ORDER = {
    "order_id": "ORD-001", "customer_id": "CUST-123", "status": "SHIPPED",
    "items": [{"name": "Wireless Headphones Pro", "qty": 1, "price": 89.99}],
    "total": 89.99, "tracking_number": "TRK987654321", "carrier": "UPS",
}
GOOD_REFUND = {
    "refund_id": "REF-R3ZXRKFH", "order_id": "ORD-002", "status": "APPROVED",
    "amount": 139.99, "message": "Refund approved. Credit appears in 3-5 business days.",
}
GOOD_DISCOUNT = {
    "order_total": 150.0, "tier": "Gold", "points_redeemed": 4000, "points_discount": 40.0,
    "tier_discount_pct": 10.0, "tier_discount": 11.0, "final_total": 99.0,
    "total_savings": 51.0, "points_earned": 99, "remaining_points": 250,
}


def check_tool(label, tool_name, payload):
    """Run a fake gateway tool result through the validation hook."""
    event = SimpleNamespace(
        tool_use={"name": tool_name},
        result={"toolUseId": "demo", "status": "success",
                "content": [{"text": json.dumps(payload)}]},
    )
    hook.validate_tool_result(event)
    verdict = "PASSED  " if event.result["status"] == "success" else "WITHHELD"
    print(f"  {verdict} {label}")


def check_discount(label, payload):
    """Validate a Code Interpreter discount breakdown, including its arithmetic."""
    try:
        main.DiscountBreakdown.model_validate(payload)
        print(f"  PASSED   {label}")
    except ValidationError as e:
        print(f"  REJECTED {label}  ({e.errors()[0]['msg']})")


print("\nGateway tool results (ToolResultValidationHook):")
check_tool("valid order ORD-001", "order-tracker___get_order", GOOD_ORDER)
check_tool("order with unknown status 'LOST_IN_SPACE'", "order-tracker___get_order",
           {**GOOD_ORDER, "status": "LOST_IN_SPACE"})
check_tool("order with negative total", "order-tracker___get_order",
           {**GOOD_ORDER, "total": -5})
check_tool("valid refund (Lambda body wrapper)", "refund-processor___initiate_refund",
           {"statusCode": 200, "body": json.dumps(GOOD_REFUND)})
check_tool("refund with malformed ID 'oops'", "refund-processor___initiate_refund",
           {"statusCode": 200, "body": json.dumps({**GOOD_REFUND, "refund_id": "oops"})})
check_tool("business error 'Order not found' (passed through)", "order-tracker___get_order",
           {"error": "Order ORD-999 not found"})

print("\nCode Interpreter discount breakdown (DiscountBreakdown):")
check_discount("valid breakdown, final_total $99.00", GOOD_DISCOUNT)
check_discount("wrong arithmetic, final_total $120.00", {**GOOD_DISCOUNT, "final_total": 120.0})
check_discount("points not in blocks of 500 (4250)", {**GOOD_DISCOUNT, "points_redeemed": 4250})
print()
