"""
ReAct Agent — Procurement Assistant
=====================================
Phase 1: Mock tools (fake data — no real API connection)
Phase 2: Replace mock function bodies with real API calls

Tools covered:
  - Procurement Cycle
  - Vendor & Supplier Management
  - Invoices & Financials
"""

import json
import random
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────────────────────

MOCK_SUPPLIERS = [
    {"id": 1, "name": "Ahmed Co",       "currency": "EGP", "rating": 4.5, "status": "active"},
    {"id": 2, "name": "Nile Trading",   "currency": "EGP", "rating": 3.8, "status": "active"},
    {"id": 3, "name": "Cairo Supplies", "currency": "EGP", "rating": 4.2, "status": "active"},
]

MOCK_ITEMS = [
    {"id": 101, "name": "كرسي مكتبي",  "unit_price": 250.0,  "unit": "piece", "stock": 200},
    {"id": 102, "name": "طاولة مكتبية","unit_price": 800.0,  "unit": "piece", "stock": 50},
    {"id": 103, "name": "ورق A4",       "unit_price": 45.0,   "unit": "ream",  "stock": 1000},
    {"id": 104, "name": "حاسوب محمول", "unit_price": 8500.0, "unit": "piece", "stock": 30},
]

MOCK_POS = [
    {
        "id": "PO-2026-001",
        "supplier": "Ahmed Co",
        "status": "pending_approval",
        "total": 12500.0,
        "currency": "EGP",
        "items": [{"name": "كرسي مكتبي", "qty": 50, "unit_price": 250.0}],
        "created_at": "2026-03-01"
    },
    {
        "id": "PO-2026-002",
        "supplier": "Nile Trading",
        "status": "approved",
        "total": 4500.0,
        "currency": "EGP",
        "items": [{"name": "ورق A4", "qty": 100, "unit_price": 45.0}],
        "created_at": "2026-03-05"
    },
]

MOCK_INVOICES = [
    {
        "id": "INV-2026-001",
        "po_id": "PO-2026-002",
        "supplier": "Nile Trading",
        "amount": 4500.0,
        "currency": "EGP",
        "status": "paid",
        "due_date": "2026-02-15",
        "paid_date": "2026-02-10"
    },
    {
        "id": "INV-2026-002",
        "po_id": "PO-2026-003",
        "supplier": "Cairo Supplies",
        "amount": 25500.0,
        "currency": "EGP",
        "status": "pending",
        "due_date": "2026-03-20",
        "paid_date": None
    },
]


# ─────────────────────────────────────────────────────────────
# TOOLS — ITEMS
# ─────────────────────────────────────────────────────────────

@tool
def get_items(name: str = "") -> str:
    """
    Get list of available items/products from the catalog.
    Optionally filter by name (partial match, supports Arabic and English).
    Returns item ID, name, unit price, unit type, and stock quantity.
    Use this to find item IDs before creating a purchase order.
    """
    results = MOCK_ITEMS
    if name:
        results = [i for i in MOCK_ITEMS if name.lower() in i["name"].lower() or name in i["name"]]
    return json.dumps(results, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# TOOLS — PROCUREMENT CYCLE
# ─────────────────────────────────────────────────────────────

@tool
def get_purchase_orders(status: str = "") -> str:
    """
    Get list of purchase orders.
    Optionally filter by status: pending_approval, approved, rejected, cancelled, completed.
    Leave status empty to get all purchase orders.
    Returns PO ID, supplier, status, total amount, and items.
    """
    results = MOCK_POS
    if status:
        results = [po for po in MOCK_POS if po["status"] == status]
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_po_details(po_id: str) -> str:
    """
    Get full details of a specific purchase order by its ID (e.g. PO-2026-001).
    Returns all fields including supplier, items, quantities, total, and status.
    """
    po = next((p for p in MOCK_POS if p["id"] == po_id), None)
    if not po:
        return json.dumps({"error": f"Purchase order {po_id} not found"})
    return json.dumps(po, ensure_ascii=False, indent=2)


@tool
def create_purchase_order(supplier_id: int, item_ids: str, quantities: str) -> str:
    """
    Preview a new purchase order BEFORE saving it.
    Always call this first to show the user a summary before confirming.

    Parameters:
    - supplier_id: numeric supplier ID (get it from get_suppliers first)
    - item_ids: comma-separated item IDs e.g. "101,102"
    - quantities: comma-separated quantities matching item_ids e.g. "50,10"

    Returns a preview with total amount. User must confirm before it is saved.
    """
    supplier = next((s for s in MOCK_SUPPLIERS if s["id"] == supplier_id), None)
    if not supplier:
        return json.dumps({"error": f"Supplier ID {supplier_id} not found. Use get_suppliers to find the correct ID."})

    try:
        ids  = [int(x.strip()) for x in item_ids.split(",")]
        qtys = [int(x.strip()) for x in quantities.split(",")]
    except ValueError:
        return json.dumps({"error": "item_ids and quantities must be comma-separated numbers"})

    order_items = []
    total = 0.0
    for item_id, qty in zip(ids, qtys):
        item = next((i for i in MOCK_ITEMS if i["id"] == item_id), None)
        if not item:
            return json.dumps({"error": f"Item ID {item_id} not found. Use get_items to find the correct ID."})
        subtotal = item["unit_price"] * qty
        total += subtotal
        order_items.append({
            "item_id":    item_id,
            "name":       item["name"],
            "qty":        qty,
            "unit_price": item["unit_price"],
            "subtotal":   subtotal
        })

    return json.dumps({
        "status":  "PREVIEW — NOT SAVED YET",
        "supplier": {"id": supplier_id, "name": supplier["name"]},
        "items":   order_items,
        "total":   total,
        "currency": supplier["currency"],
        "action_required": "Ask the user to confirm before calling confirm_create_purchase_order"
    }, ensure_ascii=False, indent=2)


@tool
def confirm_create_purchase_order(supplier_id: int, item_ids: str, quantities: str) -> str:
    """
    Actually save the purchase order after the user explicitly confirms.
    Use this ONLY when the user says 'yes', 'نعم', 'confirm', or similar confirmation.
    Do NOT call this without user confirmation first.

    Same parameters as create_purchase_order:
    - supplier_id: numeric supplier ID
    - item_ids: comma-separated item IDs
    - quantities: comma-separated quantities
    """
    supplier = next((s for s in MOCK_SUPPLIERS if s["id"] == supplier_id), None)
    if not supplier:
        return json.dumps({"error": f"Supplier ID {supplier_id} not found"})

    try:
        ids  = [int(x.strip()) for x in item_ids.split(",")]
        qtys = [int(x.strip()) for x in quantities.split(",")]
    except ValueError:
        return json.dumps({"error": "item_ids and quantities must be comma-separated numbers"})

    order_items = []
    total = 0.0
    for item_id, qty in zip(ids, qtys):
        item = next((i for i in MOCK_ITEMS if i["id"] == item_id), None)
        if not item:
            return json.dumps({"error": f"Item ID {item_id} not found"})
        subtotal = item["unit_price"] * qty
        total += subtotal
        order_items.append({
            "item_id":    item_id,
            "name":       item["name"],
            "qty":        qty,
            "unit_price": item["unit_price"],
            "subtotal":   subtotal
        })

    po_number = f"PO-2026-{random.randint(100, 999)}"
    new_po = {
        "id":       po_number,
        "supplier": supplier["name"],
        "status":   "pending_approval",
        "total":    total,
        "currency": supplier["currency"],
        "items":    order_items
    }
    MOCK_POS.append(new_po)   # ← actually save to the list

    return json.dumps({
        "status":  "created",
        "po_id":   po_number,
        "message": f"✅ تم إنشاء أمر التوريد {po_number} بنجاح — الحالة: في انتظار الموافقة"
    }, ensure_ascii=False, indent=2)


@tool
def cancel_po(po_id: str, reason: str) -> str:
    """
    Cancel a purchase order.
    Only orders with status 'pending_approval' can be cancelled.

    Parameters:
    - po_id: the PO ID to cancel (e.g. PO-2026-001)
    - reason: reason for cancellation (required)
    """
    po = next((p for p in MOCK_POS if p["id"] == po_id), None)
    if not po:
        return json.dumps({"error": f"Purchase order {po_id} not found"})
    if po["status"] != "pending_approval":
        return json.dumps({"error": f"Cannot cancel — PO status is '{po['status']}'. Only pending_approval orders can be cancelled."})
    po["status"] = "cancelled"   # ← actually update the status in the list
    return json.dumps({
        "status":  "cancelled",
        "po_id":   po_id,
        "reason":  reason,
        "message": f"✅ تم إلغاء أمر التوريد {po_id}"
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# TOOLS — VENDOR & SUPPLIER MANAGEMENT
# ─────────────────────────────────────────────────────────────

@tool
def get_suppliers(name: str = "") -> str:
    """
    Get list of all suppliers. Optionally filter by name (partial match).
    Returns supplier ID, name, currency, rating, and status.
    Always use this to find the supplier ID before creating a purchase order.
    """
    results = MOCK_SUPPLIERS
    if name:
        results = [s for s in MOCK_SUPPLIERS if name.lower() in s["name"].lower()]
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_supplier_details(supplier_id: int) -> str:
    """
    Get full profile of a specific supplier by their numeric ID.
    Returns name, currency, rating, and status.
    """
    supplier = next((s for s in MOCK_SUPPLIERS if s["id"] == supplier_id), None)
    if not supplier:
        return json.dumps({"error": f"Supplier ID {supplier_id} not found"})
    return json.dumps(supplier, ensure_ascii=False, indent=2)


@tool
def get_supplier_ratings(supplier_id: int) -> str:
    """
    Get performance rating history for a specific supplier.
    Returns overall rating, on-time delivery percentage, and quality score.
    """
    supplier = next((s for s in MOCK_SUPPLIERS if s["id"] == supplier_id), None)
    if not supplier:
        return json.dumps({"error": f"Supplier ID {supplier_id} not found"})
    return json.dumps({
        "supplier_id":      supplier_id,
        "name":             supplier["name"],
        "overall_rating":   supplier["rating"],
        "delivery_on_time": "92%",
        "quality_score":    4.3,
        "total_orders":     28,
        "last_review":      "2026-01-15"
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# TOOLS — INVOICES & FINANCIALS
# ─────────────────────────────────────────────────────────────

@tool
def get_invoices(status: str = "", supplier_name: str = "") -> str:
    """
    Get list of invoices. Optionally filter by:
    - status: paid, pending, overdue
    - supplier_name: partial name match
    Returns invoice ID, PO reference, supplier, amount, status, and due date.
    """
    results = MOCK_INVOICES
    if status:
        results = [i for i in results if i["status"] == status]
    if supplier_name:
        results = [i for i in results if supplier_name.lower() in i["supplier"].lower()]
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def get_invoice_details(invoice_id: str) -> str:
    """
    Get full details of a specific invoice by its ID (e.g. INV-2026-001).
    Returns all invoice fields including payment status and due date.
    """
    invoice = next((i for i in MOCK_INVOICES if i["id"] == invoice_id), None)
    if not invoice:
        return json.dumps({"error": f"Invoice {invoice_id} not found"})
    return json.dumps(invoice, ensure_ascii=False, indent=2)


@tool
def get_payment_status(invoice_id: str) -> str:
    """
    Check the payment status of a specific invoice.
    Returns whether it is paid, pending, or overdue, with the due date and paid date.
    """
    invoice = next((i for i in MOCK_INVOICES if i["id"] == invoice_id), None)
    if not invoice:
        return json.dumps({"error": f"Invoice {invoice_id} not found"})
    return json.dumps({
        "invoice_id": invoice_id,
        "status":     invoice["status"],
        "amount":     invoice["amount"],
        "currency":   invoice["currency"],
        "due_date":   invoice["due_date"],
        "paid_date":  invoice["paid_date"]
    }, ensure_ascii=False, indent=2)


@tool
def export_po_pdf(po_id: str) -> str:
    """
    Export a purchase order as a printable PDF document.
    Use this when the user asks to export, download, print, or get a PDF of a purchase order.
    Args:
        po_id: The purchase order ID (e.g., 'PO-2026-001')
    """
    po = next((p for p in MOCK_POS if p["id"] == po_id), None)
    if not po:
        po = next((p for p in MOCK_POS if po_id.lower() in p["id"].lower()), None)
    if not po:
        available = ", ".join(p["id"] for p in MOCK_POS)
        return f"لم يتم العثور على أمر التوريد '{po_id}'. الأوامر المتاحة: {available}"
    return "__PDF_EXPORT__" + json.dumps(po, ensure_ascii=False) + "__PDF_END__"


# ─────────────────────────────────────────────────────────────
# TOOLS LIST
# ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    # Items
    get_items,
    # Procurement
    get_purchase_orders,
    get_po_details,
    create_purchase_order,
    confirm_create_purchase_order,
    cancel_po,
    export_po_pdf,
    # Suppliers
    get_suppliers,
    get_supplier_details,
    get_supplier_ratings,
    # Invoices
    get_invoices,
    get_invoice_details,
    get_payment_status,
]

SYSTEM_PROMPT = """You are a smart procurement assistant for an ERP system.

You help users with:
- Purchase orders — إنشاء وتتبع أوامر التوريد
- Supplier management — إدارة الموردين والمقيمين
- Invoices and payments — الفواتير والمدفوعات

Rules you MUST follow:
1. Always call get_suppliers first to find the supplier ID before creating a PO.
2. Always call get_items first to find item IDs and prices before creating a PO.
3. For CREATE operations: call create_purchase_order to show a preview first. Then STOP and ask the user to confirm. Only call confirm_create_purchase_order after the user says yes/نعم/confirm.
4. For CANCEL operations: confirm with the user before calling cancel_po.
5. When the user asks for a PDF of a purchase order, call export_po_pdf with the PO ID.
6. Reply in the same language as the user. Arabic question → Arabic answer. English → English.
7. Always show amounts with currency (EGP).
8. If you are missing information, ask the user before proceeding.
9. FORMATTING: When returning a list of 2 or more items (suppliers, products, POs, invoices), always format the response as a Markdown table with clear column headers. Example for suppliers: | الاسم | الرقم | الدولة | العملة | \n |---|---|---|---| \n | ... |
"""


# ─────────────────────────────────────────────────────────────
# AGENT BUILDER & RUNNER
# ─────────────────────────────────────────────────────────────

def build_agent(llm):
    """Build the ReAct agent with all procurement tools."""
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT
    )


# Cached quota-exhausted state — avoids 30s Google SDK retry loop on every request
_quota_exhausted_until = 0.0   # Unix timestamp; 0 means not exhausted


def run_agent(agent, message: str, history: list = None) -> dict:
    """
    Run the ReAct agent.

    Args:
        agent:   compiled ReAct agent from build_agent()
        message: current user message
        history: list of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        {"answer": str, "tool_calls": int}
    """
    from langchain_core.messages import HumanMessage, AIMessage

    messages = []

    if history:
        for h in history:
            if h.get("role") == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h.get("role") == "assistant":
                messages.append(AIMessage(content=h["content"]))

    messages.append(HumanMessage(content=message))

    import time, re
    global _quota_exhausted_until

    # Fast-fail if daily quota is known to be exhausted — skips the 30s Google SDK retry loop
    if time.time() < _quota_exhausted_until:
        return {
            "answer": "انتهى الحد اليومي للـ API. يرجى تفعيل الفوترة أو انتظار إعادة الضبط منتصف الليل.",
            "tool_calls": 0
        }

    for attempt in range(3):
        try:
            result = agent.invoke({"messages": messages})
            final  = result["messages"][-1]

            # Gemini may return content as a list of blocks — extract text
            content = final.content
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()

            tool_calls = sum(1 for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls)
            tools_used = []
            for m in result["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        tools_used.append(tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown"))
            return {"answer": content, "tool_calls": tool_calls, "tools_used": tools_used}

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Daily quota exhausted — cache state so next requests fail instantly
                if "PerDay" in err or "limit: 0" in err:
                    _quota_exhausted_until = time.time() + 86400  # cache for 24 hours
                    logger.error("Daily quota exhausted — cached, fast-failing until quota resets.")
                    return {
                        "answer": "انتهى الحد اليومي للـ API. يرجى تفعيل الفوترة أو انتظار إعادة الضبط منتصف الليل.",
                        "tool_calls": 0
                    }
                # Per-minute rate limit — wait and retry
                match = re.search(r'retryDelay["\s:]+(\d+)', err)
                wait  = int(match.group(1)) + 2 if match else 15
                logger.warning(f"Rate limited — waiting {wait}s before retry {attempt + 1}/3")
                time.sleep(wait)
                continue
            logger.error(f"Agent run failed: {e}")
            return {"answer": f"Agent error: {str(e)}", "tool_calls": 0}

    return {"answer": "النظام مشغول مؤقتاً. حاول مرة أخرى بعد لحظات.", "tool_calls": 0}
