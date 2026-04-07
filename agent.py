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
# MOCK DATA — RESTAURANT INVENTORY & RECIPES
# ─────────────────────────────────────────────────────────────

MOCK_INVENTORY = [
    {"id": "INV-001", "name": "طماطم",       "unit": "kg",    "current_stock": 12,  "min_level": 20,  "cost_per_unit": 15.0,  "supplier": "Ahmed Co"},
    {"id": "INV-002", "name": "جبنة موزاريلا","unit": "kg",    "current_stock": 8,   "min_level": 10,  "cost_per_unit": 120.0, "supplier": "Nile Trading"},
    {"id": "INV-003", "name": "عجينة بيتزا",  "unit": "piece", "current_stock": 45,  "min_level": 30,  "cost_per_unit": 8.0,   "supplier": "Cairo Supplies"},
    {"id": "INV-004", "name": "دجاج",         "unit": "kg",    "current_stock": 25,  "min_level": 15,  "cost_per_unit": 95.0,  "supplier": "Ahmed Co"},
    {"id": "INV-005", "name": "أرز",          "unit": "kg",    "current_stock": 5,   "min_level": 25,  "cost_per_unit": 30.0,  "supplier": "Nile Trading"},
    {"id": "INV-006", "name": "زيت زيتون",    "unit": "liter", "current_stock": 18,  "min_level": 10,  "cost_per_unit": 85.0,  "supplier": "Cairo Supplies"},
    {"id": "INV-007", "name": "بصل",          "unit": "kg",    "current_stock": 3,   "min_level": 15,  "cost_per_unit": 12.0,  "supplier": "Ahmed Co"},
    {"id": "INV-008", "name": "فلفل أخضر",    "unit": "kg",    "current_stock": 7,   "min_level": 8,   "cost_per_unit": 20.0,  "supplier": "Ahmed Co"},
    {"id": "INV-009", "name": "لحم مفروم",    "unit": "kg",    "current_stock": 4,   "min_level": 12,  "cost_per_unit": 180.0, "supplier": "Nile Trading"},
    {"id": "INV-010", "name": "خبز",          "unit": "piece", "current_stock": 100, "min_level": 50,  "cost_per_unit": 3.0,   "supplier": "Cairo Supplies"},
]

# Recipes: each recipe has a name, category, selling_price, and ingredients (BOM)
MOCK_RECIPES = [
    {
        "id": "REC-001",
        "name": "بيتزا مارغريتا",
        "category": "بيتزا",
        "selling_price": 85.0,
        "ingredients": [
            {"inventory_id": "INV-001", "name": "طماطم",        "qty_per_unit": 0.15, "unit": "kg"},
            {"inventory_id": "INV-002", "name": "جبنة موزاريلا", "qty_per_unit": 0.12, "unit": "kg"},
            {"inventory_id": "INV-003", "name": "عجينة بيتزا",   "qty_per_unit": 1,    "unit": "piece"},
            {"inventory_id": "INV-006", "name": "زيت زيتون",     "qty_per_unit": 0.02, "unit": "liter"},
        ]
    },
    {
        "id": "REC-002",
        "name": "بيتزا دجاج",
        "category": "بيتزا",
        "selling_price": 110.0,
        "ingredients": [
            {"inventory_id": "INV-001", "name": "طماطم",        "qty_per_unit": 0.15, "unit": "kg"},
            {"inventory_id": "INV-002", "name": "جبنة موزاريلا", "qty_per_unit": 0.15, "unit": "kg"},
            {"inventory_id": "INV-003", "name": "عجينة بيتزا",   "qty_per_unit": 1,    "unit": "piece"},
            {"inventory_id": "INV-004", "name": "دجاج",          "qty_per_unit": 0.1,  "unit": "kg"},
            {"inventory_id": "INV-008", "name": "فلفل أخضر",     "qty_per_unit": 0.05, "unit": "kg"},
        ]
    },
    {
        "id": "REC-003",
        "name": "أرز باللحم",
        "category": "أطباق رئيسية",
        "selling_price": 95.0,
        "ingredients": [
            {"inventory_id": "INV-005", "name": "أرز",       "qty_per_unit": 0.25, "unit": "kg"},
            {"inventory_id": "INV-009", "name": "لحم مفروم", "qty_per_unit": 0.15, "unit": "kg"},
            {"inventory_id": "INV-007", "name": "بصل",       "qty_per_unit": 0.08, "unit": "kg"},
            {"inventory_id": "INV-006", "name": "زيت زيتون",  "qty_per_unit": 0.03, "unit": "liter"},
        ]
    },
    {
        "id": "REC-004",
        "name": "ساندويتش دجاج",
        "category": "ساندويتشات",
        "selling_price": 65.0,
        "ingredients": [
            {"inventory_id": "INV-004", "name": "دجاج",      "qty_per_unit": 0.12, "unit": "kg"},
            {"inventory_id": "INV-010", "name": "خبز",       "qty_per_unit": 1,    "unit": "piece"},
            {"inventory_id": "INV-001", "name": "طماطم",     "qty_per_unit": 0.05, "unit": "kg"},
            {"inventory_id": "INV-007", "name": "بصل",       "qty_per_unit": 0.03, "unit": "kg"},
        ]
    },
]

# Simulated yesterday's sales (number of units sold per recipe)
MOCK_DAILY_SALES = [
    {"recipe_id": "REC-001", "name": "بيتزا مارغريتا",  "qty_sold": 50},
    {"recipe_id": "REC-002", "name": "بيتزا دجاج",      "qty_sold": 35},
    {"recipe_id": "REC-003", "name": "أرز باللحم",       "qty_sold": 40},
    {"recipe_id": "REC-004", "name": "ساندويتش دجاج",   "qty_sold": 60},
]

# Delivery log: what was ordered vs what was actually received
MOCK_DELIVERIES = [
    {"id": "DEL-001", "date": "2026-04-01", "supplier": "Ahmed Co",       "item": "طماطم",   "ordered_qty": 30, "received_qty": 28, "unit": "kg"},
    {"id": "DEL-002", "date": "2026-04-01", "supplier": "Nile Trading",   "item": "جبنة موزاريلا", "ordered_qty": 15, "received_qty": 15, "unit": "kg"},
    {"id": "DEL-003", "date": "2026-04-02", "supplier": "Ahmed Co",       "item": "دجاج",    "ordered_qty": 20, "received_qty": 18, "unit": "kg"},
    {"id": "DEL-004", "date": "2026-04-02", "supplier": "Cairo Supplies", "item": "عجينة بيتزا", "ordered_qty": 60, "received_qty": 60, "unit": "piece"},
    {"id": "DEL-005", "date": "2026-04-03", "supplier": "Nile Trading",   "item": "أرز",     "ordered_qty": 40, "received_qty": 38, "unit": "kg"},
    {"id": "DEL-006", "date": "2026-04-03", "supplier": "Ahmed Co",       "item": "بصل",     "ordered_qty": 20, "received_qty": 20, "unit": "kg"},
]

# Waste log: items thrown away with reason
MOCK_WASTE_LOG = [
    {"id": "WST-001", "date": "2026-04-01", "item": "طماطم",        "qty": 2.0, "unit": "kg", "reason": "منتهية الصلاحية"},
    {"id": "WST-002", "date": "2026-04-01", "item": "خبز",          "qty": 15,  "unit": "piece", "reason": "جفاف"},
    {"id": "WST-003", "date": "2026-04-02", "item": "دجاج",         "qty": 1.5, "unit": "kg", "reason": "تلف بسبب التخزين"},
    {"id": "WST-004", "date": "2026-04-03", "item": "جبنة موزاريلا","qty": 0.5, "unit": "kg", "reason": "منتهية الصلاحية"},
    {"id": "WST-005", "date": "2026-04-03", "item": "فلفل أخضر",    "qty": 1.0, "unit": "kg", "reason": "تلف"},
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


@tool
def export_report_pdf(report_type: str) -> str:
    """
    Export any report as a printable PDF. Call this when the user asks to PRINT, EXPORT, or DOWNLOAD any report.
    IMPORTANT: You MUST call this tool — never say you can't print. The UI depends on this tool's output.

    report_type must be one of:
    - 'waste' — waste/spoilage report
    - 'stock' — current stock levels report
    - 'low_stock' — low stock alerts only
    - 'consumption' — ingredient consumption from yesterday's sales
    - 'sales' — daily sales report
    - 'deliveries' — delivery log report
    - 'recipes' — recipe/menu list with ingredients
    """
    from datetime import date
    report_data = {"type": report_type, "date": str(date.today()), "title": "", "rows": [], "summary": {}}

    if report_type == "waste":
        report_data["title"] = "تقرير الهدر والتالف"
        total_cost = 0.0
        for w in MOCK_WASTE_LOG:
            inv = next((i for i in MOCK_INVENTORY if i["name"] == w["item"]), None)
            cost = round(w["qty"] * inv["cost_per_unit"], 2) if inv else 0
            total_cost += cost
            report_data["rows"].append({"التاريخ": w["date"], "الصنف": w["item"], "الكمية": f"{w['qty']} {w['unit']}", "السبب": w["reason"], "التكلفة": f"{cost} EGP"})
        report_data["summary"] = {"إجمالي التكلفة": f"{round(total_cost, 2)} EGP", "عدد الحوادث": len(MOCK_WASTE_LOG)}

    elif report_type == "stock":
        report_data["title"] = "تقرير حالة المخزون"
        for item in MOCK_INVENTORY:
            status = "⚠️ منخفض" if item["current_stock"] < item["min_level"] else "✅ جيد"
            report_data["rows"].append({"الصنف": item["name"], "المخزون": f"{item['current_stock']} {item['unit']}", "الحد الأدنى": f"{item['min_level']} {item['unit']}", "المورد": item["supplier"], "الحالة": status})

    elif report_type == "low_stock":
        report_data["title"] = "تنبيهات المخزون المنخفض"
        total = 0.0
        for item in MOCK_INVENTORY:
            if item["current_stock"] < item["min_level"]:
                deficit = item["min_level"] - item["current_stock"]
                order_qty = round(deficit * 1.5)
                cost = round(order_qty * item["cost_per_unit"], 2)
                total += cost
                report_data["rows"].append({"الصنف": item["name"], "المخزون": f"{item['current_stock']} {item['unit']}", "النقص": f"{deficit} {item['unit']}", "الكمية المقترحة": f"{order_qty} {item['unit']}", "التكلفة": f"{cost} EGP", "المورد": item["supplier"]})
        report_data["summary"] = {"إجمالي التكلفة المتوقعة": f"{round(total, 2)} EGP"}

    elif report_type == "consumption":
        report_data["title"] = "تقرير استهلاك المكونات"
        consumption = {}
        total_rev, total_cost = 0.0, 0.0
        for sale in MOCK_DAILY_SALES:
            recipe = next((r for r in MOCK_RECIPES if r["id"] == sale["recipe_id"]), None)
            if not recipe: continue
            total_rev += recipe["selling_price"] * sale["qty_sold"]
            for ing in recipe["ingredients"]:
                used = round(ing["qty_per_unit"] * sale["qty_sold"], 2)
                inv = next((i for i in MOCK_INVENTORY if i["id"] == ing["inventory_id"]), None)
                cost = round(used * inv["cost_per_unit"], 2) if inv else 0
                total_cost += cost
                key = ing["inventory_id"]
                if key not in consumption:
                    consumption[key] = {"name": ing["name"], "used": 0, "unit": ing["unit"], "cost": 0}
                consumption[key]["used"] = round(consumption[key]["used"] + used, 2)
                consumption[key]["cost"] = round(consumption[key]["cost"] + cost, 2)
        for data in consumption.values():
            report_data["rows"].append({"المكون": data["name"], "الكمية المستهلكة": f"{data['used']} {data['unit']}", "التكلفة": f"{data['cost']} EGP"})
        report_data["summary"] = {"إجمالي الإيرادات": f"{total_rev} EGP", "تكلفة المكونات": f"{round(total_cost, 2)} EGP", "هامش الربح": f"{round((total_rev - total_cost) / total_rev * 100, 1)}%"}

    elif report_type == "sales":
        report_data["title"] = "تقرير المبيعات اليومية"
        total_rev = 0.0
        for sale in MOCK_DAILY_SALES:
            recipe = next((r for r in MOCK_RECIPES if r["id"] == sale["recipe_id"]), None)
            rev = recipe["selling_price"] * sale["qty_sold"] if recipe else 0
            total_rev += rev
            report_data["rows"].append({"الصنف": sale["name"], "الكمية": sale["qty_sold"], "سعر الوحدة": f"{recipe['selling_price']} EGP" if recipe else "?", "الإيرادات": f"{rev} EGP"})
        report_data["summary"] = {"إجمالي الإيرادات": f"{total_rev} EGP"}

    elif report_type == "deliveries":
        report_data["title"] = "تقرير التوريدات"
        for d in MOCK_DELIVERIES:
            shortage = d["ordered_qty"] - d["received_qty"]
            report_data["rows"].append({"التاريخ": d["date"], "المورد": d["supplier"], "الصنف": d["item"], "المطلوب": f"{d['ordered_qty']} {d['unit']}", "المستلم": f"{d['received_qty']} {d['unit']}", "النقص": f"{shortage} {d['unit']}" if shortage > 0 else "✅"})

    elif report_type == "recipes":
        report_data["title"] = "قائمة الوصفات والمكونات"
        for r in MOCK_RECIPES:
            ings = " | ".join([f"{i['name']}: {i['qty_per_unit']} {i['unit']}" for i in r["ingredients"]])
            report_data["rows"].append({"الوصفة": r["name"], "التصنيف": r["category"], "السعر": f"{r['selling_price']} EGP", "المكونات": ings})

    else:
        return json.dumps({"error": f"نوع التقرير '{report_type}' غير معروف"}, ensure_ascii=False)

    return "__REPORT_PDF__" + json.dumps(report_data, ensure_ascii=False) + "__REPORT_END__"


# ─────────────────────────────────────────────────────────────
# FEATURE 1: STOCK CHECK — compare stock vs min levels
# ─────────────────────────────────────────────────────────────

@tool
def check_stock_levels(item_name: str = "") -> str:
    """
    Check current inventory stock levels against minimum thresholds.
    Returns ALL items if no name given, or filters by name.
    Items below minimum level are flagged as LOW STOCK with recommended order quantity.
    Use this to identify what needs reordering.
    """
    results = MOCK_INVENTORY
    if item_name:
        results = [i for i in MOCK_INVENTORY if item_name in i["name"]]

    report = []
    for item in results:
        status = "⚠️ LOW" if item["current_stock"] < item["min_level"] else "✅ OK"
        deficit = max(0, item["min_level"] - item["current_stock"])
        # Recommend ordering 1.5x the deficit to build buffer
        recommended_order = round(deficit * 1.5) if deficit > 0 else 0
        report.append({
            "id": item["id"],
            "name": item["name"],
            "current_stock": item["current_stock"],
            "min_level": item["min_level"],
            "unit": item["unit"],
            "status": status,
            "deficit": deficit,
            "recommended_order": recommended_order,
            "supplier": item["supplier"],
            "estimated_cost": round(recommended_order * item["cost_per_unit"], 2)
        })
    return json.dumps(report, ensure_ascii=False, indent=2)


@tool
def get_low_stock_alerts() -> str:
    """
    Get only items that are BELOW minimum stock level.
    Returns a prioritized alert list with recommended order quantities and estimated costs.
    This is what runs in the nightly stock check — use it to see what needs ordering NOW.
    """
    alerts = []
    total_cost = 0.0
    for item in MOCK_INVENTORY:
        if item["current_stock"] < item["min_level"]:
            deficit = item["min_level"] - item["current_stock"]
            order_qty = round(deficit * 1.5)
            cost = round(order_qty * item["cost_per_unit"], 2)
            total_cost += cost
            alerts.append({
                "item": item["name"],
                "current": item["current_stock"],
                "minimum": item["min_level"],
                "deficit": deficit,
                "order_qty": order_qty,
                "unit": item["unit"],
                "supplier": item["supplier"],
                "estimated_cost": f"{cost} EGP"
            })

    if not alerts:
        return json.dumps({"message": "✅ جميع الأصناف فوق الحد الأدنى — لا توجد تنبيهات"}, ensure_ascii=False)

    return json.dumps({
        "alert_count": len(alerts),
        "total_estimated_cost": f"{round(total_cost, 2)} EGP",
        "alerts": alerts
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# FEATURE 2: DRAFT PO GENERATOR — auto-create POs from alerts
# ─────────────────────────────────────────────────────────────

@tool
def generate_draft_po_from_alerts() -> str:
    """
    Automatically generate draft purchase orders from low-stock alerts.
    Groups items by supplier and creates one PO per supplier.
    Returns PO previews that need user confirmation before saving.
    This is the auto-PO generator that runs after the nightly stock check.
    """
    # Group low-stock items by supplier
    supplier_groups = {}
    for item in MOCK_INVENTORY:
        if item["current_stock"] < item["min_level"]:
            deficit = item["min_level"] - item["current_stock"]
            order_qty = round(deficit * 1.5)
            supplier = item["supplier"]
            if supplier not in supplier_groups:
                supplier_groups[supplier] = {"items": [], "total": 0.0}
            cost = round(order_qty * item["cost_per_unit"], 2)
            supplier_groups[supplier]["items"].append({
                "name": item["name"],
                "qty": order_qty,
                "unit": item["unit"],
                "unit_price": item["cost_per_unit"],
                "subtotal": cost
            })
            supplier_groups[supplier]["total"] += cost

    if not supplier_groups:
        return json.dumps({"message": "✅ لا توجد أصناف تحتاج طلب — المخزون كافٍ"}, ensure_ascii=False)

    draft_pos = []
    for supplier_name, data in supplier_groups.items():
        po_number = f"DRAFT-PO-{random.randint(1000, 9999)}"
        draft_pos.append({
            "draft_po_id": po_number,
            "supplier": supplier_name,
            "items": data["items"],
            "total": f"{round(data['total'], 2)} EGP",
            "status": "DRAFT — يحتاج موافقة"
        })

    return json.dumps({
        "message": f"تم إنشاء {len(draft_pos)} أوامر توريد مسودة",
        "draft_purchase_orders": draft_pos,
        "action_required": "قل 'نعم' لتأكيد أو 'لا' للإلغاء"
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# FEATURE 3: RECIPE-TO-INGREDIENT — calculate consumption
# ─────────────────────────────────────────────────────────────

@tool
def get_recipes(name: str = "") -> str:
    """
    Get all recipes (menu items) with their ingredients (Bill of Materials).
    Optionally filter by name. Each recipe shows what ingredients are needed
    and how much per unit sold.
    """
    results = MOCK_RECIPES
    if name:
        results = [r for r in MOCK_RECIPES if name in r["name"] or name in r["category"]]
    output = []
    for r in results:
        output.append({
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "selling_price": f"{r['selling_price']} EGP",
            "ingredients": [
                {"name": ing["name"], "qty_per_unit": ing["qty_per_unit"], "unit": ing["unit"]}
                for ing in r["ingredients"]
            ]
        })
    return json.dumps(output, ensure_ascii=False, indent=2)


@tool
def calculate_consumption(recipe_id: str = "", qty_sold: int = 0) -> str:
    """
    Calculate exact ingredient consumption based on how many units of a recipe were sold.
    If recipe_id and qty_sold are empty/zero, calculates for ALL of yesterday's sales.
    This is the core AI intelligence — it links sales to raw material consumption.

    Parameters:
    - recipe_id: optional, e.g. 'REC-001'. Leave empty to calculate for all sales.
    - qty_sold: number of units sold. Set to 0 to use yesterday's sales data.
    """
    if recipe_id and qty_sold > 0:
        recipe = next((r for r in MOCK_RECIPES if r["id"] == recipe_id), None)
        if not recipe:
            return json.dumps({"error": f"Recipe {recipe_id} not found"}, ensure_ascii=False)
        sales = [{"recipe_id": recipe_id, "name": recipe["name"], "qty_sold": qty_sold}]
    else:
        sales = MOCK_DAILY_SALES

    # Aggregate consumption across all ingredients
    consumption = {}
    total_revenue = 0.0
    total_cost = 0.0

    for sale in sales:
        recipe = next((r for r in MOCK_RECIPES if r["id"] == sale["recipe_id"]), None)
        if not recipe:
            continue
        total_revenue += recipe["selling_price"] * sale["qty_sold"]

        for ing in recipe["ingredients"]:
            key = ing["inventory_id"]
            used = round(ing["qty_per_unit"] * sale["qty_sold"], 2)
            inv_item = next((i for i in MOCK_INVENTORY if i["id"] == key), None)
            cost = round(used * inv_item["cost_per_unit"], 2) if inv_item else 0

            if key not in consumption:
                consumption[key] = {
                    "name": ing["name"],
                    "total_used": 0,
                    "unit": ing["unit"],
                    "total_cost": 0,
                    "current_stock": inv_item["current_stock"] if inv_item else "?",
                    "stock_after": inv_item["current_stock"] if inv_item else 0
                }
            consumption[key]["total_used"] = round(consumption[key]["total_used"] + used, 2)
            consumption[key]["total_cost"] = round(consumption[key]["total_cost"] + cost, 2)
            total_cost += cost

    # Calculate remaining stock
    result_items = []
    for key, data in consumption.items():
        remaining = round(data["current_stock"] - data["total_used"], 2) if isinstance(data["current_stock"], (int, float)) else "?"
        inv_item = next((i for i in MOCK_INVENTORY if i["id"] == key), None)
        min_lvl = inv_item["min_level"] if inv_item else 0
        status = "⚠️ سينفد" if isinstance(remaining, (int, float)) and remaining < min_lvl else "✅ كافٍ"
        result_items.append({
            "ingredient": data["name"],
            "consumed": f"{data['total_used']} {data['unit']}",
            "cost": f"{data['total_cost']} EGP",
            "stock_before": data["current_stock"],
            "stock_after": remaining,
            "status": status
        })

    return json.dumps({
        "period": "مبيعات أمس" if not (recipe_id and qty_sold > 0) else f"{qty_sold}x {sales[0]['name']}",
        "total_revenue": f"{round(total_revenue, 2)} EGP",
        "total_ingredient_cost": f"{round(total_cost, 2)} EGP",
        "profit_margin": f"{round((total_revenue - total_cost) / total_revenue * 100, 1)}%",
        "consumption": result_items
    }, ensure_ascii=False, indent=2)


@tool
def get_daily_sales() -> str:
    """
    Get yesterday's sales data — how many of each recipe/menu item were sold.
    Use this to understand what was sold before calculating ingredient consumption.
    """
    total_revenue = 0.0
    output = []
    for sale in MOCK_DAILY_SALES:
        recipe = next((r for r in MOCK_RECIPES if r["id"] == sale["recipe_id"]), None)
        revenue = recipe["selling_price"] * sale["qty_sold"] if recipe else 0
        total_revenue += revenue
        output.append({
            "recipe": sale["name"],
            "qty_sold": sale["qty_sold"],
            "unit_price": f"{recipe['selling_price']} EGP" if recipe else "?",
            "revenue": f"{revenue} EGP"
        })
    return json.dumps({
        "date": "2026-04-04 (أمس)",
        "total_revenue": f"{total_revenue} EGP",
        "sales": output
    }, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────
# FEATURE 4: WASTE TRACKER — deliveries, waste, analysis
# ─────────────────────────────────────────────────────────────

@tool
def get_deliveries(supplier: str = "") -> str:
    """
    Get delivery log showing what was ordered vs what was actually received.
    Optionally filter by supplier name. Shows shortages between ordered and received quantities.
    """
    results = MOCK_DELIVERIES
    if supplier:
        results = [d for d in MOCK_DELIVERIES if supplier in d["supplier"]]

    output = []
    total_shortage = 0
    for d in results:
        shortage = d["ordered_qty"] - d["received_qty"]
        total_shortage += shortage
        output.append({
            "id": d["id"],
            "date": d["date"],
            "supplier": d["supplier"],
            "item": d["item"],
            "ordered": f"{d['ordered_qty']} {d['unit']}",
            "received": f"{d['received_qty']} {d['unit']}",
            "shortage": f"{shortage} {d['unit']}" if shortage > 0 else "✅ مكتمل"
        })

    return json.dumps({
        "total_deliveries": len(output),
        "total_shortages": total_shortage,
        "deliveries": output
    }, ensure_ascii=False, indent=2)


@tool
def get_waste_report() -> str:
    """
    Get the waste report showing items thrown away, quantities, reasons, and total cost.
    Analyzes waste patterns to identify top waste items and common reasons.
    After 4 weeks of data, this helps the AI adjust order quantities automatically.
    """
    waste_by_item = {}
    total_waste_cost = 0.0

    for w in MOCK_WASTE_LOG:
        inv_item = next((i for i in MOCK_INVENTORY if i["name"] == w["item"]), None)
        cost = round(w["qty"] * inv_item["cost_per_unit"], 2) if inv_item else 0
        total_waste_cost += cost

        if w["item"] not in waste_by_item:
            waste_by_item[w["item"]] = {"total_qty": 0, "total_cost": 0, "reasons": [], "unit": w["unit"]}
        waste_by_item[w["item"]]["total_qty"] = round(waste_by_item[w["item"]]["total_qty"] + w["qty"], 2)
        waste_by_item[w["item"]]["total_cost"] = round(waste_by_item[w["item"]]["total_cost"] + cost, 2)
        if w["reason"] not in waste_by_item[w["item"]]["reasons"]:
            waste_by_item[w["item"]]["reasons"].append(w["reason"])

    summary = []
    for name, data in sorted(waste_by_item.items(), key=lambda x: x[1]["total_cost"], reverse=True):
        summary.append({
            "item": name,
            "total_wasted": f"{data['total_qty']} {data['unit']}",
            "cost_lost": f"{data['total_cost']} EGP",
            "reasons": "، ".join(data["reasons"])
        })

    # Reason breakdown
    reason_counts = {}
    for w in MOCK_WASTE_LOG:
        reason_counts[w["reason"]] = reason_counts.get(w["reason"], 0) + 1

    return json.dumps({
        "period": "2026-04-01 إلى 2026-04-03",
        "total_waste_cost": f"{round(total_waste_cost, 2)} EGP",
        "total_incidents": len(MOCK_WASTE_LOG),
        "top_waste_items": summary,
        "waste_reasons": reason_counts,
        "recommendation": "يُنصح بتقليل كمية الطلب للطماطم والخبز بنسبة 10-15% لتقليل الهدر"
    }, ensure_ascii=False, indent=2)


@tool
def log_waste(item_name: str, qty: float, reason: str) -> str:
    """
    Log a new waste entry. Use this when items are thrown away or spoiled.

    Parameters:
    - item_name: name of the wasted item (e.g. 'طماطم')
    - qty: quantity wasted (number)
    - reason: reason for waste (e.g. 'منتهية الصلاحية', 'تلف', 'جفاف')
    """
    from datetime import date
    inv_item = next((i for i in MOCK_INVENTORY if item_name in i["name"]), None)
    if not inv_item:
        available = ", ".join(i["name"] for i in MOCK_INVENTORY)
        return json.dumps({"error": f"الصنف '{item_name}' غير موجود. الأصناف المتاحة: {available}"}, ensure_ascii=False)

    waste_id = f"WST-{random.randint(100, 999)}"
    entry = {
        "id": waste_id,
        "date": str(date.today()),
        "item": inv_item["name"],
        "qty": qty,
        "unit": inv_item["unit"],
        "reason": reason
    }
    MOCK_WASTE_LOG.append(entry)

    # Deduct from inventory
    inv_item["current_stock"] = max(0, round(inv_item["current_stock"] - qty, 2))

    cost = round(qty * inv_item["cost_per_unit"], 2)
    return json.dumps({
        "status": "logged",
        "message": f"✅ تم تسجيل هدر {qty} {inv_item['unit']} من {inv_item['name']}",
        "waste_id": waste_id,
        "cost_lost": f"{cost} EGP",
        "remaining_stock": f"{inv_item['current_stock']} {inv_item['unit']}"
    }, ensure_ascii=False, indent=2)


@tool
def log_delivery(supplier: str, item_name: str, ordered_qty: float, received_qty: float) -> str:
    """
    Log a delivery receipt — compare what was ordered vs what was actually received.
    This feeds the waste tracker and helps identify supplier reliability.

    Parameters:
    - supplier: supplier name
    - item_name: item delivered
    - ordered_qty: what was ordered
    - received_qty: what was actually received
    """
    from datetime import date
    inv_item = next((i for i in MOCK_INVENTORY if item_name in i["name"]), None)
    if not inv_item:
        return json.dumps({"error": f"الصنف '{item_name}' غير موجود"}, ensure_ascii=False)

    delivery_id = f"DEL-{random.randint(100, 999)}"
    entry = {
        "id": delivery_id,
        "date": str(date.today()),
        "supplier": supplier,
        "item": inv_item["name"],
        "ordered_qty": ordered_qty,
        "received_qty": received_qty,
        "unit": inv_item["unit"]
    }
    MOCK_DELIVERIES.append(entry)

    # Update inventory stock
    inv_item["current_stock"] = round(inv_item["current_stock"] + received_qty, 2)
    shortage = ordered_qty - received_qty

    result = {
        "status": "logged",
        "delivery_id": delivery_id,
        "message": f"✅ تم تسجيل استلام {received_qty} {inv_item['unit']} من {inv_item['name']}",
        "new_stock": f"{inv_item['current_stock']} {inv_item['unit']}"
    }
    if shortage > 0:
        result["warning"] = f"⚠️ نقص {shortage} {inv_item['unit']} عن الكمية المطلوبة"

    return json.dumps(result, ensure_ascii=False, indent=2)


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
    export_report_pdf,
    # Suppliers
    get_suppliers,
    get_supplier_details,
    get_supplier_ratings,
    # Invoices
    get_invoices,
    get_invoice_details,
    get_payment_status,
    # Feature 1: Stock Check
    check_stock_levels,
    get_low_stock_alerts,
    # Feature 2: Draft PO Generator
    generate_draft_po_from_alerts,
    # Feature 3: Recipe-to-Ingredient
    get_recipes,
    calculate_consumption,
    get_daily_sales,
    # Feature 4: Waste Tracker
    get_deliveries,
    get_waste_report,
    log_waste,
    log_delivery,
]

SYSTEM_PROMPT = """You are a smart procurement and restaurant inventory assistant for an ERP system.

You help users with:
- Purchase orders — إنشاء وتتبع أوامر التوريد
- Supplier management — إدارة الموردين والمقيمين
- Invoices and payments — الفواتير والمدفوعات
- Inventory & stock control — مراقبة المخزون والحد الأدنى
- Recipe management — الوصفات وحساب المكونات المستهلكة
- Waste tracking — تتبع الهدر والتالف
- Delivery tracking — متابعة التوريدات والنواقص

INVENTORY & RESTAURANT RULES:
10. When asked about stock, inventory, or what needs ordering → call check_stock_levels or get_low_stock_alerts.
11. When asked to auto-generate or draft POs from stock alerts → call generate_draft_po_from_alerts.
12. When asked about recipes, menu items, or ingredients → call get_recipes.
13. When asked about consumption, "how much X was used", or linking sales to ingredients → call calculate_consumption. With no parameters it uses yesterday's full sales.
14. When asked about waste, spoilage, or thrown away items → call get_waste_report.
15. When the user wants to log waste → call log_waste with item name, quantity, and reason.
16. When the user logs a delivery receipt → call log_delivery with supplier, item, ordered qty, received qty.
17. When asked about daily sales or what was sold → call get_daily_sales.
18. When asked about deliveries or supplier shortages → call get_deliveries.

Rules you MUST follow:
1. Always call get_suppliers first to find the supplier ID before creating a PO.
2. Always call get_items first to find item IDs and prices before creating a PO.
3. For CREATE operations: call create_purchase_order to show a preview first. Then STOP and ask the user to confirm. Only call confirm_create_purchase_order after the user says yes/نعم/confirm.
4. For CANCEL operations: confirm with the user before calling cancel_po.
5. When the user asks to print, export, download, or get a PDF of a purchase order, you MUST call the export_po_pdf tool. For ANY other report (waste, stock, consumption, sales, deliveries, recipes), you MUST call export_report_pdf with the correct report_type. NEVER say you can't print — always call the appropriate export tool.
6. Reply in the same language as the user. Arabic question → Arabic answer. English → English.
7. Always show amounts with currency (EGP).
8. If you are missing information, ask the user before proceeding.
9. FORMATTING: When returning a list of 2 or more items, always format the response as a Markdown table with clear column headers.
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

            # Scan tool messages for PDF/Report markers and prepend to answer
            # (the final AI message is a summary that loses the marker)
            for m in result["messages"]:
                msg_content = getattr(m, "content", "")
                if isinstance(msg_content, str):
                    if "__PDF_EXPORT__" in msg_content:
                        pdf_match = re.search(r"__PDF_EXPORT__.+?__PDF_END__", msg_content, re.DOTALL)
                        if pdf_match:
                            content = pdf_match.group(0) + "\n" + content
                            break
                    elif "__REPORT_PDF__" in msg_content:
                        rpt_match = re.search(r"__REPORT_PDF__.+?__REPORT_END__", msg_content, re.DOTALL)
                        if rpt_match:
                            content = rpt_match.group(0) + "\n" + content
                            break

            # Fallback: if content is empty but tools were called, extract the last tool result
            if not content.strip() and tool_calls > 0:
                for m in reversed(result["messages"]):
                    mc = getattr(m, "content", "")
                    if isinstance(mc, str) and mc.strip() and not mc.startswith("__"):
                        try:
                            data = json.loads(mc)
                            if isinstance(data, list):
                                content = "إليك النتائج:\n\n" + mc
                            elif isinstance(data, dict) and "error" not in data:
                                content = mc
                            break
                        except (json.JSONDecodeError, TypeError):
                            if len(mc) > 20:
                                content = mc
                                break

            if not content.strip():
                content = "تم تنفيذ الطلب لكن لم يتم إرجاع نتائج. حاول مرة أخرى."

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
