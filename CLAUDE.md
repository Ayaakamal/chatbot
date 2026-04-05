# Procurement Chatbot — Project Context

## What This Project Is
An Arabic-first ERP procurement assistant chatbot built with:
- **Backend**: FastAPI (`chatbot_api.py`) + LangGraph ReAct agent (`agent.py`)
- **Frontend**: Single HTML file (`test_ui.html`) — no build step, open directly in browser
- **LLM**: Google Gemini (`gemini-2.5-flash`) via `langchain_google_genai`
- **Vector DB**: ChromaDB (`chroma_db/`) built by running `ingest.py`
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (HuggingFace, local)
- **Server port**: `8002`

## Running the Project
```bash
# Start server (use the .venv312 environment)
.venv312\Scripts\python chatbot_api.py

# Or use the batch file
run.bat

# Rebuild vector DB after adding/changing documents
.venv312\Scripts\python ingest.py
```

## Environment Variables (`.env`)
```
GOOGLE_API_KEY=...       # Gemini API key (active, used for LLM)
ANTHROPIC_API_KEY=...    # Claude key (exists but $0 credits — not currently used)
```

## Architecture

### Two Chat Modes in the UI
| Mode | Tab Label | Endpoint | Data Source |
|------|-----------|----------|-------------|
| Agent | المساعد الذكي | `POST /agent/` | Mock ERP data (tools) |
| Q&A | سؤال وجواب | `POST /ask-question/stream` | ChromaDB (uploaded documents) |

**Important**: The Q&A tab searches uploaded documents only. The Agent tab uses mock data tools. They are completely separate.

### UI Layout
- **Sidebar** (left, dark `#0a3438`): navigation icons only
- **Full-page chat panel**: opens when sidebar robot icon is clicked — has both Agent + Q&A tabs
- **Floating widget** (`#fw-widget`): fixed bottom-left button, opens 370×520px chat window — same Agent + Q&A tabs, separate history
- **ERP header**: top bar with MasterCode logo (`logo.png`) and system title

### Agent Tools (`agent.py`)
The ReAct agent has these tools over mock data:
- `get_suppliers` — list all suppliers
- `get_items` — list all items/products
- `get_purchase_orders` — list POs (optionally filtered by status)
- `get_invoices` — list invoices
- `create_purchase_order` — show PO preview (requires confirmation)
- `confirm_create_purchase_order` — finalize PO creation
- `cancel_po` — cancel a PO (requires confirmation)
- `export_po_pdf` — returns `__PDF_EXPORT__{json}__PDF_END__` marker; UI intercepts and opens print window

### Mock Data (in `agent.py`)
**MOCK_SUPPLIERS** (3):
- SUP-001: Ahmed Co | SUP-002: Nile Trading | SUP-003: Cairo Supplies

**MOCK_ITEMS** (4):
- ITEM-101: كرسي مكتبي EGP 250 | ITEM-102: طاولة مكتبية EGP 800
- ITEM-103: ورق A4 EGP 45 | ITEM-104: حاسوب محمول EGP 8,500

**MOCK_POS** (2):
- PO-2026-001: Ahmed Co, pending_approval, EGP 12,500
- PO-2026-002: Nile Trading, approved, EGP 4,500

**MOCK_INVOICES** (2):
- INV-2026-001: paid, EGP 4,500 | INV-2026-002: pending, EGP 25,500

### Documents (ChromaDB / Q&A tab)
JSON files in `documents/`:
- MOD-01 through MOD-09 covering ERP modules: Settings, Financial, Users/Permissions, Procurement Cycle, Vendor Management, Pricing/PO Approvals, Shipping, Invoices, QA/Returns

## Brand Colors
| Role | Hex |
|------|-----|
| Primary (teal) | `#145055` |
| Secondary (orange) | `#DA5B30` |
| Sidebar bg | `#0a3438` (darker than primary) |
| Light teal bg | `#e0f4f4` |

**No gradients** — solid colors only throughout the UI.

## Key Frontend Details
- Page direction: `dir="rtl"` (Arabic RTL)
- Font: Cairo (Google Fonts)
- Markdown rendering: `marked.js` (CDN) — bot responses are parsed as markdown
- Tables: rendered with `overflow-x: auto` wrapper (`.md-table-wrap`) so they scroll horizontally in narrow containers
- PDF export: `openPOPdf(po)` opens a new printable Arabic window when agent returns `__PDF_EXPORT__` marker
- Conversation history: stored in `localStorage` (browser only, lost on cache clear — not a database)
- Tool badges ("استخدم X أداة") have been **removed** from responses

## PDF Export Flow
1. User asks for a PDF of a PO (e.g. "طباعة PO-2026-001")
2. Agent calls `export_po_pdf(po_id)` → returns `__PDF_EXPORT__{...json...}__PDF_END__`
3. UI detects marker, strips it, shows "📄 فتح / طباعة PDF" button, auto-opens print window

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | System status (DB loaded, doc count) |
| POST | `/ask-question/` | Q&A (non-streaming) |
| POST | `/ask-question/stream` | Q&A (SSE streaming) — used by UI |
| POST | `/agent/` | ReAct agent — used by UI |

## Pending / Future Work
- **Real ERP API**: Microsoft Dynamics 365 F&O (D365) — Postman collection exists at `C:\Users\AYA\Downloads\DSCMServicesDEV.postman_collection.json`. Needs `{{URL}}`, `client_id`, `client_secret`, `dataAreaId` from IT admin. Uses Azure AD OAuth2.
- **Anthropic/Claude LLM**: Key exists in `.env` but has $0 credits. Add credits at `console.anthropic.com` to switch. Use `ChatAnthropic(model="claude-sonnet-4-6")` from `langchain_anthropic`.
- **MasterCode logo**: Save `logo.png` to `C:\Users\AYA\Desktop\Chatbot\` (the `<img>` tag is already in the header with `onerror` fallback).

## Virtual Environments
- `.venv312` — Python 3.12, **primary environment** (use this)
- `.venv314` — Python 3.14
- `.venv` — base

## Common Issues & Fixes
- **Gemini model name**: must be all lowercase `gemini-2.5-flash` (capital letters cause 400 INVALID_ARGUMENT)
- **Quota exhausted**: `_quota_exhausted_until` cache in `agent.py` blocks for 24h after quota hit — get a new API key from a different Google account
- **ChromaDB empty**: run `ingest.py` first, then restart the server
- **`langchain_anthropic` not found**: install inside `.venv312` with `.venv312\Scripts\python -m pip install langchain-anthropic`
