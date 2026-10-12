"""AUTOSAR HLD Document Analysis Assistant.
Streamlit Web Application providing grounded architecture analysis,
RAG retrieval, entity extraction, revision comparison, and consistency verification.
Consolidated into 4 streamlined sections: Overview, Documents, HLD Assistant, Settings & Evaluation.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AUTOSAR HLD Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Confirm actual application root and safely load local .env
APP_ROOT = Path(__file__).resolve().parent
from dotenv import load_dotenv
_app_env_file = APP_ROOT / ".env"
if _app_env_file.exists():
    load_dotenv(dotenv_path=_app_env_file, override=False)

# Project imports
from src.config import (
    Settings,
    get_settings,
    test_gemini_connection,
    SUPPORTED_GEMINI_MODELS,
    get_available_gemini_models,
    STATUS_KEY_MISSING,
    STATUS_NOT_TESTED,
    STATUS_CONNECTED,
    STATUS_RATE_LIMITED,
    STATUS_MODEL_UNAVAILABLE,
    STATUS_AUTH_FAILED,
    STATUS_SERVICE_UNAVAILABLE,
    STATUS_CONNECTION_FAILED,
    PROJECT_ROOT,
)
from src.gemini_client import is_quota_blocked
from src.ingestion import (
    PDFIngestionPipeline,
    check_tesseract_available,
    DocumentData,
    PageData,
)
from src.chunking import ContextAwareChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.rag_pipeline import (
    RAGPipeline,
    validate_user_query,
    get_contextual_suggestions,
)
from src.data_manager import DataManager, classify_document, ensure_baseline_document
from src.doc_comparison import DocumentComparator
from src.export_utils import (
    export_dataframe_to_csv,
    export_dataframe_to_json,
    export_dict_to_json,
)

# Sleek Professional Dark Navy Automotive Engineering Styling
st.markdown(
    """
    <style>
    /* ── Theme Variables & Global Base ────────────────────────────── */
    :root {
        --bg-main: #0B0F17;
        --bg-panel: #111827;
        --bg-card: #0E1626;
        --bg-card-alt: #131E33;
        --border-color: #243247;
        --border-subtle: #1A2536;
        --accent-cyan: #38BDF8;
        --accent-blue: #0EA5E9;
        --accent-blue-deep: #0284C7;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        --font-mono: "JetBrains Mono", "SF Mono", Consolas, "Liberation Mono", Menlo, Courier, monospace;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.8rem;
        max-width: 1160px;
    }

    /* ── App Header Banner ───────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #0B111E 0%, #131E33 100%);
        border: 1px solid #233148;
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    .app-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    .app-header-metrics {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .header-metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #0E1626;
        border: 1px solid #233148;
        color: #CBD5E1;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
    }

    /* ── Metric Cards (Overview Dashboard) ───────────────────────── */
    .metric-card {
        background: #0E1626;
        border: 1px solid #233148;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
        min-height: 78px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #2D3F5C;
        transform: translateY(-1px);
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.07em;
        white-space: nowrap;
    }
    .metric-value {
        font-size: 1.38rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 6px;
        line-height: 1.2;
        word-break: break-word;
    }

    /* ── Status & Notification Badges ────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 600;
        white-space: nowrap;
        vertical-align: middle;
        letter-spacing: 0.02em;
    }
    .badge-success  { background:#064E3B; color:#34D399; border:1px solid #059669; }
    .badge-warning  { background:#78350F; color:#FBBF24; border:1px solid #D97706; }
    .badge-danger   { background:#7F1D1D; color:#F87171; border:1px solid #DC2626; }
    .badge-neutral  { background:#1E293B; color:#CBD5E1; border:1px solid #334155; }

    /* ── Status Row (answer mode + latency) ─────────────────────── */
    .status-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 12px 0 8px 0;
        flex-wrap: wrap;
    }
    .status-latency {
        font-size: 0.82rem;
        color: #64748B;
        font-family: var(--font-mono);
    }

    /* ── Monospace Identifiers & Code Styling ────────────────────── */
    .stMarkdown code, .evidence-text code {
        font-family: var(--font-mono) !important;
        font-size: 0.85em !important;
        background: #141E30 !important;
        color: #38BDF8 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px solid #1E2E48 !important;
    }

    /* ── Tables & Component Details Presentation ────────────────── */
    .stMarkdown table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        margin: 14px 0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #233148 !important;
        background: #0E1626 !important;
    }
    .stMarkdown table th {
        background: #162238 !important;
        color: #94A3B8 !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        padding: 10px 14px !important;
        border-bottom: 1px solid #233148 !important;
        border-right: 1px solid #1A273C !important;
        text-align: left !important;
    }
    .stMarkdown table th:last-child {
        border-right: none !important;
    }
    .stMarkdown table td {
        padding: 10px 14px !important;
        font-size: 0.86rem !important;
        color: #E2E8F0 !important;
        border-bottom: 1px solid #182338 !important;
        border-right: 1px solid #182338 !important;
        line-height: 1.5 !important;
    }
    .stMarkdown table td:last-child {
        border-right: none !important;
    }
    .stMarkdown table tr:last-child td {
        border-bottom: none !important;
    }
    .stMarkdown table tr:nth-child(even) td {
        background: #111A2E !important;
    }
    .stMarkdown table tr:hover td {
        background: #17243F !important;
    }

    /* ── Blockquotes (Notes, Conflicts, Impact) ──────────────────── */
    .stMarkdown blockquote {
        background: #0E1626 !important;
        border-left: 3px solid #0EA5E9 !important;
        border-radius: 0 6px 6px 0 !important;
        padding: 10px 16px !important;
        margin: 12px 0 !important;
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        border-top: 1px solid #1C2638 !important;
        border-bottom: 1px solid #1C2638 !important;
        border-right: 1px solid #1C2638 !important;
    }
    .stMarkdown blockquote strong {
        color: #38BDF8 !important;
    }

    /* ── Tab Navigation ───────────────────────────────────────────── */
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        border-bottom: 1px solid #233148 !important;
        padding-bottom: 0px !important;
        margin-bottom: 18px !important;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 8px 16px !important;
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        border-radius: 4px 4px 0 0 !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #E2E8F0 !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #0EA5E9 !important;
        background: rgba(14, 165, 233, 0.06) !important;
    }

    /* ── Citation Cards ───────────────────────────────────────────── */
    .citation-box {
        background: #0B111E;
        border: 1px solid #1E2C42;
        border-left: 3px solid #38BDF8;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.85rem;
        color: #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        transition: border-color 0.15s ease;
    }
    .citation-box:hover {
        border-color: #2C4060;
    }
    .citation-meta {
        font-weight: 600;
        color: #F1F5F9;
        letter-spacing: -0.01em;
    }
    .citation-verified {
        color: #34D399;
        font-weight: 600;
        font-size: 0.8rem;
        white-space: nowrap;
    }

    /* ── Evidence Cards ───────────────────────────────────────────── */
    .evidence-card {
        background: #0B111E;
        border: 1px solid #1E2C42;
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 10px;
        font-size: 0.86rem;
        color: #CBD5E1;
        line-height: 1.6;
    }
    .evidence-header {
        font-size: 0.80rem;
        color: #94A3B8;
        margin-bottom: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        padding-bottom: 8px;
        border-bottom: 1px solid #182338;
    }
    .evidence-header strong {
        color: #F8FAFC;
        font-size: 0.85rem;
    }
    .evidence-text {
        white-space: pre-wrap;
        word-break: break-word;
        overflow-wrap: break-word;
        line-height: 1.65;
        color: #CBD5E1;
        font-size: 0.86rem;
    }

    /* ── Sidebar System State ────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #090D16 !important;
        border-right: 1px solid #1E293B !important;
    }
    .sys-state-panel {
        background: #0D1322;
        border: 1px solid #1C2638;
        border-radius: 6px;
        padding: 12px 14px;
        margin-top: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .sys-state-row {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .sys-state-label {
        font-size: 0.71rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
    }
    .sys-state-value {
        font-size: 0.83rem;
        color: #CBD5E1;
        font-family: var(--font-mono);
        word-break: break-all;
        white-space: normal;
    }

    /* ── Help / Notice Box ───────────────────────────────────────── */
    .notice-box {
        background: #0E1626;
        border: 1px solid #233148;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* ── Form Inputs & Button Refinements ────────────────────────── */
    div[data-baseweb="select"] > div {
        background-color: #0E1626 !important;
        border: 1px solid #233148 !important;
        border-radius: 6px !important;
        color: #F8FAFC !important;
        transition: border-color 0.15s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #38BDF8 !important;
    }
    .stTextInput input {
        background-color: #0E1626 !important;
        border: 1px solid #233148 !important;
        border-radius: 6px !important;
        color: #F8FAFC !important;
        padding: 8px 12px !important;
    }
    .stTextInput input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 20px !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_resource(show_spinner="⚙️ Loading AI models and vector store — this takes ~10 seconds on first launch…")
def get_service_layer():
    """Cache singleton service layer across UI reruns (loaded only once per server session)."""
    init_settings = get_settings()
    emb_mgr = EmbeddingManager(model_name=init_settings.embedding_model)
    # Pre-warm embedding model now so first query is fast
    emb_mgr._lazy_load_model()
    vstore = VectorStoreManager(
        persist_directory=init_settings.chroma_persist_dir,
        embedding_manager=emb_mgr,
        collection_name="autosar_hld_chunks",
    )
    data_mgr = DataManager(vector_store=vstore)
    chunker = ContextAwareChunker(
        target_chunk_size=init_settings.chunk_size,
        chunk_overlap=init_settings.chunk_overlap,
    )
    comparator = DocumentComparator()
    # Automatically ensure baseline document is indexed on initialization if library is empty
    ensure_baseline_document(vstore=vstore, data_mgr=data_mgr, chunker=chunker)
    return emb_mgr, vstore, data_mgr, chunker, comparator


emb_mgr, vstore, data_mgr, chunker, comparator = get_service_layer()
settings = get_settings()

# Resolve effective API key and model (supporting UI testing overrides)
session_key = (st.session_state.get("active_gemini_key") or "").strip()
effective_api_key = session_key if session_key else settings.google_api_key

session_model = (st.session_state.get("active_gemini_model") or "").strip()
effective_model = session_model if session_model else settings.gemini_model

active_settings = Settings(
    google_api_key=effective_api_key,
    gemini_model=effective_model,
    embedding_model=settings.embedding_model,
    chroma_persist_dir=settings.chroma_persist_dir,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    retrieval_top_k=settings.retrieval_top_k,
)

if "gemini_status" not in st.session_state:
    if not active_settings.is_gemini_configured:
        st.session_state.gemini_status = STATUS_KEY_MISSING
        st.session_state.gemini_status_msg = "No API key configured in .env or Streamlit secrets."
    else:
        st.session_state.gemini_status = STATUS_NOT_TESTED
        st.session_state.gemini_status_msg = "Key present in environment. Test connection under Settings & Evaluation."
else:
    # Dynamically update status if configuration state changed
    if not active_settings.is_gemini_configured and st.session_state.gemini_status != STATUS_KEY_MISSING:
        st.session_state.gemini_status = STATUS_KEY_MISSING
        st.session_state.gemini_status_msg = "No API key configured in .env or Streamlit secrets."
        st.session_state.last_tested_model = None
    elif active_settings.is_gemini_configured and st.session_state.gemini_status == STATUS_KEY_MISSING:
        st.session_state.gemini_status = STATUS_NOT_TESTED
        st.session_state.gemini_status_msg = "Key present in environment. Test connection under Settings & Evaluation."

if "last_tested_model" not in st.session_state:
    st.session_state.last_tested_model = None

# If model was changed, invalidate previous connected status
if st.session_state.last_tested_model is not None and st.session_state.last_tested_model != active_settings.gemini_model and st.session_state.gemini_status == STATUS_CONNECTED:
    st.session_state.gemini_status = STATUS_NOT_TESTED
    st.session_state.gemini_status_msg = f"Model changed to {active_settings.gemini_model}; connection not tested."

if "gemini_status_msg" not in st.session_state:
    st.session_state.gemini_status_msg = ""
if "last_activity_time" not in st.session_state:
    st.session_state.last_activity_time = "—"
if "active_nav_index" not in st.session_state:
    st.session_state.active_nav_index = 0

# Sidebar Navigation (Strictly 4 Consolidated Sections)
with st.sidebar:
    st.markdown("### Navigation")
    nav_selection = st.radio(
        "Main Sections",
        ["Overview", "Documents", "HLD Assistant", "Settings & Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### System Status")
    st.markdown('<div class="sys-state-panel">', unsafe_allow_html=True)

    # Gemini availability status badge with dynamic in-place update slot
    sidebar_status_slot = st.empty()
    def render_sidebar_status_badge(curr_status: str):
        if curr_status == STATUS_CONNECTED:
            badge_cls = "badge-success"
        elif curr_status == STATUS_NOT_TESTED:
            badge_cls = "badge-warning"
        elif curr_status == STATUS_KEY_MISSING:
            badge_cls = "badge-neutral"
        else:
            badge_cls = "badge-danger"
        sidebar_status_slot.markdown(
            f'<div class="sys-state-row"><span class="sys-state-label">Gemini API</span>'
            f'<div><span class="badge {badge_cls}">{curr_status}</span></div></div>',
            unsafe_allow_html=True,
        )
    render_sidebar_status_badge(st.session_state.gemini_status)

    # Concise vector-store availability indicator
    num_indexed_docs = len(data_mgr.get_documents_summary())
    vstore_indicator = "Ready" if num_indexed_docs > 0 else "Empty"
    st.markdown(
        f'<div class="sys-state-row">'
        f'<span class="sys-state-label">Vector Store</span>'
        f'<span class="sys-state-value">{vstore_indicator}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# Header Banner
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1 class="app-title">&#x1F697; AUTOSAR HLD Assistant</h1>
            <div class="app-subtitle">Evidence-grounded automotive architecture analysis and consistency review.</div>
        </div>
        <div class="app-header-metrics">
            <span class="header-metric-badge">&#x1F4D6; {len(data_mgr.get_documents_summary())} Documents</span>
            <span class="header-metric-badge">&#x1F9E9; {vstore.get_total_chunk_count()} Chunks</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION A: OVERVIEW
# =============================================================================
if nav_selection == "Overview":
    doc_summaries = data_mgr.get_documents_summary()
    total_chunks = vstore.get_total_chunk_count()

    # 4 Compact Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Indexed Documents</div>
                <div class="metric-value">{len(doc_summaries)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Vector Chunks</div>
                <div class="metric-value">{total_chunks}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        status_text = st.session_state.gemini_status
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Gemini Connection</div>
                <div class="metric-value" style="font-size: 1.15rem;">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Last Activity</div>
                <div class="metric-value" style="font-size: 1.15rem;">{st.session_state.last_activity_time}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Document Library Snapshot (Full Width)
    st.markdown("#### Document Library Snapshot")
    if doc_summaries:
        df_recent = pd.DataFrame([
            {
                "Document Name": d["filename"],
                "Classification": d["doc_type"],
                "Page Count": d["total_pages"],
                "Indexed Chunk Count": d["chunk_count"],
            }
            for d in doc_summaries
        ])
        st.dataframe(df_recent, use_container_width=True, hide_index=True)
    else:
        st.info("No documents indexed yet. Upload an HLD document in the **Documents** section to get started.")

    st.markdown("#### Core Capabilities")
    st.markdown(
        """
        - **Evidence-Grounded Question Answering**: Natural language questions answered with exact document, page number, and section citations.
        - **Architectural Extraction**: Catalogue software components (SWCs), provided/required ports, interfaces, signals, and functional flows.
        - **Inconsistency Detection**: Detect conflicting signal cycle times, undefined interfaces, and undeclared components across specifications.
        - **Revision Comparison**: Perform semantic structural diffs across specification revisions (e.g., v1.0 vs v2.0).
        """
    )



# =============================================================================
# SECTION B: DOCUMENTS
# =============================================================================
elif nav_selection == "Documents":
    st.markdown("### Document Management & Processing Inspector")

    doc_summaries = data_mgr.get_documents_summary()

    # Upload Section
    with st.expander("Upload New HLD Specification (PDF)", expanded=(len(doc_summaries) == 0)):
        uploaded_pdf = st.file_uploader(
            "Select an AUTOSAR High-Level Design document",
            type=["pdf"],
            help="PDF documents are parsed page-by-page preserving headings and table layouts.",
        )

        if uploaded_pdf is not None:
            if st.button("Process & Index Document", type="primary"):
                with st.spinner("Parsing document, extracting architectural elements, and vectorizing..."):
                    try:
                        file_bytes = uploaded_pdf.read()
                        ocr_available = check_tesseract_available()
                        ingestor = PDFIngestionPipeline(enable_ocr=True)
                        doc_data = ingestor.process_pdf(file_bytes, uploaded_pdf.name)

                        # Contextual OCR status reporting
                        ocr_pages = [p.page_number for p in doc_data.pages if p.extraction_method == "ocr"]
                        unextracted_pages = [p.page_number for p in doc_data.pages if p.extraction_method == "scanned_page_unextracted"]
                        failed_ocr_pages = [p.page_number for p in doc_data.pages if "OCR Error" in p.ocr_status]

                        if ocr_pages:
                            st.info(f"ℹ️ Scanned-page text extracted via OCR on page(s): {', '.join(map(str, ocr_pages))}.")
                        if unextracted_pages:
                            st.warning(
                                f"⚠️ Scanned page(s) {', '.join(map(str, unextracted_pages))} could not be processed because "
                                "Tesseract OCR is not installed or configured on this system. Text-based pages were indexed normally."
                            )
                        if failed_ocr_pages:
                            st.warning(f"⚠️ OCR processing encountered errors on page(s): {', '.join(map(str, failed_ocr_pages))}.")

                        # Chunking
                        chunks = chunker.chunk_document(doc_data)

                        # Check duplicate
                        if vstore.is_document_indexed(doc_data.file_hash):
                            st.warning(f"Document '{uploaded_pdf.name}' has already been indexed. Skipping duplicate insertion.")
                        else:
                            vstore.index_chunks(chunks)

                        # Extract entities and detect issues
                        ents = data_mgr.entity_extractor.extract_from_chunks(chunks)
                        issues = data_mgr.issue_detector.detect_issues(doc_data, chunks, ents)

                        # Register into shared catalog
                        data_mgr.register_document(doc_data, chunks, ents, issues)

                        st.session_state.last_activity_time = time.strftime("%H:%M:%S")
                        st.success(f"Successfully processed '{uploaded_pdf.name}' ({doc_data.total_pages} pages, {len(chunks)} chunks, {len(ents)} entities).")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Ingestion failed: {str(err)}")

    st.markdown("---")

    # Document Library and Inspector
    if doc_summaries:
        doc_opts = [d["doc_id"] for d in doc_summaries]
        doc_map = {d["doc_id"]: d for d in doc_summaries}

        selected_doc_id = st.selectbox(
            "Select Document to Inspect:",
            doc_opts,
            format_func=lambda x: f"{doc_map[x]['filename']} [{doc_map[x]['doc_type']}] — {doc_map[x]['chunk_count']} chunks",
        )

        doc_info = doc_map[selected_doc_id]
        doc_details = data_mgr.get_document_details(selected_doc_id)

        # Summary Bar
        m1, m2, m3, m4, m5 = st.columns([2, 1, 1, 1, 1])
        m1.caption(f"**Filename:** {doc_info['filename']}")
        m2.caption(f"**Classification:** {doc_info['doc_type']}")
        m3.caption(f"**Pages:** {doc_info['total_pages']}")
        m4.caption(f"**Chunks:** {doc_info['chunk_count']}")
        with m5:
            with st.popover("🗑️ Remove Document"):
                st.markdown(f"**Remove from Library:** `{doc_info['filename']}`")
                st.caption(
                    "This action removes the document's vector chunks, extracted entities, and issue records "
                    "from the active ChromaDB index and memory catalog. "
                    "The original source PDF file on disk is NOT deleted."
                )
                del_confirmed = st.checkbox(
                    "Confirm removal from active library",
                    key=f"chk_del_{selected_doc_id}",
                )
                if st.button("Confirm Removal", type="primary", disabled=not del_confirmed, key=f"del_{selected_doc_id}"):
                    data_mgr.remove_document(selected_doc_id)
                    st.session_state.last_activity_time = time.strftime("%H:%M:%S")
                    st.success(f"Removed '{doc_info['filename']}' from active vector library.")
                    st.rerun()

        # Detailed Inspector Tabs
        tab_pages, tab_chunks = st.tabs(["Page-by-Page Extraction", "Vector Chunk Explorer"])

        with tab_pages:
            if doc_details and doc_details.get("pages"):
                for page in doc_details["pages"]:
                    with st.expander(f"Page {page['page_number']} — Method: {page['extraction_method']}"):
                        if page.get("headings"):
                            st.markdown(f"**Headings:** `{', '.join(page['headings'])}`")
                        st.text_area("Extracted Text", page["text"], height=160, key=f"ptext_{selected_doc_id}_{page['page_number']}")
            else:
                st.info("Page extraction records available on next document re-index.")

        with tab_chunks:
            chunks = vstore.search("", top_k=60, doc_id_filter=selected_doc_id)
            if chunks:
                chunk_table = [
                    {
                        "Chunk ID": c["chunk_id"],
                        "Page": c["metadata"].get("page_num"),
                        "Section": c["metadata"].get("section"),
                        "Characters": len(c["text"]),
                        "Preview": c["text"][:80] + "...",
                    }
                    for c in chunks
                ]
                st.dataframe(pd.DataFrame(chunk_table), use_container_width=True, hide_index=True)
            else:
                st.info("No chunks stored for this document.")
    else:
        baseline_pdf = PROJECT_ROOT / "Input_Data" / "synthetic_autosar_hld.pdf"
        if baseline_pdf.exists():
            st.info("The active document library is currently empty.")
            bcol1, bcol2 = st.columns([1.6, 2])
            with bcol1:
                if st.button("🔄 Re-index Baseline Document (synthetic_autosar_hld.pdf)", type="primary"):
                    with st.spinner("Restoring baseline AUTOSAR specification into active index..."):
                        success = ensure_baseline_document(vstore=vstore, data_mgr=data_mgr, chunker=chunker, force=True)
                        if success:
                            st.session_state.last_activity_time = time.strftime("%H:%M:%S")
                            st.success("Baseline specification restored into library.")
                            st.rerun()
                        else:
                            st.error("Failed to re-index baseline document.")
            with bcol2:
                st.caption("Click to re-index the default synthetic AUTOSAR HLD baseline document from disk, or upload a custom PDF above.")
        else:
            st.warning("Document library is empty. Baseline file 'synthetic_autosar_hld.pdf' was not found in Input_Data/. Please upload an HLD PDF above.")


# =============================================================================
# SECTION C: HLD ASSISTANT
# =============================================================================
elif nav_selection == "HLD Assistant":
    doc_summaries = data_mgr.get_documents_summary()

    # Document Selection Filter
    doc_filter_choices = ["All Documents"] + [d["doc_id"] for d in doc_summaries]
    doc_filter_map = {d["doc_id"]: d for d in doc_summaries}

    fcol1, fcol2 = st.columns([2.2, 1])
    with fcol1:
        selected_filter = st.selectbox(
            "Document Scope:",
            doc_filter_choices,
            format_func=lambda x: "All Documents (Cross-Specification Search)" if x == "All Documents" else f"{doc_filter_map[x]['filename']} [{doc_filter_map[x]['doc_type']}]",
        )
    with fcol2:
        effective_filter_id = None if selected_filter == "All Documents" else selected_filter
        scope_title = selected_filter if selected_filter == "All Documents" else doc_filter_map[selected_filter]['filename']
        st.markdown(
            f"""
            <div style="margin-top: 28px; font-size: 0.83rem; color: #94A3B8;">
                Scope: <code style="font-family: var(--font-mono, monospace); color: #38BDF8; background: #0E1626; padding: 3px 8px; border-radius: 4px; border: 1px solid #233148;">{scope_title}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Consolidated 5 Sub-Tabs
    sub_ask, sub_explorer, sub_issues, sub_compare, sub_export = st.tabs([
        "Ask Assistant",
        "Architecture Explorer",
        "Issues & Inconsistencies",
        "Compare Revisions",
        "Export",
    ])

    # ---------------------------------------------------------
    # TAB 1: ASK ASSISTANT
    # ---------------------------------------------------------
    with sub_ask:
        # Determine Contextual Suggestions
        curr_doc_name = doc_filter_map[effective_filter_id]["filename"] if effective_filter_id else "All Documents"
        curr_doc_type = doc_filter_map[effective_filter_id]["doc_type"] if effective_filter_id else "AUTOSAR Specification"
        suggested_queries = get_contextual_suggestions(curr_doc_name, curr_doc_type)

        scol1, scol2 = st.columns([1, 2.3])
        with scol1:
            q_choice = st.selectbox(
                "Sample Questions:",
                ["Custom Question"] + suggested_queries,
                help="Suggested questions are automatically tailored to the selected document.",
            )
        with scol2:
            initial_q = "" if q_choice == "Custom Question" else q_choice
            user_question = st.text_input(
                "Enter Question:",
                value=initial_q,
                placeholder="e.g. What is the primary function and port mapping of ThrottleControl_SWC?",
            )

        blocked, block_msg, remaining_sec = is_quota_blocked()
        if blocked:
            st.warning(f"⚠️ **Rate Limit Active:** {block_msg} ({remaining_sec}s remaining). The system will safely use local retrieval fallback to avoid consuming API requests.")

        if st.button("Search & Answer", type="primary"):
            is_valid, val_err = validate_user_query(user_question)
            if not is_valid:
                st.warning(f"⚠️ {val_err}")
            else:
                with st.spinner("Retrieving evidence passages and formulating grounded response..."):
                    t0 = time.time()
                    rag_pipe = RAGPipeline(vector_store=vstore, settings=active_settings)
                    response = rag_pipe.query(
                        question=user_question,
                        doc_id_filter=effective_filter_id,
                        top_k=active_settings.retrieval_top_k,
                    )
                    latency = round(time.time() - t0, 3)
                    st.session_state.last_activity_time = time.strftime("%H:%M:%S")

                    # Determine whether the displayed answer actually came from a successful live Gemini API response.
                    # If a live request succeeded, update connection state to Connected for the active model.
                    # If generated using a mock, cached response, or fallback, do not mark Gemini as connected.
                    # Do not infer connection success from answer mode label alone.
                    if getattr(response, "is_live_gemini", False):
                        st.session_state.gemini_status = STATUS_CONNECTED
                        st.session_state.gemini_status_msg = f"Connected to {active_settings.gemini_model}"
                        st.session_state.last_tested_model = active_settings.gemini_model
                        if "render_sidebar_status_badge" in locals():
                            render_sidebar_status_badge(STATUS_CONNECTED)
                    elif response.error_message and getattr(response, "gemini_status", None) not in (STATUS_CONNECTED, STATUS_NOT_TESTED):
                        st.session_state.gemini_status = response.gemini_status
                        st.session_state.gemini_status_msg = response.error_message
                        if "render_sidebar_status_badge" in locals():
                            render_sidebar_status_badge(response.gemini_status)

                    # Truthful Answer-Mode Labeling
                    if response.is_abstained:
                        answer_mode_label = "Insufficient Evidence"
                        badge_class = "badge-warning"
                    elif response.mode == "gemini_grounded":
                        answer_mode_label = f"Gemini-Generated Answer ({active_settings.gemini_model})"
                        badge_class = "badge-success"
                    elif response.error_message:
                        answer_mode_label = "Gemini Generation Failed — Retrieval Fallback"
                        badge_class = "badge-danger"
                    else:
                        answer_mode_label = "Retrieval-Only Evidence"
                        badge_class = "badge-neutral"

                    st.session_state.last_query_mode = answer_mode_label
                    # Answer mode status row
                    st.markdown(
                        f'<div class="status-row">'
                        f'<span class="badge {badge_class}">{answer_mode_label}</span>'
                        f'<span class="status-latency">Processed in {latency}s</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if response.error_message and response.mode != "gemini_grounded":
                        st.warning(f"⚠️ **Gemini Generation Notice:** {response.error_message}")

                    if response.is_abstained:
                        st.warning("⚠️ **Grounded Abstention:** The retrieved document passages do not contain sufficient evidence to establish an answer.")

                    # Formatted Answer Box
                    st.markdown("#### Answer")
                    st.markdown(response.answer)

                    # Traceable Source Citations (compact, deduplicated, verified against real retrieved passages)
                    if response.citations:
                        st.markdown("#### Source Citations")
                        seen_cits = set()
                        for cit in response.citations:
                            cit_key = (
                                str(cit.document).strip().lower(),
                                cit.page,
                                str(cit.section).strip().lower(),
                                cit.supporting_chunk_id or "",
                            )
                            if cit_key in seen_cits:
                                continue
                            seen_cits.add(cit_key)

                            chunk_badge = (
                                f'<code style="font-size: 0.73rem; background: #1E293B; padding: 1px 6px; border-radius: 3px; color: #94A3B8; border: 1px solid #334155;">{cit.supporting_chunk_id}</code>'
                                if cit.supporting_chunk_id
                                else ""
                            )
                            verified_label = (
                                '<span class="citation-verified" style="color: #34D399; font-weight: 600; font-size: 0.8rem;">&#x2713; Verified Passage</span>'
                                if cit.is_valid
                                else '<span class="citation-verified" style="color: #F87171; font-weight: 600; font-size: 0.8rem;">&#x26A0; Unverified Reference</span>'
                            )

                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                        <span class="citation-meta">&#x1F4C4; {cit.document}</span>
                                        <span class="badge badge-neutral" style="font-size: 0.74rem;">Page {cit.page}</span>
                                        <span style="color: #94A3B8; font-size: 0.82rem;">{cit.section}</span>
                                        {chunk_badge}
                                    </div>
                                    {verified_label}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Retrieved Evidence Drawer
                    with st.expander(f"Inspect Retrieved Evidence Passages ({len(response.retrieved_chunks)} Chunks)"):
                        for idx, chunk in enumerate(response.retrieved_chunks):
                            meta = chunk.get("metadata", {})
                            sim = chunk.get("similarity", 0.0)
                            page_label = f"Page {meta.get('page_num', '?')}"
                            section_label = meta.get('section', 'General')
                            fname_label = meta.get('filename', 'Unknown')
                            relevance = f"{sim*100:.1f}%"
                            st.markdown(
                                f'<div class="evidence-card">'
                                f'<div class="evidence-header">'
                                f'<strong>Evidence {idx+1}</strong>'
                                f'<span>&#x2014; {fname_label}</span>'
                                f'<span>{page_label} &nbsp;|&nbsp; {section_label}</span>'
                                f'<span>Relevance: <code>{relevance}</code></span>'
                                f'</div>'
                                f'<div class="evidence-text">{chunk["text"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # ---------------------------------------------------------
    # TAB 2: ARCHITECTURE EXPLORER
    # ---------------------------------------------------------
    with sub_explorer:
        entities = data_mgr.get_entities(doc_id_filter=effective_filter_id)

        if entities:
            df_ents = data_mgr.entity_extractor.entities_to_dataframe(entities)

            ecol1, ecol2 = st.columns([1, 2])
            with ecol1:
                available_types = sorted(list(df_ents["Type"].unique()))
                type_selection = st.multiselect("Filter by Type:", available_types, default=available_types)
            with ecol2:
                search_term = st.text_input("Filter by Name / Keyword:", placeholder="e.g. Throttle, Brake, Sig_")

            filtered_ents = df_ents[df_ents["Type"].isin(type_selection)]
            if search_term.strip():
                filtered_ents = filtered_ents[
                    filtered_ents["Name"].str.contains(search_term, case=False) |
                    filtered_ents["Description"].str.contains(search_term, case=False)
                ]

            st.dataframe(filtered_ents, use_container_width=True, hide_index=True)
            st.caption(f"Displaying {len(filtered_ents)} of {len(df_ents)} architectural entities.")
        else:
            st.info("No architectural entities found in the selected scope. Ingest an AUTOSAR HLD document to extract entities.")

    # ---------------------------------------------------------
    # TAB 3: ISSUES & INCONSISTENCIES
    # ---------------------------------------------------------
    with sub_issues:
        issues = data_mgr.get_issues(doc_id_filter=effective_filter_id)

        if issues:
            df_issues = data_mgr.issue_detector.issues_to_dataframe(issues)

            icol1, icol2 = st.columns([1, 2])
            with icol1:
                sev_opts = sorted(list(df_issues["Severity"].unique()))
                sev_sel = st.multiselect("Filter Severity:", sev_opts, default=sev_opts)
            with icol2:
                type_opts = sorted(list(df_issues["Type"].unique()))
                type_sel = st.multiselect("Filter Issue Type:", type_opts, default=type_opts)

            filtered_issues = df_issues[
                df_issues["Severity"].isin(sev_sel) &
                df_issues["Type"].isin(type_sel)
            ]

            st.dataframe(filtered_issues, use_container_width=True, hide_index=True)

            st.markdown("#### Issue Details & Verified Evidence")
            for iss in issues:
                if iss.severity in sev_sel and iss.issue_type in type_sel:
                    with st.expander(f"{iss.issue_id}: [{iss.severity}] {iss.issue_type} — {iss.source_section} (Page {iss.source_page})"):
                        st.markdown(f"**Description:** {iss.description}")
                        st.markdown(f"**Supporting Evidence:** `{iss.supporting_evidence}`")
                        st.markdown(f"**Review Status:** `{iss.review_status}`")
        else:
            st.success("No architectural inconsistencies detected in the selected document scope.")

    # ---------------------------------------------------------
    # TAB 4: COMPARE REVISIONS
    # ---------------------------------------------------------
    with sub_compare:
        doc_summaries_all = data_mgr.get_documents_summary()
        if len(doc_summaries_all) >= 2:
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                baseline_id = st.selectbox(
                    "Baseline Document (Revision A):",
                    [d["doc_id"] for d in doc_summaries_all],
                    format_func=lambda x: next(d["filename"] for d in doc_summaries_all if d["doc_id"] == x),
                    key="comp_base",
                )
            with rcol2:
                target_id = st.selectbox(
                    "Updated Document (Revision B):",
                    [d["doc_id"] for d in doc_summaries_all if d["doc_id"] != baseline_id],
                    format_func=lambda x: next(d["filename"] for d in doc_summaries_all if d["doc_id"] == x),
                    key="comp_target",
                )

            if st.button("Execute Revision Diff", type="primary"):
                doc_a_det = data_mgr.get_document_details(baseline_id)
                doc_b_det = data_mgr.get_document_details(target_id)

                if doc_a_det and doc_b_det:
                    # Construct minimal doc objects
                    p_a = [PageData(page_number=p["page_number"], text=p["text"], headings=p.get("headings", [])) for p in doc_a_det["pages"]]
                    p_b = [PageData(page_number=p["page_number"], text=p["text"], headings=p.get("headings", [])) for p in doc_b_det["pages"]]
                    da = DocumentData(doc_id=baseline_id, filename=doc_a_det["filename"], file_hash=doc_a_det["doc_hash"], total_pages=len(p_a), pages=p_a)
                    db = DocumentData(doc_id=target_id, filename=doc_b_det["filename"], file_hash=doc_b_det["doc_hash"], total_pages=len(p_b), pages=p_b)
                    
                    ea = data_mgr.get_entities(baseline_id)
                    eb = data_mgr.get_entities(target_id)

                    report = comparator.compare_documents(da, db, ea, eb)

                    st.markdown(f"#### Comparison: `{report.doc_a_name}` vs `{report.doc_b_name}`")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Added Sections", report.added_sections_count)
                    c2.metric("Removed Sections", report.removed_sections_count)
                    c3.metric("Modified Sections", report.modified_sections_count)

                    df_diff = comparator.diff_to_dataframe(report.section_diffs)
                    st.dataframe(df_diff, use_container_width=True, hide_index=True)

                    if report.architectural_impacts:
                        st.markdown("#### Architectural Impact Analysis")
                        for imp in report.architectural_impacts:
                            st.markdown(f"- {imp}")
        else:
            st.info("Revision comparison requires at least 2 indexed documents. Upload a second revision under the **Documents** tab.")

    # ---------------------------------------------------------
    # TAB 5: EXPORT
    # ---------------------------------------------------------
    with sub_export:
        st.markdown("#### Data Export Center")
        st.caption("All exports correspond directly to the currently loaded document data.")

        ex_ents = data_mgr.get_entities(doc_id_filter=effective_filter_id)
        ex_issues = data_mgr.get_issues(doc_id_filter=effective_filter_id)

        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.markdown("##### Architectural Entities")
            if ex_ents:
                df_ex_e = data_mgr.entity_extractor.entities_to_dataframe(ex_ents)
                st.download_button(
                    f"Download Entities as CSV ({len(df_ex_e)} rows)",
                    export_dataframe_to_csv(df_ex_e),
                    file_name="architectural_entities.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.download_button(
                    f"Download Entities as JSON ({len(df_ex_e)} rows)",
                    export_dataframe_to_json(df_ex_e),
                    file_name="architectural_entities.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.caption("No entities available to export.")

        with col_ex2:
            st.markdown("##### Consistency Findings")
            if ex_issues:
                df_ex_i = data_mgr.issue_detector.issues_to_dataframe(ex_issues)
                st.download_button(
                    f"Download Issues as CSV ({len(df_ex_i)} findings)",
                    export_dataframe_to_csv(df_ex_i),
                    file_name="consistency_findings.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.download_button(
                    f"Download Issues as JSON ({len(df_ex_i)} findings)",
                    export_dataframe_to_json(df_ex_i),
                    file_name="consistency_findings.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.caption("No issues available to export.")


# =============================================================================
# SECTION D: SETTINGS & EVALUATION
# =============================================================================
elif nav_selection == "Settings & Evaluation":
    st.markdown("### Gemini LLM Diagnostics & Connectivity")
    diag_col1, diag_col2 = st.columns([3, 2])
    with diag_col1:
        key_is_present = active_settings.is_gemini_configured
        key_status_label = (
            '<span class="badge badge-success">Configured (Present)</span>'
            if key_is_present
            else '<span class="badge badge-neutral">Not Configured</span>'
        )
        st.markdown(f"**API Key Detection:** {key_status_label}", unsafe_allow_html=True)

        avail_models = get_available_gemini_models(api_key=effective_api_key)
        if active_settings.gemini_model not in avail_models:
            avail_models = [active_settings.gemini_model] + [m for m in avail_models if m != active_settings.gemini_model]
        curr_idx = avail_models.index(active_settings.gemini_model)

        chosen_model = st.selectbox(
            "Configured Gemini Model:",
            options=avail_models,
            index=curr_idx,
            help="Active model for grounded architecture reasoning.",
        )
        if chosen_model != active_settings.gemini_model:
            st.session_state.active_gemini_model = chosen_model
            st.session_state.gemini_status = STATUS_NOT_TESTED
            st.session_state.gemini_status_msg = f"Model switched to {chosen_model}; connection not tested."
            st.rerun()

    with diag_col2:
        st.markdown(
            f"**Sidebar Status:** `{st.session_state.gemini_status}`\n\n"
            f"*{st.session_state.get('gemini_status_msg', '')}*"
        )
        if st.button("Test Gemini Connection", type="secondary"):
            with st.spinner(f"Testing connection to {active_settings.gemini_model}..."):
                test_status, test_msg = test_gemini_connection(
                    api_key=effective_api_key,
                    model_name=active_settings.gemini_model,
                )
                st.session_state.gemini_status = test_status
                st.session_state.gemini_status_msg = test_msg
                if test_status == STATUS_CONNECTED:
                    st.session_state.last_tested_model = active_settings.gemini_model
                    st.success(f"✅ {test_msg}")
                elif test_status == STATUS_KEY_MISSING:
                    st.warning(f"⚠️ {test_msg}")
                else:
                    st.error(f"❌ {test_status}: {test_msg}")
                st.rerun()

    with st.expander("API Key Configuration / Session Override (Safe — No Credentials Leaked)", expanded=False):
        st.markdown(
            "Provide your API key in `.env` (`GOOGLE_API_KEY=AIzaSy...` or `GEMINI_API_KEY=AIzaSy...`), "
            "or temporarily override it in this browser session. Keys are never logged or stored to disk."
        )
        session_key_in = st.text_input(
            "Session API Key:",
            type="password",
            value=st.session_state.get("active_gemini_key", ""),
            help="Overrides environment key for this session without saving to disk.",
        )
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("Apply Session Key"):
                if session_key_in.strip():
                    st.session_state.active_gemini_key = session_key_in.strip()
                    st.session_state.gemini_status = STATUS_NOT_TESTED
                    st.session_state.gemini_status_msg = "Session key set. Ready to test connection."
                else:
                    if "active_gemini_key" in st.session_state:
                        del st.session_state["active_gemini_key"]
                st.rerun()
        with btn_col2:
            if st.button("Clear Session Key"):
                if "active_gemini_key" in st.session_state:
                    del st.session_state["active_gemini_key"]
                st.rerun()

    st.markdown("---")
    st.markdown("### Evaluation Overview")

    eval_metrics_path = settings.evaluation_results_dir / "metrics.json"
    eval_report_path = settings.evaluation_results_dir / "Evaluation_Report.md"
    eval_answers_path = settings.evaluation_results_dir / "answer_results.json"
    eval_questions_path = settings.evaluation_results_dir / "evaluation_questions.json"

    if eval_metrics_path.exists():
        with open(eval_metrics_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)

        m = m_data.get("metrics", {})
        exec_mode = m_data.get("execution_mode", "retrieval_only")
        is_live_gen = m_data.get("is_live_generation", False)

        # Consistent execution-mode determination
        if exec_mode == "live_generation" or (exec_mode == "gemini_grounded" and is_live_gen):
            exec_mode_label = f"Live Answer Generation ({m_data.get('gemini_model', settings.gemini_model)})"
            is_live = True
        elif exec_mode == "mixed_mode":
            exec_mode_label = "Mixed Mode (Live Generation & Retrieval Fallback)"
            is_live = True
        else:
            exec_mode_label = "Offline Retrieval Evaluation"
            is_live = False

        # Dynamic metric evaluations against targets
        recall_val = m.get("retrieval_recall_at_k", 0)
        citation_val = m.get("citation_validity_rate", 0)
        citation_cov = m.get("citation_coverage_rate", 0)
        abstention_val = m.get("abstention_accuracy", 0)
        fact_val = m.get("factual_match_rate", 0)
        avg_latency = m_data.get("average_query_latency_seconds", 0)
        ingest_time = m_data.get("document_processing_time_seconds", 0)

        # Latency check: Target is < 2.0s
        latency_fails = avg_latency >= 2.0

        # Dynamic metric statuses against target standards
        status_recall = "PASS" if recall_val >= 0.85 else "FAIL"
        status_cit_val = "PASS" if citation_val >= 1.0 else "FAIL"
        status_cit_cov = "PASS" if citation_cov >= 0.90 else "FAIL"
        status_abs = "PASS" if abstention_val >= 0.80 else "FAIL"
        status_fact = "PASS" if fact_val >= 0.85 else "FAIL"
        status_lat = "FAIL (Above Target)" if latency_fails else "PASS"
        status_ingest = "PASS" if ingest_time < 10.0 else "FAIL"

        all_passed = (
            status_recall == "PASS"
            and status_cit_val == "PASS"
            and status_cit_cov == "PASS"
            and status_abs == "PASS"
            and status_fact == "PASS"
            and not latency_fails
            and status_ingest == "PASS"
        )
        overall_badge = "✅ PASS" if all_passed else "❌ FAIL"

        # 1. Main Evaluation Overview Header
        st.markdown(
            f"**Benchmark Name:** `{m_data.get('dataset_name', 'AUTOSAR HLD Synthetic Benchmark')}` &nbsp;|&nbsp; "
            f"**Execution Mode:** `{exec_mode_label}` &nbsp;|&nbsp; "
            f"**Overall Status:** {overall_badge}"
        )

        # Aligned, Consistent Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval Recall@4", f"{recall_val*100:.1f}%")
        c2.metric("Citation Validity Rate", f"{citation_val*100:.1f}%")
        c3.metric("Citation Coverage", f"{citation_cov*100:.1f}%")
        c4.metric("Abstention Accuracy", f"{abstention_val*100:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Factual Match Rate", f"{fact_val*100:.1f}%")
        c6.metric("Total Evaluated Questions", m_data.get("total_questions", 0))
        c7.metric("Average Query Latency", f"{avg_latency:.3f}s")
        c8.metric("Doc Ingestion Time", f"{ingest_time:.2f}s")

        # 2. Evaluation Scope Notice
        scope_notice = (
            "Evaluation results apply to the tested synthetic benchmark. "
            "Retrieval, citation, and factual-match metrics measure their respective evaluation stages "
            "and do not establish production readiness or certified AUTOSAR/ISO 26262 compliance."
        )
        if is_live:
            mode_detail = (
                " This benchmark run executed live Gemini answer generation; factual-match and citation metrics "
                "evaluate the generated responses against ground-truth architectural specifications."
            )
        else:
            mode_detail = (
                " This benchmark was conducted in offline retrieval evaluation mode and does not measure "
                "live Gemini answer-generation quality."
            )
        st.caption(f"{scope_notice}{mode_detail}")

        # 3. Expandable Section 1: Detailed Evaluation Results
        with st.expander("Detailed Evaluation Results", expanded=False):
            if eval_answers_path.exists():
                with open(eval_answers_path, "r", encoding="utf-8") as f_ans:
                    answers_data = json.load(f_ans)

                eq_map = {}
                if eval_questions_path.exists():
                    with open(eval_questions_path, "r", encoding="utf-8") as f_eq:
                        eq_map = {q["id"]: q for q in json.load(f_eq)}

                q_table = []
                for a in answers_data:
                    qid = a.get("question_id")
                    q_info = eq_map.get(qid, {})
                    q_type = q_info.get("type", "factual_lookup")
                    exp_page = q_info.get("expected_page")
                    exp_page_str = str(exp_page) if exp_page is not None else "N/A (Abstain)"

                    ret_chunks = a.get("retrieved_chunks", [])
                    if exp_page is not None:
                        hit = any(c.get("metadata", {}).get("page_num") == exp_page for c in ret_chunks)
                        hit_str = "Hit" if hit else "Miss"
                    else:
                        hit_str = "N/A"

                    is_abs = a.get("is_abstained", False)
                    should_abs = a.get("should_abstain", False)
                    if should_abs:
                        abs_str = "Correct Abstention" if is_abs else "Failed to Abstain"
                    else:
                        abs_str = "Answered" if not is_abs else "False Abstention"

                    q_table.append({
                        "Question ID": qid,
                        "Query Type": q_type,
                        "Expected Page": exp_page_str,
                        "Retrieval Hit": hit_str,
                        "Abstention Result": abs_str,
                        "Latency (s)": round(a.get("latency_seconds", 0), 3),
                    })
                st.dataframe(pd.DataFrame(q_table), use_container_width=True, hide_index=True)

        # 4. Expandable Section 2: Metric Definitions and Thresholds
        with st.expander("Metric Definitions and Thresholds", expanded=False):
            st.markdown(
                f"""
| Metric | Target Threshold | Measured Value | Status | Formula / Description |
|---|---|---|---|---|
| **Retrieval Recall@4** | >= 85.0% | **{recall_val*100:.1f}%** | `{status_recall}` | Proportion of factual queries where target page/section was in top-4 chunks |
| **Citation Validity Rate** | 100.0% | **{citation_val*100:.1f}%** | `{status_cit_val}` | Valid emitted citations referencing verified passages / Total emitted citations |
| **Citation Coverage** | >= 90.0% | **{citation_cov*100:.1f}%** | `{status_cit_cov}` | Factual questions with verified citations / Total factual questions |
| **Abstention Accuracy** | >= 80.0% | **{abstention_val*100:.1f}%** | `{status_abs}` | Correct abstentions on unanswerable/negative queries / Total negative queries |
| **Factual Match Rate** | >= 85.0% | **{fact_val*100:.1f}%** | `{status_fact}` | Ground-truth engineering facts retrieved and presented / Expected facts |
| **Average Query Latency** | < 2.0 seconds | **{avg_latency:.3f} s** | `{status_lat}` | Mean retrieval and citation generation response time |
| **Document Ingestion Time** | < 10.0 seconds | **{ingest_time:.2f} s** | `{status_ingest}` | Wall-clock time for PDF parsing, chunking, and embedding |
                """
            )

        # 5. Expandable Section 3: Comprehensive Evaluation Report
        with st.expander("Comprehensive Evaluation Report", expanded=False):
            if eval_report_path.exists():
                report_text = eval_report_path.read_text(encoding="utf-8")
                st.markdown(report_text)
                st.download_button(
                    label="Export Full Evaluation Report (Markdown)",
                    data=report_text,
                    file_name="Evaluation_Report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info("Comprehensive evaluation report artifact will be generated upon executing the benchmark.")

    if st.button("Execute Automated Evaluation Benchmark"):
        with st.spinner("Running 20-question automated evaluation benchmark..."):
            from evaluation.run_evaluation import run_evaluation_benchmark
            run_evaluation_benchmark()
            st.session_state.last_activity_time = time.strftime("%H:%M:%S")
            st.success("Evaluation benchmark completed. Metrics recomputed and updated.")
            st.rerun()
