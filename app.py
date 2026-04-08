import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from extractor import (
    extract_text_from_pdf,
    extract_financial_metrics_from_text,
    clean_llm_json,
    categorize_metric,
    summarize_document,
    convert_value_to_numeric,
    answer_financial_question,
    parse_period_to_index,
)
from rag_utils import index_document, search_documents, reset_collection
from forecast_utils import prepare_metric_history, forecast_next_value
from market_utils import (
    get_market_snapshot,
    get_price_history,
    compare_market_performance,
    company_name_to_ticker,
)
from sentiment_utils import analyze_earnings_sentiment
from anomaly_utils import detect_metric_anomalies
from research_report_utils import generate_equity_research_report
from valuation_utils import build_valuation_summary
from analyst_agent_utils import run_financial_analyst_agent
from autonomous_agent_utils import run_autonomous_financial_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial AI Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px; }
    .stExpander { border-radius: 8px; }
    div[data-testid="stSidebarContent"] { background: #1a1a2e; color: white; }
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .section-divider { border-top: 2px solid #e0e0e0; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Financial AI Research Platform")
st.caption(
    "Upload financial PDFs · Extract metrics · AI summaries · RAG Q&A · "
    "Forecasting · Market data · Sentiment analysis · Valuation models · Research reports"
)

# ── Session state ─────────────────────────────────────────────────────────────
def initialize_session_state() -> None:
    defaults = {
        "history": [],
        "latest_results": [],
        "latest_combined_metrics": [],
        "latest_summaries": [],
        "latest_documents": [],
        "rag_ready": False,
        "latest_answer": None,
        "latest_chunks": [],
        "latest_sources": [],
        "latest_market_snapshot": None,
        "detected_ticker": "",
        "detected_company": "",
        "detected_companies": [],
        "selected_detected_ticker": "",
        "auto_market_loaded": False,
        "latest_sentiment_result": None,
        "latest_valuation_result": None,
        "latest_agent_output": None,
        "latest_autonomous_agent_output": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_processing_state() -> None:
    keys = [
        "latest_results", "latest_combined_metrics", "latest_summaries",
        "latest_documents", "latest_answer", "latest_chunks", "latest_sources",
        "latest_market_snapshot", "detected_ticker", "detected_company",
        "detected_companies", "selected_detected_ticker", "auto_market_loaded",
        "latest_sentiment_result", "latest_valuation_result",
        "latest_agent_output", "latest_autonomous_agent_output",
    ]
    for key in keys:
        st.session_state[key] = [] if isinstance(st.session_state.get(key), list) else (
            "" if isinstance(st.session_state.get(key), str) else
            False if isinstance(st.session_state.get(key), bool) else None
        )


initialize_session_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    max_pages = st.slider("Max PDF pages per file", 1, 50, 10)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.0, 0.05)
    show_text = st.checkbox("Show extracted raw text", value=False)
    generate_ai_summary = st.checkbox("Generate AI summary", value=True)
    enable_rag = st.checkbox("Enable document Q&A (RAG)", value=True)
    anomaly_threshold = st.slider("Anomaly detection threshold (%)", 10, 100, 30)
    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown("""
1. Upload PDFs or paste text
2. Click **Process Documents**
3. Explore metrics & charts
4. Ask questions via RAG
5. Load market data
6. Run valuation & agents
""")

# ── File upload ───────────────────────────────────────────────────────────────
st.markdown("### 📁 Document Input")
upload_col, text_col = st.columns([1, 1])

with upload_col:
    uploaded_files = st.file_uploader(
        "Upload financial PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload annual reports, earnings releases, 10-K, 10-Q, etc.",
    )

with text_col:
    manual_text = st.text_area(
        "Or paste financial text directly",
        height=160,
        placeholder="Paste earnings call text, financial statement excerpts, press releases...",
    )

# ── Action buttons ────────────────────────────────────────────────────────────
action_col1, action_col2, action_col3 = st.columns(3)
with action_col1:
    run_extract = st.button("🚀 Process Documents", type="primary", use_container_width=True)
with action_col2:
    clear_history = st.button("🗑️ Clear History", use_container_width=True)
with action_col3:
    reset_rag = st.button("🔄 Reset RAG Index", use_container_width=True)

if clear_history:
    st.session_state.history = []
    clear_processing_state()
    st.success("✅ History cleared.")

if reset_rag:
    try:
        reset_collection()
        st.session_state.rag_ready = False
        st.session_state.latest_answer = None
        st.session_state.latest_chunks = []
        st.session_state.latest_sources = []
        st.success("✅ RAG index reset.")
    except Exception as e:
        st.error(f"Failed to reset RAG index: {e}")

# ── Document Processing ───────────────────────────────────────────────────────
if run_extract:
    has_uploaded = bool(uploaded_files)
    has_text = bool(manual_text.strip())

    if not has_uploaded and not has_text:
        st.warning("⚠️ Upload one or more PDFs or paste text first.")
    else:
        clear_processing_state()
        if enable_rag:
            reset_collection()

        documents_to_process = []
        if has_uploaded:
            for f in uploaded_files:
                try:
                    pdf_bytes = f.read()
                    source_text = extract_text_from_pdf(pdf_bytes, max_pages=max_pages)
                    if show_text:
                        with st.expander(f"📄 Raw text: {f.name}"):
                            st.text(source_text[:15000])
                    documents_to_process.append({"name": f.name, "text": source_text, "source_type": "pdf"})
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")

        if has_text:
            documents_to_process.append({"name": "pasted_text", "text": manual_text.strip(), "source_type": "text"})

        indexed_total = 0
        progress_bar = st.progress(0, text="Starting processing...")
        total_docs = len(documents_to_process)

        for doc_idx, doc in enumerate(documents_to_process):
            doc_name = doc["name"]
            doc_text = doc["text"]

            if not doc_text.strip():
                continue

            progress_pct = int((doc_idx / total_docs) * 100)
            progress_bar.progress(progress_pct, text=f"Processing {doc_name} ({doc_idx + 1}/{total_docs})...")

            try:
                result = extract_financial_metrics_from_text(doc_text, temperature=temperature)
                result = clean_llm_json(result)
            except Exception as e:
                st.error(f"Extraction failed for {doc_name}: {e}")
                result = {"metrics": [], "company_name": None, "document_type": None, "summary": None}

            summary_text = None
            if generate_ai_summary:
                try:
                    progress_bar.progress(progress_pct + 10, text=f"Generating summary for {doc_name}...")
                    summary_text = summarize_document(doc_text, temperature=temperature)
                except Exception as e:
                    summary_text = f"Summary failed: {e}"

            safe_doc_id = (
                doc_name.replace(" ", "_").replace(".", "_").replace("/", "_").lower()[:60]
            )

            if enable_rag:
                try:
                    chunks = index_document(doc_text, document_name=safe_doc_id)
                    indexed_total += chunks
                except Exception as e:
                    st.warning(f"RAG indexing failed for {doc_name}: {e}")

            metrics = result.get("metrics", [])
            for m in metrics:
                m["document_name"] = doc_name
                m["numeric_value"] = convert_value_to_numeric(m.get("value"))
                m["period_index"] = parse_period_to_index(m.get("period"))
                m["category"] = categorize_metric(m.get("metric", ""))

            st.session_state.latest_results.append({
                "document_name": doc_name,
                "source_type": doc["source_type"],
                "text": doc_text,
                "result": result,
                "summary": summary_text,
            })
            st.session_state.latest_combined_metrics.extend(metrics)
            st.session_state.latest_summaries.append({"document_name": doc_name, "summary": summary_text})
            st.session_state.latest_documents.append(doc_name)
            st.session_state.history.insert(0, {
                "preview": f"{doc_name}: {doc_text.replace(chr(10), ' ')[:120]}",
                "result": result,
                "summary": summary_text,
            })

        progress_bar.progress(100, text="✅ Processing complete!")

        # Detect companies
        detected_pairs = []
        seen_tickers = set()
        for item in st.session_state.latest_results:
            company_name = (item["result"].get("company_name") or "").strip()
            ticker = company_name_to_ticker(company_name)
            if company_name and ticker and ticker not in seen_tickers:
                detected_pairs.append({"company_name": company_name, "ticker": ticker})
                seen_tickers.add(ticker)

        st.session_state.detected_companies = detected_pairs
        if detected_pairs:
            st.session_state.detected_company = detected_pairs[0]["company_name"]
            st.session_state.detected_ticker = detected_pairs[0]["ticker"]
            st.session_state.selected_detected_ticker = detected_pairs[0]["ticker"]
        else:
            st.session_state.detected_company = ""
            st.session_state.detected_ticker = ""
            st.session_state.selected_detected_ticker = ""

        st.session_state.history = st.session_state.history[:10]
        st.session_state.rag_ready = enable_rag and indexed_total > 0

        if enable_rag and indexed_total > 0:
            st.success(f"✅ Indexed {indexed_total} chunks across {len(st.session_state.latest_documents)} document(s).")

# ── Results Display ───────────────────────────────────────────────────────────
results = st.session_state.latest_results
combined_metrics = st.session_state.latest_combined_metrics

if results:
    st.markdown("---")
    st.subheader("📋 Processed Documents")

    overview_cols = st.columns(4)
    with overview_cols[0]:
        st.metric("📄 Documents", len(results))
    with overview_cols[1]:
        st.metric("📊 Metrics Extracted", len(combined_metrics))
    with overview_cols[2]:
        detected_companies = {
            item["result"].get("company_name", "").strip()
            for item in results if item["result"].get("company_name")
        }
        st.metric("🏢 Companies", len(detected_companies))
    with overview_cols[3]:
        rag_status = "✅ Ready" if st.session_state.rag_ready else "❌ Not Ready"
        st.metric("🔍 RAG", rag_status)

    # Document Summaries
    st.markdown("### 📝 Document Summaries")
    for item in results:
        doc_name = item["document_name"]
        result = item["result"]
        company_name = result.get("company_name") or "Unknown"
        document_type = result.get("document_type") or "Unknown"

        with st.expander(f"📄 {doc_name} | 🏢 {company_name} | 📑 {document_type}"):
            if result.get("summary"):
                st.markdown("**Document Summary**")
                st.info(result["summary"])
            if item["summary"]:
                st.markdown("**AI Financial Analysis**")
                st.markdown(item["summary"])

    # Metrics Table
    st.markdown("### 📊 Extracted Financial Metrics")
    if combined_metrics:
        df = pd.DataFrame(combined_metrics)

        preferred_cols = ["document_name", "metric", "value", "period", "currency",
                          "page", "evidence", "confidence", "category", "numeric_value", "period_index"]
        existing_cols = [c for c in preferred_cols if c in df.columns]
        df = df[existing_cols].copy()

        if "confidence" in df.columns:
            df = df.sort_values("confidence", ascending=False)

        display_cols = [c for c in df.columns if c not in ["numeric_value", "period_index"]]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if "category" in df.columns:
                st.markdown("#### Metric Categories")
                fig_cat = px.pie(
                    df, names="category", title="Metrics by Category",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        with chart_col2:
            if "document_name" in df.columns:
                st.markdown("#### Metrics per Document")
                doc_counts = df["document_name"].value_counts().reset_index()
                doc_counts.columns = ["Document", "Count"]
                fig_doc = px.bar(doc_counts, x="Document", y="Count",
                                 color="Count", color_continuous_scale="Blues")
                st.plotly_chart(fig_doc, use_container_width=True)

        # Bar chart of numeric metrics
        chart_df = df.dropna(subset=["numeric_value"]) if "numeric_value" in df.columns else pd.DataFrame()
        if not chart_df.empty:
            st.markdown("#### 💹 Financial Metrics Comparison")
            fig = px.bar(
                chart_df, x="metric", y="numeric_value",
                color="document_name" if "document_name" in chart_df.columns else "category",
                hover_data=[c for c in ["period", "currency", "confidence", "document_name"] if c in chart_df.columns],
                title="Extracted Financial Metrics",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # Forecasting
        st.markdown("### 📈 Trend Detection & Forecasting")
        forecastable_df = (
            df.dropna(subset=["numeric_value", "period_index"])
            if "numeric_value" in df.columns and "period_index" in df.columns
            else pd.DataFrame()
        )

        if not forecastable_df.empty and "metric" in forecastable_df.columns:
            available_metrics = sorted(forecastable_df["metric"].dropna().unique().tolist())
            selected_metric = st.selectbox("Select a metric to analyze & forecast", available_metrics)
            metric_history = prepare_metric_history(forecastable_df, selected_metric)

            if metric_history.empty:
                st.info("No usable time-series data for this metric.")
            else:
                st.dataframe(
                    metric_history[[c for c in ["document_name", "metric", "period", "numeric_value"] if c in metric_history.columns]],
                    use_container_width=True, hide_index=True,
                )
                forecast_result = forecast_next_value(metric_history)
                if forecast_result:
                    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
                    with fcol1:
                        st.metric("Forecast Value", f"{forecast_result['forecast_value']:,.2f}")
                    with fcol2:
                        st.metric("Trend", forecast_result.get("trend", "N/A"))
                    with fcol3:
                        r2 = forecast_result.get("r2_score", 0)
                        st.metric("R² Score", f"{r2:.3f}")
                    with fcol4:
                        lb = forecast_result.get("lower_bound", 0)
                        ub = forecast_result.get("upper_bound", 0)
                        st.metric("95% Range", f"{lb:,.0f} – {ub:,.0f}")

                    chart_history = metric_history.copy()
                    forecast_row = {
                        "metric": selected_metric,
                        "period": f"Forecast ({forecast_result['next_period_index']})",
                        "numeric_value": forecast_result["forecast_value"],
                        "document_name": "📈 Forecast",
                    }
                    chart_plot_df = pd.concat([chart_history, pd.DataFrame([forecast_row])], ignore_index=True)
                    fig_fc = px.line(
                        chart_plot_df, x="period", y="numeric_value",
                        color="document_name", markers=True,
                        title=f"{selected_metric} — Historical Trend & Forecast",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)
                else:
                    st.info("Need at least 2 data points to forecast.")
        else:
            st.info("No trend-ready metrics. Use documents with periods like 2022, 2023, Q1 2024.")

        # Anomaly Detection
        st.markdown("### 🚨 Anomaly Detection")
        anomaly_df = (
            detect_metric_anomalies(df, threshold=anomaly_threshold / 100)
            if "numeric_value" in df.columns and "period_index" in df.columns
            else pd.DataFrame()
        )

        if not anomaly_df.empty:
            st.warning(f"⚠️ {len(anomaly_df)} anomal{'y' if len(anomaly_df) == 1 else 'ies'} detected.")
            st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
            fig_anom = px.bar(
                anomaly_df, x="metric", y="change_pct",
                color="anomaly_type",
                hover_data=[c for c in ["document_name", "period", "previous_value", "current_value", "severity"] if c in anomaly_df.columns],
                title="Detected Metric Anomalies",
                barmode="group",
                color_discrete_map={"Spike": "#2ecc71", "Drop": "#e74c3c"},
            )
            st.plotly_chart(fig_anom, use_container_width=True)
        else:
            st.success("✅ No major anomalies detected with current threshold.")

        # Downloads
        st.markdown("### 📥 Downloads")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(results, indent=2),
                file_name="financial_research_results.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Download CSV",
                data=df.to_csv(index=False),
                file_name="financial_research_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Evidence
        st.markdown("### 🔍 Evidence by Metric")
        for _, row in df.iterrows():
            metric = row.get("metric", "Unknown")
            value = row.get("value", "N/A")
            doc_name = row.get("document_name", "Unknown")
            page = row.get("page", "N/A")
            confidence = row.get("confidence", "N/A")
            evidence = row.get("evidence", "No evidence provided.")
            with st.expander(f"{doc_name} | **{metric}** — {value} | page {page} | confidence {confidence}"):
                st.write(evidence)

# ── RAG Q&A ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Ask Questions Across Documents")

question = st.text_input(
    "Ask a financial research question",
    placeholder="Which company had the highest revenue? Compare margins. What risks were mentioned?",
)

if st.button("🔍 Ask Research Platform", use_container_width=True):
    if not results:
        st.warning("Process documents first.")
    elif not question.strip():
        st.warning("Enter a question first.")
    elif not st.session_state.rag_ready:
        st.warning("RAG not ready. Reprocess documents with Q&A enabled.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                retrieval = search_documents(question, top_k=6)
                retrieved_chunks = retrieval.get("documents", [])
                retrieved_sources = retrieval.get("sources", [])
                if not retrieved_chunks:
                    st.info("No relevant context found.")
                else:
                    answer = answer_financial_question(
                        query=question, context_chunks=retrieved_chunks, temperature=temperature
                    )
                    st.session_state.latest_answer = answer
                    st.session_state.latest_chunks = retrieved_chunks
                    st.session_state.latest_sources = retrieved_sources
            except Exception as e:
                st.error(f"Q&A failed: {e}")

if st.session_state.latest_answer:
    st.markdown("### 💬 Answer")
    st.markdown(st.session_state.latest_answer)

if st.session_state.latest_chunks:
    st.markdown("### 📚 Retrieved Context Chunks")
    for i, chunk in enumerate(st.session_state.latest_chunks, 1):
        source = st.session_state.latest_sources[i - 1] if i - 1 < len(st.session_state.latest_sources) else "Unknown"
        with st.expander(f"Chunk {i} | Source: {source}"):
            st.write(chunk)

# ── Market Data ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Live Financial Market Data")

detected_companies = st.session_state.detected_companies
if detected_companies:
    detected_text = ", ".join([f"**{item['ticker']}** ({item['company_name']})" for item in detected_companies])
    st.success(f"🏢 Detected: {detected_text}")
    selected_detected_ticker = st.selectbox(
        "Select detected company ticker",
        options=[item["ticker"] for item in detected_companies],
        index=0,
        key="selected_detected_ticker_widget",
    )
    selected_company_name = next(
        (item["company_name"] for item in detected_companies if item["ticker"] == selected_detected_ticker), ""
    )
    st.session_state.selected_detected_ticker = selected_detected_ticker
    st.session_state.detected_ticker = selected_detected_ticker
    st.session_state.detected_company = selected_company_name

default_symbol = st.session_state.selected_detected_ticker or st.session_state.detected_ticker or "AAPL"
market_col1, market_col2 = st.columns(2)
with market_col1:
    market_symbol = st.text_input(
        "Enter stock ticker", value=default_symbol,
        placeholder="AAPL, TSLA, AMZN, MSFT", key="market_symbol_input",
    )
with market_col2:
    compare_symbols_text = st.text_input("Compare tickers (comma-separated)", value="AAPL,TSLA,AMZN")

auto_col1, auto_col2 = st.columns(2)
with auto_col1:
    auto_detect_toggle = st.checkbox("Auto-load detected company market data", value=True)
with auto_col2:
    load_market_clicked = st.button("📊 Load Market Data", use_container_width=True)

should_auto_load = (
    auto_detect_toggle
    and bool(st.session_state.selected_detected_ticker or st.session_state.detected_ticker)
    and not st.session_state.auto_market_loaded
)

if load_market_clicked or should_auto_load:
    target_symbol = market_symbol.strip() or st.session_state.selected_detected_ticker or st.session_state.detected_ticker
    if not target_symbol:
        st.warning("Enter a ticker first.")
    else:
        with st.spinner(f"Loading market data for {target_symbol}..."):
            try:
                snapshot = get_market_snapshot(target_symbol)
                st.session_state.latest_market_snapshot = snapshot
                st.session_state.auto_market_loaded = True
                st.session_state.latest_valuation_result = None
                st.session_state.latest_agent_output = None
                st.session_state.latest_autonomous_agent_output = None

                st.markdown("### 📊 Market Snapshot")
                snap_cols = st.columns(5)
                metrics_data = [
                    ("Symbol", snapshot["symbol"]),
                    ("Price", f"${snapshot['price']:.2f}" if snapshot["price"] else "N/A"),
                    ("Change", f"{snapshot['change']:.2f}" if snapshot["change"] is not None else "N/A"),
                    ("Change %", f"{snapshot['change_pct']:.2f}%" if snapshot["change_pct"] is not None else "N/A"),
                    ("Market Cap", f"${int(snapshot['market_cap']):,}" if snapshot["market_cap"] else "N/A"),
                ]
                for col, (label, val) in zip(snap_cols, metrics_data):
                    with col:
                        st.metric(label, val)

                # Extra metrics
                extra_cols = st.columns(3)
                with extra_cols[0]:
                    vol = snapshot.get("volume")
                    st.metric("Volume", f"{int(vol):,}" if vol else "N/A")
                with extra_cols[1]:
                    wh = snapshot.get("fifty_two_week_high")
                    st.metric("52W High", f"${wh:.2f}" if wh else "N/A")
                with extra_cols[2]:
                    wl = snapshot.get("fifty_two_week_low")
                    st.metric("52W Low", f"${wl:.2f}" if wl else "N/A")

                # Price history with MA
                hist_df = get_price_history(target_symbol, period="6mo", interval="1d")
                if not hist_df.empty and "Close" in hist_df.columns:
                    st.markdown("### 📉 Price History")
                    fig_market = go.Figure()
                    fig_market.add_trace(go.Scatter(
                        x=hist_df["Date"], y=hist_df["Close"],
                        name="Close", line=dict(color="#2563eb", width=2)
                    ))
                    if "MA20" in hist_df.columns:
                        fig_market.add_trace(go.Scatter(
                            x=hist_df["Date"], y=hist_df["MA20"],
                            name="20-day MA", line=dict(color="#f59e0b", width=1.5, dash="dash")
                        ))
                    if "MA50" in hist_df.columns:
                        fig_market.add_trace(go.Scatter(
                            x=hist_df["Date"], y=hist_df["MA50"],
                            name="50-day MA", line=dict(color="#ef4444", width=1.5, dash="dot")
                        ))
                    fig_market.update_layout(
                        title=f"{target_symbol.upper()} — 6 Month Price History",
                        xaxis_title="Date", yaxis_title="Price ($)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_market, use_container_width=True)

                # Comparative performance
                compare_symbols = [s.strip().upper() for s in compare_symbols_text.split(",") if s.strip()]
                compare_df = compare_market_performance(compare_symbols, period="6mo")
                if not compare_df.empty:
                    st.markdown("### 🔀 Comparative Market Performance")
                    fig_cmp = px.line(
                        compare_df, x="Date", y="Normalized", color="Symbol",
                        title="Normalized 6-Month Performance (Base = 100)",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_cmp.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
                    st.plotly_chart(fig_cmp, use_container_width=True)

            except Exception as e:
                st.error(f"Market data failed: {e}")

# Market + Document comparison
if st.session_state.latest_market_snapshot and results:
    snapshot = st.session_state.latest_market_snapshot
    st.markdown("### 🔗 Market + Document Insight")
    if st.button("🔗 Compare Reports with Market Data", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                market_context = "\n".join([
                    f"Ticker: {snapshot['symbol']}",
                    f"Price: {snapshot['price']}",
                    f"Market Cap: {snapshot['market_cap']}",
                    f"Change %: {snapshot['change_pct']}",
                    f"52W High: {snapshot.get('fifty_two_week_high')}",
                    f"52W Low: {snapshot.get('fifty_two_week_low')}",
                ])
                comparison_context = (
                    st.session_state.latest_chunks or
                    [item["result"].get("summary") or item["text"][:2000] for item in results[:3]]
                )
                comparison_context.append(market_context)
                answer = answer_financial_question(
                    query=(
                        f"Compare the uploaded financial reports with current market data for "
                        f"{snapshot['symbol']}. Highlight alignment or mismatch between "
                        f"reported fundamentals and market performance."
                    ),
                    context_chunks=comparison_context,
                    temperature=temperature,
                )
                st.markdown("#### 💡 Comparison Insight")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Comparison failed: {e}")

# ── Valuation Model ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔢 AI Valuation Model")

if st.button("🔢 Run Valuation Model", use_container_width=True):
    if not combined_metrics:
        st.warning("Process documents first.")
    elif not st.session_state.latest_market_snapshot:
        st.warning("Load market data first.")
    else:
        with st.spinner("Running valuation analysis..."):
            try:
                valuation = build_valuation_summary(
                    metrics=combined_metrics,
                    market_snapshot=st.session_state.latest_market_snapshot,
                )
                st.session_state.latest_valuation_result = valuation

                signal = valuation.get("signal", "N/A")
                signal_emoji = {"Potentially Undervalued": "🟢", "Potentially Overvalued": "🔴",
                                "Fairly Valued": "🟡", "Slight Upside": "🟢", "Slight Downside": "🟠"}.get(signal, "⚪")

                vcols = st.columns(3)
                with vcols[0]:
                    mc = valuation.get("market_cap")
                    st.metric("Market Cap", f"${mc:,.0f}" if mc else "N/A")
                with vcols[1]:
                    efv = valuation.get("estimated_fair_value")
                    st.metric("Est. Fair Value", f"${efv:,.0f}" if efv else "N/A")
                with vcols[2]:
                    gap = valuation.get("valuation_gap_pct")
                    st.metric("Valuation Gap", f"{gap:.2f}%" if gap is not None else "N/A")

                scols = st.columns(4)
                vals_data = [
                    ("Revenue", valuation.get("revenue")),
                    ("Net Income", valuation.get("net_income")),
                    ("EBITDA", valuation.get("ebitda")),
                    ("Free Cash Flow", valuation.get("free_cash_flow")),
                ]
                for col, (label, val) in zip(scols, vals_data):
                    with col:
                        st.metric(label, f"${val:,.0f}" if val else "N/A")

                st.markdown(f"**Valuation Signal: {signal_emoji} {signal}** ({valuation.get('methods_used', 0)} method(s) used)")

                detail_df = pd.DataFrame([
                    {"Method": "P/E Fair Value (20x)", "Value": valuation.get("pe_fair_value")},
                    {"Method": "Revenue Multiple (3x)", "Value": valuation.get("revenue_multiple_value")},
                    {"Method": "EV/EBITDA (12x)", "Value": valuation.get("ev_ebitda_value")},
                    {"Method": "DCF Valuation", "Value": valuation.get("dcf_value")},
                    {"Method": "Avg. Fair Value", "Value": valuation.get("estimated_fair_value")},
                    {"Method": "Current Market Cap", "Value": valuation.get("market_cap")},
                ])
                chart_df_val = detail_df.dropna(subset=["Value"])
                if not chart_df_val.empty:
                    fig_val = px.bar(
                        chart_df_val, x="Method", y="Value",
                        title="Valuation Method Comparison",
                        color="Value", color_continuous_scale="RdYlGn",
                    )
                    st.plotly_chart(fig_val, use_container_width=True)
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)

                st.info("📚 Simplified educational valuation model. Not investment advice.")
            except Exception as e:
                st.error(f"Valuation failed: {e}")

# ── AI Analyst Agent ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 AI Financial Analyst Agent")

if st.button("🤖 Run AI Analyst Agent", use_container_width=True):
    if not results:
        st.warning("Process documents first.")
    elif not st.session_state.latest_market_snapshot:
        st.warning("Load market data first.")
    else:
        with st.spinner("Running AI analyst agent..."):
            try:
                agent_output = run_financial_analyst_agent(
                    company_name=st.session_state.detected_company or "Unknown Company",
                    ticker=st.session_state.detected_ticker or "N/A",
                    metrics=combined_metrics,
                    market_snapshot=st.session_state.latest_market_snapshot,
                    sentiment_result=st.session_state.latest_sentiment_result,
                )
                st.session_state.latest_agent_output = agent_output
                st.session_state.latest_valuation_result = agent_output.get("valuation_result")
            except Exception as e:
                st.error(f"Analyst agent failed: {e}")

if st.session_state.latest_agent_output:
    agent_output = st.session_state.latest_agent_output
    valuation_result = agent_output.get("valuation_result")
    investment_thesis = agent_output.get("investment_thesis")
    research_report = agent_output.get("research_report")

    if valuation_result:
        acols = st.columns(3)
        with acols[0]:
            efv = valuation_result.get("estimated_fair_value")
            st.metric("Fair Value", f"${efv:,.0f}" if efv else "N/A")
        with acols[1]:
            gap = valuation_result.get("valuation_gap_pct")
            st.metric("Gap %", f"{gap:.2f}%" if gap is not None else "N/A")
        with acols[2]:
            st.metric("Signal", valuation_result.get("signal", "N/A"))

    if investment_thesis:
        st.markdown(investment_thesis)

    if research_report:
        with st.expander("📄 Full Research Report"):
            st.markdown(research_report)
        combined_dl = research_report + ("\n\n" + investment_thesis if investment_thesis else "")
        st.download_button(
            "⬇️ Download Analyst Report", data=combined_dl,
            file_name="ai_analyst_report.txt", mime="text/plain", use_container_width=True,
        )

# ── Autonomous Agent ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 Autonomous Financial Analyst")

if st.button("🧠 Run Autonomous Agent", use_container_width=True):
    if not results:
        st.warning("Process documents first.")
    elif not st.session_state.latest_market_snapshot:
        st.warning("Load market data first.")
    else:
        with st.spinner("Running autonomous financial agent..."):
            try:
                auto_output = run_autonomous_financial_agent(
                    company_name=st.session_state.detected_company or "Unknown Company",
                    ticker=st.session_state.detected_ticker or "N/A",
                    metrics=combined_metrics,
                    market_snapshot=st.session_state.latest_market_snapshot,
                    sentiment_result=st.session_state.latest_sentiment_result,
                )
                st.session_state.latest_autonomous_agent_output = auto_output
                st.session_state.latest_valuation_result = auto_output.get("valuation_result")
            except Exception as e:
                st.error(f"Autonomous agent failed: {e}")

if st.session_state.latest_autonomous_agent_output:
    auto_output = st.session_state.latest_autonomous_agent_output
    rec = auto_output.get("recommendation", "N/A")
    rec_color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(rec, "⚪")

    top_cols = st.columns(3)
    with top_cols[0]:
        st.metric("Recommendation", f"{rec_color} {rec}")
    with top_cols[1]:
        vr = auto_output.get("valuation_result") or {}
        gap = vr.get("valuation_gap_pct")
        st.metric("Valuation Gap %", f"{gap:.2f}%" if gap is not None else "N/A")
    with top_cols[2]:
        st.metric("Risk Flags", len(auto_output.get("risks", [])))

    if auto_output.get("final_brief"):
        st.markdown(auto_output["final_brief"])

    with st.expander("📄 Full Autonomous Agent Report"):
        if auto_output.get("research_report"):
            st.markdown(auto_output["research_report"])

    st.download_button(
        "⬇️ Download Agent Brief", data=auto_output.get("final_brief", ""),
        file_name="autonomous_agent_brief.txt", mime="text/plain", use_container_width=True,
    )

# ── Research Report ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📄 AI Equity Research Report")

if st.button("📄 Generate Research Report", use_container_width=True):
    if not results:
        st.warning("Process documents first.")
    else:
        with st.spinner("Generating research report..."):
            company_name = st.session_state.detected_company or "Unknown Company"
            ticker = st.session_state.detected_ticker or "N/A"
            market_snapshot = st.session_state.latest_market_snapshot
            valuation_result = st.session_state.latest_valuation_result
            if valuation_result is None and market_snapshot and combined_metrics:
                valuation_result = build_valuation_summary(combined_metrics, market_snapshot)

            report = generate_equity_research_report(
                company_name=company_name, ticker=ticker,
                financial_metrics=combined_metrics, market_snapshot=market_snapshot,
                sentiment_result=st.session_state.latest_sentiment_result,
                valuation_result=valuation_result,
            )
            st.markdown(report)
            st.download_button(
                "⬇️ Download Report", data=report,
                file_name="equity_research_report.md", mime="text/markdown", use_container_width=True,
            )

# ── Sentiment Analysis ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎙️ Earnings Call Sentiment Analysis")

transcript_text = st.text_area(
    "Paste earnings call transcript",
    height=200,
    placeholder="Paste CEO/CFO commentary or full earnings transcript here...",
)

if st.button("🎙️ Analyze Sentiment", use_container_width=True):
    if not transcript_text.strip():
        st.warning("Paste transcript text first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                sentiment_result = analyze_earnings_sentiment(transcript_text)
                st.session_state.latest_sentiment_result = sentiment_result
                st.session_state.latest_agent_output = None
                st.session_state.latest_autonomous_agent_output = None

                label = sentiment_result["sentiment_label"]
                label_emoji = {"Bullish": "🟢", "Mildly Bullish": "🟢", "Neutral": "🟡",
                               "Mildly Bearish": "🟠", "Bearish": "🔴"}.get(label, "⚪")

                s_cols = st.columns(5)
                sentiment_metrics = [
                    ("Sentiment", f"{label_emoji} {label}"),
                    ("Score", str(sentiment_result["sentiment_score"])),
                    ("Positive", str(sentiment_result["positive_hits"])),
                    ("Negative", str(sentiment_result["negative_hits"])),
                    ("Guidance", str(sentiment_result["guidance_hits"])),
                ]
                for col, (lbl, val) in zip(s_cols, sentiment_metrics):
                    with col:
                        st.metric(lbl, val)

                e_cols = st.columns(3)
                with e_cols[0]:
                    st.metric("Risk Mentions", sentiment_result["risk_hits"])
                with e_cols[1]:
                    st.metric("Words Analyzed", sentiment_result["token_count"])
                with e_cols[2]:
                    st.metric("Normalized Score", f"{sentiment_result.get('normalized_score', 0):.3f}")

                sent_df = pd.DataFrame([
                    {"Category": "Positive", "Count": sentiment_result["positive_hits"]},
                    {"Category": "Negative", "Count": sentiment_result["negative_hits"]},
                    {"Category": "Guidance", "Count": sentiment_result["guidance_hits"]},
                    {"Category": "Risk", "Count": sentiment_result["risk_hits"]},
                ])
                fig_sent = px.bar(
                    sent_df, x="Category", y="Count",
                    title="Transcript Signal Breakdown",
                    color="Category",
                    color_discrete_map={
                        "Positive": "#2ecc71", "Negative": "#e74c3c",
                        "Guidance": "#3498db", "Risk": "#f39c12",
                    },
                )
                st.plotly_chart(fig_sent, use_container_width=True)

                top_terms = sentiment_result.get("top_terms", {})
                if top_terms:
                    st.markdown("### 🔤 Top Terms")
                    terms_df = pd.DataFrame([{"Term": k, "Count": v} for k, v in top_terms.items()])
                    fig_terms = px.bar(
                        terms_df, x="Term", y="Count",
                        title="Most Frequent Terms in Transcript",
                        color="Count", color_continuous_scale="Blues",
                    )
                    st.plotly_chart(fig_terms, use_container_width=True)

            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")

# ── History ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🕓 Recent Runs")
if not st.session_state.history:
    st.info("No history yet. Process a document to get started.")
else:
    for i, item in enumerate(st.session_state.history, 1):
        with st.expander(f"Run #{i}: {item['preview'][:100]}"):
            if item.get("summary"):
                st.markdown("**AI Summary**")
                st.markdown(item["summary"])
            st.json(item["result"])
