.python-version
import json
import pandas as pd
import plotly.express as px
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

st.set_page_config(
    page_title="Financial AI Research Platform",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Financial AI Research Platform")
st.caption(
    "Upload one or more financial PDFs, extract metrics, generate summaries, "
    "ask cross-document questions, forecast trends, compare with live market data, "
    "analyze earnings-call sentiment, detect KPI anomalies, run valuation models, "
    "generate AI research reports, and produce autonomous AI analyst briefs."
)


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
    st.session_state.latest_results = []
    st.session_state.latest_combined_metrics = []
    st.session_state.latest_summaries = []
    st.session_state.latest_documents = []
    st.session_state.latest_answer = None
    st.session_state.latest_chunks = []
    st.session_state.latest_sources = []
    st.session_state.latest_market_snapshot = None
    st.session_state.detected_ticker = ""
    st.session_state.detected_company = ""
    st.session_state.detected_companies = []
    st.session_state.selected_detected_ticker = ""
    st.session_state.auto_market_loaded = False
    st.session_state.latest_sentiment_result = None
    st.session_state.latest_valuation_result = None
    st.session_state.latest_agent_output = None
    st.session_state.latest_autonomous_agent_output = None


initialize_session_state()

with st.sidebar:
    st.header("Settings")
    max_pages = st.slider("Max PDF pages to read per file", 1, 50, 8)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)
    show_text = st.checkbox("Show extracted raw text", value=False)
    generate_ai_summary = st.checkbox("Generate AI summary", value=True)
    enable_rag = st.checkbox("Enable document Q&A (RAG)", value=True)
    anomaly_threshold = st.slider("Anomaly threshold (%)", 10, 100, 30)

uploaded_files = st.file_uploader(
    "Upload financial PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

manual_text = st.text_area(
    "Or paste financial text",
    height=220,
    placeholder="Paste earnings text, annual report text, or financial statement text here...",
)

action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

with action_col1:
    run_extract = st.button("Process Documents", type="primary", use_container_width=True)
with action_col2:
    clear_history = st.button("Clear History", use_container_width=True)
with action_col3:
    reset_rag = st.button("Reset RAG Index", use_container_width=True)

if clear_history:
    st.session_state.history = []
    clear_processing_state()
    st.success("History cleared.")

if reset_rag:
    try:
        reset_collection()
        st.session_state.rag_ready = False
        st.session_state.latest_answer = None
        st.session_state.latest_chunks = []
        st.session_state.latest_sources = []
        st.success("RAG index reset successfully.")
    except Exception as e:
        st.error(f"Failed to reset RAG index: {e}")

if run_extract:
    has_uploaded_files = uploaded_files is not None and len(uploaded_files) > 0
    has_manual_text = bool(manual_text.strip())

    if not has_uploaded_files and not has_manual_text:
        st.warning("Upload one or more PDFs or paste text first.")
    else:
        with st.spinner("Processing financial documents with AI..."):
            try:
                clear_processing_state()

                if enable_rag:
                    reset_collection()

                documents_to_process = []

                if has_uploaded_files:
                    for uploaded_file in uploaded_files:
                        try:
                            pdf_bytes = uploaded_file.read()
                            source_text = extract_text_from_pdf(pdf_bytes, max_pages=max_pages)

                            if show_text:
                                with st.expander(f"Extracted text: {uploaded_file.name}"):
                                    st.text(source_text[:15000])

                            documents_to_process.append(
                                {
                                    "name": uploaded_file.name,
                                    "text": source_text,
                                    "source_type": "pdf",
                                }
                            )
                        except Exception as file_error:
                            st.error(f"Failed to read {uploaded_file.name}: {file_error}")

                if has_manual_text:
                    documents_to_process.append(
                        {
                            "name": "manual_text",
                            "text": manual_text.strip(),
                            "source_type": "text",
                        }
                    )

                indexed_total = 0

                for doc in documents_to_process:
                    doc_name = doc["name"]
                    doc_text = doc["text"]

                    if not doc_text.strip():
                        continue

                    result = extract_financial_metrics_from_text(
                        doc_text,
                        temperature=temperature,
                    )
                    result = clean_llm_json(result)

                    summary_text = None
                    if generate_ai_summary:
                        try:
                            summary_text = summarize_document(doc_text, temperature=temperature)
                        except Exception as summary_error:
                            summary_text = f"Summary generation failed: {summary_error}"

                    safe_doc_id = (
                        doc_name.replace(" ", "_")
                        .replace(".", "_")
                        .replace("/", "_")
                        .lower()[:60]
                    )

                    if enable_rag:
                        try:
                            indexed_chunks = index_document(doc_text, document_name=safe_doc_id)
                            indexed_total += indexed_chunks
                        except Exception as index_error:
                            st.warning(f"Indexing failed for {doc_name}: {index_error}")

                    metrics = result.get("metrics", [])
                    for metric in metrics:
                        metric["document_name"] = doc_name

                    st.session_state.latest_results.append(
                        {
                            "document_name": doc_name,
                            "source_type": doc["source_type"],
                            "text": doc_text,
                            "result": result,
                            "summary": summary_text,
                        }
                    )

                    st.session_state.latest_combined_metrics.extend(metrics)
                    st.session_state.latest_summaries.append(
                        {
                            "document_name": doc_name,
                            "summary": summary_text,
                        }
                    )
                    st.session_state.latest_documents.append(doc_name)

                    preview = doc_text.replace("\n", " ")[:120]
                    st.session_state.history.insert(
                        0,
                        {
                            "preview": f"{doc_name}: {preview}",
                            "result": result,
                            "summary": summary_text,
                        },
                    )

                detected_pairs = []
                seen_tickers = set()

                for item in st.session_state.latest_results:
                    company_name = (item["result"].get("company_name") or "").strip()
                    ticker = company_name_to_ticker(company_name)

                    if company_name and ticker and ticker not in seen_tickers:
                        detected_pairs.append(
                            {
                                "company_name": company_name,
                                "ticker": ticker,
                            }
                        )
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
                    st.success(
                        f"Indexed {indexed_total} chunks across {len(st.session_state.latest_documents)} document(s)."
                    )

            except Exception as e:
                st.error(f"Processing failed: {e}")

results = st.session_state.latest_results
combined_metrics = st.session_state.latest_combined_metrics

if results:
    st.subheader("Processed Documents")

    overview_cols = st.columns(4)
    with overview_cols[0]:
        st.metric("Documents", len(results))
    with overview_cols[1]:
        st.metric("Metrics Extracted", len(combined_metrics))
    with overview_cols[2]:
        detected_companies = {
            item["result"].get("company_name", "").strip()
            for item in results
            if item["result"].get("company_name")
        }
        st.metric("Companies Detected", len(detected_companies))
    with overview_cols[3]:
        st.metric("RAG Status", "Ready" if st.session_state.rag_ready else "Not Ready")

    st.markdown("### Document Summaries")
    for item in results:
        doc_name = item["document_name"]
        result = item["result"]
        summary_text = item["summary"]
        company_name = result.get("company_name") or "Unknown"
        document_type = result.get("document_type") or "Unknown"

        with st.expander(f"{doc_name} | {company_name} | {document_type}"):
            doc_summary = result.get("summary")
            if doc_summary:
                st.markdown("**Document Summary**")
                st.write(doc_summary)

            if summary_text:
                st.markdown("**AI Financial Summary**")
                st.write(summary_text)

    st.markdown("### Combined Extracted Metrics")

    if combined_metrics:
        df = pd.DataFrame(combined_metrics)

        preferred_cols = [
            "document_name",
            "metric",
            "value",
            "period",
            "currency",
            "page",
            "evidence",
            "confidence",
        ]
        existing_cols = [c for c in preferred_cols if c in df.columns]
        df = df[existing_cols].copy()

        if "metric" in df.columns:
            df["category"] = df["metric"].apply(categorize_metric)

        if "value" in df.columns:
            df["numeric_value"] = df["value"].apply(convert_value_to_numeric)

        if "period" in df.columns:
            df["period_index"] = df["period"].apply(parse_period_to_index)

        if "confidence" in df.columns:
            df = df.sort_values(by="confidence", ascending=False)

        display_cols = [col for col in df.columns if col not in ["numeric_value", "period_index"]]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("### Metric Categories")
            if "category" in df.columns:
                category_counts = df["category"].value_counts()
                st.bar_chart(category_counts)

        with chart_col2:
            st.markdown("### Metrics by Document")
            if "document_name" in df.columns:
                doc_counts = df["document_name"].value_counts()
                st.bar_chart(doc_counts)

        chart_df = df.dropna(subset=["numeric_value"]) if "numeric_value" in df.columns else pd.DataFrame()

        if not chart_df.empty and "metric" in chart_df.columns:
            st.markdown("### Financial Metrics Visualization")
            fig = px.bar(
                chart_df,
                x="metric",
                y="numeric_value",
                color="document_name" if "document_name" in chart_df.columns else "category",
                hover_data=["period", "currency", "confidence", "document_name"],
                title="Extracted Financial Metrics Across Documents",
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Trend Detection and Forecasting")
        forecastable_df = (
            df.dropna(subset=["numeric_value", "period_index"])
            if "numeric_value" in df.columns and "period_index" in df.columns
            else pd.DataFrame()
        )

        if not forecastable_df.empty and "metric" in forecastable_df.columns:
            available_metrics = sorted(forecastable_df["metric"].dropna().unique().tolist())

            selected_metric = st.selectbox(
                "Select a metric to analyze",
                available_metrics,
            )

            metric_history = prepare_metric_history(forecastable_df, selected_metric)

            if metric_history.empty:
                st.info("No usable time-series data found for this metric.")
            else:
                st.dataframe(
                    metric_history[
                        [col for col in ["document_name", "metric", "period", "numeric_value"] if col in metric_history.columns]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                forecast_result = forecast_next_value(metric_history)

                if forecast_result:
                    st.success(
                        f"Forecasted next value for {selected_metric}: {forecast_result['forecast_value']:.2f}"
                    )

                    chart_history = metric_history.copy()
                    forecast_row = {
                        "metric": selected_metric,
                        "period": f"Forecast ({forecast_result['next_period_index']})",
                        "numeric_value": forecast_result["forecast_value"],
                        "document_name": "Forecast",
                    }

                    chart_plot_df = pd.concat(
                        [chart_history, pd.DataFrame([forecast_row])],
                        ignore_index=True,
                    )

                    fig_forecast = px.line(
                        chart_plot_df,
                        x="period",
                        y="numeric_value",
                        color="document_name" if "document_name" in chart_plot_df.columns else None,
                        markers=True,
                        title=f"{selected_metric} Trend and Forecast",
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                else:
                    st.info("Need at least two time points to create a forecast.")
        else:
            st.info("No trend-ready metrics found. Use documents with periods like 2023, FY2024, or Q1 2025.")

        st.markdown("### AI Anomaly Detection")
        anomaly_df = (
            detect_metric_anomalies(df, threshold=anomaly_threshold / 100)
            if "numeric_value" in df.columns and "period_index" in df.columns
            else pd.DataFrame()
        )

        if not anomaly_df.empty:
            st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

            fig_anomaly = px.bar(
                anomaly_df,
                x="metric",
                y="change_pct",
                color="anomaly_type",
                hover_data=["document_name", "period", "previous_value", "current_value"],
                title="Detected Metric Anomalies",
                barmode="group",
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
        else:
            st.info("No major anomalies detected with the current threshold.")

        st.markdown("### Downloads")
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download Combined JSON",
                data=json.dumps(results, indent=2),
                file_name="financial_research_results.json",
                mime="application/json",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                "Download Combined CSV",
                data=df.to_csv(index=False),
                file_name="financial_research_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("### Evidence by Document")
        for _, row in df.iterrows():
            metric = row.get("metric", "Unknown metric")
            value = row.get("value", "N/A")
            doc_name = row.get("document_name", "Unknown document")
            page = row.get("page", "N/A")
            confidence = row.get("confidence", "N/A")
            evidence = row.get("evidence", "No evidence provided.")

            with st.expander(f"{doc_name} | {metric} — {value} | page {page} | confidence {confidence}"):
                st.write(evidence)

st.markdown("---")
st.subheader("Ask Questions Across Documents")

question = st.text_input(
    "Ask a financial research question",
    placeholder="Which company had the highest revenue? What risks were mentioned across reports? Compare profitability across the uploaded documents.",
)

if st.button("Ask Research Platform", use_container_width=True):
    if not results:
        st.warning("Process one or more documents first.")
    elif not question.strip():
        st.warning("Enter a question first.")
    elif not st.session_state.rag_ready:
        st.warning("RAG is not ready. Process the documents again with document Q&A enabled.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                retrieval = search_documents(question, top_k=6)
                retrieved_chunks = retrieval.get("documents", [])
                retrieved_sources = retrieval.get("sources", [])

                if not retrieved_chunks:
                    st.info("No relevant context found.")
                else:
                    answer = answer_financial_question(
                        query=question,
                        context_chunks=retrieved_chunks,
                        temperature=temperature,
                    )
                    st.session_state.latest_answer = answer
                    st.session_state.latest_chunks = retrieved_chunks
                    st.session_state.latest_sources = retrieved_sources

            except Exception as qa_error:
                st.error(f"Question answering failed: {qa_error}")

if st.session_state.latest_answer:
    st.markdown("### Answer")
    st.write(st.session_state.latest_answer)

if st.session_state.latest_chunks:
    st.markdown("### Retrieved Context")
    for i, chunk in enumerate(st.session_state.latest_chunks, start=1):
        source_name = (
            st.session_state.latest_sources[i - 1]
            if i - 1 < len(st.session_state.latest_sources)
            else "Unknown"
        )
        with st.expander(f"Chunk {i} | Source: {source_name}"):
            st.write(chunk)

st.markdown("---")
st.subheader("Live Financial Market Data")

detected_companies = st.session_state.detected_companies

if detected_companies:
    detected_text = ", ".join(
        [f"{item['ticker']} ({item['company_name']})" for item in detected_companies]
    )
    st.success(f"Detected companies: {detected_text}")

    selected_detected_ticker = st.selectbox(
        "Select detected company ticker",
        options=[item["ticker"] for item in detected_companies],
        index=0,
        key="selected_detected_ticker_widget",
    )

    selected_company_name = next(
        (
            item["company_name"]
            for item in detected_companies
            if item["ticker"] == selected_detected_ticker
        ),
        "",
    )

    st.session_state.selected_detected_ticker = selected_detected_ticker
    st.session_state.detected_ticker = selected_detected_ticker
    st.session_state.detected_company = selected_company_name

default_symbol = (
    st.session_state.selected_detected_ticker
    or st.session_state.detected_ticker
    or "AAPL"
)

market_col1, market_col2 = st.columns(2)

with market_col1:
    market_symbol = st.text_input(
        "Enter stock ticker",
        value=default_symbol,
        placeholder="AAPL, TSLA, AMZN, MSFT",
        key="market_symbol_input",
    )

with market_col2:
    compare_symbols_text = st.text_input(
        "Compare tickers (comma-separated)",
        value="AAPL,TSLA,AMZN",
    )

auto_load_col1, auto_load_col2 = st.columns(2)

with auto_load_col1:
    auto_detect_toggle = st.checkbox(
        "Auto-load detected company market data",
        value=True,
    )

with auto_load_col2:
    load_market_clicked = st.button("Load Market Data", use_container_width=True)

should_auto_load = (
    auto_detect_toggle
    and bool(st.session_state.selected_detected_ticker or st.session_state.detected_ticker)
    and not st.session_state.auto_market_loaded
)

if load_market_clicked or should_auto_load:
    target_symbol = market_symbol.strip() or st.session_state.selected_detected_ticker or st.session_state.detected_ticker

    if not target_symbol:
        st.warning("Enter a ticker symbol first.")
    else:
        try:
            snapshot = get_market_snapshot(target_symbol)
            st.session_state.latest_market_snapshot = snapshot
            st.session_state.auto_market_loaded = True
            st.session_state.latest_valuation_result = None
            st.session_state.latest_agent_output = None
            st.session_state.latest_autonomous_agent_output = None

            st.markdown("### Market Snapshot")
            snap_cols = st.columns(5)
            with snap_cols[0]:
                st.metric("Symbol", snapshot["symbol"])
            with snap_cols[1]:
                st.metric("Price", f"{snapshot['price']:.2f}" if snapshot["price"] is not None else "N/A")
            with snap_cols[2]:
                st.metric("Change", f"{snapshot['change']:.2f}" if snapshot["change"] is not None else "N/A")
            with snap_cols[3]:
                st.metric("Change %", f"{snapshot['change_pct']:.2f}%" if snapshot["change_pct"] is not None else "N/A")
            with snap_cols[4]:
                st.metric("Market Cap", f"{int(snapshot['market_cap']):,}" if snapshot["market_cap"] is not None else "N/A")

            hist_df = get_price_history(target_symbol, period="6mo", interval="1d")

            if not hist_df.empty and "Close" in hist_df.columns:
                st.markdown("### Price History")
                fig_market = px.line(
                    hist_df,
                    x="Date",
                    y="Close",
                    title=f"{target_symbol.upper()} Closing Price (6 Months)",
                )
                st.plotly_chart(fig_market, use_container_width=True)

            compare_symbols = [s.strip().upper() for s in compare_symbols_text.split(",") if s.strip()]
            compare_df = compare_market_performance(compare_symbols, period="6mo")

            if not compare_df.empty:
                st.markdown("### Comparative Market Performance")
                fig_compare = px.line(
                    compare_df,
                    x="Date",
                    y="Normalized",
                    color="Symbol",
                    title="Normalized 6-Month Market Performance",
                )
                st.plotly_chart(fig_compare, use_container_width=True)

        except Exception as market_error:
            st.error(f"Market data load failed: {market_error}")

if st.session_state.latest_market_snapshot:
    snapshot = st.session_state.latest_market_snapshot

    st.markdown("### Market + Document Insight")

    if results:
        st.caption("Optional: compare the uploaded reports with current market data.")

        if st.button("Compare Uploaded Reports with Market Data", use_container_width=True):
            try:
                market_context = (
                    f"Ticker: {snapshot['symbol']}\n"
                    f"Price: {snapshot['price']}\n"
                    f"Previous Close: {snapshot['previous_close']}\n"
                    f"Change: {snapshot['change']}\n"
                    f"Change %: {snapshot['change_pct']}\n"
                    f"Day High: {snapshot['day_high']}\n"
                    f"Day Low: {snapshot['day_low']}\n"
                    f"Market Cap: {snapshot['market_cap']}"
                )

                comparison_context = []

                if st.session_state.latest_chunks:
                    comparison_context.extend(st.session_state.latest_chunks)
                else:
                    for item in results[:3]:
                        doc_summary = item["result"].get("summary")
                        if doc_summary:
                            comparison_context.append(doc_summary)
                        else:
                            comparison_context.append(item["text"][:2000])

                comparison_context.append(market_context)

                answer = answer_financial_question(
                    query=(
                        f"Compare the uploaded financial reports with the current market data for "
                        f"{snapshot['symbol']}. Highlight any alignment or mismatch between "
                        f"reported fundamentals and market performance."
                    ),
                    context_chunks=comparison_context,
                    temperature=temperature,
                )

                st.markdown("#### Comparison Insight")
                st.write(answer)

            except Exception as compare_error:
                st.error(f"Comparison failed: {compare_error}")
    else:
        st.info("Process at least one document first to use market + document comparison.")

st.markdown("---")
st.subheader("AI Valuation Model")

if st.button("Run Valuation Model", use_container_width=True):
    if not combined_metrics:
        st.warning("Process documents first.")
    elif not st.session_state.latest_market_snapshot:
        st.warning("Load market data first.")
    else:
        try:
            valuation = build_valuation_summary(
                metrics=combined_metrics,
                market_snapshot=st.session_state.latest_market_snapshot,
            )
            st.session_state.latest_valuation_result = valuation
            st.session_state.latest_agent_output = None
            st.session_state.latest_autonomous_agent_output = None

            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                market_cap = valuation.get("market_cap")
                st.metric(
                    "Market Cap",
                    f"{market_cap:,.0f}" if market_cap is not None else "N/A",
                )
            with vcol2:
                fair_value = valuation.get("estimated_fair_value")
                st.metric(
                    "Estimated Fair Value",
                    f"{fair_value:,.0f}" if fair_value is not None else "N/A",
                )
            with vcol3:
                gap = valuation.get("valuation_gap_pct")
                st.metric(
                    "Valuation Gap %",
                    f"{gap:.2f}%" if gap is not None else "N/A",
                )

            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                revenue = valuation.get("revenue")
                st.metric(
                    "Revenue Used",
                    f"{revenue:,.0f}" if revenue is not None else "N/A",
                )
            with scol2:
                net_income = valuation.get("net_income")
                st.metric(
                    "Net Income Used",
                    f"{net_income:,.0f}" if net_income is not None else "N/A",
                )
            with scol3:
                st.metric("AI Valuation Signal", valuation.get("signal", "N/A"))

            detail_df = pd.DataFrame(
                [
                    {"Method": "P/E Fair Value", "Value": valuation.get("pe_fair_value")},
                    {"Method": "Revenue Multiple Value", "Value": valuation.get("revenue_multiple_value")},
                    {"Method": "Estimated Fair Value", "Value": valuation.get("estimated_fair_value")},
                    {"Method": "Current Market Cap", "Value": valuation.get("market_cap")},
                ]
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

            chart_df = detail_df.dropna(subset=["Value"])
            if not chart_df.empty:
                fig_val = px.bar(
                    chart_df,
                    x="Method",
                    y="Value",
                    title="Valuation Comparison",
                )
                st.plotly_chart(fig_val, use_container_width=True)

            st.info(
                "This is a simplified educational valuation model using extracted metrics and "
                "basic multiples. It is not investment advice."
            )

        except Exception as valuation_error:
            st.error(f"Valuation model failed: {valuation_error}")

st.markdown("---")
st.subheader("AI Financial Analyst Agent")

if st.button("Run AI Financial Analyst Agent", use_container_width=True):
    if not results:
        st.warning("Process documents first.")
    elif not st.session_state.latest_market_snapshot:
        st.warning("Load market data first.")
    else:
        try:
            company_name = st.session_state.detected_company or "Unknown Company"
            ticker = st.session_state.detected_ticker or "N/A"
            market_snapshot = st.session_state.latest_market_snapshot
            sentiment_result = st.session_state.latest_sentiment_result
            metrics = combined_metrics

            agent_output = run_financial_analyst_agent(
                company_name=company_name,
                ticker=ticker,
                metrics=metrics,
                market_snapshot=market_snapshot,
                sentiment_result=sentiment_result,
            )

            st.session_state.latest_agent_output = agent_output
            st.session_state.latest_valuation_result = agent_output.get("valuation_result")
            st.session_state.latest_autonomous_agent_output = None

        except Exception as agent_error:
            st.error(f"AI analyst agent failed: {agent_error}")

if st.session_state.latest_agent_output:
    agent_output = st.session_state.latest_agent_output

    valuation_result = agent_output.get("valuation_result")
    investment_thesis = agent_output.get("investment_thesis")
    research_report = agent_output.get("research_report")

    if valuation_result:
        st.markdown("### Agent Valuation Summary")
        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            fair_value = valuation_result.get("estimated_fair_value")
            st.metric(
                "Fair Value",
                f"{fair_value:,.0f}" if fair_value is not None else "N/A"
            )
        with acol2:
            gap = valuation_result.get("valuation_gap_pct")
            st.metric(
                "Gap %",
                f"{gap:.2f}%" if gap is not None else "N/A"
            )
        with acol3:
            st.metric("Signal", valuation_result.get("signal", "N/A"))

    if investment_thesis:
        st.markdown(investment_thesis)

    if research_report:
        with st.expander("Full AI Research Report"):
            st.markdown(research_report)

        combined_download = research_report
        if investment_thesis:
            combined_download += "\n\n" + investment_thesis

        st.download_button(
            "Download Analyst Report",
            data=combined_download,
            file_name="ai_financial_analyst_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.markdown("---")
st.subheader("Autonomous Financial Analyst")

if st.button("Run Autonomous Financial Agent", use_container_width=True):
    if not results:
        st.warning("Process documents first.")
    elif not st.session_state.latest_market_snapshot:
        st.warning("Load market data first.")
    else:
        try:
            company_name = st.session_state.detected_company or "Unknown Company"
            ticker = st.session_state.detected_ticker or "N/A"
            market_snapshot = st.session_state.latest_market_snapshot
            sentiment_result = st.session_state.latest_sentiment_result
            metrics = combined_metrics

            autonomous_output = run_autonomous_financial_agent(
                company_name=company_name,
                ticker=ticker,
                metrics=metrics,
                market_snapshot=market_snapshot,
                sentiment_result=sentiment_result,
            )

            st.session_state.latest_autonomous_agent_output = autonomous_output
            st.session_state.latest_valuation_result = autonomous_output.get("valuation_result")

        except Exception as autonomous_error:
            st.error(f"Autonomous financial agent failed: {autonomous_error}")

if st.session_state.latest_autonomous_agent_output:
    auto_output = st.session_state.latest_autonomous_agent_output

    top_cols = st.columns(3)
    with top_cols[0]:
        st.metric("Recommendation", auto_output.get("recommendation", "N/A"))
    with top_cols[1]:
        valuation_result = auto_output.get("valuation_result") or {}
        gap = valuation_result.get("valuation_gap_pct")
        st.metric("Valuation Gap %", f"{gap:.2f}%" if gap is not None else "N/A")
    with top_cols[2]:
        st.metric("Risk Flags", len(auto_output.get("risks", [])))

    if auto_output.get("final_brief"):
        st.markdown(auto_output["final_brief"])

    with st.expander("Full Autonomous Agent Report"):
        if auto_output.get("research_report"):
            st.markdown(auto_output["research_report"])

    st.download_button(
        "Download Autonomous Agent Brief",
        data=auto_output.get("final_brief", ""),
        file_name="autonomous_financial_agent_brief.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.markdown("---")
st.subheader("AI Equity Research Report")

if st.button("Generate AI Research Report", use_container_width=True):
    if not results:
        st.warning("Process documents first.")
    else:
        company_name = st.session_state.detected_company or "Unknown Company"
        ticker = st.session_state.detected_ticker or "N/A"
        market_snapshot = st.session_state.latest_market_snapshot
        financial_metrics = combined_metrics
        sentiment_result = st.session_state.latest_sentiment_result

        valuation_result = st.session_state.latest_valuation_result
        if valuation_result is None and market_snapshot and financial_metrics:
            valuation_result = build_valuation_summary(
                metrics=financial_metrics,
                market_snapshot=market_snapshot,
            )

        report = generate_equity_research_report(
            company_name=company_name,
            ticker=ticker,
            financial_metrics=financial_metrics,
            market_snapshot=market_snapshot,
            sentiment_result=sentiment_result,
            valuation_result=valuation_result,
        )

        st.markdown(report)

        st.download_button(
            "Download Research Report",
            data=report,
            file_name="equity_research_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

st.markdown("---")
st.subheader("Earnings Call Sentiment Analysis")

transcript_text = st.text_area(
    "Paste earnings call transcript text",
    height=250,
    placeholder="Paste CEO/CFO commentary or earnings call transcript here...",
)

if st.button("Analyze Earnings Sentiment", use_container_width=True):
    if not transcript_text.strip():
        st.warning("Paste transcript text first.")
    else:
        try:
            sentiment_result = analyze_earnings_sentiment(transcript_text)
            st.session_state.latest_sentiment_result = sentiment_result
            st.session_state.latest_agent_output = None
            st.session_state.latest_autonomous_agent_output = None

            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Label", sentiment_result["sentiment_label"])
            with metric_cols[1]:
                st.metric("Score", sentiment_result["sentiment_score"])
            with metric_cols[2]:
                st.metric("Positive Hits", sentiment_result["positive_hits"])
            with metric_cols[3]:
                st.metric("Negative Hits", sentiment_result["negative_hits"])
            with metric_cols[4]:
                st.metric("Guidance Hits", sentiment_result["guidance_hits"])

            extra_cols = st.columns(2)
            with extra_cols[0]:
                st.metric("Risk Mentions", sentiment_result["risk_hits"])
            with extra_cols[1]:
                st.metric("Words Analyzed", sentiment_result["token_count"])

            sentiment_df = pd.DataFrame(
                [
                    {"Category": "Positive", "Count": sentiment_result["positive_hits"]},
                    {"Category": "Negative", "Count": sentiment_result["negative_hits"]},
                    {"Category": "Guidance", "Count": sentiment_result["guidance_hits"]},
                    {"Category": "Risk", "Count": sentiment_result["risk_hits"]},
                ]
            )

            fig_sentiment = px.bar(
                sentiment_df,
                x="Category",
                y="Count",
                title="Transcript Signal Breakdown",
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

            top_terms = sentiment_result.get("top_terms", {})
            if top_terms:
                st.markdown("### Frequent Terms")
                terms_df = pd.DataFrame(
                    [{"Term": k, "Count": v} for k, v in top_terms.items()]
                )
                st.dataframe(terms_df, use_container_width=True, hide_index=True)

        except Exception as sentiment_error:
            st.error(f"Sentiment analysis failed: {sentiment_error}")

st.subheader("Recent Runs")
if not st.session_state.history:
    st.info("No history yet.")
else:
    for i, item in enumerate(st.session_state.history, start=1):
        with st.expander(f"Run #{i}: {item['preview']}"):
            if item.get("summary"):
                st.markdown("**AI Summary**")
                st.write(item["summary"])
            st.json(item["result"])
