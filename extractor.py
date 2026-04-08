import json
import os
import re
import fitz
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"


class FinancialMetric(BaseModel):
    metric: str = Field(description="Metric name such as Revenue, Net Income, EPS, EBITDA, Cash Flow.")
    value: str | None = Field(description="Metric value exactly as stated in the document.")
    period: str | None = Field(description="Relevant period such as Q4 2025, FY2025, 2024, etc.")
    currency: str | None = Field(description="Currency symbol or code if present.")
    page: int | None = Field(description="1-based page number where the metric appears.")
    evidence: str | None = Field(description="Short supporting quote from the document.")
    confidence: float | None = Field(description="Confidence score between 0 and 1.")


class ExtractionResult(BaseModel):
    company_name: str | None = Field(description="Company name if detected.")
    document_type: str | None = Field(
        description="Type of document: annual report, earnings release, 10-K, 10-Q, financial statement, etc."
    )
    summary: str | None = Field(description="Short summary of what the document contains.")
    metrics: list[FinancialMetric] = Field(description="List of extracted financial metrics.")


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 8) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    try:
        total_pages = min(len(doc), max_pages)
        for i in range(total_pages):
            page = doc.load_page(i)
            page_text = page.get_text("text").strip()
            if page_text:
                pages.append(f"[PAGE {i + 1}]\n{page_text}")
    finally:
        doc.close()
    return "\n\n".join(pages).strip()


def _get_llm(temperature: float = 0.0, use_fallback: bool = False) -> ChatGroq:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to your .env file or Streamlit secrets.")
    model = FALLBACK_MODEL if use_fallback else PRIMARY_MODEL
    return ChatGroq(model=model, temperature=temperature, api_key=groq_api_key)


def extract_financial_metrics_from_text(text: str, temperature: float = 0.0) -> dict:
    parser = JsonOutputParser(pydantic_object=ExtractionResult)

    prompt = ChatPromptTemplate.from_template(
        """You are a senior financial analyst and document extraction specialist.

Your task is to extract ALL financial metrics from the document below with high precision.

Return ONLY valid JSON matching this schema:
{format_instructions}

EXTRACTION RULES:
1. Extract every financial metric explicitly stated — do not invent or estimate values.
2. Prioritize: Revenue, Net Income, EPS, Operating Income, Gross Profit, Gross Margin,
   EBITDA, Free Cash Flow, Total Assets, Total Liabilities, Total Equity, Debt,
   Cash & Equivalents, CapEx, Dividends, Guidance/Outlook figures.
3. Preserve exact values as written (e.g. "$4.2B", "12.3%", "$(450M)").
4. For each metric include a short evidence quote (max 20 words) from the source text.
5. Use [PAGE X] markers to assign page numbers.
6. Set confidence: 1.0 = explicitly stated, 0.7 = clearly implied, 0.5 = uncertain.
7. Detect currency from symbols ($, €, £, ¥) or stated text.
8. For periods: standardize to formats like Q1 2024, FY2024, 2023, H1 2024.
9. Use null only when genuinely unknown — never guess.

Document text:
{text}
"""
    )

    try:
        llm = _get_llm(temperature, use_fallback=False)
        chain = prompt | llm | parser
        return chain.invoke({
            "text": text[:28000],
            "format_instructions": parser.get_format_instructions(),
        })
    except Exception:
        llm = _get_llm(temperature, use_fallback=True)
        chain = prompt | llm | parser
        return chain.invoke({
            "text": text[:20000],
            "format_instructions": parser.get_format_instructions(),
        })


def summarize_document(text: str, temperature: float = 0.0) -> str:
    prompt = ChatPromptTemplate.from_template(
        """You are a senior equity research analyst writing for institutional investors.

Analyze the following financial document and provide a structured summary:

**Performance Highlights:** 2-3 key financial results (revenue, profit, margins) with actual numbers.
**Cash Flow & Balance Sheet:** Key liquidity and leverage observations.
**Guidance & Outlook:** Any forward-looking statements or management guidance.
**Key Risks:** 1-2 most significant risks mentioned.
**Analyst Take:** One sentence overall assessment of financial health.

Be concise, factual, and data-driven. Use actual numbers from the document where possible.

Document text:
{text}
"""
    )

    try:
        llm = _get_llm(temperature, use_fallback=False)
        chain = prompt | llm
        response = chain.invoke({"text": text[:16000]})
    except Exception:
        llm = _get_llm(temperature, use_fallback=True)
        chain = prompt | llm
        response = chain.invoke({"text": text[:12000]})

    return response.content if hasattr(response, "content") else str(response)


def answer_financial_question(
    query: str, context_chunks: list[str], temperature: float = 0.0
) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    prompt = ChatPromptTemplate.from_template(
        """You are a senior financial research analyst with deep expertise in equity analysis.

Answer the question below using ONLY the provided context from uploaded financial documents.
Be specific, cite figures where available, and structure your answer clearly.
If the answer cannot be found, state: "This information was not found in the uploaded documents."

Context from documents:
{context}

Question: {query}

Provide a thorough, analyst-quality answer:
"""
    )

    try:
        llm = _get_llm(temperature, use_fallback=False)
        chain = prompt | llm
        response = chain.invoke({"context": context[:16000], "query": query})
    except Exception:
        llm = _get_llm(temperature, use_fallback=True)
        chain = prompt | llm
        response = chain.invoke({"context": context[:12000], "query": query})

    return response.content if hasattr(response, "content") else str(response)


def categorize_metric(metric: str) -> str:
    metric = (metric or "").lower()
    if any(k in metric for k in ["revenue", "sales", "turnover"]):
        return "Revenue"
    if any(k in metric for k in ["income", "profit", "earnings", "eps", "ebitda", "ebit"]):
        return "Profitability"
    if any(k in metric for k in ["cash", "free cash flow", "operating cash"]):
        return "Cash Flow"
    if any(k in metric for k in ["asset", "liabilit", "equity", "debt", "leverage"]):
        return "Balance Sheet"
    if any(k in metric for k in ["expense", "cost", "capex", "depreciation"]):
        return "Operations"
    if any(k in metric for k in ["margin", "return", "roe", "roa", "roic"]):
        return "Margins & Returns"
    if any(k in metric for k in ["guidance", "outlook", "forecast", "target"]):
        return "Guidance"
    return "Other"


def convert_value_to_numeric(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    is_negative = ("(" in text and ")" in text) or text.lstrip("$€£¥ ").startswith("-")

    multiplier = 1.0
    upper = text.upper()
    if "BILLION" in upper or upper.rstrip("BILLION").endswith("B"):
        multiplier = 1_000_000_000
    elif "MILLION" in upper or upper.rstrip("MILLION").endswith("M"):
        multiplier = 1_000_000
    elif "THOUSAND" in upper or upper.rstrip("THOUSAND").endswith("K"):
        multiplier = 1_000

    cleaned = re.sub(r"[^0-9.]", "", text)
    if not cleaned or cleaned.count(".") > 1:
        return None

    try:
        number = float(cleaned) * multiplier
        if is_negative and number > 0:
            number = -number
        return number
    except ValueError:
        return None


def parse_period_to_index(period: str):
    if not period:
        return None
    text = str(period).strip().upper()

    quarter_match = re.match(r"Q([1-4])\s*[\s\-]?\s*(\d{4})", text)
    if quarter_match:
        return int(quarter_match.group(2)) * 4 + int(quarter_match.group(1))

    half_match = re.match(r"H([12])\s*(\d{4})", text)
    if half_match:
        return int(half_match.group(2)) * 2 + int(half_match.group(1))

    fy_match = re.match(r"FY\s*(\d{4})", text)
    if fy_match:
        return int(fy_match.group(1))

    year_match = re.match(r"(\d{4})", text)
    if year_match:
        return int(year_match.group(1))

    return None


def clean_llm_json(result):
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        text = result.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text.strip())
    raise ValueError(f"Unexpected LLM response format: {type(result)}")
