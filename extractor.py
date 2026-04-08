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
        description="Type of document such as annual report, earnings release, or financial statement."
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
            pages.append(f"[PAGE {i + 1}]\n{page_text}")
    finally:
        doc.close()
    return "\n\n".join(pages).strip()


def _get_llm(temperature: float = 0.0) -> ChatGroq:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Add it to your .env file or Streamlit secrets.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        api_key=groq_api_key,
    )


def extract_financial_metrics_from_text(text: str, temperature: float = 0.0) -> dict:
    parser = JsonOutputParser(pydantic_object=ExtractionResult)

    prompt = ChatPromptTemplate.from_template(
        """
You are a financial document extraction assistant.

Return ONLY valid JSON matching this schema:
{format_instructions}

Rules:
- Extract only metrics explicitly present in the text.
- Keep values exactly as written where possible.
- Include page number when the source contains [PAGE X].
- Include a short evidence quote for each metric.
- Use null when unknown.
- Do not invent data.
- Prefer important finance metrics such as Revenue, Net Income, EPS, Operating Income,
  Gross Profit, Gross Margin, EBITDA, Free Cash Flow, Total Assets, Total Liabilities, Guidance.

Document text:
{text}
"""
    )

    llm = _get_llm(temperature)
    chain = prompt | llm | parser
    result = chain.invoke(
        {
            "text": text[:24000],
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return result


def summarize_document(text: str, temperature: float = 0.0) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
You are a financial analyst assistant.

Summarize the key financial insights from the following document in 4 concise bullet points.
Focus on performance, profitability, cash flow, guidance, trends, and important risks if present.

Document text:
{text}
"""
    )

    llm = _get_llm(temperature)
    chain = prompt | llm
    response = chain.invoke({"text": text[:12000]})

    if hasattr(response, "content"):
        return response.content
    return str(response)


def answer_financial_question(
    query: str, context_chunks: list[str], temperature: float = 0.0
) -> str:
    context = "\n\n".join(context_chunks)

    prompt = ChatPromptTemplate.from_template(
        """
You are a financial analyst assistant.

Answer the user's question using only the context below.
If the answer is not in the context, say clearly that you could not find it in the uploaded documents.

Context:
{context}

Question:
{query}
"""
    )

    llm = _get_llm(temperature)
    chain = prompt | llm
    response = chain.invoke({"context": context[:12000], "query": query})

    if hasattr(response, "content"):
        return response.content
    return str(response)


def categorize_metric(metric: str) -> str:
    metric = (metric or "").lower()
    if "revenue" in metric or "sales" in metric:
        return "Revenue"
    if "income" in metric or "profit" in metric or "earnings" in metric or "eps" in metric:
        return "Profitability"
    if "cash" in metric or "free cash flow" in metric:
        return "Cash Flow"
    if "asset" in metric or "liabilit" in metric or "equity" in metric:
        return "Balance Sheet"
    if "expense" in metric or "cost" in metric or "operating" in metric:
        return "Operations"
    return "Other"


def convert_value_to_numeric(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    is_negative = "(" in text and ")" in text
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned.count(".") > 1:
        return None
    try:
        number = float(cleaned)
        if is_negative and number > 0:
            number = -number
        return number
    except ValueError:
        return None


def parse_period_to_index(period: str):
    if not period:
        return None
    text = str(period).strip().upper()

    quarter_match = re.match(r"Q([1-4])\s+(\d{4})", text)
    if quarter_match:
        q = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        return year * 4 + q

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
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
        return json.loads(text)
    raise ValueError(f"Unexpected LLM response format: {type(result)}")
