# Financial AI Research Platform

AI-powered financial research platform for document extraction, multi-document RAG, forecasting, market data analysis, sentiment detection, valuation modeling, and equity research report generation.

## Overview

This project is an end-to-end AI financial analysis system built for research, portfolio demonstration, and practical experimentation. It combines document intelligence, retrieval-augmented generation, forecasting, live market data, sentiment analysis, anomaly detection, valuation, and automated equity research reporting in a single Streamlit application.

## Features

### Financial document intelligence
- Upload one or more financial PDFs
- Extract structured financial metrics with AI
- Capture evidence and source context
- Generate AI summaries from uploaded reports

### Multi-document RAG
- Index multiple financial documents with ChromaDB
- Retrieve relevant chunks using semantic search
- Ask cross-document financial research questions
- Generate grounded answers from uploaded reports

### Financial analytics dashboard
- Compare metrics across multiple documents
- Categorize financial KPIs automatically
- Visualize extracted values interactively
- Inspect metric-level evidence by source file

### Trend detection and forecasting
- Detect trend-ready metrics from extracted data
- Forecast the next KPI value
- Visualize historical and predicted values

### Live market data integration
- Pull market data using Yahoo Finance
- Auto-detect company tickers where possible
- Display price, change, market cap, and price history
- Compare multiple ticker performances

### Earnings call sentiment analysis
- Analyze pasted transcript or commentary text
- Identify bullish, bearish, or neutral sentiment
- Count positive, negative, guidance, and risk signals
- Highlight frequent management terms

### AI anomaly detection
- Flag unusual spikes and drops in KPIs
- Detect abnormal period-over-period changes
- Visualize anomalies by document and metric

### AI valuation model
- Estimate fair value using simple multiple-based methods
- Compare estimated fair value to market capitalization
- Generate an AI valuation signal:
  - Potentially Undervalued
  - Fairly Valued
  - Potentially Overvalued

### AI equity research report generation
- Generate AI-written equity research reports
- Combine:
  - extracted metrics
  - market data
  - sentiment analysis
  - valuation output

### AI financial analyst agent
- Detect company
- Load market context
- Analyze metrics
- Incorporate sentiment
- Run valuation
- Generate an investment thesis and analyst-style report

## Tech Stack

### Frontend
- Streamlit

### AI and LLM
- Groq
- LangChain

### Vector database
- ChromaDB

### Data and ML
- pandas
- scikit-learn

### Market data
- yfinance

### Visualization
- Plotly

### PDF processing
- PyMuPDF

### Deployment
- Docker
- Streamlit Community Cloud

## Project Structure

```text
financial-ai-research-platform
│
├── app.py
├── extractor.py
├── rag_utils.py
├── forecast_utils.py
├── market_utils.py
├── sentiment_utils.py
├── anomaly_utils.py
├── valuation_utils.py
├── research_report_utils.py
├── analyst_agent_utils.py
├── requirements.txt
├── runtime.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── .env

How It Works

Upload one or more financial PDFs

Extract structured metrics and summaries

Index documents into a vector database

Ask cross-document research questions with RAG

Load live market data for detected companies

Run sentiment analysis on earnings call text

Detect anomalies across KPIs

Run valuation analysis

Generate AI equity research reports

Generate an AI analyst investment thesis

Example Questions
Document Q&A

Which company had the highest revenue growth?

Compare Apple and Tesla profitability.

What risks were mentioned across the uploaded reports?

Which company showed stronger net income performance?

Market comparison

Compare Tesla’s uploaded report with current TSLA market performance.

How does Apple’s reported growth align with current market sentiment?

Sentiment analysis

Is this earnings call bullish or bearish?

How many risk signals appear in this transcript?

Analyst workflow

Generate an investment thesis for Tesla.

Is the company potentially undervalued based on extracted financial metrics?

Installation
1. Clone the repository
git clone https://github.com/BaselAtiyire/financial-ai-research-platform.git
cd financial-ai-research-platform
2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Add environment variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here
Run Locally
python -m streamlit run app.py
Run with Docker
Build image
docker build -t financial-ai-research-platform .
Run container
docker run -p 8502:8501 --env-file .env financial-ai-research-platform

Then open:

http://localhost:8502
Deploy to Streamlit Community Cloud
Required files

Make sure these are in the repo root:

app.py

requirements.txt

runtime.txt

runtime.txt
python-3.12
Streamlit secrets

Add this in Streamlit Cloud secrets:

GROQ_API_KEY="your_actual_groq_api_key"
Recommended GitHub Topics
ai
machine-learning
financial-analysis
equity-research
rag
langchain
streamlit
chromadb
sentiment-analysis
forecasting
Resume Description

Financial AI Research Platform
Built an AI-powered financial research platform that extracts structured metrics from corporate reports using LLMs, supports multi-document RAG analysis, performs KPI forecasting and anomaly detection, integrates live market data, analyzes earnings-call sentiment, runs valuation models, and generates AI equity research reports and investment theses.

Notes

This platform is for educational and research purposes

The valuation model is simplified and is not investment advice

Market data availability depends on Yahoo Finance

ChromaDB works best in Python 3.11 or 3.12 environments

Future Improvements

SEC 10-K and 10-Q auto-ingestion

DCF valuation model

Intrinsic value benchmarking

Financial news sentiment ingestion

Portfolio optimization module

AI buy/hold/sell recommendation system

User authentication and saved workspaces

Author

Basil Atiyire
AI Engineer | Data Scientist | ML Engineer | Financial AI Enthusiast

GitHub: BaselAtiyire


For the very top of the repo, use this short description too:

**Description**  
`AI-powered financial research platform for document extraction, multi-document RAG, forecasting, market data analysis, sentiment detection, valuation modeling, and equity research report generation.`