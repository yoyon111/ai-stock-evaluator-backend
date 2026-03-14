"""
Investment Research AI Committee - LangGraph Multi-Agent System
Now with streaming progress callbacks via `emit`.
"""

import os
from typing import TypedDict, Callable, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import yfinance as yf
from tavily import TavilyClient
from dotenv import load_dotenv


MAX_RESEARCH_PASSES = 3
MAX_CYCLES = 5

INTENT_DATA_MAP = {
    "outlook":        ["stock_info", "earnings", "price_history", "analyst_recommendations", "news_outlook"],
    "comparison":     ["stock_info", "earnings", "price_history", "news_general"],
    "catalyst":       ["stock_info", "news_catalyst", "news_general"],
    "historical":     ["stock_info", "price_history_5y", "earnings"],
    "recommendation": ["stock_info", "earnings", "price_history", "analyst_recommendations", "news_general"],
}

ALL_INTENTS = list(INTENT_DATA_MAP.keys())


class AgentState(TypedDict):
    messages: list
    user_query: str
    plan: dict
    research_data: dict
    data_checklist: dict
    research_pass_count: int
    quant_analysis: str
    qual_analysis: str
    moderator_decision: str
    moderator_request: str
    moderator_verdict: str
    cycle_count: int
    writer_verdict: str
    final_report: str
    risk_flags: str
    sources: list
    _emit: object   # streaming callback injected at runtime


class FinancialDataTools:

    @staticmethod
    def get_stock_info(ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "current_price": info.get("currentPrice", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "peg_ratio": info.get("pegRatio", "N/A"),
                "price_to_book": info.get("priceToBook", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "beta": info.get("beta", "N/A"),
                "revenue": info.get("totalRevenue", "N/A"),
                "revenue_growth": info.get("revenueGrowth", "N/A"),
                "gross_margins": info.get("grossMargins", "N/A"),
                "profit_margin": info.get("profitMargins", "N/A"),
                "operating_margins": info.get("operatingMargins", "N/A"),
                "ebitda": info.get("ebitda", "N/A"),
                "free_cashflow": info.get("freeCashflow", "N/A"),
                "debt_to_equity": info.get("debtToEquity", "N/A"),
                "current_ratio": info.get("currentRatio", "N/A"),
                "return_on_equity": info.get("returnOnEquity", "N/A"),
                "return_on_assets": info.get("returnOnAssets", "N/A"),
                "short_ratio": info.get("shortRatio", "N/A"),
                "analyst_target_price": info.get("targetMeanPrice", "N/A"),
                "recommendation_mean": info.get("recommendationMean", "N/A"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions", "N/A"),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    @staticmethod
    def get_earnings(ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.earnings_history
            quarterly = stock.quarterly_earnings
            income = stock.quarterly_income_stmt
            return {
                "ticker": ticker,
                "earnings_history": hist.to_dict() if hist is not None else {},
                "quarterly_earnings": quarterly.to_dict() if quarterly is not None else {},
                "quarterly_income": income.to_dict() if income is not None else {},
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    @staticmethod
    def get_price_history(ticker: str, period: str = "1y") -> dict:
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period)
            if len(history) == 0:
                return {"ticker": ticker, "error": "No data"}
            closes = history["Close"]
            return {
                "ticker": ticker,
                "period": period,
                "start_price": round(float(closes.iloc[0]), 2),
                "end_price": round(float(closes.iloc[-1]), 2),
                "high": round(float(closes.max()), 2),
                "low": round(float(closes.min()), 2),
                "price_change_pct": round(
                    (float(closes.iloc[-1]) - float(closes.iloc[0])) / float(closes.iloc[0]) * 100, 2
                ),
                "avg_volume": int(history["Volume"].mean()),
                "volatility_30d": round(
                    float(closes.pct_change().rolling(30).std().iloc[-1]) * 100, 2
                ) if len(closes) > 30 else "N/A",
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    @staticmethod
    def get_analyst_recommendations(ticker: str) -> dict:
        try:
            stock = yf.Ticker(ticker)
            recs = stock.recommendations
            upgrades = stock.upgrades_downgrades
            result = {"ticker": ticker}
            if recs is not None and len(recs) > 0:
                result["recent_recommendations"] = recs.tail(10).to_dict()
            if upgrades is not None and len(upgrades) > 0:
                result["recent_upgrades_downgrades"] = upgrades.head(10).to_dict()
            return result
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}


class WebSearchTools:

    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, days: int = 30, max_results: int = 5) -> dict:
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                days=days,
            )
            results = response.get("results", [])
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0),
                        "published_date": r.get("published_date", ""),
                    }
                    for r in results
                ],
            }
        except Exception as e:
            return {"query": query, "results": [], "error": str(e)}


class PlannerAgent:

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        emit = state.get("_emit")

        if emit:
            emit("agent_start", {
                "agent": "planner", "label": "Planner", "icon": "🎯",
                "message": "Identifying tickers and research intents..."
            })

        prompt = f"""You are a financial research planner. Analyze this query.

Query: {state['user_query']}

STEP 1 - Extract ticker symbols. Convert names to official tickers (Google→GOOGL, Meta→META).
STEP 2 - Identify ALL intents present. Options: outlook, comparison, catalyst, historical, recommendation
STEP 3 - Output in EXACTLY this format, nothing else:
TICKERS: AAPL, MSFT
INTENTS: outlook, comparison
TIMEFRAME: 1y"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        plan = {"tickers": [], "intents": [], "timeframe": "1y", "raw": raw}
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("TICKERS:"):
                plan["tickers"] = [t.strip().upper() for t in line.replace("TICKERS:", "").split(",") if t.strip()]
            elif line.startswith("INTENTS:"):
                raw_intents = [i.strip().lower() for i in line.replace("INTENTS:", "").split(",")]
                plan["intents"] = [i for i in raw_intents if i in ALL_INTENTS] or ["recommendation"]
            elif line.startswith("TIMEFRAME:"):
                plan["timeframe"] = line.replace("TIMEFRAME:", "").strip()

        checklist = {}
        for ticker in plan["tickers"]:
            for intent in plan["intents"]:
                for data_type in INTENT_DATA_MAP.get(intent, []):
                    checklist[f"{ticker}__{data_type}"] = False

        if emit:
            emit("agent_done", {
                "agent": "planner", "label": "Planner", "icon": "🎯",
                "message": f"Found {len(plan['tickers'])} ticker(s): {', '.join(plan['tickers'])} · {len(checklist)} data points needed",
                "detail": {"tickers": plan["tickers"], "intents": plan["intents"]}
            })

        state["plan"] = plan
        state["data_checklist"] = checklist
        state["research_data"] = {}
        state["research_pass_count"] = 0
        state["quant_analysis"] = ""
        state["qual_analysis"] = ""
        state["moderator_decision"] = ""
        state["moderator_request"] = ""
        state["moderator_verdict"] = ""
        state["cycle_count"] = 0
        state["writer_verdict"] = ""
        state["final_report"] = ""
        state["sources"] = []
        state["messages"].append(AIMessage(
            content=f"Planner: {plan['tickers']} | intents: {plan['intents']} | {len(checklist)} data points needed"
        ))
        return state


class ResearchAgent:

    def __init__(self, llm, tavily_api_key: str):
        self.llm = llm
        self.fin = FinancialDataTools()
        self.web = WebSearchTools(tavily_api_key)

    def process(self, state: AgentState) -> AgentState:
        emit = state.get("_emit")
        pass_num = state["research_pass_count"] + 1
        targeted = bool(state.get("moderator_request"))

        checklist = state["data_checklist"]
        tickers = state["plan"]["tickers"]
        intents = state["plan"]["intents"]
        sources = state.get("sources", [])
        total = len(checklist)
        collected_before = sum(1 for v in checklist.values() if v)

        if emit:
            msg = (
                f"Targeted fetch: {state['moderator_request'][:70]}"
                if targeted
                else f"Pass {pass_num}/{MAX_RESEARCH_PASSES} · collecting {total - collected_before} data points"
            )
            emit("agent_start", {
                "agent": "researcher", "label": "Researcher", "icon": "🔍",
                "message": msg,
            })

        if targeted:
            self._fetch_targeted(state, tickers, sources, emit)
            state["moderator_request"] = ""
        else:
            self._fetch_checklist(state, checklist, intents, sources, emit)

        state["research_pass_count"] = pass_num
        state["sources"] = sources

        collected = sum(1 for v in checklist.values() if v)

        if emit:
            emit("agent_done", {
                "agent": "researcher", "label": "Researcher", "icon": "🔍",
                "message": f"Collected {collected}/{total} data points · {len(sources)} sources found",
            })

        state["messages"].append(AIMessage(
            content=f"Researcher {'targeted' if targeted else f'pass {pass_num}'}: {collected}/{total} collected"
        ))
        return state

    def _fetch_checklist(self, state, checklist, intents, sources, emit=None):
        items = [(k, v) for k, v in checklist.items() if not v]
        for i, (key, _) in enumerate(items):
            ticker, data_type = key.split("__", 1)
            label = data_type.replace("_", " ").title()
            if emit:
                emit("researcher_step", {
                    "message": f"Fetching {label} for {ticker}",
                    "progress": i + 1,
                    "total": len(items),
                })
            self._fetch_item(state, checklist, key, ticker, data_type, intents, sources)

    def _fetch_targeted(self, state, tickers, sources, emit=None):
        request = state["moderator_request"].lower()
        intents = state["plan"]["intents"]

        for ticker in tickers:
            if "recommendation" in request or "analyst" in request or "target" in request:
                key = f"{ticker}__analyst_recommendations"
                if emit:
                    emit("researcher_step", {"message": f"Fetching analyst recommendations for {ticker}"})
                state["research_data"][key] = self.fin.get_analyst_recommendations(ticker)
                state["data_checklist"][key] = True

            if "earnings" in request or "eps" in request or "revenue" in request:
                key = f"{ticker}__earnings_targeted"
                if emit:
                    emit("researcher_step", {"message": f"Fetching earnings data for {ticker}"})
                state["research_data"][key] = self.fin.get_earnings(ticker)

            if "price" in request or "history" in request or "performance" in request:
                key = f"{ticker}__price_history_targeted"
                if emit:
                    emit("researcher_step", {"message": f"Fetching 5Y price history for {ticker}"})
                state["research_data"][key] = self.fin.get_price_history(ticker, "5y")

        for ticker in tickers:
            query = f"{ticker} {state['moderator_request']}"
            key = f"{ticker}__targeted_search_{state['cycle_count']}"
            if emit:
                emit("researcher_step", {"message": f'Searching: "{query[:55]}"'})
            result = self.web.search(query, days=14)
            state["research_data"][key] = result
            for r in result.get("results", []):
                sources.append({"title": r["title"], "url": r["url"], "ticker": ticker})

    def _fetch_item(self, state, checklist, key, ticker, data_type, intents, sources):
        company = state["research_data"].get(f"{ticker}__stock_info", {}).get("company_name", ticker)

        if data_type == "stock_info":
            state["research_data"][key] = self.fin.get_stock_info(ticker)
        elif data_type == "earnings":
            state["research_data"][key] = self.fin.get_earnings(ticker)
        elif data_type == "price_history":
            period = "2y" if "historical" in intents else "1y"
            state["research_data"][key] = self.fin.get_price_history(ticker, period)
        elif data_type == "price_history_5y":
            state["research_data"][key] = self.fin.get_price_history(ticker, "5y")
        elif data_type == "analyst_recommendations":
            state["research_data"][key] = self.fin.get_analyst_recommendations(ticker)
        elif data_type == "news_general":
            result = self.web.search(f"{company} {ticker} stock news latest", days=30)
            state["research_data"][key] = result
            for r in result.get("results", []):
                sources.append({"title": r["title"], "url": r["url"], "ticker": ticker})
        elif data_type == "news_catalyst":
            result = self.web.search(f"why is {ticker} {company} stock moving", days=14)
            state["research_data"][key] = result
            for r in result.get("results", []):
                sources.append({"title": r["title"], "url": r["url"], "ticker": ticker})
        elif data_type == "news_outlook":
            result = self.web.search(f"{ticker} {company} outlook forecast next quarter analyst", days=30)
            state["research_data"][key] = result
            for r in result.get("results", []):
                sources.append({"title": r["title"], "url": r["url"], "ticker": ticker})

        checklist[key] = True


def route_after_research(state: AgentState) -> str:
    checklist = state["data_checklist"]
    gaps = [k for k, v in checklist.items() if not v]
    passes = state["research_pass_count"]

    if state["cycle_count"] > 0:
        return "quant_analyst"
    if gaps and passes < MAX_RESEARCH_PASSES:
        return "researcher"
    return "quant_analyst"


class QuantAnalyst:

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        emit = state.get("_emit")
        cycle = state["cycle_count"]

        if emit:
            emit("agent_start", {
                "agent": "quant", "label": "Quant Analyst", "icon": "📊",
                "message": (
                    f"Re-analyzing numbers (cycle {cycle})..."
                    if cycle > 0
                    else "Crunching valuation ratios, margins, and price performance..."
                )
            })

        focus = ""
        if state.get("moderator_request"):
            focus = f"\nThe Moderator flagged this: {state['moderator_request']}\nAddress this directly."

        prompt = f"""You are a quantitative financial analyst. Analyze the data independently and objectively.

User Query: {state['user_query']}
Intents: {state['plan']['intents']}
Tickers: {state['plan']['tickers']}
{focus}

Research Data:
{state['research_data']}

Analyze using only hard data. Cover:
- Key ratios with exact numbers (P/E, PEG, P/B, margins, ROE, D/E)
- Price performance with specific percentages and timeframes
- Earnings trends — accelerating or decelerating? By exactly how much?
- Valuation: expensive, fair, or cheap vs historical averages and sector?
- Balance sheet: debt, cash, free cash flow trends
- If comparison: explicit side-by-side numbers
- If outlook: what do current numbers imply about trajectory?
- If catalyst: what does the financial data say about the price move?

No vague statements. Exact numbers only. Use clear subheadings."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        state["quant_analysis"] = response.content

        if emit:
            emit("agent_done", {
                "agent": "quant", "label": "Quant Analyst", "icon": "📊",
                "message": "Quantitative analysis complete",
            })

        state["messages"].append(AIMessage(content="Quant Analyst: Analysis complete"))
        return state


class QualAnalyst:

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        emit = state.get("_emit")
        cycle = state["cycle_count"]

        if emit:
            emit("agent_start", {
                "agent": "qual", "label": "Qual Analyst", "icon": "📰",
                "message": (
                    f"Re-analyzing narrative (cycle {cycle})..."
                    if cycle > 0
                    else "Reading news, sentiment, and competitive signals..."
                )
            })

        focus = ""
        if state.get("moderator_request"):
            focus = f"\nThe Moderator flagged this: {state['moderator_request']}\nAddress this directly."

        news_content = ""
        for key, val in state["research_data"].items():
            if "news" in key and isinstance(val, dict):
                for r in val.get("results", []):
                    news_content += f"\n[{r.get('title','')}]({r.get('url','')})\n{r.get('content','')[:400]}\n"

        prompt = f"""You are a qualitative investment analyst. Analyze context, narrative, and market perception independently.

User Query: {state['user_query']}
Intents: {state['plan']['intents']}
Tickers: {state['plan']['tickers']}
{focus}

News & Web Research:
{news_content}

Research Data:
{state['research_data']}

Analyze:
- Recent news: key stories and what they signal
- Industry/competitive landscape: tailwinds or headwinds?
- Management signals: guidance, executive moves, strategy shifts
- Market sentiment and macro factors
- If catalyst: what's driving price action?
- If outlook: what narrative factors shape the next quarter?
- If comparison: qualitative competitive moats and weaknesses

Cite news inline as [Source: Title](URL)."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        state["qual_analysis"] = response.content

        if emit:
            emit("agent_done", {
                "agent": "qual", "label": "Qual Analyst", "icon": "📰",
                "message": "Qualitative analysis complete",
            })

        state["messages"].append(AIMessage(content="Qual Analyst: Analysis complete"))
        return state


class ModeratorAgent:

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        emit = state.get("_emit")
        cycle = state["cycle_count"]

        if emit:
            emit("agent_start", {
                "agent": "moderator", "label": "Moderator", "icon": "⚖️",
                "message": f"Reviewing both analyses for gaps and conflicts (cycle {cycle})..."
            })

        if cycle >= MAX_CYCLES:
            state["moderator_decision"] = "VERDICT"
            state["moderator_verdict"] = self._force_verdict(state)
            if emit:
                emit("agent_done", {
                    "agent": "moderator", "label": "Moderator", "icon": "⚖️",
                    "message": "Max cycles reached — writing final verdict",
                })
            state["messages"].append(AIMessage(content="Moderator: Safety cap — forcing verdict"))
            return state

        prompt = f"""You are a senior investment research moderator.

User Query: {state['user_query']}
Intents: {state['plan']['intents']}
Tickers: {state['plan']['tickers']}
Cycle: {cycle} (max {MAX_CYCLES})

QUANTITATIVE ANALYSIS:
{state['quant_analysis']}

QUALITATIVE ANALYSIS:
{state['qual_analysis']}

STEP 1 — Identify agreements, conflicts, and gaps.
STEP 2 — Decide: NEED_DATA | NEED_ANALYSIS | VERDICT

Output in EXACTLY this format:

---AGREEMENTS---
[bullet list]

---CONFLICTS---
[bullet list with your call on who is right]

---GAPS---
[bullet list, or "None"]

---DECISION---
NEED_DATA: [exact data needed]
or
NEED_ANALYSIS: [exact question]
or
VERDICT

---SYNTHESIS---
[Only if VERDICT: full synthesized conclusion for each intent. Make a call.]"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content

        decision = "VERDICT"
        request = ""
        verdict = ""

        if "---DECISION---" in raw:
            block = raw.split("---DECISION---")[1].split("---")[0].strip()
            if block.startswith("NEED_DATA:"):
                decision = "NEED_DATA"
                request = block.replace("NEED_DATA:", "").strip()
            elif block.startswith("NEED_ANALYSIS:"):
                decision = "NEED_ANALYSIS"
                request = block.replace("NEED_ANALYSIS:", "").strip()
            else:
                decision = "VERDICT"

        if "---SYNTHESIS---" in raw:
            verdict = raw.split("---SYNTHESIS---")[1].strip()

        state["moderator_decision"] = decision
        state["moderator_request"] = request
        state["cycle_count"] = cycle + 1
        if verdict:
            state["moderator_verdict"] = verdict

        if emit:
            msgs = {
                "NEED_DATA": f"Needs more data: {request[:65]}",
                "NEED_ANALYSIS": f"Needs deeper analysis: {request[:65]}",
                "VERDICT": "Confident — proceeding to write the report",
            }
            emit("agent_done", {
                "agent": "moderator", "label": "Moderator", "icon": "⚖️",
                "message": msgs.get(decision, decision),
                "detail": {"decision": decision}
            })

        state["messages"].append(AIMessage(
            content=f"Moderator cycle {cycle}: {decision}{(' — ' + request) if request else ''}"
        ))
        return state

    def _force_verdict(self, state: AgentState) -> str:
        prompt = f"""Synthesize the best possible verdict from the available analysis. Be direct.
Query: {state['user_query']}
Intents: {state['plan']['intents']}
Quant: {state['quant_analysis']}
Qual: {state['qual_analysis']}
Give a clear, specific conclusion for each intent. Make a call."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


def route_after_moderator(state: AgentState) -> str:
    decision = state.get("moderator_decision", "VERDICT")
    if decision == "NEED_DATA":
        return "researcher"
    elif decision == "NEED_ANALYSIS":
        return "quant_analyst"
    return "writer"


class WriterAgent:

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        emit = state.get("_emit")

        if emit:
            emit("agent_start", {
                "agent": "writer", "label": "Writer", "icon": "📄",
                "message": "Drafting the final investment research report..."
            })

        intents = state["plan"]["intents"]
        tickers = state["plan"]["tickers"]
        sources = state.get("sources", [])
        sources_text = "\n".join([
            f"[{i+1}] {s['title']} — {s['url']}"
            for i, s in enumerate(sources[:20])
        ])
        section_instructions = self._get_sections(intents, tickers)

        prompt = f"""You are a professional investment research writer.

User Query: {state['user_query']}
Tickers: {tickers}
Intents: {intents}

MODERATOR'S SYNTHESIZED VERDICT:
{state['moderator_verdict']}

QUANTITATIVE ANALYSIS:
{state['quant_analysis']}

QUALITATIVE ANALYSIS:
{state['qual_analysis']}

AVAILABLE SOURCES (cite inline as [1], [2] etc.):
{sources_text}

Write a clean markdown report:
{section_instructions}

Requirements: cite every news claim, include key numbers, be specific.

End with:
## Sources
[numbered list with full URLs]

---SELF-EVALUATION---
COMPLETE
or
WEAK: [what's missing]"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content

        report = raw
        writer_verdict = "COMPLETE"

        if "---SELF-EVALUATION---" in raw:
            parts = raw.split("---SELF-EVALUATION---")
            report = parts[0].strip()
            if parts[1].strip().upper().startswith("WEAK"):
                writer_verdict = "WEAK"

        state["final_report"] = report
        state["writer_verdict"] = writer_verdict

        if emit:
            emit("agent_done", {
                "agent": "writer", "label": "Writer", "icon": "📄",
                "message": "Report complete!" if writer_verdict == "COMPLETE" else "Needs revision...",
            })

        state["messages"].append(AIMessage(
            content=f"Writer: {'Complete' if writer_verdict == 'COMPLETE' else 'Weak'}"
        ))
        return state

    def _get_sections(self, intents: list, tickers: list) -> str:
        sections = ["## Executive Summary\n[2-3 sentences. Direct answer upfront.]"]
        if "comparison" in intents:
            sections.append("## Company Snapshots")
            sections.append("## Head-to-Head Metrics\n[Table: Price, Market Cap, P/E, Forward P/E, Revenue Growth, Profit Margin, 52W Return, Analyst Target]")
            sections.append("## Relative Strengths & Weaknesses")
        if "historical" in intents:
            sections.append("## Performance Track Record")
        if "catalyst" in intents:
            sections.append("## What's Driving Price Action\n[Specific recent events — all cited.]")
        if "outlook" in intents:
            sections.append("## Forward Outlook\n[Next quarter: EPS estimates, analyst targets, catalysts and risks]")
        if "recommendation" in intents:
            sections.append("## Investment Case\n[Bull case and bear case with specific reasons]")
        sections.append("## Quant vs Qual: Key Agreements and Conflicts")
        sections.append("## Conclusion\n[Direct answer with confidence level and key condition to watch.]")
        return "\n\n".join(sections)


class RiskAnalyst:
    """
    Adversarial agent — runs after the Writer and specifically looks for
    what could go wrong. Separate from the main committee which is optimized
    to build a coherent thesis, not poke holes in it.
    """

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        emit = state.get("_emit")

        if emit:
            emit("agent_start", {
                "agent": "risk", "label": "Risk Analyst", "icon": "🚨",
                "message": "Scanning for red flags the main committee may have missed..."
            })

        prompt = f"""You are an adversarial risk analyst. Your only job is to find what could go wrong.
The main research committee has already built a thesis — your job is to stress-test it.

User Query: {state['user_query']}
Tickers: {state['plan']['tickers']}

FINAL REPORT (the thesis to stress-test):
{state['final_report']}

RAW RESEARCH DATA:
{state['research_data']}

Look specifically for:
- **Valuation risk**: Is the stock pricing in perfection? What happens if growth slows even slightly?
- **Balance sheet risk**: Debt levels, interest coverage, cash burn rate
- **Earnings quality**: Any one-time items inflating results? Revenue recognition concerns?
- **Insider activity**: Any significant insider selling signals in the data?
- **Upcoming catalysts that could hurt**: Earnings dates, lock-up expirations, regulatory decisions
- **Competitive threats**: Specific named competitors gaining ground
- **Macro sensitivity**: How exposed is this to rate changes, recession, or sector rotation?
- **What the thesis assumes that might not be true**: Call out the key assumptions baked into the bullish case

Be specific and blunt. No hedging. If a risk is serious, say so clearly.
If the data genuinely doesn't support a particular risk, skip it — don't manufacture concerns.

Format as a list of discrete risk items. Each item should be:
**[Risk Category]**: One sentence on what the risk is. One sentence on why it matters or what triggers it.

End with a one-line overall risk summary: LOW / MEDIUM / HIGH risk profile and the single biggest reason why."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        state["risk_flags"] = response.content

        if emit:
            emit("agent_done", {
                "agent": "risk", "label": "Risk Analyst", "icon": "🚨",
                "message": "Risk analysis complete",
            })

        state["messages"].append(AIMessage(content="Risk Analyst: Complete"))
        return state


def route_after_writer(state: AgentState) -> str:
    if state.get("writer_verdict") == "WEAK":
        return "moderator"
    return "risk_analyst"


class InvestmentResearchCommittee:

    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key,
            timeout=120,
            max_retries=2
        )

        self.planner    = PlannerAgent(self.llm)
        self.researcher = ResearchAgent(self.llm, tavily_api_key)
        self.quant      = QuantAnalyst(self.llm)
        self.qual       = QualAnalyst(self.llm)
        self.moderator  = ModeratorAgent(self.llm)
        self.writer     = WriterAgent(self.llm)
        self.risk       = RiskAnalyst(self.llm)
        self.graph      = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("planner",       self.planner.process)
        workflow.add_node("researcher",    self.researcher.process)
        workflow.add_node("quant_analyst", self.quant.process)
        workflow.add_node("qual_analyst",  self.qual.process)
        workflow.add_node("moderator",     self.moderator.process)
        workflow.add_node("writer",        self.writer.process)
        workflow.add_node("risk_analyst",  self.risk.process)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")

        workflow.add_conditional_edges("researcher", route_after_research,
            {"researcher": "researcher", "quant_analyst": "quant_analyst"})

        workflow.add_edge("quant_analyst", "qual_analyst")
        workflow.add_edge("qual_analyst",  "moderator")

        workflow.add_conditional_edges("moderator", route_after_moderator,
            {"researcher": "researcher", "quant_analyst": "quant_analyst", "writer": "writer"})

        workflow.add_conditional_edges("writer", route_after_writer,
            {"moderator": "moderator", "risk_analyst": "risk_analyst"})

        workflow.add_edge("risk_analyst", END)

        return workflow.compile()

    def research(self, user_query: str, emit: Optional[Callable] = None) -> dict:
        """
        Run the full agent committee.
        Pass an `emit(event_type, payload)` callable for live streaming updates.
        """
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "plan": {},
            "research_data": {},
            "data_checklist": {},
            "research_pass_count": 0,
            "quant_analysis": "",
            "qual_analysis": "",
            "moderator_decision": "",
            "moderator_request": "",
            "moderator_verdict": "",
            "cycle_count": 0,
            "writer_verdict": "",
            "final_report": "",
            "risk_flags": "",
            "sources": [],
            "_emit": emit,
        }

        return self.graph.invoke(initial_state)


if __name__ == "__main__":
    load_dotenv()
    committee = InvestmentResearchCommittee(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    while True:
        q = input("\n> ").strip()
        if q.lower() in ["quit", "q"]:
            break
        committee.research(q)
