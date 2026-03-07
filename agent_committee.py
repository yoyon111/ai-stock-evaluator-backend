"""
Investment Research AI Committee - LangGraph Multi-Agent System

Agents:
1. Planner      - Detects intents, extracts tickers, builds data checklist
2. Researcher   - Fetches data (loops until checklist satisfied or cap hit)
3. Quant        - Analyzes numbers independently
4. Qual         - Analyzes narrative/news independently
5. Moderator    - Reads both, identifies conflicts, decides what to do next:
                    → NEED_DATA: specific data gap → back to Researcher
                    → NEED_ANALYSIS: specific question → back to Quant+Qual
                    → VERDICT: confident enough → Writer
6. Writer       - Produces intent-matched report with citations

Key design:
- Moderator owns loop termination, not a round counter
- Quant and Qual always analyze independently (no debate rounds)
- Moderator identifies WHAT is blocking confidence and routes accordingly
- Hard cap of MAX_CYCLES=5 as safety net only

Graph:
  Planner → Researcher [loop if checklist incomplete]
                      → Quant → Qual → Moderator
                                           ↓
                              NEED_DATA → Researcher → Quant → Qual → Moderator
                              NEED_ANALYSIS → Quant → Qual → Moderator
                              VERDICT → Writer → END
"""

import os
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import yfinance as yf
from tavily import TavilyClient
from dotenv import load_dotenv


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_RESEARCH_PASSES = 3   # Max Researcher loops for initial checklist
MAX_CYCLES = 5            # Hard safety cap on Moderator → loop cycles

INTENT_DATA_MAP = {
    "outlook":        ["stock_info", "earnings", "price_history", "analyst_recommendations", "news_outlook"],
    "comparison":     ["stock_info", "earnings", "price_history", "news_general"],
    "catalyst":       ["stock_info", "news_catalyst", "news_general"],
    "historical":     ["stock_info", "price_history_5y", "earnings"],
    "recommendation": ["stock_info", "earnings", "price_history", "analyst_recommendations", "news_general"],
}

ALL_INTENTS = list(INTENT_DATA_MAP.keys())


# ============================================================================
# STATE
# ============================================================================

class AgentState(TypedDict):
    messages: list
    user_query: str

    plan: dict
    research_data: dict
    data_checklist: dict
    research_pass_count: int

    quant_analysis: str
    qual_analysis: str

    # Moderator drives the loop
    moderator_decision: str     # NEED_DATA | NEED_ANALYSIS | VERDICT
    moderator_request: str      # what specifically is needed
    moderator_verdict: str      # final synthesis (set when decision = VERDICT)
    cycle_count: int            # how many Moderator loops have happened

    writer_verdict: str
    final_report: str
    sources: list


# ============================================================================
# TOOLS
# ============================================================================

class FinancialDataTools:

    @staticmethod
    def get_stock_info(ticker: str) -> dict:
        try:
            print(f"      📊 Stock info: {ticker}")
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
            print(f"      ❌ {e}")
            return {"ticker": ticker, "error": str(e)}

    @staticmethod
    def get_earnings(ticker: str) -> dict:
        try:
            print(f"      📈 Earnings: {ticker}")
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
            print(f"      ❌ {e}")
            return {"ticker": ticker, "error": str(e)}

    @staticmethod
    def get_price_history(ticker: str, period: str = "1y") -> dict:
        try:
            print(f"      📉 Price history ({period}): {ticker}")
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
            print(f"      ❌ {e}")
            return {"ticker": ticker, "error": str(e)}

    @staticmethod
    def get_analyst_recommendations(ticker: str) -> dict:
        try:
            print(f"      🏦 Analyst recs: {ticker}")
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
            print(f"      🔍 '{query}'")
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                days=days,
            )
            results = response.get("results", [])
            print(f"      ✅ {len(results)} results")
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
            print(f"      ❌ {e}")
            return {"query": query, "results": [], "error": str(e)}


# ============================================================================
# PLANNER
# ============================================================================

class PlannerAgent:

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        print("\n" + "="*70)
        print("🎯 PLANNER")
        print("="*70)

        prompt = f"""You are a financial research planner. Analyze this query.

Query: {state['user_query']}

STEP 1 - Extract ticker symbols. Convert names to official tickers (Google→GOOGL, Meta→META).

STEP 2 - Identify ALL intents present. A query can have multiple.
Options: outlook, comparison, catalyst, historical, recommendation

STEP 3 - Output in EXACTLY this format, nothing else:
TICKERS: AAPL, MSFT
INTENTS: outlook, comparison
TIMEFRAME: 1y"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        print(f"   {raw}")

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

        # Build checklist — union of all data required across all intents
        checklist = {}
        for ticker in plan["tickers"]:
            for intent in plan["intents"]:
                for data_type in INTENT_DATA_MAP.get(intent, []):
                    checklist[f"{ticker}__{data_type}"] = False

        print(f"\n   Tickers: {plan['tickers']}")
        print(f"   Intents: {plan['intents']}")
        print(f"   Checklist: {len(checklist)} items")

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


# ============================================================================
# RESEARCHER
# ============================================================================

class ResearchAgent:
    """
    Two modes:
    1. Checklist mode (initial pass): works through all unchecked items
    2. Targeted mode (Moderator requested): fetches the specific data asked for
    """

    def __init__(self, llm, tavily_api_key: str):
        self.llm = llm
        self.fin = FinancialDataTools()
        self.web = WebSearchTools(tavily_api_key)

    def process(self, state: AgentState) -> AgentState:
        pass_num = state["research_pass_count"] + 1
        targeted = bool(state.get("moderator_request"))

        print("\n" + "="*70)
        if targeted:
            print(f"🔍 RESEARCHER (targeted — cycle {state['cycle_count']})")
            print(f"   Moderator requested: {state['moderator_request']}")
        else:
            print(f"🔍 RESEARCHER (checklist pass {pass_num}/{MAX_RESEARCH_PASSES})")
        print("="*70)

        checklist = state["data_checklist"]
        intents = state["plan"]["intents"]
        tickers = state["plan"]["tickers"]
        sources = state.get("sources", [])

        if targeted:
            self._fetch_targeted(state, tickers, sources)
            state["moderator_request"] = ""  # clear after fulfilling
        else:
            self._fetch_checklist(state, checklist, intents, sources)

        state["research_pass_count"] = pass_num
        state["sources"] = sources

        collected = sum(1 for v in checklist.values() if v)
        total = len(checklist)
        print(f"\n   Checklist: {collected}/{total} collected")

        state["messages"].append(AIMessage(
            content=f"Researcher {'targeted' if targeted else f'pass {pass_num}'}: {collected}/{total} collected"
        ))
        return state

    def _fetch_checklist(self, state, checklist, intents, sources):
        for key, collected in checklist.items():
            if collected:
                continue
            ticker, data_type = key.split("__", 1)
            self._fetch_item(state, checklist, key, ticker, data_type, intents, sources)

    def _fetch_targeted(self, state, tickers, sources):
        """Fetch what the Moderator specifically asked for."""
        request = state["moderator_request"].lower()
        intents = state["plan"]["intents"]

        for ticker in tickers:
            if "recommendation" in request or "analyst" in request or "target" in request:
                key = f"{ticker}__analyst_recommendations"
                state["research_data"][key] = self.fin.get_analyst_recommendations(ticker)
                state["data_checklist"][key] = True

            if "earnings" in request or "eps" in request or "revenue" in request:
                key = f"{ticker}__earnings_targeted"
                state["research_data"][key] = self.fin.get_earnings(ticker)

            if "price" in request or "history" in request or "performance" in request:
                key = f"{ticker}__price_history_targeted"
                state["research_data"][key] = self.fin.get_price_history(ticker, "5y")

        # Always do a targeted web search for exactly what was asked
        for ticker in tickers:
            query = f"{ticker} {state['moderator_request']}"
            key = f"{ticker}__targeted_search_{state['cycle_count']}"
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
    targeted = state.get("moderator_request", "") == ""  # request already cleared means we came from Moderator

    # If this was a targeted fetch (Moderator sent us here), always go to analysis
    if state["cycle_count"] > 0:
        print(f"\n   ✅ Targeted fetch done → analysis")
        return "quant_analyst"

    # Initial checklist mode
    if gaps and passes < MAX_RESEARCH_PASSES:
        print(f"\n   🔄 {len(gaps)} gaps → looping Researcher")
        return "researcher"
    if gaps:
        print(f"\n   ⚠️  {len(gaps)} gaps but hit pass cap → proceeding")
    else:
        print(f"\n   ✅ Checklist complete → analysis")
    return "quant_analyst"


# ============================================================================
# QUANT ANALYST
# ============================================================================

class QuantAnalyst:
    """
    Always analyzes independently — no debate framing.
    Sees the Moderator's specific question if this is a follow-up cycle.
    """

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        cycle = state["cycle_count"]
        print("\n" + "="*70)
        print(f"📊 QUANT ANALYST {'(cycle ' + str(cycle) + ')' if cycle > 0 else ''}")
        print("="*70)

        focus = ""
        if state.get("moderator_request"):
            focus = f"\nThe Moderator has flagged this specific question for you to address:\n{state['moderator_request']}\nMake sure your analysis directly answers this."

        prompt = f"""You are a quantitative financial analyst. Analyze the data independently and objectively.

User Query: {state['user_query']}
Intents: {state['plan']['intents']}
Tickers: {state['plan']['tickers']}
{focus}

Research Data:
{state['research_data']}

Analyze using only hard data. Cover what's relevant to the intents:
- Key ratios with exact numbers (P/E, PEG, P/B, margins, ROE, D/E)
- Price performance with specific percentages and timeframes  
- Earnings trends — accelerating or decelerating? By exactly how much?
- Valuation: expensive, fair, or cheap vs historical averages and sector?
- Balance sheet strength: debt, cash, free cash flow trends
- If comparison: explicit side-by-side numbers, clear winner per metric
- If outlook: what do current numbers imply about trajectory?
- If catalyst: what does the financial data say about the price move?

No vague statements. Exact numbers only. Use clear subheadings."""

        print("   ⏳ Analyzing...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state["quant_analysis"] = response.content
        print(f"   ✅ Done ({len(response.content)} chars)")
        state["messages"].append(AIMessage(content=f"Quant Analyst: Analysis complete"))
        return state


# ============================================================================
# QUAL ANALYST
# ============================================================================

class QualAnalyst:
    """
    Always analyzes independently — no debate framing.
    Sees the Moderator's specific question if this is a follow-up cycle.
    """

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        cycle = state["cycle_count"]
        print("\n" + "="*70)
        print(f"📰 QUAL ANALYST {'(cycle ' + str(cycle) + ')' if cycle > 0 else ''}")
        print("="*70)

        focus = ""
        if state.get("moderator_request"):
            focus = f"\nThe Moderator has flagged this specific question for you to address:\n{state['moderator_request']}\nMake sure your analysis directly answers this."

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

Analyze the qualitative picture. Cover what's relevant to the intents:
- Recent news: what are the key stories and what do they signal?
- Industry and competitive landscape: tailwinds or headwinds?
- Management signals: guidance changes, executive moves, strategy shifts
- Market sentiment: how is the market perceiving this company right now?
- Macro factors: rates, sector rotation, regulatory environment
- If catalyst: what specifically is driving price action?
- If outlook: what narrative factors will shape the next quarter?
- If comparison: qualitative competitive moats and weaknesses

Cite news inline as [Source: Title](URL).
Be specific about what each piece of news means for the thesis."""

        print("   ⏳ Analyzing...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state["qual_analysis"] = response.content
        print(f"   ✅ Done ({len(response.content)} chars)")
        state["messages"].append(AIMessage(content=f"Qual Analyst: Analysis complete"))
        return state


# ============================================================================
# MODERATOR — owns the loop
# ============================================================================

class ModeratorAgent:
    """
    The quality gate. Reads Quant and Qual independently.
    Decides one of three things:
      NEED_DATA     → specific data gap exists → send Researcher back out
      NEED_ANALYSIS → specific question unanswered → send analysts back with a focused prompt
      VERDICT       → confident enough → proceed to Writer

    This is what drives the loop — not a round counter.
    """

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        cycle = state["cycle_count"]
        print("\n" + "="*70)
        print(f"⚖️  MODERATOR (cycle {cycle}/{MAX_CYCLES})")
        print("="*70)

        # Force VERDICT if we've hit the safety cap
        if cycle >= MAX_CYCLES:
            print(f"   ⚠️  Safety cap reached — forcing VERDICT")
            state["moderator_decision"] = "VERDICT"
            state["moderator_verdict"] = self._force_verdict(state)
            state["messages"].append(AIMessage(content="Moderator: Safety cap hit — forcing verdict"))
            return state

        prompt = f"""You are a senior investment research moderator. Your job is to ensure the final answer is genuinely confident and complete.

User Query: {state['user_query']}
Intents: {state['plan']['intents']}
Tickers: {state['plan']['tickers']}
Cycle: {cycle} (max {MAX_CYCLES})

QUANTITATIVE ANALYSIS:
{state['quant_analysis']}

QUALITATIVE ANALYSIS:
{state['qual_analysis']}

STEP 1 — Identify agreements and conflicts:
- What do both analysts agree on? (high confidence)
- Where do they disagree or contradict each other?
- What important questions does neither analyst answer?

STEP 2 — Decide what to do next. Pick EXACTLY ONE:

NEED_DATA if:
- A specific data point is missing that would resolve a key disagreement
- Neither analyst could address an important aspect due to missing data
- Example: "analyst price targets for GOOGL are missing"

NEED_ANALYSIS if:
- The data exists but analysts didn't address a specific important question
- A conflict between them needs deeper examination
- Example: "neither analyst addressed margin compression impact on next quarter EPS"

VERDICT if:
- Both analysts have addressed all key aspects of the query
- Any remaining disagreements are minor or preference-based
- You can confidently answer the user's query right now

STEP 3 — Output in EXACTLY this format:

---AGREEMENTS---
[bullet list of what both agree on]

---CONFLICTS---
[bullet list of disagreements, with your call on who is right and why]

---GAPS---
[bullet list of unanswered questions, or "None"]

---DECISION---
NEED_DATA: [exact data needed]
or
NEED_ANALYSIS: [exact question analysts should address]
or
VERDICT

---SYNTHESIS---
[Your full synthesized verdict — only write this if decision is VERDICT]
[For each intent: specific conclusion with confidence level]
[Make a call. Don't hedge everything.]"""

        print("   ⏳ Evaluating analyses...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content

        # Parse decision
        decision = "VERDICT"
        request = ""
        verdict = ""

        if "---DECISION---" in raw:
            decision_block = raw.split("---DECISION---")[1].split("---")[0].strip()
            if decision_block.startswith("NEED_DATA:"):
                decision = "NEED_DATA"
                request = decision_block.replace("NEED_DATA:", "").strip()
            elif decision_block.startswith("NEED_ANALYSIS:"):
                decision = "NEED_ANALYSIS"
                request = decision_block.replace("NEED_ANALYSIS:", "").strip()
            else:
                decision = "VERDICT"

        if "---SYNTHESIS---" in raw:
            verdict = raw.split("---SYNTHESIS---")[1].strip()

        state["moderator_decision"] = decision
        state["moderator_request"] = request
        state["cycle_count"] = cycle + 1

        if verdict:
            state["moderator_verdict"] = verdict

        print(f"\n   Decision: {decision}")
        if request:
            print(f"   Request:  {request}")

        state["messages"].append(AIMessage(
            content=f"Moderator cycle {cycle}: {decision}{(' — ' + request) if request else ''}"
        ))
        return state

    def _force_verdict(self, state: AgentState) -> str:
        """Called only when safety cap is hit. Synthesizes best possible verdict from what we have."""
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
        print(f"\n   🔄 NEED_DATA → Researcher")
        return "researcher"
    elif decision == "NEED_ANALYSIS":
        print(f"\n   🔄 NEED_ANALYSIS → Quant + Qual")
        return "quant_analyst"
    else:
        print(f"\n   ✅ VERDICT → Writer")
        return "writer"


# ============================================================================
# WRITER
# ============================================================================

class WriterAgent:

    def __init__(self, llm):
        self.llm = llm

    def process(self, state: AgentState) -> AgentState:
        print("\n" + "="*70)
        print("📄 WRITER")
        print("="*70)

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

Write a clean markdown report using these sections:
{section_instructions}

Requirements:
- Every news claim must have an inline citation [1][2]
- Every key number must appear (P/E, price change %, revenue, etc.)
- Be specific — no vague statements
- Tone matches query: outlook = forward-looking, comparison = analytical tables, catalyst = news-driven

End with:
## Sources
[numbered list with full URLs]

---SELF-EVALUATION---
Is every section specific and well-supported?
COMPLETE
or
WEAK: [section name and what's missing]"""

        print("   ⏳ Writing...")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        raw = response.content

        report = raw
        writer_verdict = "COMPLETE"

        if "---SELF-EVALUATION---" in raw:
            parts = raw.split("---SELF-EVALUATION---")
            report = parts[0].strip()
            eval_block = parts[1].strip()
            if eval_block.upper().startswith("WEAK"):
                writer_verdict = "WEAK"
                print(f"   ⚠️  WEAK: {eval_block}")
            else:
                print("   ✅ COMPLETE")

        state["final_report"] = report
        state["writer_verdict"] = writer_verdict
        state["messages"].append(AIMessage(
            content=f"Writer: {'Complete' if writer_verdict == 'COMPLETE' else 'Weak — back to Moderator'}"
        ))
        return state

    def _get_sections(self, intents: list, tickers: list) -> str:
        sections = [
            "## Executive Summary\n[2-3 sentences. Direct answer to the query upfront.]"
        ]
        if "comparison" in intents:
            sections.append("## Company Snapshots\n[Brief overview of each company]")
            sections.append("## Head-to-Head Metrics\n[Markdown table: Price, Market Cap, P/E, Forward P/E, Revenue Growth, Profit Margin, 52W Return, Analyst Target]")
            sections.append("## Relative Strengths & Weaknesses\n[Per company: where it wins, where it loses]")
        if "historical" in intents:
            sections.append("## Performance Track Record\n[Price performance across timeframes, earnings history, key inflection points]")
        if "catalyst" in intents:
            sections.append("## What's Driving Price Action\n[Specific recent events and news — all cited. Root cause clearly stated.]")
        if "outlook" in intents:
            sections.append("## Forward Outlook\n[Next quarter expectations: EPS estimates, analyst targets, key catalysts and risks]")
        if "recommendation" in intents:
            sections.append("## Investment Case\n[Bull case with specific reasons. Bear case with specific risks.]")
        sections.append("## Quant vs Qual: Key Agreements and Conflicts\n[Where the numbers and narrative aligned. Where they clashed and who was right.]")
        sections.append("## Conclusion\n[Direct answer. Comparison → winner. Outlook → trajectory + confidence. Catalyst → root cause. Recommendation → stance + confidence level + key condition to watch.]")
        return "\n\n".join(sections)


def route_after_writer(state: AgentState) -> str:
    if state.get("writer_verdict") == "WEAK":
        print(f"\n   🔄 Weak report → back to Moderator")
        return "moderator"
    print(f"\n   ✅ Complete → END")
    return END


# ============================================================================
# COMMITTEE
# ============================================================================

class InvestmentResearchCommittee:

    def __init__(self, openai_api_key: str, tavily_api_key: str):
        print("🚀 Initializing Investment Research Committee...")

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

        self.graph = self._build_graph()
        print("✅ Committee ready\n")
        print("   Planner → Researcher [checklist loop]")
        print("          → Quant → Qual → Moderator")
        print("                             ↓ NEED_DATA    → Researcher → Quant → Qual → Moderator")
        print("                             ↓ NEED_ANALYSIS → Quant → Qual → Moderator")
        print("                             ↓ VERDICT       → Writer → END\n")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner",      self.planner.process)
        workflow.add_node("researcher",   self.researcher.process)
        workflow.add_node("quant_analyst",self.quant.process)
        workflow.add_node("qual_analyst", self.qual.process)
        workflow.add_node("moderator",    self.moderator.process)
        workflow.add_node("writer",       self.writer.process)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")

        workflow.add_conditional_edges(
            "researcher",
            route_after_research,
            {
                "researcher":    "researcher",
                "quant_analyst": "quant_analyst",
            }
        )

        workflow.add_edge("quant_analyst", "qual_analyst")
        workflow.add_edge("qual_analyst",  "moderator")

        workflow.add_conditional_edges(
            "moderator",
            route_after_moderator,
            {
                "researcher":    "researcher",
                "quant_analyst": "quant_analyst",
                "writer":        "writer",
            }
        )

        workflow.add_conditional_edges(
            "writer",
            route_after_writer,
            {
                "moderator": "moderator",
                END:         END,
            }
        )

        return workflow.compile()

    def research(self, user_query: str) -> dict:
        print(f"\n{'='*70}")
        print(f"Query: {user_query}")
        print(f"{'='*70}")

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
            "sources": [],
        }

        result = self.graph.invoke(initial_state)

        print("\n" + "="*70)
        print("AGENT ACTIVITY LOG:")
        print("="*70)
        for i, msg in enumerate(result["messages"][1:], 1):
            print(f"  {i}. {msg.content}")

        print("\n" + "="*70)
        print("FINAL REPORT:")
        print("="*70)
        print(result["final_report"])

        return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        print("ERROR: Set OPENAI_API_KEY and TAVILY_API_KEY in your .env file")
        exit(1)

    committee = InvestmentResearchCommittee(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY
    )

    print("Example queries:")
    print("  1. What's the outlook for NVDA next quarter?")
    print("  2. Compare TSLA vs competitors on valuation and news flow")
    print("  3. What's driving META's stock price lately?")
    print("  4. How has AAPL performed since its last earnings call?")
    print("  5. Should I invest in Microsoft right now?")
    print("\nEnter your query (or 'quit' to exit):\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not user_input:
            continue
        try:
            committee.research(user_input)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
