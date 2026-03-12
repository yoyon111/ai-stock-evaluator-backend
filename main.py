"""
Investment Research AI Committee - FastAPI Backend

Endpoints:
  POST /research          → run the agent committee, save report to Supabase
  GET  /reports           → fetch all past reports for the authenticated user
  GET  /reports/{id}      → fetch a single report by ID
  DELETE /reports/{id}    → delete a report

Auth:
  All endpoints (except /health) require a Supabase JWT in the Authorization header
  Header format: Authorization: Bearer <supabase_access_token>

Run with:
  uvicorn main:app --reload
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

from agent_committee import InvestmentResearchCommittee

load_dotenv()

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Investment Research AI Committee",
    description="Multi-agent stock research API",
    version="1.0.0"
)

# Allow requests from your frontend (update origin in Phase 3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.48:3000", "https://ai-stock-evaluator.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CLIENTS
# ============================================================================

# Supabase client (uses anon key — row level security handles permissions)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

# Agent committee (initialised once at startup, reused for all requests)
committee = InvestmentResearchCommittee(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


# ============================================================================
# AUTH HELPER
# ============================================================================

async def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Validates the Supabase JWT from the Authorization header.
    Returns the user dict if valid, raises 401 if not.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.replace("Bearer ", "")

    try:
        # Verify the token with Supabase and get the user
        response = supabase.auth.get_user(token)
        if not response or not response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": response.user.id, "email": response.user.email, "token": token}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth failed: {str(e)}")


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class ResearchRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Compare Google and Meta's latest earnings"
            }
        }


class ResearchResponse(BaseModel):
    id: str
    query: str
    report: str
    sources: list
    tickers: list
    intents: list
    created_at: str


class ReportSummary(BaseModel):
    id: str
    query: str
    tickers: list
    intents: list
    created_at: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Basic health check — no auth required."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/research", response_model=ResearchResponse)
async def run_research(
    request: ResearchRequest,
    user: dict = Depends(get_current_user)
):
    """
    Run the AI agent committee on a query.
    Saves the report to Supabase and returns it.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")

    print(f"\n[API] Research request from user {user['id']}")
    print(f"[API] Query: {request.query}")

    # Run the agent committee
    try:
        result = committee.research(request.query)
    except Exception as e:
        print(f"[API] Committee error: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

    # Extract what we need from the result
    report      = result.get("final_report", "")
    sources     = result.get("sources", [])
    tickers     = result.get("plan", {}).get("tickers", [])
    intents     = result.get("plan", {}).get("intents", [])

    if not report:
        raise HTTPException(status_code=500, detail="No report was generated")

    # Save to Supabase
    # Note: we use the user's token so RLS policies apply correctly
    try:
        user_supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )
        user_supabase.auth.set_session(user["token"], "")

        insert_response = user_supabase.table("reports").insert({
            "user_id":    user["id"],
            "query":      request.query,
            "report":     report,
            "sources":    sources,
            "tickers":    tickers,
            "intents":    intents,
        }).execute()

        saved = insert_response.data[0]
        print(f"[API] Report saved with id: {saved['id']}")

    except Exception as e:
        print(f"[API] Supabase save error: {e}")
        # Don't fail the whole request if saving fails — return report anyway
        return ResearchResponse(
            id="unsaved",
            query=request.query,
            report=report,
            sources=sources,
            tickers=tickers,
            intents=intents,
            created_at=datetime.utcnow().isoformat()
        )

    return ResearchResponse(
        id=saved["id"],
        query=saved["query"],
        report=saved["report"],
        sources=saved["sources"],
        tickers=saved["tickers"],
        intents=saved["intents"],
        created_at=saved["created_at"]
    )


@app.post("/research/stream")
async def run_research_stream(
    request: ResearchRequest,
    user: dict = Depends(get_current_user)
):
    """
    SSE streaming version of /research.
    Emits agent progress events in real-time, then a final `done` event with the full result.
    
    Event format:  data: {"type": "...", "payload": {...}}\n\n
    Event types:
      agent_start      — agent just started working
      agent_done       — agent finished
      researcher_step  — granular sub-step inside Researcher
      complete         — final result (report, sources, tickers, id, etc.)
      error            — something went wrong
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 characters)")

    print(f"\n[API] Stream research from user {user['id']}: {request.query}")

    # Queue to bridge sync LangGraph callbacks → async SSE generator
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(event_type: str, payload: dict):
        """Called from sync LangGraph threads — put onto the async queue."""
        loop.call_soon_threadsafe(
            queue.put_nowait,
            json.dumps({"type": event_type, "payload": payload})
        )

    async def run_committee():
        """Run the blocking committee.research() in a thread pool."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: committee.research(request.query, emit=emit)
            )
            return result
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "error",
                "payload": {"message": str(e)}
            }))
            return None

    async def event_stream():
        # Kick off the committee in the background
        task = asyncio.create_task(run_committee())

        SENTINEL = "__DONE__"

        # Stream queue events until the task finishes
        while True:
            try:
                # Poll with a timeout so we can detect task completion
                msg = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {msg}\n\n"
            except asyncio.TimeoutError:
                if task.done():
                    # Drain any remaining events
                    while not queue.empty():
                        msg = queue.get_nowait()
                        yield f"data: {msg}\n\n"
                    break
                # Keep-alive ping so proxy/browser doesn't time out
                yield ": ping\n\n"
                continue

        # Task finished — get the result and save it
        result = task.result()
        if result is None:
            return

        report   = result.get("final_report", "")
        sources  = result.get("sources", [])
        tickers  = result.get("plan", {}).get("tickers", [])
        intents  = result.get("plan", {}).get("intents", [])

        if not report:
            yield f"data: {json.dumps({'type': 'error', 'payload': {'message': 'No report generated'}})}\n\n"
            return

        # Save to Supabase
        saved_id = "unsaved"
        saved_at = datetime.utcnow().isoformat()
        try:
            user_supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_ANON_KEY")
            )
            user_supabase.auth.set_session(user["token"], "")
            insert_response = user_supabase.table("reports").insert({
                "user_id": user["id"],
                "query":   request.query,
                "report":  report,
                "sources": sources,
                "tickers": tickers,
                "intents": intents,
            }).execute()
            saved = insert_response.data[0]
            saved_id = saved["id"]
            saved_at = saved["created_at"]
            print(f"[API] Stream report saved: {saved_id}")
        except Exception as e:
            print(f"[API] Save error: {e}")

        # Final event — frontend can now render the report
        yield f"data: {json.dumps({'type': 'complete', 'payload': {'id': saved_id, 'query': request.query, 'report': report, 'sources': sources, 'tickers': tickers, 'intents': intents, 'created_at': saved_at}})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/reports", response_model=list[ReportSummary])
async def get_reports(user: dict = Depends(get_current_user)):
    """
    Fetch all past reports for the authenticated user.
    Returns summaries (no full report text) for the dashboard list.
    """
    try:
        user_supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )
        user_supabase.auth.set_session(user["token"], "")

        response = user_supabase.table("reports")\
            .select("id, query, tickers, intents, created_at")\
            .eq("user_id", user["id"])\
            .order("created_at", desc=True)\
            .execute()

        return response.data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reports: {str(e)}")


@app.get("/reports/{report_id}", response_model=ResearchResponse)
async def get_report(
    report_id: str,
    user: dict = Depends(get_current_user)
):
    """
    Fetch a single full report by ID.
    RLS ensures users can only fetch their own reports.
    """
    try:
        user_supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )
        user_supabase.auth.set_session(user["token"], "")

        response = user_supabase.table("reports")\
            .select("*")\
            .eq("id", report_id)\
            .eq("user_id", user["id"])\
            .single()\
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Report not found")

        return response.data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch report: {str(e)}")


@app.delete("/reports/{report_id}")
async def delete_report(
    report_id: str,
    user: dict = Depends(get_current_user)
):
    """Delete a report. Users can only delete their own."""
    try:
        user_supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )
        user_supabase.auth.set_session(user["token"], "")

        user_supabase.table("reports")\
            .delete()\
            .eq("id", report_id)\
            .eq("user_id", user["id"])\
            .execute()

        return {"success": True, "deleted_id": report_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")
