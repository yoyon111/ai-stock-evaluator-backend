This one is the backend/API for the first project. Your code specifically describes it as a FastAPI backend for an "Investment Research AI Committee," with /research, /reports, and /health endpoints and Supabase JWT authentication.

AI Stock Evaluator — Backend

FastAPI backend for an AI-powered investment research application. The backend orchestrates a multi-agent research committee, gathers external information, generates investment research reports, and stores user reports in Supabase.

Overview

This service powers the backend of the AI Stock Evaluator application.

Users submit natural-language research questions such as:

"Compare Google's and Meta's latest earnings."

The backend passes the request through an AI research committee, gathers relevant information, generates a structured research report, and saves the result for the authenticated user.

Features
🤖 Multi-agent investment research
🔎 External web research
📈 Stock and company analysis
📝 AI-generated research reports
🔐 Supabase JWT authentication
💾 Persistent report storage
⚡ FastAPI REST API
❤️ Health-check endpoint
🌊 Streaming support for research responses
API Endpoints
Method	Endpoint	Description
GET	/health	Check API status
POST	/research	Run an AI research request
GET	/reports	Retrieve the user's previous reports
GET	/reports/{id}	Retrieve a specific report
DELETE	/reports/{id}	Delete a report

All endpoints except /health require a valid Supabase access token.

Example Research Request
{
  "query": "Compare Google's and Meta's latest earnings"
}
Architecture
Frontend
   │
   │ HTTP / JWT
   ▼
FastAPI Backend
   │
   ├── Authentication
   │      └── Supabase
   │
   ├── Research API
   │
   ├── AI Research Committee
   │      ├── Research agents
   │      ├── Analysis agents
   │      └── Report generation
   │
   └── Report Storage
          └── Supabase
