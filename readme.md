# AI Stock Evaluator Backend

FastAPI backend for an AI-powered investment research application.

## Overview

This repository contains the backend for the AI Stock Evaluator application.

The backend receives research questions from the frontend, coordinates AI research agents, gathers external information, generates investment research reports, and stores reports for authenticated users.

## Features

- FastAPI REST API
- AI-powered investment research
- Multi-agent research pipeline
- External web research
- AI-generated research reports
- Supabase authentication
- Report storage
- Streaming research responses
- Health-check endpoint

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | /health | Check API status |
| POST | /research | Run an AI research request |
| GET | /reports | Get the user's saved reports |
| GET | /reports/{id} | Get a specific report |
| DELETE | /reports/{id} | Delete a report |

## Architecture

Frontend
↓
FastAPI Backend
↓
AI Research Agents
↓
External Research Sources
↓
Generated Research Report
↓
Supabase

## Tech Stack

- Python
- FastAPI
- Pydantic
- Supabase
- OpenAI
- Tavily
- Uvicorn

## Setup

### Clone the repository

    git clone https://github.com/yoyon111/ai-stock-evaluator-backend.git
    cd ai-stock-evaluator-backend

### Install dependencies

    pip install -r requirements.txt

### Environment Variables

Create a `.env` file and add the required API credentials:

    SUPABASE_URL=your_supabase_url
    SUPABASE_ANON_KEY=your_supabase_anon_key
    OPENAI_API_KEY=your_openai_api_key
    TAVILY_API_KEY=your_tavily_api_key

### Run the server

    uvicorn main:app --reload

The API will be available at:

    http://localhost:8000

FastAPI's interactive documentation is available at:

    http://localhost:8000/docs

## Example Research Request

    {
      "query": "Compare Google's and Meta's latest earnings"
    }

## Related Project

Frontend:
https://github.com/yoyon111/AI-stock-evaluator

## Disclaimer

This project is intended for educational and research purposes. Generated analysis should not be considered financial advice.
