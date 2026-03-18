# ReviveAgent Design Document (v1.0 - Week 1)

## 1. Problem Statement
Sales reps spend 4–8 hours/week manually hunting ghosted deals across CRM, email, calendar, and notes.
Result: 30–40% of pipeline goes silent, deals die, revenue is lost.
Current tools (HubSpot/Salesforce alerts) are rule-based and miss nuanced context.

## 2. Goals & Success Metrics (MVP)
- Automate 80% of revival detection and drafting.
- Ghosting prediction accuracy ≥ 85% (validated on synthetic + real test data).
- End-to-end latency < 30 seconds per deal.
- User gets daily digest + one-click “Send revival” (with approval).
- Portfolio deliverables: live demo, RAGAS eval scores, agent trace logs, cost-per-revival tracking.

## 3. Target Users & Value Proposition
- B2B SaaS sales teams (10–200 reps).
- Value: “Save 5 hours/week per rep + 15–25% more closed-won deals.”
- Pricing (later): Free (50 deals/mo), Pro ($29/user/mo), Enterprise (unlimited + custom fine-tune).

## 4. MVP Scope (Weeks 1–6)
- Connect to mock CRM + local email files (real APIs in Week 6+).
- Daily batch job: scan stalled deals → retrieve context → predict ghost risk → generate revival message.
- Simple web dashboard (Streamlit/Gradio for now → Next.js later).
- Output: “Deal X is 78% likely to ghost. Suggested revival: [drafted email]”

## 5. High-Level Architecture (see diagram below)

## 6. Data Sources (MVP → Production)
- Mock CRM JSON + sample email .eml files (Week 1–3)
- Real HubSpot/Salesforce + Gmail API + Google Calendar (Week 6+)
- Vector store: local Chroma → pgvector or Qdrant in prod

## 7. Non-Functional Requirements
- Multi-tenancy ready (user_id isolation from day 1)
- Cost tracking (tokens + API calls)
- Observability hooks (logging, tracing)
- GDPR-ready (user data deletion endpoint)

## 8. Risks & Trade-offs (to be updated weekly)
- Hallucinations → mitigated by RAG (Week 3)
- Agent looping forever → max 5 steps + human approval (Week 4)
- API rate limits → caching + batching (Week 6)

## 9. Tech Stack (high-level)
- LLM: Groq (Llama-3.3-70B or Mixtral) – free tier for dev
- Backend: FastAPI + Python
- Vector DB: Chroma (local)
- Agent framework: LangGraph or LlamaIndex Workflows
- Orchestration: Celery + Redis (later)
- Frontend: Streamlit (quick) → Next.js + shadcn
- MLOps: MLflow, Prometheus (Week 6)
- Deployment: Render / Fly.io / Vercel

## 10. Week-by-Week Roadmap
(Will be updated live in this document)
