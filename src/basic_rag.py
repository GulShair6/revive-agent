# src/reviveagent_rag.py
"""
ReviveAgent - RAG Pipeline
- Embeddings: BAAI/bge-small-en-v1.5
- Vector store: Chroma (local, persistent)
- Chunking: RecursiveCharacterTextSplitter
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: Groq (Llama 3.1 8B or similar)
"""

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

load_dotenv()

# ───────────────────────────────────────────────────────────────
#  CONFIG
# ───────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PATH = "./chroma_db_reviveagent"
COLLECTION_NAME = "reviveagent_deals_v3"

# ───────────────────────────────────────────────────────────────
#  EMBEDDINGS
# ───────────────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},  # change to "cuda" if you have GPU
)

# ───────────────────────────────────────────────────────────────
#  SAMPLE DOCUMENTS (later replace with real CRM / email loader)
# ───────────────────────────────────────────────────────────────

raw_documents = [
    Document(
        page_content="""Deal: Acme Corp - SaaS subscription renewal $15k ARR
Stage: Proposal sent 2025-12-10
Champion: Sarah Johnson (VP Marketing)
Last contact: Email from prospect 'Budget approval pending Q1 budget cycle' received 2026-01-05
Notes: Great demo on Jan 3 — they especially loved the real-time analytics dashboard and custom reporting.
Competitor mentioned: HubSpot — we have better pricing flexibility and faster implementation.
Budget: confirmed $14–18k range.""",
        metadata={"deal_id": "acme-001", "source": "crm_notes", "date": "2026-01-05"},
    ),
    Document(
        page_content="""Email thread - Acme Corp:
2026-02-01   You sent: follow-up email + Metro case study PDF
2026-02-01   Prospect opened email (tracked)
2026-02-03   Prospect opened email again
No reply since 2026-02-01 → 18 days of silence as of 2026-02-19
Previous pattern: Sarah usually replies within 4 business days.""",
        metadata={"deal_id": "acme-001", "source": "email_thread", "silence_days": 18},
    ),
    Document(
        page_content="""Deal: Beta Inc - Pilot expired 2026-02-15
ARR potential: $28k
Stage: Legal review
Last touch: 2026-02-10 email from their counsel: "Legal is reviewing the MSA, expect 2–3 weeks."
No update since. Champion: Michael Reyes (Procurement Lead)
Known issue: They are very risk-averse — legal delays common in past deals.""",
        metadata={"deal_id": "beta-002", "source": "crm", "status": "stalled_legal"},
    ),
    Document(
        page_content="""Deal: Gamma Ltd - Negotiation phase $42k ARR
Last meeting: 2026-02-05 Zoom — pricing objection on setup fee
We offered to waive 50% of implementation fee if signed by Feb 28.
No response since Feb 10. CFO (decision maker) was cc'd but silent.""",
        metadata={"deal_id": "gamma-003", "source": "crm_notes"},
    ),
]

# ───────────────────────────────────────────────────────────────
#  CHUNKING
# ───────────────────────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=480,
    chunk_overlap=120,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

split_docs = text_splitter.split_documents(raw_documents)

# ───────────────────────────────────────────────────────────────
#  VECTOR STORE
# ───────────────────────────────────────────────────────────────

vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings, collection_name=COLLECTION_NAME, persist_directory=CHROMA_PATH)

# If already exists, you can also just load:
# vectorstore = Chroma(
#     collection_name=COLLECTION_NAME,
#     embedding_function=embeddings,
#     persist_directory=CHROMA_PATH
# )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},  # get more → then rerank
)

# ───────────────────────────────────────────────────────────────
#  RERANKER
# ───────────────────────────────────────────────────────────────

reranker = CrossEncoder(RERANK_MODEL)


def rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    sorted_items = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_items][:3]  # keep top 3 after reranking


# ───────────────────────────────────────────────────────────────
#  LLM & PROMPT
# ───────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=450,
)

rag_prompt = ChatPromptTemplate.from_template(
    """You are ReviveAgent — expert at reviving stalled B2B SaaS deals.

Use ONLY the provided context. Do NOT make up information.

CONTEXT:
{context}

USER QUERY:
{question}

Instructions:
- Be professional, concise and empathetic
- Identify the most likely reason for the stall from context
- Suggest ONE concrete next action (usually an email or LinkedIn message)
- If possible, include a short email draft / message skeleton
- If context is insufficient, say so clearly

Answer:"""
)


def format_context_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}" for doc in docs)


# ───────────────────────────────────────────────────────────────
#  RAG CHAIN
# ───────────────────────────────────────────────────────────────

rag_chain = (
    {"docs": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(lambda x: {"reranked_docs": rerank_documents(x["question"], x["docs"]), "question": x["question"]})
    | {"context": lambda x: format_context_docs(x["reranked_docs"]), "question": lambda x: x["question"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ───────────────────────────────────────────────────────────────
#  TEST / DEMO
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ReviveAgent RAG Demo\n" + "=" * 50 + "\n")

    test_queries = [
        "Suggest revival strategy for Acme Corp — silent for 18 days after proposal",
        "What should I do about Beta Inc pilot that expired?",
        "Help me with Gamma Ltd — pricing objection and no reply since Feb 10",
        "Any recent activity on Acme Corp deal?",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: {q}")
        print("-" * 60)
        try:
            answer = rag_chain.invoke(q)
            print(answer.strip())
        except Exception as e:
            print(f"Error: {e}")
        print("\n")
