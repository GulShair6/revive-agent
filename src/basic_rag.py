from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Embeddings (local, no API calls)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # change to "cuda" if you have GPU
)

# 2. Sample sales context documents (in real app → load from CRM/email files)
sample_docs = [
    Document(
        page_content="""Deal: Acme Corp - SaaS subscription $15k ARR
Stage: Proposal sent 2025-12-10
Last contact: Email from prospect 'Budget approval pending Q1 budget cycle' on 2026-01-05
Notes: Demo went well, they loved the analytics dashboard. Competitor is HubSpot but we have better pricing flexibility.""",
        metadata={"deal_id": "acme-001", "source": "crm_notes"},
    ),
    Document(
        page_content="""Email thread summary:
Prospect: "We are comparing options, will get back by end of week."
You: Follow-up sent 2026-02-01 with case study attached.
No reply since then (18 days silent). Opened email 2 times.""",
        metadata={"deal_id": "acme-001", "source": "email_thread"},
    ),
    Document(
        page_content="""Another stalled deal: Beta Inc - Pilot expired 2026-02-15
Reason for stall: Waiting on legal review.
Last email: Legal is reviewing, expect 2-3 weeks.""",
        metadata={"deal_id": "beta-002", "source": "crm"},
    ),
]

# 3. Create / load Chroma collection (persists to ./chroma_db folder)
vectorstore = Chroma.from_documents(
    documents=sample_docs,
    embedding=embeddings,
    collection_name="reviveagent_deals_v1",
    persist_directory="./chroma_db",  # creates folder automatically
)

# Or if already exists → just load:
# vectorstore = Chroma(
#     collection_name="reviveagent_deals_v1",
#     embedding_function=embeddings,
#     persist_directory="./chroma_db"
# )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},  # retrieve top 2 most relevant chunks
)

# 4. LLM setup (same Groq as before)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_tokens=400,
)

# 5. RAG prompt (forces grounding)
rag_prompt = ChatPromptTemplate.from_template(
    """You are ReviveAgent, a sales revival expert.
Use ONLY the following context to suggest a personalized revival follow-up.
If context doesn't help, say so honestly.

Context:
{context}

User query / deal description:
{question}

Respond professionally, empathetically, and suggest 1 concrete next action (e.g. email draft)."""
)


# 6. Simple RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()

# Test it!
if __name__ == "__main__":
    query = "Suggest revival for Acme Corp deal silent 18 days after proposal"
    print("Query:", query)
    print("\nReviveAgent (with RAG):")
    print(rag_chain.invoke(query))
