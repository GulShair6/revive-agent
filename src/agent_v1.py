from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field
from src.logger import logger

load_dotenv()


class RevivalPlan(BaseModel):
    """Final revival recommendation"""

    risk_score: str = Field(..., description="Ghosting risk summary")
    key_context: str = Field(..., description="2-3 most important facts pulled")
    suggested_action: str = Field(..., description="One concrete next step")
    email_draft: str = Field(..., description="Personalized email draft if appropriate")


# ----------------------
# 1. Reuse existing RAG setup (embeddings + vectorstore)
# ----------------------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(collection_name="reviveagent_deals_v1", embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ----------------------
# 2. Define Tools
# ----------------------
@tool
def search_deal_context(query: str) -> str:
    """Search internal CRM notes, emails, and past interactions for relevant deal context."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."


@tool
def predict_ghosting_risk(deal_description: str) -> str:
    """Simple stub: Predict how likely the deal is to ghost (0-100%). In real version use XGBoost."""
    # Dummy logic — replace with real model later
    if "silent" in deal_description.lower() or "no reply" in deal_description.lower():
        return "78% risk of ghosting (high urgency)"
    return "42% risk of ghosting (moderate)"


tools = [search_deal_context, predict_ghosting_risk]

# ----------------------
# 3. LLM with tool binding
# ----------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

system_prompt = """You are a helpful sales revival agent.
When you need information, use the provided tools exactly once per step.
To call a tool, output ONLY the tool call in valid JSON format — nothing else.
Do NOT wrap in <function>, XML tags, or extra text.
Tool call format must be: {{"name": "tool_name", "arguments": {{"arg1": "value1", ...}}}}

After receiving tool results, reason step-by-step and decide next action or final answer.
Never repeat the same tool call multiple times unless new information justifies it.
ONLY use facts explicitly present in the provided context.
If the answer is not fully supported by context, say: "I don't have enough specific information from past interactions to give a precise revival suggestion."
Cite sources like: [from CRM note] or [from email 2026-03-01]"""

# Then use this in your agent node instead of raw llm_with_tools.invoke(messages)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chain it
chain = prompt | llm.bind_tools(tools, tool_choice="auto")


# ----------------------
# 4. Agent state
# ----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]  # built-in reducer


# ----------------------
# 5. Agent node (reason + decide tool or final answer)
# ----------------------
def agent(state: AgentState):
    messages = state["messages"]
    logger.debug(f"Agent called with {len(messages)} messages")
    logger.debug(f"Last message: {messages[-1].content[:200] if messages else 'empty'}")

    try:
        # Pass a DICT with the key "messages"
        response = chain.invoke({"messages": messages})
        logger.info("Raw LLM response:", response)
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.debug("→ Tool calls:", [c["name"] for c in response.tool_calls])
        return {"messages": [response]}
    except Exception as e:
        logger.info("LLM call failed:", str(e))
        # Fallback: return a message saying error
        return {"messages": [AIMessage(content=f"Tool calling failed: {str(e)}. Trying to recover...")]}


# ----------------------
# 6. Build graph
# ----------------------
workflow = StateGraph(state_schema=AgentState)

# 1. Add all nodes FIRST
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))


# Define and add the generate_draft node BEFORE any edge references it
def generate_draft(state: AgentState):
    structured_llm = llm.with_structured_output(RevivalPlan)
    last_messages = state["messages"]
    plan = structured_llm.invoke(last_messages)
    return {"messages": [AIMessage(content=plan.model_dump_json())]}


workflow.add_node("generate_draft", generate_draft)

# 2. Now add edges and conditional edges
workflow.add_edge(START, "agent")


# Custom router that can send to generate_draft
def route_after_agent(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    # Heuristic: look for signal in agent's last message or message count
    if "draft" in last_msg.content.lower() or len(state["messages"]) > 6:
        return "generate_draft"
    return END


workflow.add_conditional_edges(
    "agent",
    route_after_agent,
    {"tools": "tools", "generate_draft": "generate_draft", END: END},
)

workflow.add_edge("tools", "agent")  # loop back after tools
workflow.add_edge("generate_draft", END)  # final output node


memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["generate_draft"])

# logger.info(graph.get_graph().draw_mermaid())

config = RunnableConfig(recursion_limit=12)
# ----------------------
# 7. Run example
# ----------------------
if __name__ == "__main__":
    query = "Help revive the TechNova deal that went silent after proposal about integration concerns."
    inputs = {"messages": [HumanMessage(content=query)]}

    config = {
        "configurable": {"thread_id": "test_thread_1"},
        "recursion_limit": 20,  # give some room
    }

    logger.info("Starting agent...\n")

    # First run: goes until interrupt_before generate_draft
    state_snapshot = None
    for chunk in graph.stream(inputs, config=config, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        if isinstance(last_msg, AIMessage):
            logger.info("Agent:", last_msg.content.strip())
            if last_msg.tool_calls:
                logger.info("→ Calling tools:", [c["name"] for c in last_msg.tool_calls])
        logger.info("-" * 60)

        # Keep the last chunk to check if interrupted
        state_snapshot = chunk

    # Check if we hit the interrupt
    current_state = graph.get_state(config)
    if current_state.next == ("generate_draft",):  # means paused before generate_draft
        logger.info("\n=== HUMAN-IN-THE-LOOP PAUSE ===")
        logger.info("The agent wants to generate & send a revival email draft.")
        logger.info("Last agent message:", current_state.values["messages"][-1].content)

        # In real app: show draft preview in UI/dashboard/email/Slack
        # Here: simulate human review via console
        decision = input("\nApprove draft? (y / n / edit <feedback>): ").strip().lower()

        if decision.startswith("y"):
            resume_input = None  # None = just continue to execute generate_draft
        elif decision.startswith("n"):
            resume_input = {"messages": [HumanMessage(content="Human rejected the draft. Stop and explain why.")]}
        else:
            # edit/feedback
            resume_input = {"messages": [HumanMessage(content=f"Human feedback: {decision}. Revise the draft accordingly.")]}

        logger.info("\nResuming graph with human decision...\n")

        # Resume the graph (re-runs from the interrupt point)
        for chunk in graph.stream(resume_input, config=config, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, AIMessage):
                logger.info("Agent (after resume):", last_msg.content.strip())
            logger.info("-" * 60)
    else:
        logger.info("No interrupt reached in this run.")
