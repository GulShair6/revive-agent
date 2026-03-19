from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig

load_dotenv()

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
If you have enough context, give a final answer instead of calling tools again."""

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
    print("Input messages to LLM:", [m.content[:200] for m in messages])  # truncate

    try:
        # Pass a DICT with the key "messages"
        response = chain.invoke({"messages": messages})
        print("Raw LLM response:", response)
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("→ Tool calls:", [c["name"] for c in response.tool_calls])
        return {"messages": [response]}
    except Exception as e:
        print("LLM call failed:", str(e))
        # Fallback: return a message saying error
        return {"messages": [AIMessage(content=f"Tool calling failed: {str(e)}. Trying to recover...")]}


# ----------------------
# 6. Build graph
# ----------------------
workflow = StateGraph(state_schema=AgentState)

workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # built-in: if tool calls → "tools", else END
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")  # loop back after tool execution

graph = workflow.compile(checkpointer=MemorySaver())

config = RunnableConfig(recursion_limit=12)
# ----------------------
# 7. Run example
# ----------------------
if __name__ == "__main__":
    query = "Help revive the TechNova deal that went silent after proposal about integration concerns."
    inputs = {"messages": [HumanMessage(content=query)]}

    config = {"recursion_limit": 12, "configurable": {"thread_id": "test_thread_1"}}

    print("Starting agent...\n")
    try:
        for chunk in graph.stream(inputs, config=config, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print("Agent:", last_msg.content)
                if last_msg.tool_calls:
                    print("→ Calling tools:", [c["name"] for c in last_msg.tool_calls])
            print("-" * 60)
    except Exception as e:
        print("Execution failed:", str(e))
