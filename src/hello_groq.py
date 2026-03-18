import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Initialize LLM (change model if rate-limited)
llm = ChatGroq(
    model="llama-3.1-8b-instant",          # fast & strong agent model
    temperature=0.7,
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Simple conversational prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=(
        "You are ReviveAgent, an expert B2B sales revival assistant. "
        "Your tone is professional, empathetic, concise, and action-oriented. "
        "You help sales reps revive ghosted deals by suggesting personalized follow-ups."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
])

# Chain: prompt → LLM
chain = prompt | llm

# Simple REPL chat loop
def chat_loop():
    print("ReviveAgent v0.1 – type 'exit' to quit\n")
    history = []  # will hold AIMessage + HumanMessage

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Invoke chain with history
        response = chain.invoke({
            "input": user_input,
            "chat_history": history,
        })

        ai_reply = response.content
        print("\nReviveAgent:", ai_reply, "\n")

        # Append to history (keeps context)
        history.append(HumanMessage(content=user_input))
        history.append(response)  # AIMessage

if __name__ == "__main__":
    chat_loop()