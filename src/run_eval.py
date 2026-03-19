import csv
from pathlib import Path
from agent_v1 import graph  # import your compiled graph from previous file
from langchain_core.messages import HumanMessage, AIMessage
from src.logger import logger

# Adjust if you changed file name
CONFIG = {"configurable": {"thread_id": "eval_run"}}


def run_agent_on_query(query: str):
    inputs = {"messages": [HumanMessage(content=query)]}
    final_output = ""
    tool_calls_count = 0

    for event in graph.stream(inputs, CONFIG, stream_mode="values"):
        last_msg = event["messages"][-1]

        # Only check tool_calls on AIMessage
        if isinstance(last_msg, AIMessage):
            if last_msg.tool_calls:
                tool_calls_count += len(last_msg.tool_calls)
            else:
                final_output = last_msg.content

    return {
        "response": final_output or "(no final answer generated)",
        "tool_calls": tool_calls_count,
    }


if __name__ == "__main__":
    eval_file = Path("eval_set.csv")
    if not eval_file.exists():
        logger.info("Create eval_set.csv first!")
        exit(1)

    results = []
    with open(eval_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"]
            logger.info(f"\nRunning: {query}")
            result = run_agent_on_query(query)
            row.update(
                {
                    "actual_response": result["response"][:500] + "..." if len(result["response"]) > 500 else result["response"],
                    "agent_looped?": f"{result['tool_calls']} calls",
                }
            )
            results.append(row)
            logger.info(result["response"][:300] + "...\n")

    # Optional: save updated CSV
    with open("eval_results_updated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
