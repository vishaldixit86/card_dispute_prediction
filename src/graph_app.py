import os
import pandas as pd
from langgraph.graph import StateGraph, END

from agent1 import agent1_node
from agent2_runtime import load_agent2_artifacts, agent2_node
from agent3_explainer import agent3_node
from interactive_pipeline import (
    human_review_enrichment,
    human_review_explanation,
)

DATA_DIR = "data"


def human_enrich_node(state: dict) -> dict:
    return human_review_enrichment(state)


def human_explain_node(state: dict) -> dict:
    return human_review_explanation(state)


def route_after_human_enrich(state: dict) -> str:
    feedback = state.get("human_feedback", {})
    approved = feedback.get("approved_enrichment", True)

    if approved:
        return "agent2_predict"
    else:
        return "agent1_enrich"


def route_after_human_explain(state: dict) -> str:
    # For now always END, but we have approved_explanation in state if needed
    return "END"


def build_graph():
    graph = StateGraph(dict)

    graph.add_node("agent1_enrich", agent1_node)
    graph.add_node("human_enrich", human_enrich_node)
    graph.add_node("agent2_predict", agent2_node)
    graph.add_node("agent3_explain", agent3_node)
    graph.add_node("human_explain", human_explain_node)

    graph.set_entry_point("agent1_enrich")

    graph.add_edge("agent1_enrich", "human_enrich")
    graph.add_edge("agent2_predict", "agent3_explain")
    graph.add_edge("agent3_explain", "human_explain")

    graph.add_conditional_edges(
        "human_enrich",
        route_after_human_enrich,
        {
            "agent2_predict": "agent2_predict",
            "agent1_enrich": "agent1_enrich",
        },
    )

    graph.add_conditional_edges(
        "human_explain",
        route_after_human_explain,
        {
            "END": END,
        },
    )

    app = graph.compile()
    return app


def main():
    load_agent2_artifacts()

    app = build_graph()

    data_path = os.path.join(DATA_DIR, "disputes_test.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run train_agent2.py first."
        )

    df_test = pd.read_csv(data_path)
    sample_row = df_test.iloc[0].to_dict()

    initial_state = {"raw_input": sample_row}

    final_state = app.invoke(initial_state)

    print("\n=== Final state keys ===")
    print(final_state.keys())
    print("\nPipeline via LangGraph with conditional edges finished.")


if __name__ == "__main__":
    main()
