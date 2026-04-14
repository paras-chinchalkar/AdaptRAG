from langgraph.graph import END, StateGraph
from src.rag.state import GraphState
from src.rag.nodes import (
    retrieve,
    generate,
    grade_documents,
    web_search_node,
    route_question,
    decide_to_generate,
    grade_generation_v_documents_and_question,
    rewrite_question,
)

def create_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search_node)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("rewrite", rewrite_question)  # rewrite question

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    # Both retrieve goes to grade_documents
    workflow.add_edge("retrieve", "grade_documents")
    
    # After rewriting, we go to websearch
    workflow.add_edge("rewrite", "websearch")
    
    # Web search generates
    workflow.add_edge("websearch", "generate")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",  # retry generation
            "useful": END,
            "not useful": "rewrite",  # rewrite and do web search
            "max_retries": END, # safe escape
        },
    )

    # Compile
    app = workflow.compile()
    return app

# Initialize the graph
app = create_graph()
