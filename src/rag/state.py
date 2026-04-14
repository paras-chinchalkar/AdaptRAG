from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
        retries: number of times we have routed back
    """
    question: str
    generation: str
    search: str
    documents: List[str]
    retries: int
