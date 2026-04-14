import os
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from src.rag.state import GraphState
from src.rag.retriever import get_retriever
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Retriever setup
retriever = get_retriever()
web_search_tool = DuckDuckGoSearchResults()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# --- Node Functions ---

def rewrite_question(state: GraphState):
    """
    Rewrite the question to produce an optimized keyword search query.
    """
    print("---REWRITE QUESTION---")
    question = state["question"]
    
    system = """You are a question re-writer that converts an input question to a better version that is optimized 
for web search and vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. 
Respond with the improved question and nothing else."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )

    question_rewriter = re_write_prompt | llm | (lambda x: x.content)
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}

def retrieve(state: GraphState):
    """
    Retrieve documents from vectorstore
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state: GraphState):
    """
    Generate answer using RAG on retrieved documents
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retries = state.get("retries", 0)

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context}",
            ),
        ]
    )

    # Post-processing
    context = "\n\n".join(doc.page_content if hasattr(doc, "page_content") else doc for doc in documents)
    
    chain = prompt | llm | (lambda x: x.content)
    generation = chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "retries": retries + 1}


def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    structured_llm_grader = llm.with_structured_output(GradeDocuments, method="json_mode")

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    You MUST respond with a pure JSON object with the key 'binary_score' and value 'yes' or 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content if hasattr(d, 'page_content') else str(d)}
        )
        grade = score.binary_score
        
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue

    return {"documents": filtered_docs, "question": question, "search": web_search}


def web_search_node(state: GraphState):
    """
    Web search based on the question
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    if type(docs) is str:
        # Some web search tools just return str
        docs = [docs]
    elif len(docs) > 0 and type(docs[0]) is dict:
        docs = [d["snippet"] for d in docs]
        
    documents.extend(docs)
    return {"documents": documents, "question": question}


# --- Routing/Edge Functions ---

def route_question(state: GraphState):
    """
    Route question to web search or RAG.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    structured_llm_router = llm.with_structured_output(RouteQuery, method="json_mode")
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. For all else, use websearch.
    You MUST respond with a pure JSON object with the key 'datasource' and value 'vectorstore' or 'websearch'."""
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    source = question_router.invoke({"question": question})
    
    if source.datasource == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or add web search
    """
    print("---ASSESS GRADED DOCUMENTS---")
    search = state["search"]

    if search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "rewrite"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers question.
    Also handles retry logic to break infinite loops.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retries = state.get("retries", 0)

    # Max retries hit, break loop
    if retries >= 3:
        print("---MAX RETRIES REACHED, STOPPING---")
        return "max_retries"

    structured_llm_grader = llm.with_structured_output(GradeHallucinations, method="json_mode")
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
     You MUST respond with a pure JSON object with the key 'binary_score' and value 'yes' or 'no'."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    hallucination_grader = hallucination_prompt | structured_llm_grader

    context = "\n\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in documents)
    score = hallucination_grader.invoke(
        {"documents": context, "generation": generation}
    )
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---CHECK ANSWER V QUESTION---")
        
        structured_llm_grader_ans = llm.with_structured_output(GradeAnswer, method="json_mode")
        system_ans = """You are a grader assessing whether an answer addresses / resolves a question. \n 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
         You MUST respond with a pure JSON object with the key 'binary_score' and value 'yes' or 'no'."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_ans),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        answer_grader = answer_prompt | structured_llm_grader_ans

        score_ans = answer_grader.invoke({"question": question, "generation": generation})
        grade_ans = score_ans.binary_score
        
        if grade_ans == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
