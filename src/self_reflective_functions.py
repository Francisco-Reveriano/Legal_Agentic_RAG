from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import Any, List, Dict
import concurrent.futures

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_full_context(state:Dict[str, Any]) -> str:
    # variables
    openai_api_key = state["openai_api_key"]
    original_question = state["original_question"]
    original_context = state["original_context"]

    # LLM with function call
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", f"Retrieved document: \n\n {original_context} \n\n User question: {original_question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    response = retrieval_grader.invoke({"original_context": original_context, "original_question": original_question})
    print(response)
    return response



def grade_documents(state:Dict[str, Any]) -> Dict[str, Any]:
    # Get Variables
    question = state["updated_question"]
    doc_chunks = state["updated_chunk_list"]
    openai_api_key = state["openai_api_key"]
    query_results = state["query_results"]

    # Prepare Chunk Grader
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Setup Chunk Checker
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader

    # Run Checker through the whole space
    ## Helper function to process each document chunk concurrently
    def process_chunk(doc):
        try:
            score = retrieval_grader.invoke({"document": doc, "question": question})
            return doc, score.binary_score
        except Exception as e:
            print(f"Error processing document: {e}")
            return doc, None

    ## Use a ThreadPoolExecutor to process documents concurrently
    filtered_docs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, doc_chunks))
    ## Filter documents based on the binary score returned
    for doc, binary_score in results:
        if binary_score == "yes":
            filtered_docs.append(doc)

    # Make sure we have a dictionary with only available cleaned data
    updated_query_results = query_results
    for match in updated_query_results:
        if match["metadata"]["content"] in filtered_docs:
            del match
    print("NUmber of Documents before Relevance Check: ", len(doc_chunks))
    print("Number of Documents after Relevance Check: ", len(filtered_docs))
    print("NUmber of Documents deleted: ", len(doc_chunks)-len(filtered_docs))


    # Update State
    state.update({
        "filtered_documents": filtered_docs,
        "filtered_query_results": updated_query_results,
        "filtered_context": " ".join(filtered_docs)
    })

    return state