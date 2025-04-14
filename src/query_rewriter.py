import asyncio
import warnings
import os
from typing import Any, List, Dict
from io import StringIO
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")


async def a_query_rewriter(state:Dict[str, Any]) -> Dict[str, Any]:

    query_rewriter_prompt = '''
    # **Instructions**
    - You are a question re-writer your job is to rewrite the question in a clear, concise, and accurate manner.
    - Make sure you keep the underlying semantic intent and meaning of the original question.
    - Ensure that output is no more than 1 sentence. 

    # **Input Sections**
    ## **Original Question**
    {original_question}
    
    '''

    # Bring Original Question
    original_question = state["original_question"]
    concise_summary = state["concise_summary"]
    openai_api_key = state["openai_api_key"]

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", query_rewriter_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=openai_api_key)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "concise_summary": concise_summary,
        "original_question": original_question,
    })

    state["updated_question"] = generation

    return state