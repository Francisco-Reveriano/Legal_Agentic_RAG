import asyncio
import warnings
import os
from typing import Any, List, Dict

import pandas as pd
from io import StringIO
import tiktoken

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")


async def a_concise_summary(state:Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronously generates a concise and well-structured summary of a regulation
    based on the provided context and question. It uses a language model pipeline
    to process a detailed prompt, format the response into Markdown, and
    returns the resulting summarized text.

    :param state: A dictionary containing the necessary inputs for the generation
                  process including:
                  - original_question: The specific question regarding the regulation.
                  - original_context: The context or background information about
                    the regulation.
                  - openai_api_key: The API key for accessing the OpenAI language model.
    :return: A dictionary (`state`) updated with the concise summary of the
             regulation under the key `"concise_summary"`.
    """
    concise_summary_prompt = '''
    ## **Instructions**
    - Provide a **concise, well-structured paragraph** summarizing the regulation in **Original Question**.  
    - Use **clear, plain language** to explain the regulation’s **core purpose, scope, and impact**.  
    - Break response into **Authority and Purpose**, **Scope**, and **Impact** sections. 
    - Ensure output is in Markdown format.
    - Ensure that summary is less than three sentences.
 
    ## **Input Sections**
    ### **Context**
    {context}

    ### **Original Question**
    {original_question}

    '''
    # Bring Original Question
    original_question = state["original_question"]
    original_context = state["original_context"]
    openai_api_key = state["openai_api_key"]

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", concise_summary_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": original_context,
        "original_question": original_question,
    })

    state["concise_summary"] = generation

    return state

def concise_summary(original_question: str, context: str, OPENAI_API_KEY:str) -> str:
    concise_summary_prompt = '''
    ## **Instructions**
    - Provide a **concise, well-structured paragraph** summarizing the regulation in **Original Question**.  
    - Use **clear, plain language** to explain the regulation’s **core purpose, scope, and impact**.  
    - Ensure output is in Markdown format.

    ## **Input Sections**
    ### **Context**
    {context}

    ### **Original Question**
    {original_question}

    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", concise_summary_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = rag_chain.invoke({
        "context": context,
        "original_question": original_question,
    })

    return generation
