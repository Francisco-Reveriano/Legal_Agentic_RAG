import asyncio
import warnings
import os
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


async def a_concise_summary(original_question: str, context: str, OPENAI_API_KEY:str) -> str:
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
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": context,
        "original_question": original_question,
    })

    return generation

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
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = rag_chain.invoke({
        "context": context,
        "original_question": original_question,
    })

    return generation
