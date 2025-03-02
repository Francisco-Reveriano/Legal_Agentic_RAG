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

from src import detailed_business_requirements
from src.business_requirements import retriever  # Ensure retriever is imported once
from src.prompts import *

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")


async def concise_summary(original_question: str, OPENAI_API_KEY:str, PINECONE_API_KEY:str, top_k:int=25) -> str:
    """
    Provides a concise summary of a regulation based on an input question and relevant context.
    This function uses a pipeline consisting of a language model, an output parser, and a prompt
    template to generate a Markdown-formatted summary of the specified regulation.

    :param original_question: A string representing the input question or regulation-related
        text for which a structured summary is generated.
    :param OPENAI_API_KEY: A string representing the API key for accessing OpenAI's language
        model functionalities.
    :param PINECONE_API_KEY: A string representing the API key for accessing Pinecone services
        used for document retrieval.
    :param top_k: An integer specifying the maximum number of documents to retrieve as
        relevant context for generating the summary. The default value is 25.

    :return: A string containing the Markdown-formatted summary of the regulation, generated
        based on the provided question and contextual information.
    """
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

    # Retrieve documents relevant to the query.
    # If `retriever` is a blocking function, run it in a separate thread.
    docs = await asyncio.to_thread(
        retriever,
        query=" ".join([original_question]),
        PINECONE_API_KEY=PINECONE_API_KEY,
        OPENAI_API_KEY=OPENAI_API_KEY,
        top_k=top_k,
    )

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": docs,
        "original_question": original_question,
    })

    return generation
