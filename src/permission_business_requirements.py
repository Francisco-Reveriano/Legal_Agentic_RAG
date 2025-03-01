"""
Module: detailed_business_requirements.py

Description:
    This module provides an asynchronous function `permissions_buss_req` that generates a detailed list of
    business permissions by combining prompt engineering, document retrieval, and language model inference.
    The function constructs a specialized prompt tailored to extracting permitted actions within a regulatory
    framework, retrieves context documents related to the input query, and then invokes a language model chain
    to generate a clear, step-by-step bullet list of permissions based on an original question and a verbatim business requirement.

Dependencies:
    - asyncio: For asynchronous execution.
    - warnings: To suppress non-critical warnings.
    - os: For interacting with environment variables.
    - dotenv: For loading environment variables from a .env file.
    - pandas: Imported for potential data manipulation.
    - io.StringIO: For string-based I/O operations.
    - tiktoken: For tokenization tasks.
    - langchain (and submodules): For building prompt templates, handling language model chains, and parsing outputs.
    - pinecone: For integrating with Pinecone’s vector database.
    - src.business_requirements.retriever: A module for retrieving relevant documents based on a query.
    - src.prompts: Additional prompt configurations.

Usage:
    This module is intended to be used in asynchronous contexts. For example:

        import asyncio
        result = asyncio.run(permissions_buss_req(
            "What are the compliance requirements?",
            "Ensure full regulatory compliance"
        ))
        print(result)

Environment Setup:
    - The OpenAI API key is loaded from an environment variable ('OPENAI_API_KEY').
    - The Pinecone API key is similarly loaded from 'PINECONE_API_KEY'.
    - Any warnings are suppressed for cleaner output.
"""

import asyncio
import warnings
import os
from dotenv import load_dotenv
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

# Load environment variables from the .env file
load_dotenv()

# Set API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


async def permissions_buss_req(original_question: str, business_requirement: str, detailed_business_requirement:str) -> str:
    """
    Asynchronously processes a query related to business permissions using context,
    business requirements, and detailed descriptions. This function utilizes a
    language model to generate a detailed list of actions permitted by regulations
    in a well-structured and detailed format. It ensures legal and contextual clarity
    in the output while leveraging retrieval and processing techniques.

    :param original_question: The initial question asking for clarification or
        guidance on specific permissions related to business operations.
    :type original_question: str
    :param business_requirement: Verbatim input specifying the overarching
        requirement or directive related to business operations within a regulatory
        or operational context.
    :type business_requirement: str
    :param detailed_business_requirement: A more comprehensive breakdown or
        specification of the business requirement, providing additional insight
        or fine-grained details regarding the scope and constraints of the original
        question.
    :type detailed_business_requirement: str

    :return: A structured, detailed, and clear representation of permissions and
        actions permitted based on provided context, business requirements, and
        detailed descriptions.
    :rtype: str
    """

    permission_requirement_detailed_prompt = '''
    #  **Detailed Prompt for Permissions*

    ## **Your Task**
    - Create a list of bullet points directly outlining what is permitted based on the provided **Context**, **Verbatim Business Requirement**, **Original Question**, and **Detailed Business Requirements* while ensuring clarity, accuracy, and legal precision
    - Provide output in correctly formatted Markdown
    
    ## **Key Characteristics of Permissions**
    - In a regulatory context, "permissions" refer to the specific actions, activities, or operations that a company is legally authorized to perform under applicable laws, regulations, and industry standards.

    ## **Input Sections**
    ### **Context**
    {context}

    ### **Verbatim Business Requirements**
    {business_requirement}

    ### **Original Question**
    {original_question}
    
    ### **Detailed Business Requirements**
    {detailed_business_requirement}

    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", permission_requirement_detailed_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve documents relevant to the query.
    # If `retriever` is a blocking function, run it in a separate thread.
    docs = await asyncio.to_thread(
        retriever,
        query=" ".join([original_question, business_requirement, detailed_business_requirement]),
        top_k=20,
    )

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": docs,
        "original_question": original_question,
        "business_requirement": business_requirement,
        "detailed_business_requirement": detailed_business_requirement,
    })

    return generation
