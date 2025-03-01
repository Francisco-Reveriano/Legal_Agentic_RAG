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


async def prohibitions_buss_req(original_question: str, business_requirement: str,
                               detailed_business_requirement: str) -> str:
    """
    Generate a list of regulatory prohibitions with clarity and legal precision based on the
    provided regulatory context, business requirements, and detailed business requirements.

    :param original_question: Original query or question being addressed to generate regulatory prohibitions.
    :type original_question: str
    :param business_requirement: Verbatim business requirement text outlining the regulatory context or
                                  specific requirements.
    :type business_requirement: str
    :param detailed_business_requirement: Expounded or detailed explanation of the business requirements which
                                           complements the verbatim requirements.
    :type detailed_business_requirement: str
    :return: A generated structured output in markdown format detailing the list of prohibited actions
             or operations based on the input provided.
    :rtype: str
    """
    prohibitions_requirement_detailed_prompt = '''
    #  **Detailed Prompt for Prohibitions*

    ## **Your Task**
    - Create a list of bullet points directly outlining what is prohibited based on the provided regulatory text **Context**, **Verbatim Business Requirement**, **Original Question**, and **Detailed Business Requirements* while ensuring clarity, accuracy, and legal precision
    - Provide output in correctly formatted Markdown
    - Keep response succinct and brief
    - Double-check that output is correct and in Markdown format

    ## **Key Characteristics of Prohibitions**
    - In a regulatory context, "prohibitions" refer to the specific actions, activities, or operations that a company is legally barred from performing under applicable laws, regulations, and industry standards.
    - Applicable low should be based on the provided **Context**, **Verbatim Business Requirement**, and **Original Question**.
    
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
        ("system", prohibitions_requirement_detailed_prompt),
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
