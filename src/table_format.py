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


async def table_to_markdown(df: str) -> str:

    table_to_markdown_prompt = '''
    # Instructions
    - Convert the provided DataFrame **Original Dataframe** into a three column table
    - Table output should be proper Markdown format
    - Double-check that the output is in proper markdown
    
    # **Table Structure**
    |**Business Requirements**|**Simplified Business Requirements**|**Requirements, Permissions, Prohibitions**|
    |-----------------------|----------------------------------|-----------------------------------------|
    | Business Requirements from the dataframe column | Simplified Requirement from the dataframe column |  <br> - **Requirement:** Detailed Business Requirements column in bullet points \n <br> - **Permission:** Permission Business Requirements column in bullet points \n <br> - **Prohibition:** Prohibitions Business Requirements in bullet points \n|
    
    # **Original Dataframe**
    {original_dataframe}
    
    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", table_to_markdown_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "original_dataframe": df,
    })

    return generation
