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


async def key_terms(original_question: str, response:str, top_k: int = 25):

    key_terms_prompt = '''
    # **Instructions**
    - Extract and identify the most important **key terms** from the provided **Response**.
    - Use the **Context** to define each **key term* clearly and accurately.
    - Present definitions in a **bullet-point list**, using **concise, one-sentence explanations** in **simple English**.
    - **Do not define** the primary regulation mentioned in the **Original Question**.
    - Format the output in **correct Markdown** (ensure proper spacing, bolding, and bullet points).  
    
    ## **Output Format Example**
    ```markdown
    - **Key Term 1**: Definition based on the Context.
    - **Key Term 2**: Definition based on the Context.

    
    # **User Inputs**
    ## ** Response**
    {response}
    
    ## **Context**
    {context}
    
    ## **Original Question**
    {original_question}

    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", key_terms_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve documents relevant to the query.
    # If `retriever` is a blocking function, run it in a separate thread.
    docs = await asyncio.to_thread(
        retriever,
        query=original_question,
        top_k=top_k,
    )

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "response": response,
        "context": docs,
        "original_question": original_question,
    })

    return generation, docs
