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
from src.business_requirements import retriever  # Ensure retriever is imported once
from src.prompts import *

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")

# Load environment variables from the .env file
load_dotenv()

# Set API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


async def simplified_buss_req(original_question: str, business_requirement: str) -> str:
    """
    Asynchronously synthesizes a simplified business requirement response using a language model chain.

    This function builds a synthesis prompt that integrates the context (retrieved documents),
    the verbatim business requirement, and the original question. It then constructs a chain of operations
    using a prompt template, an asynchronous language model, and an output parser to generate a concise,
    plain English synthesis adhering to legal and business constraints.

    Parameters:
        original_question (str): The original query or question to be synthesized.
        business_requirement (str): The verbatim business requirement to be incorporated.

    Returns:
        str: A synthesized response addressing the original question while following the provided
             business requirement.
    """
    business_requirement_synthesis_prompt = '''
    # **Synthesis Prompt for RAG**

    ## **Your Task**
    Synthesize a response based on the provided **Context**, **Verbatim Business Requirement**, and **Original Question** while ensuring clarity, accuracy, and legal precision.

    ## **Instructions**

    ### 1. **Read and Analyze**
    - Carefully review the **Context**, **Verbatim Business Requirement**, and **Question to Synthesize**.
    - Identify key points, constraints, and any legal aspects that must be addressed.

    ### 2. **Step-by-Step Synthesis (Chain of Thought)**
    - Extract the essential information needed to answer the question.
    - Cross-check the synthesis to make sure it is related to the original question.
    - Cross-check the synthesis against the **Verbatim Business Requirement** to ensure it is **fully aligned** and **verbatim where required**.
    - Verify that any **legal considerations** are correctly represented.
    - Ensure there are **no logical inconsistencies** or missing details.
    - Ensure that synthesis is related to the original question asked.

    ### 3. **Plain English Summary**
    - Provide a **clear and concise synthesis** that directly synthesizes the business requirement in **simple, plain English**.
    - Keep the synthesis **under four sentences** while preserving accuracy and completeness.
    - Synthesis should only be on the provided verbatim business requirement.
    - Use natural language and avoid using any technical terms.

    ### 4. **Accuracy and Validation**
    - Before finalizing, **double-check** that:
      - The synthesis **directly addresses** the original question.
      - The **Verbatim Business Requirement is used exactly as provided**.
      - No critical details from the **Context** are omitted or misrepresented.
    - If inconsistencies arise, **adjust and revalidate** the synthesis before finalizing.

    ---

    ## **Input Sections**

    ### **Context**
    {context}

    ### **Verbatim Business Requirement**
    {business_requirement}

    ### **Original Question**
    {original_question}
    '''

    # Create a chat prompt template using the synthesis prompt.
    prompt = ChatPromptTemplate([
        ("system", business_requirement_synthesis_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve documents relevant to the query.
    # If `retriever` is a blocking function, we run it in a separate thread.
    docs = await asyncio.to_thread(
        retriever,
        query=" ".join([original_question, business_requirement]),
        top_k=15
    )

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": docs,
        "original_question": original_question,
        "business_requirement": business_requirement
    })

    return generation
