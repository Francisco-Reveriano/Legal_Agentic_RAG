import asyncio
import warnings
import os

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.business_requirements import retriever
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")

async def simplified_buss_req(original_question: str, business_requirement: str, OPENAI_API_KEY:str, PINECONE_API_KEY:str, model_name:str="gpt-4o-mini", top_k:int=25) -> str:
    """
    Simplifies a business requirement by synthesizing a plain English response that aligns
    with the provided context, verbatim business requirement, and original question. This
    function processes inputs through a pre-defined synthesis prompt, retrieves relevant
    documents, and generates a concise synthesis using a language model.

    :param original_question: The original query or question that needs to be addressed.
    :type original_question: str
    :param business_requirement: A verbatim business requirement to guide the synthesis.
    :type business_requirement: str
    :param OPENAI_API_KEY: API key for accessing OpenAI services.
    :type OPENAI_API_KEY: str
    :param PINECONE_API_KEY: API key for accessing Pinecone services.
    :type PINECONE_API_KEY: str
    :param model_name: Name of the OpenAI model to use (default: "gpt-4o-mini").
    :type model_name: str
    :param top_k: The number of top documents to retrieve for context (default: 25).
    :type top_k: int
    :return: A synthesized plain English response addressing the original query while
             adhering to the constraints of the business requirement and context.
    :rtype: str
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
    llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve documents relevant to the query.
    # If `retriever` is a blocking function, we run it in a separate thread.
    docs = await asyncio.to_thread(
        retriever,
        query=" ".join([original_question, business_requirement]),
        PINECONE_API_KEY=PINECONE_API_KEY,
        OPENAI_API_KEY=OPENAI_API_KEY,
        top_k=top_k
    )

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": docs,
        "original_question": original_question,
        "business_requirement": business_requirement
    })

    return generation
