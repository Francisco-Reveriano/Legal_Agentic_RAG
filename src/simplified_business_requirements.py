import asyncio
import warnings
import os

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")
from typing import Any, List, Dict

async def simplified_buss_req(business_requirement: str, state:Dict[str, Any], model_name:str="gpt-4o-mini") -> str:

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

    # Inputs
    updated_question:str = state["updated_question"]
    filtered_context:str = state["filtered_context"]


    # Create a chat prompt template using the synthesis prompt.
    prompt = ChatPromptTemplate([
        ("system", business_requirement_synthesis_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": filtered_context,
        "original_question": updated_question,
        "business_requirement": business_requirement
    })

    return generation
