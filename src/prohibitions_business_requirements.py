import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Dict

async def prohibitions_buss_req(state:Dict[str, Any], business_requirement: str,
                               detailed_business_requirement: str, model_name:str="gpt-4o-mini") -> str:

    prohibitions_requirement_detailed_prompt = '''
    #  **Detailed Prompt for Prohibitions*

    ## **Your Task**
    - Create a list of bullet points directly outlining what is prohibited based on the provided regulatory text **Context**, **Verbatim Business Requirement**, **Original Question**, and **Detailed Business Requirements* while ensuring clarity, accuracy, and legal precision
    - Provide output in correctly formatted Markdown
    - Keep response succinct and brief
    - Double-check that output is correct and in Markdown format
    - Remove all actions or next steps 

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
    # Inputs
    updated_question: str = state["updated_question"]
    filtered_context: str = state["filtered_context"]

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", prohibitions_requirement_detailed_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": filtered_context,
        "original_question": updated_question,
        "business_requirement": business_requirement,
        "detailed_business_requirement": detailed_business_requirement,
    })

    return generation
