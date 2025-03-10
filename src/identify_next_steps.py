import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Dict


async def identify_detailed_actions(state: Dict[str, Any],
                                detailed_business_requirements: str,
                                detailed_business_permissions: str,
                                detailed_business_prohibitions: str) -> str:
    # Inputs
    filtered_context: str = state["filtered_context"]

    # Identify Prompt
    identify_detailed_actions_prompt = f'''
    #  **Detailed Prompt for Prohibitions*

    ## **Your Task**
    - From the provided **Input Section** identify the key next steps to meet Business Requirements, Business Permissions, and Business Prohibitions.
    - Provide output in correctly formatted Markdown
    - Keep response succinct and brief
    - Double-check that output is correct and in Markdown format

    ## **Key Characteristics of Prohibitions**
    - In a regulatory context, "prohibitions" refer to the specific actions, activities, or operations that a company is legally barred from performing under applicable laws, regulations, and industry standards.
    - Applicable low should be based on the provided **Context**, **Verbatim Business Requirement**, and **Original Question**.

    ## **Input Sections**
    ### **Context**
    {filtered_context}

    ### **Detailed Business Requirements**
    {detailed_business_requirements}

    ### **Detailed Business Permissions**
    {detailed_business_permissions}

    ### **Detailed Business Prohibitions**
    {detailed_business_prohibitions}

    '''
    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", identify_detailed_actions_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model="o3-mini", reasoning_effort="high", api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "filtered_context": filtered_context,
        "detailed_business_requirement": detailed_business_requirements,
        "detailed_business_permissions": detailed_business_permissions,
        "detailed_business_prohibitions": detailed_business_prohibitions,
    })

    return generation
