import asyncio
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Dict
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")

async def permissions_buss_req(state:Dict[str, Any], business_requirement: str, detailed_business_requirement:str, model_name:str="gpt-4o-mini") -> str:

    permission_requirement_detailed_prompt = '''
    #  **Detailed Prompt for Permissions*

    ## **Your Task**
    - Create a list of bullet points directly outlining what is permitted based on the provided **Context**, **Verbatim Business Requirement**, **Original Question**, and **Detailed Business Requirements* while ensuring clarity, accuracy, and legal precision
    - Provide a detailed description of each permission and action in bullet points
    - Provide output in correctly formatted Markdown
    - Do not include a summary or synthesis
    
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

    # Inputs
    updated_question: str = state["updated_question"]
    filtered_context: str = state["filtered_context"]

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", permission_requirement_detailed_prompt),
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
