import asyncio
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Dict

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

# Extracted constant for detailed prompt template
BUSINESS_REQUIREMENT_DETAILED_PROMPT = '''
    #  **Detailed Prompt for Requirements**
    
    ## **Your Task**
        - Provide list of detailed requirements and actions necessary to meet the **Verbatim Business Requirements*** based on the provided **Context** and **Verbatim Business Requirement**
        - Provide a detailed description of each requirement and action in bullet points
        - Ensure that each requirement follows **Instructions & Key Characteristics**
        - Do not include a summary or synthesis
    
    ## **Instructions & Key Characteristics**
        1. Specific – Clearly define what needs to be done.
        2. Actionable – Outline actions step-by-step.
        3. Measurable – Ensure progress can be tracked.
        4. Ordered – Steps must follow a logical sequence.
        5. Achievable – Feasible within given constraints.
        6. Time-bound – Provide deadlines where applicable.
    ## **Input Sections**
    
    ### **Context**
    {context}
    
    ### **Verbatim Business Requirements**
    {business_requirement}
    
    ### **Original Question**
    {original_question}
'''


def create_chat_prompt_template() -> ChatPromptTemplate:
    """Helper function to create and return a ChatPromptTemplate instance."""
    return ChatPromptTemplate([("system", BUSINESS_REQUIREMENT_DETAILED_PROMPT)])


async def generate_detailed_requirements(business_requirement: str, state:Dict[str, Any], model_name:str="gpt-4o-mini") -> str:
    # Inputs
    updated_question: str = state["updated_question"]
    filtered_context: str = state["filtered_context"]

    # Create a chat prompt template
    prompt = create_chat_prompt_template()

    # Initialize the ChatOpenAI instance
    llm = ChatOpenAI(model=model_name, temperature=0.3, api_key=state["openai_api_key"])

    # Combine prompt, language model, and parser into a chain
    rag_chain = prompt | llm | StrOutputParser()

    # Invoke the processing chain with required inputs
    output = await rag_chain.ainvoke({
        "context": filtered_context,
        "business_requirement": business_requirement,
        "original_question": updated_question,
    })

    return output
