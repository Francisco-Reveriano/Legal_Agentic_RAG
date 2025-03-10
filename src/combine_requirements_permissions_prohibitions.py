import asyncio
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Dict
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")


async def combined_req_perm_prohibition_markdown(detailed_business_requirements: str, permission_business_requirements: str, prohibitions_business_requirements: str,
                               state:Dict[str, Any], model_name: str = "gpt-4o-mini") -> str:

    combined_table_prompt = '''
        # Instructions
        - Combine **Detailed_Business_Requirements**, **Permission_Business_Requirements**, and "Prohibitions_Business_Requirements** into a single list with three sections titled *Requirements*, *Permissions*, and *Prohibitions*
        - Ensure that output is in proper markdown

        # **Detailed_Business_Requirements**
        {detailed_business_requirements}

        # **Permission_Business_Requirements**
        {permission_business_requirements}

        # **Prohibitions_Business_Requirements**
        {prohibitions_business_requirements}

        '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", combined_table_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Asynchronously invoke the chain with the provided inputs.
    generation = rag_chain.invoke({
        "detailed_business_requirements": detailed_business_requirements,
        "permission_business_requirements": permission_business_requirements,
        "prohibitions_business_requirements": prohibitions_business_requirements,
    })

    return generation
