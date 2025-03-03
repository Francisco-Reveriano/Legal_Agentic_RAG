import asyncio
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.business_requirements import retriever

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")


async def combined_req_perm_prohibition_markdown(detailed_business_requirements: str, permission_business_requirements: str, prohibitions_business_requirements: str,
                               OPENAI_API_KEY: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Asynchronously generates a single markdown document combining
    detailed business requirements, permission business requirements,
    and prohibitions business requirements into three distinct sections:
    *Requirements*, *Permissions*, and *Prohibitions*. The function takes
    input strings for each of the business requirements and uses an
    OpenAI language model to produce a structured markdown output.

    :param detailed_business_requirements: The input string containing
        detailed business requirements.
    :type detailed_business_requirements: str
    :param permission_business_requirements: The input string containing
        permission-related business requirements.
    :type permission_business_requirements: str
    :param prohibitions_business_requirements: The input string containing
        prohibitions-related business requirements.
    :type prohibitions_business_requirements: str
    :param OPENAI_API_KEY: The API key string for authenticating
        with the OpenAI API.
    :type OPENAI_API_KEY: str
    :param model_name: The name of the specific language model to
        leverage. Defaults to "gpt-4o-mini".
    :type model_name: str, optional
    :return: A markdown-formatted string containing the combined
        and structured list of business requirements.
    :rtype: str
    """
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
    llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Asynchronously invoke the chain with the provided inputs.
    generation = rag_chain.invoke({
        "detailed_business_requirements": detailed_business_requirements,
        "permission_business_requirements": permission_business_requirements,
        "prohibitions_business_requirements": prohibitions_business_requirements,
    })

    return generation
