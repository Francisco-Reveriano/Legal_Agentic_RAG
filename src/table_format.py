import asyncio
import warnings
import asyncio

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")


async def table_to_markdown(df: pd.DataFrame, OPENAI_API_KEY:str, model_name:str="o3-mini", temperature:float=0) -> str:
    """
    Asynchronously converts a given Pandas DataFrame into a Markdown table format using
    the OpenAI language model. This function utilizes a detailed prompt to guide the
    language model in producing a properly formatted Markdown table. It leverages
    asynchronous processing and constructs the Markdown table from specific columns
    in the DataFrame while preserving all verbatim content.

    :param df: The Pandas DataFrame containing data to be converted into a Markdown table.
    :param OPENAI_API_KEY: The API key required to authenticate with the OpenAI service.
    :param model_name: The specific model name used for conversion, defaults to "gpt-4o-mini".
    :param temperature: The temperature setting for the OpenAI model, defaults to 0.
    :return: A string representation of the DataFrame as a Markdown table.
    :rtype: str
    """
    table_to_markdown_prompt = '''
    # Instructions
    - Convert the provided DataFrame **Original DataFrame** into a properly formatted Markdown table.
    - "Business Requirements" from the Dataframe into **Business Requirements** table column
    - "Simplified Business Requirements" from the Dataframe into **Simplified Business Requirements** table column
    - "Requirements, Permissions, Prohibitions" from the Dataframe into **Requirements, Permissions, Prohibitions** table column
    - Do not synthesize and keep language verbatim.
    - Ensure that **Requirements, Permissions, Prohibitions** has a proper bullet point format and structure. 
    - Ensure Table is in markdown format
    - Ensure that all instructions are followed.
    
    
    # **Example Table Output Format**
    | **Business Requirements** | **Simplified Business Requirements** | **Requirements, Permissions, Prohibitions** |
    |---------------------------|--------------------------------------|---------------------------------------------|
    |                           |                                      |                                             |

    # **Original DataFrame**
    {original_dataframe}

    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", table_to_markdown_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name=model_name, reasoning_effort="low", api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "original_dataframe": df[["Business_Requirements", "Simplified_Business_Requirements","Combined_Requirements_Permissions_Prohibitions"]].to_json(),
    })

    return generation
