import asyncio
import warnings
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")


async def table_to_markdown(df: str, OPENAI_API_KEY:str, model_name:str="gpt-4o", temperature:float=0) -> str:
    """
    Converts a given DataFrame to a Markdown table using a structured prompt and an AI language model.

    This function takes an input DataFrame, processes it into the required Markdown table format
    according to strict mapping rules, and outputs the result. Each AI initialization and processing
    is performed asynchronously to ensure efficiency. The Markdown table includes specific columns,
    mapped from the input DataFrame with bullet points structured in the required format.

    :param df: The input dataframe as a string representation of the data to be processed.
    :param OPENAI_API_KEY: API key for accessing OpenAI's GPT-based models for processing.
    :param model_name: Name of the OpenAI model to be used (default is "gpt-4o-mini").
    :param temperature: The temperature parameter to control randomness in AI outputs.
    :return: A string containing the Markdown formatted table built from the input dataframe.
    """
    table_to_markdown_prompt = '''
# Instructions
- Convert the provided DataFrame **Original DataFrame** into a properly formatted three-column Markdown table.
- Ensure the table output is in valid **Markdown format**.
- Maintain bullet points for the **Requirements, Permissions, Prohibitions** column without synthesizing the content.
- **Ensure a blank line** separates **Requirement**, **Permission**, and **Prohibition** in the **Requirements, Permissions, Prohibitions** column.

# **Table Structure**
1. **Business Requirements** → Maps to the "Business_Requirements" column from the **Original DataFrame**.
2. **Simplified Business Requirements** → Maps to the "Simplified_Business_Requirements" column from the **Original DataFrame**.
3. **Requirements, Permissions, Prohibitions** → Aggregates content from:
   - "Detailed_Business_Requirements"
   - "Permission_Business_Requirements"
   - "Prohibition_Business_Requirements"
   - **Each section (Requirement, Permission, Prohibition) must be separated by a blank line.**

# **Example Table Output Format**
| **Business Requirements** | **Simplified Business Requirements** | **Requirements, Permissions, Prohibitions** |
|---------------------------|--------------------------------------|--------------------------------------------|
| Business_Requirements from the DataFrame | Simplified_Business_Requirements from the DataFrame | - **Requirement:**  <br>   - "Detailed_Business_Requirements" in bullet points  <br> <br> - **Permission:**  <br>   - "Permission_Business_Requirements" in bullet points  <br> <br> - **Prohibition:**  <br>   - "Prohibition_Business_Requirements" in bullet points |

# **Original DataFrame**
{original_dataframe}


    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", table_to_markdown_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()


    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "original_dataframe": df,
    })

    return generation
