import time
import asyncio
from io import StringIO

from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI
from src.data_processing import *
import pandas as pd
import warnings
from typing import Any, List, Dict
from io import StringIO
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
warnings.filterwarnings("ignore")

DEVELOPER_PROMPT_TEMPLATE = '''
        # **Regulatory Requirement Extraction & Formatting**  
        
        ## **Instructions**  
        - Your task is to extract and format **clear, explicit legal requirements** from the given regulatory text. \n
        - You must maintain the **exact working* from the regulation \n
        - Do not paraphrase, summarize, or interpret the regulatory text. \n
        - Do not include general guidance, definitions, explanations, or recommendations. \n
        - Do not alter phrasing. \n
        - Use the **exact wording** from the regulation. \n
        - Ensure that extracted requirements are **properly formatted** and **CSV-compatible**. \n
        - Ensure that extracted requirements are **complete coverage** of the regulatory text. \n
        - Ensure that extracted requirements are **delimiter & formatting** compatible with CSV parsing. \n
        - Ensure that extracted requirements are **final validation** before outputting. \n
        - Ensure that extracted requirements are **ranked by importance and potential impact**. \n
        - Ensure that extracted requirements are **limited to max 15 highest-ranking requirements**. \n
    
        ---  
        
        ## **Regulatory Text**  
        {}
        
        ## **Output Format**  
        Provide the extracted requirements as a **CSV-parsable table** using the following structure:  
        
        | Business Requirement |  
        |----------------------|  
        | [Extracted requirement 1] |  
        | [Extracted requirement 2] |  
        | … |  
        
        - Each row must contain **one standalone regulatory requirement**.  
        - Maintain the **exact wording** from the regulation—**do not paraphrase, summarize, or interpret**.  
        
        ---  
        
        ## **Extraction Guidelines**  
        1. **Explicit Legal Requirements Only**: Extract only **clearly defined legal obligations**—statements that mandate, prohibit, or require specific actions.  
        2. **Rank them by importance and potential impact**
        3. **Limit to max 15 highest-ranking requirements*
        4. **No Contextual or Advisory Text**: Exclude general guidance, definitions, explanations, or recommendations.  
        5. **Verbatim Extraction**: Use the **exact wording** from the regulatory text. **Do not alter phrasing**.  
        6. **Structural Integrity**: Ensure the extracted requirements are **properly formatted** and **CSV-compatible**.  
        7. **Complete Coverage**: Review the entire regulatory text to extract **all** relevant requirements.  
        8. **Delimiter & Formatting**:  
           - Use `"^^"` as the delimiter for the CSV output.  
           - Ensure **each row is properly formatted** and compatible with CSV parsing.  
        9. **Final Validation**: Before outputting, **verify** that:  
           - The delimiter `"^^"` is consistently applied.  
           - Each requirement is properly structured in its own row.  
        
        ---  
        
        ## **Output Format Specification**  
        - Output must be a **CSV string** formatted with `"^^"` as the delimiter.  
        - Ensure **no additional text** is included outside of the CSV table.  
        - Verify **correct formatting** before returning the output.  
        
        ---  
'''

def verbatim_business_requirements_2(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the provided state dictionary to generate verbatim business requirements
    based on the updated question and filtered context. This function uses token
    validation, constructs a formatted developer prompt, queries the OpenAI API,
    and updates the state with the resulting requirements.

    :param state: A dictionary containing the necessary keys such as
        'updated_question', 'filtered_context', and 'openai_api_key' to process
        and retrieve the business requirements.
    :type state: Dict[str, Any]
    :return: The updated state dictionary with the added key
        'verbatim_business_requirements' containing the generated requirements.
    :rtype: Dict[str, Any]
    :raises ValueError: If the combined token count of the context exceeds 150,000.
    """
    # Define variables
    updated_question = state["updated_question"]
    filtered_context = state["filtered_context"]

    chunk_token_count = count_tokens_gpt4(filtered_context)

    # Error handling: raise an error if the token count exceeds 100k tokens
    if chunk_token_count > 150000:
        raise ValueError(
            f"Combined context token count ({chunk_token_count}) exceeds the 100k limit. "
            "Please provide a shorter input or adjust the context."
        )

    # Format the prompt using the precompiled template
    developer_prompt = DEVELOPER_PROMPT_TEMPLATE.format(filtered_context)

    # Query O3-mini to get the best resu lts
    client = OpenAI(api_key=state["openai_api_key"])
    completion = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": updated_question}
        ]
    )

    # Output
    state["verbatim_business_requirements"] = completion.choices[0].message.content

    return state

def verbatim_business_requirements(state: Dict[str, Any]) -> Dict[str, Any]:
    # Define variables
    updated_question = state["updated_question"]
    filtered_context = state["filtered_context"]
    openai_api_key = state["openai_api_key"]

    verbatim_business_requirement_prompt = '''
        # **Regulatory Requirement Extraction & Formatting**  
        
        ## Objectives
        - Extract **explicit legal requirements** from the regulatory text below, using **verbatim wording only**, and present them in a **CSV-parsable format** for downstream compliance processing.
        - Keep verbatim the original language and do not translate to English. 
        
        ## **Instructions**  
        1. Extract only **clear, specific legal obligations** (e.g., mandates, prohibitions, required actions).  
        2. Use the **exact wording** from the regulation.  
           - **Do not paraphrase**, summarize, or interpret.  
           - **Do not include** contextual, advisory, or explanatory content.  
        3. Ensure extracted requirements are:  
           - **Properly formatted** for CSV export.  
           - **Delimited** with `"^^"` (double caret).  
           - **Standalone**, each as a complete requirement.  
        4. Perform a **final validation** before output:  
           - No broken rows or line breaks.  
           - All rows follow the CSV structure.  
           - Delimiters are used **consistently**.  
        5. **Rank** the requirements by:  
           - **Importance**  
           - **Potential business impact**  
        6. **Limit** the final list to **15 top-ranked requirements**.

        ---  
        
        ## **Regulatory Text**  
        {regulatory_text}
        
        ## **Output Format**  
        - Provide the output as a **CSV string**, using `"^^"` as the delimiter.
        - No headings or text outside the CSV block.  
        - Each row must contain **one standalone requirement**.  
        - Validate all formatting before final output.

        ### **Example Output Format**
        Business Requirement^^
        [Requirement 1]^^
        [Requirement 2]^^
        ...
        [Requirement 15]^^
        ---  
        
        ## **Requirements Guidelines**  
        1. **Explicit Legal Requirements Only**: Extract only **clearly defined legal obligations**—statements that mandate, prohibit, or require specific actions.  
        2. **Rank them by importance and potential impact**
        3. **Limit to max 15 highest-ranking requirements*
        4. **No Contextual or Advisory Text**: Exclude general guidance, definitions, explanations, or recommendations.  
        5. **Verbatim Extraction**: Use the **exact wording** from the regulatory text. **Do not alter phrasing**.  
        6. **Structural Integrity**: Ensure the extracted requirements are **properly formatted** and **CSV-compatible**.  
        7. **Complete Coverage**: Review the entire regulatory text to extract **all** relevant requirements.  
        8. **Delimiter & Formatting**:  
           - Use `"^^"` as the delimiter for the CSV output.  
           - Ensure **each row is properly formatted** and compatible with CSV parsing.  
        9. **Final Validation**: Before outputting, **verify** that:  
           - The delimiter `"^^"` is consistently applied.  
           - Each requirement is properly structured in its own row.  
        
        ---  
        '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", verbatim_business_requirement_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model="o3-mini", reasoning_effort="low", api_key=openai_api_key)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Asynchronously invoke the chain with the provided inputs.
    generation = rag_chain.invoke({
        "regulatory_text": filtered_context,
    })

    state["verbatim_business_requirements"] = generation

    return state

async def a_convert_str_to_df(state: Dict[str, Any], retries=5) -> pd.DataFrame:
    # Define Inputs
    data_str = state["verbatim_business_requirements"]
    openai_api_key = state["openai_api_key"]

    # Run Loop to transform
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            # Use the python engine for better error tolerance
            df = await asyncio.to_thread(pd.read_csv, StringIO(data_str), sep="^^", engine='python')
            return df
        except pd.errors.ParserError as pe:
            print(f"Attempt {attempt}: ParserError encountered: {pe}")
            # On the final attempt, try GPT conversion
            if attempt == retries:
                try:
                    client = OpenAI(api_key=openai_api_key)
                    completion = asyncio.to_thread(
                        client.chat.completions.create,
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "developer", "content": markdown_to_csv_prompt},
                            {"role": "user", "content": data_str}
                        ]
                    )
                    new_csv = completion.choices[0].message.content
                    df = await asyncio.to_thread(
                        pd.read_csv, StringIO(new_csv), sep="^^", engine='python', on_bad_lines='skip'
                    )
                    return df
                except Exception as gpt_e:
                    last_exception = gpt_e
                    print(f"GPT conversion attempt failed: {gpt_e}")
                    break
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt}: Unexpected error encountered: {e}")

    # Final attempt: skip bad lines if previous attempts fail
    try:
        print("Final attempt: trying to read by skipping bad lines.")
        df = pd.read_csv(StringIO(data_str), sep="^^", on_bad_lines='skip', engine='python')
        print("DataFrame created by skipping problematic lines.")
        return df
    except Exception as final_e:
        print("Final attempt failed.")
        raise Exception(f"Failed to convert string to DataFrame after {retries} retries and skipping bad lines.") from final_e
