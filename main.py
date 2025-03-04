import time
import warnings
import os
import asyncio
import pandas as pd

# Importing project-specific modules for business requirements and data processing
from src.business_requirements import *
from src.data_processing import *
from src.prompts import *
from src.simplified_business_requirements import *
from src.detailed_business_requirements import *
from src.permission_business_requirements import *
from src.prohibitions_business_requirements import *
from src.concise_summary import *
from src.table_format import *
from src.key_terms import *
from src.combine_requirements_permissions_prohibitions import *

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

# Asynchronous function to generate simplified business requirements for a DataFrame of business requirements
async def simplified_business_requirement(query: str, df: pd.DataFrame, OPENAI_API_KEY:str, PINECONE_API_KEY:str, top_k:int=20):
    """
    Asynchronously processes a set of business requirements provided in a DataFrame, simplifying them
    based on a given query. The function uses OpenAI's API and Pinecone for natural language processing
    and retrieval tasks. The updated DataFrame is then returned with an additional column containing
    the simplified business requirements.

    :param query: A string representing the main query or context for simplifying the
        business requirements.
    :param df: A pandas DataFrame containing a column `Business_Requirements` which holds
        the business requirements to be simplified.
    :param OPENAI_API_KEY: The API key for accessing OpenAI's services.
    :param PINECONE_API_KEY: The API key for accessing Pinecone's vector similarity
        service.
    :param top_k: An integer representing the number of top results to consider during
        processing. Default value is 20.
    :return: Returns a pandas DataFrame with an added column `Simplified_Business_Requirements`
        that contains the simplified text corresponding to the original business requirements.
    """
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        simplified_buss_req(original_question=query, business_requirement=row, OPENAI_API_KEY=OPENAI_API_KEY, PINECONE_API_KEY=PINECONE_API_KEY, top_k=top_k)
        for row in df["Business_Requirements"]
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the simplified business requirements back into the DataFrame
    df["Simplified_Business_Requirements"] = results
    return df


# Asynchronous function to generate detailed business requirements for a DataFrame
async def detailed_business_requirement(query: str, df: pd.DataFrame, OPENAI_API_KEY:str, PINECONE_API_KEY:str, top_k:int=20):
    """
    Generates detailed business requirements for each row in the provided DataFrame by
    feeding it through an asynchronous process that utilizes external APIs for processing
    the input query and business requirements.

    The function creates and executes asynchronous tasks for each row in the DataFrame's
    "Business_Requirements" column, generates detailed requirements using external services,
    and updates the DataFrame with the results in a new column titled
    "Detailed_Business_Requirements".

    :param query: Query string used to enhance and refine the business requirements.
    :param df: DataFrame containing the column "Business_Requirements" with original
        business requirement details.
    :param OPENAI_API_KEY: API key for accessing the OpenAI service.
    :param PINECONE_API_KEY: API key needed for Pinecone API integration.
    :param top_k: Maximum number of top results to consider for query enhancement. Defaults to 20.
    :type top_k: int
    :return: Updated DataFrame with an additional column "Detailed_Business_Requirements"
        containing the generated detailed business requirements.
    :rtype: pd.DataFrame
    """
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        generate_detailed_requirements(original_question=query, business_requirement=row, OPENAI_API_KEY=OPENAI_API_KEY, PINECONE_API_KEY=PINECONE_API_KEY, top_k=top_k)
        for row in df["Business_Requirements"]
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the detailed business requirements back into the DataFrame
    df["Detailed_Business_Requirements"] = results
    return df


# Asynchronous function to generate permissions for each business requirement in the DataFrame
async def permissions_business_requirement(query: str, df: pd.DataFrame, OPENAI_API_KEY:str, PINECONE_API_KEY:str, top_k:int=20):
    """
    Executes asynchronous tasks to process business requirements based on input query
    and returns the modified DataFrame containing permissions associated with each
    requirement.

    The function iterates over each row of the input DataFrame and creates async tasks
    to generate permissions by processing detailed business requirements using external
    services. These tasks run concurrently, and the resulting permissions are appended
    to a new column in the original DataFrame.

    :param query: The initial query or prompt as a string used for processing business
        requirements.
    :param df: A pandas DataFrame containing business and detailed business requirements
        in its rows.
    :param OPENAI_API_KEY: The API key to access the OpenAI services for generating
        permissions.
    :param PINECONE_API_KEY: The API key to access Pinecone service for embeddings or
        vector-related tasks.
    :param top_k: The number of top results to use during processing. Defaults to 20.
    :return: The original DataFrame with an additional column:
        "Permission_Business_Requirements", which contains the generated permissions
        for each business requirement.
    """
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        permissions_buss_req(
            original_question=query,
            business_requirement=row["Business_Requirements"],
            detailed_business_requirement=row["Detailed_Business_Requirements"],
            OPENAI_API_KEY=OPENAI_API_KEY,
            PINECONE_API_KEY=PINECONE_API_KEY,
            top_k=top_k,
        )
        for _, row in df.iterrows()
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the generated permissions back into the DataFrame
    df["Permission_Business_Requirements"] = results
    return df


# Asynchronous function to generate prohibitions for each business requirement in the DataFrame
async def prohibitions_business_requirement(query: str, df: pd.DataFrame, OPENAI_API_KEY:str, PINECONE_API_KEY:str, top_k:int=20):
    """
    Executes asynchronous tasks to process and generate prohibitions for business
    requirements provided in the input DataFrame. This function leverages OpenAI
    and Pinecone APIs for analyzing the relationship between the input query and
    business requirements. Results will be appended to the DataFrame as a new column.

    :param query: The input question or query to be analyzed against business
        requirements.
    :type query: str
    :param df: A pandas DataFrame containing columns for business requirements
        and detailed descriptions. Each row represents a separate business
        requirement.
    :type df: pd.DataFrame
    :param OPENAI_API_KEY: The API key for accessing OpenAI services for
        generating responses.
    :type OPENAI_API_KEY: str
    :param PINECONE_API_KEY: The API key for accessing Pinecone services to
        retrieve necessary embeddings or search results.
    :type PINECONE_API_KEY: str
    :param top_k: The number of top search results or embeddings to consider
        when analyzing prohibitions. Defaults to 20.
    :type top_k: int, optional
    :return: The updated DataFrame with a new column "Prohibitions_Business_Requirements"
        containing results of the prohibitions for each business requirement.
    :rtype: pd.DataFrame
    """
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        prohibitions_buss_req(
            original_question=query,
            business_requirement=row["Business_Requirements"],
            detailed_business_requirement=row["Detailed_Business_Requirements"],
            OPENAI_API_KEY=OPENAI_API_KEY,
            PINECONE_API_KEY=PINECONE_API_KEY,
            top_k=top_k,
        )
        for _, row in df.iterrows()
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the generated prohibitions back into the DataFrame
    df["Prohibitions_Business_Requirements"] = results
    return df


# Asynchronous function to generate permissions for each business requirement in the DataFrame
async def requirement_permission_prohibition_markdown(df: pd.DataFrame, OPENAI_API_KEY: str, model_name: str = "gpt-4o-mini"):
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        combined_req_perm_prohibition_markdown(
            detailed_business_requirements=row["Detailed_Business_Requirements"],
            permission_business_requirements=row["Permission_Business_Requirements"],
            prohibitions_business_requirements=row["Prohibitions_Business_Requirements"],
            OPENAI_API_KEY=OPENAI_API_KEY,
            model_name=model_name,
        )
        for _, row in df.iterrows()
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the generated permissions back into the DataFrame
    df["Combined_Requirements_Permissions_Prohibitions"] = results
    return df

# Function to generate a full table of requirements, including simplified, detailed, permissions, and prohibitions
async def create_table(query: str, OPENAI_API_KEY:str, PINECONE_API_KEY:str, output_path:str="Data/Results/Query_Results.xlsx", top_k: int = 25):
    max_retries = 3

    # Step 1: Identify business requirements using the provided query
    for attempt in range(1, max_retries + 1):
        try:
            response = verbatim_business_requirements(query, top_k=top_k, OPENAI_API_KEY=OPENAI_API_KEY, PINECONE_API_KEY=PINECONE_API_KEY)
            print("First Response Created")
            break
        except Exception as error:
            print(f"Attempt {attempt} failed with error: {error}")
            if attempt == max_retries:
                raise ValueError("Function did not succeed after multiple attempts")

    # Step 2: Convert the response into a DataFrame
    for attempt in range(1, max_retries + 1):
        try:
            df = convert_str_to_df(data_str=response, OPENAI_API_KEY=OPENAI_API_KEY)
            df = df.dropna(axis=1, how='all')  # Drop empty columns
            df = clean_dataframe(df, df.columns[0])  # Clean the DataFrame

            # Check if the DataFrame has zero rows and force a retry if so
            if df.shape[0] == 0:
                raise ValueError("DataFrame is empty after cleaning (0 rows).")

            df["Business_Requirements"] = df[df.columns[0]]  # Add business requirements to the DataFrame
            df = df.drop(df.columns[0], axis=1)  # Remove the original column after assigning
            print("First DataFrame Cleaned")
            break
        except Exception as error:
            print(f"Attempt {attempt} failed with error: {error}")
            if attempt == max_retries:
                raise ValueError("DataFrame cleaning function did not succeed after multiple attempts")

    # Step 3: Process the business requirements asynchronously
    df = await simplified_business_requirement(query=query, df=df, OPENAI_API_KEY=OPENAI_API_KEY, PINECONE_API_KEY=PINECONE_API_KEY, top_k=top_k)
    print("Simplified Requirements Created")
    df = await detailed_business_requirement(query=query, df=df, OPENAI_API_KEY=OPENAI_API_KEY, PINECONE_API_KEY=PINECONE_API_KEY, top_k=top_k)
    print("Detailed Requirements Created")
    df = await permissions_business_requirement(query=query, df=df, OPENAI_API_KEY=OPENAI_API_KEY, PINECONE_API_KEY=PINECONE_API_KEY, top_k=top_k)
    print("Permissions Requirements Created")
    df = await prohibitions_business_requirement(query=query, df=df, OPENAI_API_KEY=OPENAI_API_KEY, PINECONE_API_KEY=PINECONE_API_KEY, top_k=top_k)
    print("Prohibitions Requirements Created")
    df = await requirement_permission_prohibition_markdown(df=df, OPENAI_API_KEY=OPENAI_API_KEY, model_name="gpt-4o-mini")
    print("Requirements and Permissions Created")

    # Step 4: Save the final DataFrame as an Excel file
    save = False
    if save:
        df.to_excel(output_path, index=False)

    # Step 5: Convert the DataFrame into Markdown table format
    table_markdown = await table_to_markdown(df=df, OPENAI_API_KEY=OPENAI_API_KEY)
    print("Final Markdown Table Created")

    # Return the results
    return df, table_markdown