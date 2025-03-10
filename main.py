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
from src.identify_next_steps import *
from src.combine_requirements_permissions_prohibitions import *
from typing import Any, List, Dict

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

# Asynchronous function to generate simplified business requirements for a DataFrame of business requirements
async def simplified_business_requirement(state:Dict[str, Any], df:pd.DataFrame) -> pd.DataFrame:
    # Inputs
    tasks = [
        simplified_buss_req(business_requirement=row, state=state, model_name="gpt-4o")
        for row in df["Business_Requirements"]
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the simplified business requirements back into the DataFrame
    df["Simplified_Business_Requirements"] = results
    return df


# Asynchronous function to generate detailed business requirements for a DataFrame
async def detailed_business_requirement(state:Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        generate_detailed_requirements(business_requirement=row, state=state, model_name="gpt-4o")
        for row in df["Business_Requirements"]
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the detailed business requirements back into the DataFrame
    df["Detailed_Business_Requirements"] = results
    return df


# Asynchronous function to generate permissions for each business requirement in the DataFrame
async def permissions_business_requirement(state:Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        permissions_buss_req(
            state=state,
            business_requirement=row["Business_Requirements"],
            detailed_business_requirement=row["Detailed_Business_Requirements"],
            model_name="gpt-4o",
        )
        for _, row in df.iterrows()
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the generated permissions back into the DataFrame
    df["Permission_Business_Requirements"] = results
    return df


# Asynchronous function to generate prohibitions for each business requirement in the DataFrame
async def prohibitions_business_requirement(state:Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:

    # Create asynchronous tasks for each row's business requirement
    tasks = [
        prohibitions_buss_req(
            state=state,
            business_requirement=row["Business_Requirements"],
            detailed_business_requirement=row["Detailed_Business_Requirements"],
            model_name="gpt-4o",
        )
        for _, row in df.iterrows()
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the generated prohibitions back into the DataFrame
    df["Prohibitions_Business_Requirements"] = results
    return df


# Asynchronous function to generate permissions for each business requirement in the DataFrame
async def requirement_permission_prohibition_markdown(df: pd.DataFrame, state:Dict[str, Any], model_name: str = "gpt-4o") -> pd.DataFrame:
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        combined_req_perm_prohibition_markdown(
            detailed_business_requirements=row["Detailed_Business_Requirements"],
            permission_business_requirements=row["Permission_Business_Requirements"],
            prohibitions_business_requirements=row["Prohibitions_Business_Requirements"],
            state=state,
            model_name=model_name,
        )
        for _, row in df.iterrows()
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the generated permissions back into the DataFrame
    df["Combined_Requirements_Permissions_Prohibitions"] = results
    return df


# Asynchronous function to generate permissions for each business requirement in the DataFrame
async def identify_next_steps_markdown(df: pd.DataFrame, state:Dict[str, Any], model_name: str = "gpt-4o") -> pd.DataFrame:
    # Create asynchronous tasks for each row's business requirement
    tasks = [
        identify_detailed_actions(
            detailed_business_requirements=row["Detailed_Business_Requirements"],
            detailed_business_permissions=row["Permission_Business_Requirements"],
            detailed_business_prohibitions=row["Prohibitions_Business_Requirements"],
            state=state,
        )
        for _, row in df.iterrows()
    ]

    # Execute all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)

    # Store the generated permissions back into the DataFrame
    df["Next_Steps"] = results
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