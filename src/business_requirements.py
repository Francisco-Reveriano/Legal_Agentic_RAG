import time

from IPython.display import Markdown, display  # Import
from openai import OpenAI
import warnings
import tiktoken
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
from src.embeddings import *
from src.data_processing import *
import pandas as pd
from io import StringIO
from src.prompts import *
import asyncio
warnings.filterwarnings("ignore")
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


async def retriever(query: str, top_k: int = 10):
    """
    Asynchronously retrieves context for a given query by performing a vector similarity search
    on a Pinecone index using embeddings from OpenAI.

    This function performs the following steps asynchronously:

    1. **Initialize Pinecone Index:**
       - Creates a Pinecone client using the global `pinecone_api_key`.
       - Loads the Pinecone index named "legal-assistant-rag".

    2. **Initialize OpenAI Client:**
       - Retrieves the OpenAI API key from the environment variable `OPENAI_API_KEY`.
       - Creates an OpenAI client for embedding generation.

    3. **Generate Query Embedding Asynchronously:**
       - Offloads the synchronous call to create the embedding using OpenAI's
         "text-embedding-3-large" model to a separate thread.
       - Extracts the embedding vector from the result.

    4. **Query the Pinecone Index Asynchronously:**
       - Offloads the blocking query call to the executor.
       - Retrieves the top `top_k` matches along with their metadata.

    5. **Process the Results:**
       - Sorts the matches based on the integer value of each match's 'id' field.
       - Iterates over each match to extract the `prechunk`, `content`, and `postchunk`
         from the metadata.
       - Removes duplicates and concatenates the unique chunks into a single string.

    Parameters:
        query (str): The input query for which context is to be retrieved.
        top_k (int, optional): The number of top matching results to retrieve from the index.
                               Defaults to 10.

    Returns:
        str: A string containing the combined, unique text chunks that serve as the context for the query.
    """

    # Load Pinecone Index
    pc = Pinecone(pinecone_api_key)
    index = pc.Index("legal-assistant-rag")

    # Load OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    loop = asyncio.get_event_loop()

    # Asynchronously create embedding
    embedding_result = await loop.run_in_executor(
        None,
        lambda: client.embeddings.create(input=query, model="text-embedding-3-large")
    )
    xq = embedding_result.data[0].embedding

    # Asynchronously query the Pinecone index
    res = await loop.run_in_executor(
        None,
        lambda: index.query(vector=xq, top_k=top_k, include_metadata=True)
    )
    res.matches = sorted(res.matches, key=lambda chunk: int(chunk['id']))

    # Combine Chunks
    chunk_list = []
    for r in res.matches:
        chunk_list.append(r["metadata"]["prechunk"])
        chunk_list.append(r["metadata"]["content"])
        chunk_list.append(r["metadata"]["postchunk"])
    unique_chunks = list(set(chunk_list))
    question_context = " ".join(unique_chunks)
    return question_context


def verbatim_business_requirements(query:str, top_k:int=40):
    """
       Extracts explicit business requirements from regulatory text by querying a vector store
       and prompting an LLM for a CSV-formatted output.

       Steps:
         1. Loads the Pinecone index using the provided API key.
         2. Generates an embedding for the input query using OpenAI's embeddings.
         3. Queries the Pinecone index to retrieve top-k matching regulatory text chunks.
         4. Combines unique text chunks into a single context string.
         5. Checks if the combined context exceeds 100k tokens and raises an error if so.
         6. Constructs a detailed developer prompt instructing the LLM to extract business requirements.
         7. Sends the prompt along with the original query to the chat completion endpoint.
         8. Returns the formatted output containing business requirements in CSV format.

       Args:
           query (str): The user query or regulatory text to process.
           top_k (int, optional): The number of top matches to retrieve from Pinecone. Defaults to 40.

       Returns:
           str: The LLM-generated response containing a CSV-parsable table of business requirements.

       Raises:
           ValueError: If the combined context token count exceeds 100,000 tokens.
       """

    # Load Pinecone Index
    pc = Pinecone(pinecone_api_key)
    index = pc.Index("legal-assistant-rag")

    # Load OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    xq = client.embeddings.create(input=query, model="text-embedding-3-large").data[0].embedding
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    res.matches = sorted(res.matches, key=lambda chunk: int(chunk['id']))
    # Combine Chunks
    chunk_list = []
    for r in res.matches:
        chunk_list.append(r["metadata"]["prechunk"])
        chunk_list.append(r["metadata"]["content"])
        chunk_list.append(r["metadata"]["postchunk"])
    unique_chunks = list(set(chunk_list))
    question_context = " ".join(unique_chunks)
    chunk_token_count = count_tokens_gpt4(question_context)

    # Error handling: raise an error if the token count exceeds 100k tokens
    if chunk_token_count > 100000:
        raise ValueError(
            f"Combined context token count ({chunk_token_count}) exceeds the 100k limit. "
            "Please provide a shorter input or adjust the context."
        )


    developer_prompt = '''
    # **Regulatory Requirement Extraction & Formatting**  

    ## **Instructions**  
    Your task is to extract and format **clear, explicit legal requirements** from the given regulatory text. These requirements must be presented in a structured format for easy reference and compliance tracking.  
    
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
    2. **No Contextual or Advisory Text**: Exclude general guidance, definitions, explanations, or recommendations.  
    3. **Verbatim Extraction**: Use the **exact wording** from the regulatory text. **Do not alter phrasing**.  
    4. **Structural Integrity**: Ensure the extracted requirements are **properly formatted** and **CSV-compatible**.  
    5. **Complete Coverage**: Review the entire regulatory text to extract **all** relevant requirements.  
    6. **Delimiter & Formatting**:  
       - Use `"^^"` as the delimiter for the CSV output.  
       - Ensure **each row is properly formatted** and compatible with CSV parsing.  
    7. **Final Validation**: Before outputting, **verify** that:  
       - The delimiter `"^^"` is consistently applied.  
       - Each requirement is properly structured in its own row.  
    
    ---  
    
    ## **Output Format Specification**  
    - Output must be a **CSV string** formatted with `"^^"` as the delimiter.  
    - Ensure **no additional text** is included outside of the CSV table.  
    - Verify **correct formatting** before returning the output.  
    
    ---  
    '''.format(question_context)

    completion = client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="medium",
        messages=[
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": query}
        ]
    )

    return completion.choices[0].message.content

def convert_str_to_df(data_str, retries=5):
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            # Use the python engine for better error tolerance
            df = pd.read_csv(StringIO(data_str), sep="^^", engine='python')
            return df
        except pd.errors.ParserError as pe:
            print(f"Attempt {attempt}: ParserError encountered: {pe}")
            # On the final attempt, try GPT conversion
            if attempt == retries:
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "developer", "content": markdown_to_csv_prompt},
                            {"role": "user", "content": data_str}
                        ]
                    )
                    new_csv = completion.choices[0].message.content
                    df = pd.read_csv(StringIO(new_csv), sep="^^", engine='python', on_bad_lines='skip')
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
