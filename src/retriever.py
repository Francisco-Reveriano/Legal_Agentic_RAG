import time
import asyncio
from io import StringIO
import os
from typing import Any, List, Dict
from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI
import pandas as pd
import warnings
import re
warnings.filterwarnings("ignore")

EMBEDDING_MODEL = "text-embedding-3-large"

def remove_toc_tokens(text: str) -> str:
    """
    Removes all occurrences of table-of-contents tokens of the form {#_Toc<number>}
    from the input text.
    """
    # The regex pattern matches an opening curly brace, then "#_Toc",
    # followed by one or more digits, and a closing curly brace.
    token_pattern = r'\{?#_(?:Toc|Hlk)\d+\}?'
    text = re.sub(token_pattern, '', text)

    # Remove width and height attributes (e.g., width="1.8402777777777777in")
    # This regex matches the attribute name followed by = and a quoted value.
    dimension_pattern = r'\s*(width|height)\s*=\s*"[^"]+"'
    text = re.sub(dimension_pattern, '', text)
    text = text.replace('''{''}.''', "")
    text = text.replace('{}',"")
    return text

def retriever(state:Dict[str, Any], original_context:bool=True) -> Dict[str, Any]:
    pinecone_api_key = state["pinecone_api_key"]
    openai_api_key = state["openai_api_key"]
    top_k = state["top_k"]

    # Load Pinecone Index
    pc = Pinecone(pinecone_api_key)
    index = pc.Index("legal-assistant-rag")
    #index = pc.Index("australia-broadcasting-services-act")
    #index = pc.Index("japan-foreign-exchange-act")

    # Load OpenAI Client
    client = OpenAI(api_key=openai_api_key)

    # Create embedding
    if original_context:
        embedding_input = state["original_question"]
        embedding_result = client.embeddings.create(input=embedding_input, model=EMBEDDING_MODEL)
        xq = embedding_result.data[0].embedding
    else:
        embedding_input = state["updated_question"]
        embedding_result = client.embeddings.create(input=embedding_input, model=EMBEDDING_MODEL)
        xq = embedding_result.data[0].embedding

    # Asynchronously query the Pinecone index
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)

    res.matches = sorted(res.matches, key=lambda chunk: int(chunk['id']))

    # Combine Chunks
    chunk_list = []
    for r in res.matches:
        chunk_list.append(r["metadata"]["content"])
    unique_chunks = list(set(chunk_list))
    question_context = " ".join(unique_chunks)
    question_context = remove_toc_tokens(question_context)

    # Update Dictionary
    if original_context:
        state.update({
            "original_context": question_context,
            "original_chunk_list": unique_chunks,
            "query_results": res.matches
        })
    else:
        state.update({
            "updated_context": question_context,
            "updated_chunk_list": unique_chunks,
            "updated_query_results": res.matches
        })

    return state