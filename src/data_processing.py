import os
from tqdm import tqdm
from typing import List, Dict, Any
import tiktoken

# Import the text splitter from its library.
# Adjust the import as needed; for example, if using LangChain:
from langchain.text_splitter import RecursiveCharacterTextSplitter


def count_tokens_gpt4(text: str) -> int:
    """
    Count the number of tokens in the given text using GPT-4's encoding.

    Args:
        text (str): The text to tokenize.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Get the encoding for the GPT-4 model.
        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception as e:
        # If GPT-4 encoding is unavailable, fallback to a default encoding.
        print(f"Error fetching GPT-4 encoding: {e}. Falling back to default encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the text into tokens.
    tokens = encoding.encode(text)
    return len(tokens)

def load_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load and process text files from a specified folder.

    This function iterates over all files in the provided folder and processes
    only those with a ".txt" extension. For each text file, it reads the content,
    splits it into chunks using a recursive character-based splitter, and constructs
    metadata for each chunk. The metadata includes:
      - id: A unique identifier for the chunk.
      - title: The base filename.
      - content: The text content of the chunk.
      - prechunk: Content of the previous chunk (empty string for the first chunk).
      - postchunk: Content of the next chunk (empty string for the last chunk).

    Args:
        folder_path (str): The path to the folder containing text files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing metadata for a text chunk.
    """
    metadata: List[Dict[str, Any]] = []
    chunk_id = 0

    # Iterate over each file in the folder with a progress bar.
    for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
        # Process only files that end with '.txt'
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Open and read the file content.
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            # Initialize the text splitter with desired parameters.
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=1200,
                chunk_overlap=400,
            )

            # Create document chunks from the file's text.
            document_chunks = text_splitter.create_documents([text])
            file_title = os.path.basename(filename)

            # Build metadata for each chunk including context from adjacent chunks.
            for i, chunk in enumerate(document_chunks):
                prechunk = document_chunks[i - 1].page_content if i > 0 else ""
                postchunk = (
                    document_chunks[i + 1].page_content
                    if i < len(document_chunks) - 1
                    else ""
                )
                metadata.append({
                    "id": chunk_id,
                    "title": file_title,
                    "content": document_chunks[i].page_content,
                    "prechunk": prechunk,
                    "postchunk": postchunk,
                    "token_count": count_tokens_gpt4(document_chunks[i].page_content)
                })
                chunk_id += 1
        else:
            print(f"File not processed (unsupported format): {filename}")

    return metadata


if __name__ == "__main__":
    data = load_files("../Data/Raw")
    print(f"Total chunks processed: {len(data)}")
