import asyncio
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.business_requirements import retriever  # Ensure retriever is imported once
from langchain_core.prompts import ChatPromptTemplate

# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")



async def key_terms(original_question: str, response:str, OPENAI_API_KEY:str, PINECONE_API_KEY:str, model_name:str="gpt-4o", top_k: int = 25):
    """
    Asynchronously extracts key terms from a given response and generates a markdown
    formatted output with concise definitions for each term, based on the provided context.

    This function combines a detailed prompt, a language model, and a chain to process
    data and retrieve key term definitions with simple explanations. It uses inputs such as
    the original question, response, context, and an external retriever for relevant documents.

    :param original_question: The question asked by the user that the response addresses.
    :type original_question: str
    :param response: The generated response from which key terms need to be extracted.
    :type response: str
    :param OPENAI_API_KEY: API key for authenticating the OpenAI service.
    :type OPENAI_API_KEY: str
    :param PINECONE_API_KEY: API key for authenticating the Pinecone service.
    :type PINECONE_API_KEY: str
    :param model_name: Optional; Name of the language model to use. Defaults to "gpt-4o".
    :type model_name: str, optional
    :param top_k: Optional; Number of top relevant documents to retrieve for context
        enrichment. Defaults to 25.
    :type top_k: int, optional
    :return: A tuple containing the processed markdown output with key term definitions
        and the list of retrieved documents.
    :rtype: tuple[str, list]
    """
    key_terms_prompt = '''
    # **Instructions**
    - Extract and identify the most important **key terms** from the provided **Response**.
    - Use the **Context** to define each **key term* clearly and accurately.
    - Present definitions in a **bullet-point list**, using **concise, one-sentence explanations** in **simple English**.
    - **Do not define** the primary regulation mentioned in the **Original Question**.
    - Format the output in **correct Markdown** (ensure proper spacing, bolding, and bullet points).  
    
    ## **Output Format Example**
    ```markdown
    - **Key Term 1**: Definition based on the Context.
    - **Key Term 2**: Definition based on the Context.

    
    # **User Inputs**
    ## ** Response**
    {response}
    
    ## **Context**
    {context}
    
    ## **Original Question**
    {original_question}

    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", key_terms_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name=model_name, temperature=0.1, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve documents relevant to the query.
    # If `retriever` is a blocking function, run it in a separate thread.
    docs = await asyncio.to_thread(
        retriever,
        query=original_question,
        top_k=top_k,
        OPENAI_API_KEY=OPENAI_API_KEY,
        PINECONE_API_KEY=PINECONE_API_KEY,
    )

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "response": response,
        "context": docs,
        "original_question": original_question,
    })

    return generation, docs
