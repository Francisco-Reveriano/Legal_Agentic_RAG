import asyncio
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Dict
# Suppress any warnings for cleaner output
warnings.filterwarnings("ignore")



async def key_terms(state: Dict[str, Any], model_name:str="gpt-4o-mini") -> Dict[str, Any]:

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

    # Inputs
    updated_question: str = state["updated_question"]
    filtered_context: str = state["filtered_context"]
    final_response: str = state["final_response"]

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", key_terms_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model=model_name, temperature=0.1, api_key=state["openai_api_key"])

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "response": final_response,
        "context": filtered_context,
        "original_question": updated_question,
    })

    # Save output
    state["key_terms"] = generation

    return state
