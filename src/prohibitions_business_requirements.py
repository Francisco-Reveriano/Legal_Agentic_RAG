import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.business_requirements import retriever

async def prohibitions_buss_req(original_question: str, business_requirement: str,
                               detailed_business_requirement: str, OPENAI_API_KEY:str, PINECONE_API_KEY:str, model_name:str="gpt-4o", top_k:int=20) -> str:
    """
    Generates a Markdown-formatted list of prohibitions based on provided regulatory and business content.
    The function uses provided inputs to outline prohibited actions, ensuring clarity and legal precision
    in a succinct format. Integrates a language model to process and refine the content.

    :param original_question: The original question posed for clarification or guidance on the regulatory
        or business matter.
    :type original_question: str
    :param business_requirement: Business-related requirement provided verbatim for the extraction of
        prohibitions as per the regulatory context.
    :type business_requirement: str
    :param detailed_business_requirement: A detailed description or rewrite of the business requirement
        to add context or clarification.
    :type detailed_business_requirement: str
    :param OPENAI_API_KEY: API key required for authenticating and accessing the OpenAI language model.
    :type OPENAI_API_KEY: str
    :param PINECONE_API_KEY: API key required for accessing the Pinecone vector similarity service.
    :type PINECONE_API_KEY: str
    :param model_name: Name of the specific language model to use for generating content, defaulting to "gpt-4o".
    :type model_name: str, optional
    :param top_k: The maximum number of relevant documents to retrieve for creating the output.
    :type top_k: int, optional
    :return: A Markdown-formatted string containing a bullet-point list of prohibitions derived from the
        provided inputs.
    :rtype: str
    """
    prohibitions_requirement_detailed_prompt = '''
    #  **Detailed Prompt for Prohibitions*

    ## **Your Task**
    - Create a list of bullet points directly outlining what is prohibited based on the provided regulatory text **Context**, **Verbatim Business Requirement**, **Original Question**, and **Detailed Business Requirements* while ensuring clarity, accuracy, and legal precision
    - Provide output in correctly formatted Markdown
    - Keep response succinct and brief
    - Double-check that output is correct and in Markdown format

    ## **Key Characteristics of Prohibitions**
    - In a regulatory context, "prohibitions" refer to the specific actions, activities, or operations that a company is legally barred from performing under applicable laws, regulations, and industry standards.
    - Applicable low should be based on the provided **Context**, **Verbatim Business Requirement**, and **Original Question**.
    
    ## **Input Sections**
    ### **Context**
    {context}

    ### **Verbatim Business Requirements**
    {business_requirement}

    ### **Original Question**
    {original_question}

    ### **Detailed Business Requirements**
    {detailed_business_requirement}

    '''

    # Create a chat prompt template using the detailed prompt.
    prompt = ChatPromptTemplate([
        ("system", prohibitions_requirement_detailed_prompt),
    ])

    # Initialize the ChatOpenAI language model with a specific model name and temperature.
    llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=OPENAI_API_KEY)

    # Combine the prompt, the language model, and the output parser into a processing chain.
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve documents relevant to the query.
    # If `retriever` is a blocking function, run it in a separate thread.
    docs = await asyncio.to_thread(
        retriever,
        query=" ".join([original_question, business_requirement, detailed_business_requirement]),
        top_k=top_k,
        OPENAI_API_KEY=OPENAI_API_KEY,
        PINECONE_API_KEY=PINECONE_API_KEY,
    )

    # Asynchronously invoke the chain with the provided inputs.
    generation = await rag_chain.ainvoke({
        "context": docs,
        "original_question": original_question,
        "business_requirement": business_requirement,
        "detailed_business_requirement": detailed_business_requirement,
    })

    return generation
