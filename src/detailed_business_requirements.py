import asyncio
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.business_requirements import retriever  # Ensure retriever is imported once

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

# Extracted constant for detailed prompt template
BUSINESS_REQUIREMENT_DETAILED_PROMPT = '''
    #  **Detailed Prompt for Requirements**
    
    ## **Your Task**
        - Provide list of detailed requirements and actions necessary to meet the **Verbatim Business Requirements*** based on the provided **Context** and **Verbatim Business Requirement**
        - Provide a detailed description of each requirement and action in bullet points
    
    ## **Instructions & Key Characteristics**
        1. Specific – Clearly define what needs to be done.
        2. Actionable – Outline actions step-by-step.
        3. Measurable – Ensure progress can be tracked.
        4. Ordered – Steps must follow a logical sequence.
        5. Achievable – Feasible within given constraints.
        6. Time-bound – Provide deadlines where applicable.
    ## **Input Sections**
    
    ### **Context**
    {context}
    
    ### **Verbatim Business Requirements**
    {business_requirement}
    
    ### **Original Question**
    {original_question}
'''


def create_chat_prompt_template() -> ChatPromptTemplate:
    """Helper function to create and return a ChatPromptTemplate instance."""
    return ChatPromptTemplate([("system", BUSINESS_REQUIREMENT_DETAILED_PROMPT)])


async def generate_detailed_requirements(
        original_question: str,
        business_requirement: str,
        OPENAI_API_KEY: str,
        PINECONE_API_KEY: str,
        model_name: str = "gpt-4o-mini",
        top_k: int = 25
) -> str:
    """
    Asynchronously generates a detailed list of business requirements using a prompt,
    a language model, and context document retrieval from a vector database.

    :param original_question: User's question or query for generating detailed requirements.
    :param business_requirement: A high-level business need to convert into actionable steps.
    :param openai_api_key: OpenAI API key for the language model.
    :param pinecone_api_key: Pinecone API key for document retrieval.
    :param model_name: OpenAI model name (default: 'gpt-4o-mini').
    :param top_k: Number of context documents to retrieve (default: 25).
    :return: A Markdown-formatted detailed list of requirements as a string.
    """

    # Create a chat prompt template
    prompt = create_chat_prompt_template()

    # Initialize the ChatOpenAI instance
    llm = ChatOpenAI(model_name=model_name, temperature=0.3, api_key=OPENAI_API_KEY)

    # Combine prompt, language model, and parser into a chain
    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve relevant context documents asynchronously
    context_docs = await asyncio.to_thread(
        retriever,
        query=original_question,
        top_k=top_k,
        OPENAI_API_KEY=OPENAI_API_KEY,
        PINECONE_API_KEY=PINECONE_API_KEY,
    )

    # Invoke the processing chain with required inputs
    output = await rag_chain.ainvoke({
        "context": context_docs,
        "business_requirement": business_requirement,
        "original_question": original_question,
    })
    return output
