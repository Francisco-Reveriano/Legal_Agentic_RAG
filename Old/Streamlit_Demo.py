import io
import json
import warnings
import asyncio

import streamlit as st
import pandas as pd
from openai import OpenAI

from src.retriever import *
from src.concise_summary import *
from src.self_reflective_functions import *
from src.query_rewriter import *
from src.key_terms import *
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
for key, default in {
    'conversation': [],
    'new_conversation_flag': 0,
    'download_buffer': None,
    'download_available': False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar controls
st.sidebar.title('LRR Chatbot Options')
if st.sidebar.button("New Conversation"):
    st.session_state.conversation = []
    st.session_state.download_buffer = None
    st.session_state.download_available = False
    st.session_state.new_conversation_flag = 0

# Main interface
st.title("LRR Chatbot")
st.write("This is a Generative AI tool to assist in Control Assessment.")

# Display conversation history
for msg in st.session_state.conversation:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# User input
user_input = st.chat_input("Hello! How can I assist you today?")
if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        # Initialize state
        state = {
            "openai_api_key": st.secrets["OPENAI_API_KEY"],
            "pinecone_api_key": st.secrets["PINECONE_API_KEY"],
            "top_k": 20,
            "original_question": user_input,
        }
        placeholder = st.empty()
        # Retrieval and summary
        state = retriever(state, original_context=True)
        summary = asyncio.run(a_concise_summary(state))
        placeholder.markdown(f"**Assistant:**\n{state['concise_summary']}", unsafe_allow_html=True)
        with st.status("Running analysis..."):
            # Document grading and query rewriting
            result = grade_full_context(state)
            if 'yes' in str(result).lower():
                state = a_query_rewriter(state)
            else:
                st.error("No relevant documents found.")
            state = retriever(state, original_context=False)
            state = grade_documents(state)
            # Business requirements extraction
            state = verbatim_business_requirements(state)
            # DataFrame creation and cleaning
            df = asyncio.run(a_convert_str_to_df(state)).dropna(axis=1, how='all')
            df['Business_Requirements'] = df.iloc[:, 0]
            df = df.iloc[:, 1:]
            # Further processing
            for func in [
                simplified_business_requirement,
                detailed_business_requirement,
                permissions_business_requirement,
                prohibitions_business_requirement,
                requirement_permission_prohibition_markdown,
                identify_next_steps_markdown,
            ]:
                df = asyncio.run(func(state, df))
            state['final_response'] = df.to_json()
            # Key terms
            state = asyncio.run(key_terms(state, "gpt-4o"))

        st.dataframe(df)
        with st.expander("See key terms:"):
            st.write(state['key_terms'])

        # Prepare download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        buffer.seek(0)
        st.session_state.download_buffer = buffer
        st.session_state.download_available = True

# Download button
if st.session_state.download_available:
    st.download_button(
        label="Download Table",
        data=st.session_state.download_buffer,
        file_name="Output_Table.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
