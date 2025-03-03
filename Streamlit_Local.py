import os
import io
import time

import streamlit as st
st. set_page_config(layout="wide")
from dotenv import load_dotenv
from openai import OpenAI
from main import *
import asyncio
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# -----------------------------------------------------------------------------
# INITIAL SETUP: Initialize the OpenAI client and conversation history in session state
# -----------------------------------------------------------------------------
if 'model' not in st.session_state:
    # Instantiate the OpenAI client with your API key.
    st.session_state['model'] = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if 'conversation' not in st.session_state:
    # This list will hold the conversation messages.
    st.session_state['conversation'] = []

if 'new_conversation_flag' not in st.session_state:
    st.session_state['new_conversation_flag'] = 0

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.title('LRR Chatbot Options')

# Button to start a new conversation
if st.sidebar.button("New Conversation"):
    st.session_state['conversation'] = []  # Clear conversation history
    st.session_state['new_conversation_flag'] = 0  # Reset flag

# -----------------------------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------------------------
st.title("LRR Chatbot")

# Display all previous conversation messages in order (rendered as Markdown)
for msg in st.session_state['conversation']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# -----------------------------------------------------------------------------
# USER INPUT AND RESPONSE HANDLING
# -----------------------------------------------------------------------------
# Get the user's legal query.
user_input = st.chat_input("Hello! How can I assist you today?:")

if user_input:
    # Append the user's message to the conversation history.
    st.session_state['conversation'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**User:** {user_input}")

    # Retrieve and display the legal response.
    with st.chat_message("assistant"):
        # Create an in-memory binary buffer
        buffer = io.BytesIO()

        placeholder = st.empty()
        full_response = ""
        query = st.session_state['conversation'][-1]["content"]
        # Get Regulation Summary
        regulation_summary = asyncio.run(concise_summary(original_question=query,
                                                         top_k=25,
                                                         OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"],
                                                         PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]))
        full_response += regulation_summary
        placeholder.markdown(f"**Assistant:** {full_response}")
        df, table_markdown = asyncio.run(create_table(query=query,
                                                      top_k=25,
                                                      OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"],
                                                      PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]))
        full_response += "\n\n"
        full_response += table_markdown
        full_response = full_response.replace("```markdown\n", "\n")
        full_response = full_response.replace("```", "")
        placeholder.markdown(f"**Assistant:** {full_response}")
        key_term_response, context = asyncio.run(key_terms(original_question=query,
                                                top_k=15,
                                               response=full_response,
                                               OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"],
                                               PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]
                                                ))
        full_response += "\n\n"
        full_response += key_term_response
        full_response = full_response.replace("```markdown\n", "\n")
        full_response = full_response.replace("```", "")
        placeholder.markdown(f"**Assistant:** {full_response}")

        st.dataframe(df[["Business_Requirements", "Simplified_Business_Requirements","Combined_Requirements_Permissions_Prohibitions"]])
        # Append the Assistant's reply to the conversation history
        st.session_state['conversation'].append({"role": "assistant", "content": full_response})

        # Save DataFrame to an Excel file in memory
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()

        # Set the buffer position to the beginning
        buffer.seek(0)

        # Streamlit download button
        st.download_button(
            label="Download Table",
            data=buffer,
            file_name="Output_Table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )