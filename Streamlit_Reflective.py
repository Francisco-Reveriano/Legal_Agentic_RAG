import os
import io
import time

import streamlit as st
st. set_page_config(layout="wide")
from openai import OpenAI
from main import *
from src.retriever import *
from src.concise_summary import *
from src.self_reflective_functions import *
from src.query_rewriter import *
from src.key_terms import *
import asyncio
import warnings
import pickle
import json
warnings.filterwarnings("ignore")

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

# Initialize download file storage in session state
if 'download_buffer' not in st.session_state:
    st.session_state['download_buffer'] = None
if 'download_available' not in st.session_state:
    st.session_state['download_available'] = False

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
st.sidebar.title('LRR Chatbot Options')

# Button to start a new conversation
if st.sidebar.button("New Conversation"):
    st.session_state['conversation'] = []  # Clear conversation history
    st.session_state['new_conversation_flag'] = 0  # Reset flag
    # Also clear stored download data
    st.session_state['download_buffer'] = None
    st.session_state['download_available'] = False

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
        # Create an in-memory binary buffer for Excel File
        buffer = io.BytesIO()
        # Create State Dictionary
        state = {"openai_api_key": st.secrets["OPENAI_API_KEY"],
                 "pinecone_api_key": st.secrets["PINECONE_API_KEY"],
                 "top_k": 20,
        }

        placeholder = st.empty()
        full_response = ""
        state["original_question"] = user_input # Use the latest query
        # Begin Retrieval Process
        state = retriever(state, original_context=True)
        # Get Regulation Summary
        regulation_summary = asyncio.run(a_concise_summary(state))
        ## Update Visualization
        full_response += state["concise_summary"]
        placeholder.markdown(f"**Assistant:** \n {full_response}", unsafe_allow_html=True)
        with st.status("Running analysis..."):
            ### Grade Full Context Documents
            st.write("*Grading retrieved data...*")
            response = grade_full_context(state)
            if 'yes' in str(response).lower():
                st.write("*Retrieved relevant documents*")
            else:
                st.write("*Retrieved documents not relevant*")
                raise Exception("Retrieved documents not relevant")

            ### Adjust Query (Question + Document)
            st.write("*Adjusting query...*")
            state = asyncio.run(a_query_rewriter(state))
            st.markdown(f"**Assistant: **  {state['updated_question']}", unsafe_allow_html=True)
            print("Updated Query: ", state["updated_question"])

            ### Get Updated Context
            st.write("*Getting updated context...*")
            state = retriever(state, original_context=False)

            ### Filter Context
            st.write("*Filtering chunks...*")
            state = grade_documents(state)

            ### Obtain Verbatim Business Requirements
            st.write("*Obtaining verbatim business requirements...*")
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    state = verbatim_business_requirements(state)
                    break
                except Exception as error:
                    print(f"Attempt {attempt} failed with error: {error}")
                    if attempt == max_retries:
                        raise ValueError("Function did not succeed after multiple attempts")

            ### Convert Response into a Dataframe
            st.write("*Converting response into a dataframe...*")
            for attempt in range(1, max_retries + 1):
                try:
                    df = asyncio.run(a_convert_str_to_df(state))
                    df = df.dropna(axis=1, how='all')  # Drop empty columns
                    #df = clean_dataframe(df, df.columns[0])
                    # Check if the DataFrame has zero rows and force a retry if so
                    if df.shape[0] == 0:
                        raise ValueError("DataFrame is empty after cleaning (0 rows).")
                    df["Business_Requirements"] = df[df.columns[0]]  # Add business requirements to the DataFrame
                    df = df.drop(df.columns[0], axis=1)  # Remove the original column after assigning
                    break
                except Exception as error:
                    print(f"Attempt {attempt} failed with error: {error}")
                    if attempt == max_retries:
                        raise ValueError("DataFrame cleaning function did not succeed after multiple attempts")

            ### Create Simplified Business Requirements
            st.write("*Creating simplified business requirements...*")
            df = asyncio.run(simplified_business_requirement(state, df))

            ## Create Detailed Business Requirements
            st.write("*Creating detailed business requirements...*")
            df = asyncio.run(detailed_business_requirement(state, df))

            ## Create Detailed Business Permissions
            st.write("*Creating detailed business permissions...*")
            df = asyncio.run(permissions_business_requirement(state, df))

            ## Created Detailed Business Prohibitions
            st.write("*Creating detailed business prohibitions...*")
            df = asyncio.run(prohibitions_business_requirement(state, df))

            ## Combine Business Requirements, Permissions, and Prohibitions
            st.write("*Combining business requirements, permissions, and prohibitions...*")
            df = asyncio.run(requirement_permission_prohibition_markdown(df, state))

            ## Identify Key Next Steps
            st.write("*Identifying key next steps...*")
            df = asyncio.run(identify_next_steps_markdown(df, state))

            ## Determine Definitions
            st.write("*Determining definitions...*")
            state["final_response"] = df.to_json()
            state = asyncio.run(key_terms(state, "gpt-4o"))


        # Showcase dataframe
        st.dataframe(df)

        # Showcase Key Terms
        expander = st.expander("See key terms:")
        expander.write(state["key_terms"])

        save = True
        if save:
            df.to_excel("Data/Results/Results_Dataframe.xlsx", index=False, sheet_name='Sheet1')
            with open('Data/Results/State.json', 'w') as f:
                json.dump({"filtered_contest": state["filtered_context"]}, f)

        # Download button
        ## Create an in-memory binary buffer for Excel File
        buffer = io.BytesIO()
        # Save DataFrame to an Excel file in memory
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
            # Set the buffer position to the beginning
        buffer.seek(0)
        # Store the buffer in session state so the download remains available.
        st.session_state['download_buffer'] = buffer
        st.session_state['download_available'] = True
    # -----------------------------------------------------------------------------
    # PERSISTING THE DOWNLOAD BUTTON
    # -----------------------------------------------------------------------------
    # If a download file exists, display the download button.
if st.session_state.get('download_available', False):
    st.download_button(
        label="Download Table",
        data=st.session_state['download_buffer'],
        file_name="Output_Table.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
