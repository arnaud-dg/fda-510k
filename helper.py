import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
from collections import deque

# Snowflake connection
conn = st.connection("snowflake")

# Constants
MODEL_NAME = 'mistral-7b'
NUM_CHUNKS = 3  # Number of chunks provided as context
SLIDE_WINDOW = 7  # Number of last conversations to remember
MAX_CACHE_SIZE = 10  # Maximum number of cached conversations

# Initialize cache
conversation_cache = deque(maxlen=MAX_CACHE_SIZE)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = MODEL_NAME
    if "use_chat_history" not in st.session_state:
        st.session_state.use_chat_history = False
    if "debug" not in st.session_state:
        st.session_state.debug = False

def config_options():
    """Configure sidebar options."""
    st.sidebar.image("https://raw.githubusercontent.com/arnaud-dg/fda-510k/main/assets/510k.png", width=250)
    st.sidebar.write("This web application is a personalized LLM chatbot. Unlike other general-purpose LLMs, it is specifically designed to help you explore FDA submission files for medical devices utilizing artificial intelligence.")
    st.sidebar.write("780 documents serve as the response base for the LLM.") 
    st.sidebar.selectbox('Select your LLM model:', (
        'mistral-7b',
        'llama3-8b',
        'gemma-7b'
    ), key="model_name")
    st.sidebar.checkbox('Do you want that I remember the chat history?', key="use_chat_history", value=False)
    st.sidebar.checkbox('Debug: Click to see summary generated of previous conversation', key="debug", value=False)
    if st.sidebar.button("Start Over"):
        st.session_state.messages = []
    st.sidebar.expander("Session State").write(st.session_state)

def _get_similar_chunks(question):
    """Retrieve similar chunks from the database."""
    cmd = """
        WITH results AS (
            SELECT RELATIVE_PATH,
                   VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec,
                   SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)) as similarity,
                   chunk
            FROM docs_chunks_table
            ORDER BY similarity DESC
            LIMIT ?
        )
        SELECT chunk, relative_path FROM results 
    """
    df_chunks = conn.query(cmd, params=[question, NUM_CHUNKS])
    similar_chunks = " ".join(df_chunks["CHUNK"].replace("'", ""))
    return similar_chunks

def _get_chat_history():
    """Retrieve recent chat history."""
    chat_history = []
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    for i in range(start_index, len(st.session_state.messages) - 1):
        chat_history.append(st.session_state.messages[i])
    return chat_history

def _summarize_question_with_history(chat_history, question):
    """Summarize the question with chat history."""
    prompt = f"""
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natural language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
    """
    cmd = """
        SELECT snowflake.cortex.complete(?, ?) as response
    """
    
    df_response = conn.query(cmd, params=[st.session_state.model_name, prompt])
    summary = df_response['RESPONSE'].iloc[0]

    if st.session_state.debug:
        st.sidebar.text("Summary to be used to find similar chunks in the docs:")
        st.sidebar.caption(summary)

    return summary

def _create_prompt(question):
    """Create a prompt for the LLM."""
    if st.session_state.use_chat_history:
        chat_history = _get_chat_history()
        if chat_history:
            question_summary = _summarize_question_with_history(chat_history, question)
            prompt_context = _get_similar_chunks(question_summary)
        else:
            prompt_context = _get_similar_chunks(question)
    else:
        prompt_context = _get_similar_chunks(question)
        chat_history = ""
  
    prompt = f"""
        You are an expert chat assistant that extracts information from the CONTEXT provided
        between <context> and </context> tags.
        You offer a chat experience considering the information included in the CHAT HISTORY
        provided between <chat_history> and </chat_history> tags.
        When answering the question contained between <question> and </question> tags
        be concise and do not hallucinate. 
        If you don't have the information, just say so.
           
        Do not mention the CONTEXT used in your answer.
        Do not mention the CHAT HISTORY used in your answer.
           
        <chat_history>
        {chat_history}
        </chat_history>
        <context>          
        {prompt_context}
        </context>
        <question>  
        {question}
        </question>
        Answer: 
    """
    return prompt

def complete_query(question):
    """Complete the query using the LLM."""
    st.cache_data.clear()

    prompt = _create_prompt(question)
    cmd = """
        SELECT snowflake.cortex.complete(?, ?) as response
    """
    df_response = conn.query(cmd, params=[st.session_state.model_name, prompt])

    # Add to cache
    conversation_cache.append((question, df_response['RESPONSE'].iloc[0]))

    return df_response

def generate_submission_report(name, applicant_name, description, indication, usage_context, algorithm_type, training_dataset):
    """Generate a submission report based on user inputs."""
    prompt = f"""
    Generate a detailed FDA 510(k) submission report using the following information:

    Product Name: {name}
    Applicant Name: {applicant_name}
    Device Description: {description}
    Proposed Indications for Use: {indication}
    Usage Context: {usage_context}
    Algorithm Type: {algorithm_type}
    Training Dataset: {training_dataset}

    Please structure the report according to the following sections:

    1. Product Name (80 characters max)
    2. Applicant Information (200 characters max)
    3. Consensus Standards (300 characters max)
    4. Device Description (500 characters max)
    5. Proposed Indications for Use (200 characters max)
    6. Classification (500 characters max)
    7. Equivalent Medical Devices (500 characters max)
    8. Description of Verification Tests (200-1000 characters)

    Ensure that each section adheres to the specified character limits and provides relevant, concise information based on the input provided and your knowledge of FDA 510(k) submissions.
    """

    cmd = """
        SELECT snowflake.cortex.complete(?, ?) as response
    """
    df_response = conn.query(cmd, params=[st.session_state.model_name, prompt])
    return df_response['RESPONSE'].iloc[0]