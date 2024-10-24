import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
from collections import deque

# Initialize Snowflake connection
conn = st.connection("snowflake")

# Constants
MODEL_NAME = 'mistral-7b'
NUM_CHUNKS = 3
SLIDE_WINDOW = 7
MAX_CACHE_SIZE = 10

# Initialize cache
conversation_cache = deque(maxlen=MAX_CACHE_SIZE)

def get_translation(language):
    """Return translations based on selected language."""
    translations = {
        'ENG': {
            'title': 'FDA 510k form Knowledge Base',
            'tab_chat': 'Chat',
            'tab_report': 'Generate Report',
            'chat_placeholder': 'How can I help you concerning FDA medical devices submissions?',
            'thinking': 'thinking',
            'report_header': 'Generate FDA 510(k) Submission Report',
            'sidebar_title': 'Options:',
            'sidebar_description': 'This web application is a personalized LLM chatbot. Unlike other general-purpose LLMs, it is specifically designed to help you explore FDA submission files for medical devices utilizing artificial intelligence.',
            'sidebar_doc_count': '880 documents serve as the response base for the LLM.',
            'temperature_label': 'Model Temperature:',
            'temperature_help': 'Higher values make the output more creative, lower values make it more focused and deterministic.',
            'select_language': 'Select your language:',
            'select_model': 'Select your LLM model:',
            'use_history': 'Do you want to use the chat history?',
            'description_header': 'Description'
        },
        'FR': {
            'title': 'Base de Connaissances FDA 510k',
            'tab_chat': 'Discussion',
            'tab_report': 'Générer un Rapport',
            'chat_placeholder': 'Comment puis-je vous aider concernant les soumissions FDA pour les dispositifs médicaux ?',
            'thinking': 'réfléchit',
            'report_header': 'Générer un Rapport de Soumission FDA 510(k)',
            'sidebar_title': 'Options :',
            'sidebar_description': "Cette application web est un chatbot LLM personnalisé. Contrairement aux LLM généraux, il est spécifiquement conçu pour vous aider à explorer les dossiers de soumission FDA pour les dispositifs médicaux utilisant l'intelligence artificielle.",
            'sidebar_doc_count': '880 documents servent de base de réponse pour le LLM.',
            'temperature_label': 'Température du modèle :',
            'temperature_help': 'Des valeurs plus élevées rendent la sortie plus créative, des valeurs plus basses la rendent plus concentrée et déterministe.',
            'select_language': 'Sélectionnez votre langue :',
            'select_model': 'Sélectionnez votre modèle LLM :',
            'use_history': 'Voulez-vous utiliser l\'historique des conversations ?',
            'description_header': 'Description'
        }
    }
    return translations[language]

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
    if "language" not in st.session_state:
        st.session_state.language = 'ENG'
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

def config_options():
    """Configure sidebar options."""
    st.sidebar.image("https://raw.githubusercontent.com/arnaud-dg/fda-510k/main/assets/510k.png", width=250)
    st.sidebar.write("")
    st.sidebar.divider()
    
    translations = get_translation(st.session_state.language)
    st.sidebar.markdown(f"__{translations['description_header']}__")
    st.sidebar.write(translations["sidebar_description"])
    st.sidebar.write(translations["sidebar_doc_count"])
    
    st.sidebar.divider()
    st.sidebar.markdown(f"<strong>{translations['sidebar_title']}</strong>", unsafe_allow_html=True)
    
    st.sidebar.selectbox(
        translations['select_language'],
        options=['ENG', 'FR'],
        key="language",
        index=0
    )
    
    st.sidebar.selectbox(
        translations['select_model'],
        options=['mistral-7b', 'llama3-8b', 'gemma-7b'],
        key="model_name",
        index=0
    )
    
    st.sidebar.checkbox(
        translations['use_history'],
        key="use_chat_history",
        value=False
    )
    
    st.sidebar.slider(
        translations['temperature_label'],
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help=translations['temperature_help'],
        key="temperature"
    )

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
    
    # Utiliser des paramètres nommés pour l'appel à la fonction Snowflake
    cmd = """
        SELECT snowflake.cortex.complete(
            model => ?,
            prompt => ?
        ) as response
    """
    
    df_response = conn.query(
        cmd,
        params=[
            st.session_state.model_name,
            prompt,
            st.session_state.temperature
        ]
    )

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

    1. Product Name
    2. Applicant Information
    3. Consensus Standards
    4. Device Description
    5. Proposed Indications for Use 
    6. Classification
    7. Equivalent Medical Devices
    8. Description of Verification Tests

    Ensure that each section adheres to the specified character limits and provides relevant, concise information based on the input provided and your knowledge of FDA 510(k) submissions.
    """

    cmd = """
        SELECT snowflake.cortex.complete(?, ?) as response
    """
    df_response = conn.query(cmd, params=[st.session_state.model_name, prompt])
    return df_response['RESPONSE'].iloc[0]