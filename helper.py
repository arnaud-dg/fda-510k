import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
from collections import deque
import json

# Initialize Snowflake connection
conn = st.connection("snowflake")

# Constants
MODEL_NAME = 'mistral-7b'
NUM_CHUNKS = 3
SLIDE_WINDOW = 7
MAX_CACHE_SIZE = 10
DEFAULT_TEMPERATURE = 0.9

# Initialize cache
conversation_cache = deque(maxlen=MAX_CACHE_SIZE)

def get_translation(language):
    """Return translations based on selected language."""
    translations = {
        'ENG': {
            'title': 'FDA 510k Knowledge Base Chatbot',
            'tab_chat': 'Chat',
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
            'use_history': 'Consider the chat history?',
            'description_header': 'Description',
            'clear_cache': 'Clear Cache and Reset',
            'clear_cache_help': 'Clear the conversation history and reset the application.'
        },
        'FR': {
            'title': 'Assistant - Base de Connaissances FDA 510k',
            'tab_chat': 'Discussion',
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
            'use_history': 'Considérer l\'historique des conversations ?',
            'description_header': 'Description',
            'clear_cache': 'Effacer le Cache et Réinitialiser',
            'clear_cache_help': 'Effacer l\'historique des conversations et réinitialiser l\'application.'
        }
    }
    return translations[language]

def clear_session_state():
    """Clear session state and reset the application."""
    # Clear conversation cache
    conversation_cache.clear()
    
    # Reset only the messages and debug state
    st.session_state.messages = []
    st.session_state.debug = False
    
    # Clear suggestions to regenerate them
    if "suggestions" in st.session_state:
        del st.session_state.suggestions

    # Clear Streamlit cache
    st.cache_data.clear()

def generate_dynamic_suggestions(messages, language):
    """Generate new suggestions using the LLM based on conversation context."""
    # If there are no previous messages, return None to use default suggestions
    if not messages:
        return None
        
    # Get the last exchange
    last_exchange = messages[-2:] if len(messages) >= 2 else messages
    last_topic = last_exchange[-1]["content"]
    
    # Define language-specific prompt
    if language == 'ENG':
        prompt = f"""
            Based on this last conversation about: "{last_topic}"
            Generate 3 relevant follow-up questions that a user might want to ask about this topic.
            The questions should be specifically about FDA medical device submissions and regulations.
            Return only the questions, one per line, without any numbering or additional text.
            Keep each question under 75 characters.
            """
    else:
        prompt = f"""
            Basé sur cette dernière conversation à propos de : "{last_topic}"
            Générez 3 questions de suivi pertinentes qu'un utilisateur pourrait vouloir poser sur ce sujet.
            Les questions doivent porter spécifiquement sur les soumissions et réglementations FDA des dispositifs médicaux.
            Retournez uniquement les questions, une par ligne, sans numérotation ni texte supplémentaire.
            Gardez chaque question sous 75 caractères.
            """
    
    # Use Snowflake Cortex to generate suggestions
    cmd = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            ?, 
            ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('role', 'user', 'content', ?)),
            OBJECT_CONSTRUCT('temperature', 0.7, 'max_tokens', 256)
        ) AS response
    """
    
    try:
        # Execute query with parameters
        df_response = conn.query(
            cmd,
            params=[MODEL_NAME, prompt]
        )
        
        # Get response and split into lines
        raw_response = df_response['RESPONSE'].iloc[0]
        response_json = json.loads(raw_response)
        suggestions_text = response_json['choices'][0]['messages'].strip()
        
        # Split into separate questions and take first 3
        suggestions = [q.strip() for q in suggestions_text.split('\n') if q.strip()][:3]
        
        # If we don't get exactly 3 questions, return None to use defaults
        if len(suggestions) != 3:
            return None
            
        return suggestions
        
    except Exception as e:
        st.error(f"Error generating suggestions: {e}")
        return None

def generate_suggestions(messages, language):
    """Generate suggested questions based on chat history and language."""
    # Initial default suggestions for each language
    initial_suggestions = {
        'ENG': [
            "What is a 510(k) submission?",
            "Give me details about medical devices using computer vision",
            "What is a good training dataset?"
        ],
        'FR': [
            "Qu'est-ce qu'une soumission 510(k) ?",
            "Donnez-moi des détails sur les dispositifs médicaux utilisant la vision par ordinateur",
            "Quel est un bon jeu de données d'entraînement ?"
        ]
    }
    
    # If there are no messages, return initial suggestions
    if not messages:
        return initial_suggestions[language]
        
    # Try to generate dynamic suggestions
    dynamic_suggestions = generate_dynamic_suggestions(messages, language)
    
    # If dynamic generation fails or returns None, use initial suggestions
    if not dynamic_suggestions:
        return initial_suggestions[language]
        
    return dynamic_suggestions

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = MODEL_NAME
    if "debug" not in st.session_state:
        st.session_state.debug = False
    if "language" not in st.session_state:
        st.session_state.language = 'ENG'

def config_options():
    """Configure sidebar options."""
    translations = get_translation(st.session_state.language)
    
    # Add image above the expander
    st.sidebar.image("https://raw.githubusercontent.com/arnaud-dg/fda-510k/main/assets/510k.png", width=250)
    st.sidebar.write("")
    
    st.sidebar.markdown(f"__{translations['description_header']}__")
    st.sidebar.write(translations["sidebar_description"])
    st.sidebar.write(translations["sidebar_doc_count"])
    
    st.sidebar.divider()

    # Create expander for options
    with st.sidebar.expander(translations['sidebar_title'], expanded=False):
        
        st.selectbox(
            translations['select_language'],
            options=['ENG', 'FR'],
            key="language",
            index=0
        )
        
        st.selectbox(
            translations['select_model'],
            options=['mistral-7b', 'llama3-8b', 'gemma-7b'],
            key="model_name",
            index=0
        )

        st.slider(
            translations['temperature_label'],
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help=translations['temperature_help'],
            key="temperature"
        )
        
        st.checkbox(
            translations['use_history'],
            key="use_chat_history",
            value=True
        )
        
        st.divider()
        
        # Add clear cache button
        if st.button(
            translations['clear_cache'],
            help=translations['clear_cache_help'],
            type="primary"
        ):
            clear_session_state()
            st.rerun() 

def complete_query(question):
    """Complete the query using the LLM."""
    # Clear data cache
    st.cache_data.clear()

    # Create prompt from question and history
    prompt = _create_prompt(question)

    # Build SQL query for Snowflake Cortex
    cmd = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            ?, 
            ARRAY_CONSTRUCT(OBJECT_CONSTRUCT('role', 'user', 'content', ?)),
            OBJECT_CONSTRUCT('temperature', ?, 'max_tokens', 1024)
        ) AS response
    """

    # Execute query with parameters
    df_response = conn.query(
        cmd,
        params=[
            st.session_state.model_name,
            prompt,
            st.session_state.temperature
        ]
    )
    
    # Get response text and parse JSON
    raw_response = df_response['RESPONSE'].iloc[0]
    
    try:
        # Parse the JSON response
        response_json = json.loads(raw_response)
        # Extract the message text from the choices array
        response_text = response_json['choices'][0]['messages'].strip()
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        # Fallback in case of parsing errors
        st.error(f"Error parsing response: {e}")
        response_text = "Je suis désolé, mais je n'ai pas pu traiter correctement la réponse. Pourriez-vous reformuler votre question?"
    
    # Add to conversation cache
    conversation_cache.append((question, response_text))

    return response_text

# Helper functions for complete_query
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

def _create_prompt(question):
    """Create a prompt for the LLM."""
    # Get chat history and context
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
    
    # Define language instruction
    language_instruction = "Respond in English only." if st.session_state.language == 'ENG' else "Répondez exclusivement en français."
  
    prompt = f"""
        You are an expert chat assistant that extracts information from the CONTEXT provided
        between <context> and </context> tags.
        You offer a chat experience considering the information included in the CHAT HISTORY
        provided between <chat_history> and </chat_history> tags.
        When answering the question contained between <question> and </question> tags
        be concise and do not hallucinate. 
        If you don't have the information, just say so.
        
        IMPORTANT INSTRUCTION: {language_instruction}
           
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

def _summarize_question_with_history(chat_history, question):
    """Summarize the question with chat history."""
    # Define language instruction
    language_instruction = "Respond in English only." if st.session_state.language == 'ENG' else "Répondez exclusivement en français."
    
    prompt = f"""
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natural language. 
        Answer with only the query. Do not add any explanation.
        
        IMPORTANT INSTRUCTION: {language_instruction}
        
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