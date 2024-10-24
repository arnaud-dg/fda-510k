import streamlit as st

st.set_page_config(layout="wide")

import pandas as pd
from helper import initialize_session_state, config_options, complete_query, get_translation, generate_suggestions

# Set options for pandas to display full text in columns
pd.set_option("max_colwidth", None)


def main():
    initialize_session_state()
    config_options()
    
    # Get translations based on selected language
    translations = get_translation(st.session_state.language)
    
    st.title(f":brain: {translations['title']} :robot_face:")
    
    # Create tabs with translated names
    tab1 = st.tabs([translations["tab_chat"]])[0]
    
    with tab1:
        display_chat_interface(translations)

def display_chat_interface(translations):
    # Display suggested questions
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = generate_suggestions(st.session_state.messages, st.session_state.language)
    
    # Create columns for suggestion buttons
    col1, col2, col3 = st.columns(3)
    
    # Display suggestion buttons
    with col1:
        if st.button(st.session_state.suggestions[0], key="sug1"):
            question = st.session_state.suggestions[0]
            process_question(question, translations)
    
    with col2:
        if st.button(st.session_state.suggestions[1], key="sug2"):
            question = st.session_state.suggestions[1]
            process_question(question, translations)
    
    with col3:
        if st.button(st.session_state.suggestions[2], key="sug3"):
            question = st.session_state.suggestions[2]
            process_question(question, translations)
    
    st.divider()
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input(translations["chat_placeholder"]):
        process_question(question, translations)

def process_question(question, translations):
    """Process a question from either chat input or suggestion buttons."""
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        question = question.replace("'", "")

        with st.spinner(f"{st.session_state.model_name} {translations['thinking']}..."):
            response = complete_query(question)
            message_placeholder.markdown(response)  # response is now a string, not a DataFrame
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Generate new suggestions after each interaction
    st.session_state.suggestions = generate_suggestions(st.session_state.messages, st.session_state.language)
    st.experimental_rerun()

if __name__ == "__main__":
    main()