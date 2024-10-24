import streamlit as st
import pandas as pd
from helper import initialize_session_state, config_options, complete_query, get_translation

# Set options for pandas to display full text in columns
pd.set_option("max_colwidth", None)

def main():
    initialize_session_state()
    config_options()
    
    # Get translations based on selected language
    translations = get_translation(st.session_state.language)
    
    st.title(f":brain: {translations['title']}")
    
    # Create tabs with translated names
    tab1, tab2 = st.tabs([translations["tab_chat"], translations["tab_report"]])
    
    with tab1:
        display_chat_interface(translations)
    
    with tab2:
        display_report_interface(translations)

def display_chat_interface(translations):
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input(translations["chat_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            question = question.replace("'", "")
    
            with st.spinner(f"{st.session_state.model_name} {translations['thinking']}..."):
                response = complete_query(question)
                res_text = response['RESPONSE'].iloc[0]
                message_placeholder.markdown(res_text)
        
        st.session_state.messages.append({"role": "assistant", "content": res_text})

def display_report_interface(translations):
    # Implementation of the report generation interface
    st.header(translations["report_header"])
    # Add your report generation form here

if __name__ == "__main__":
    main()