import streamlit as st
import pandas as pd
from helper import (
    initialize_session_state,
    config_options,
    complete_query,
    generate_submission_report
)

# Set options for pandas to display full text in columns
pd.set_option("max_colwidth", None)

def main():
    st.title(":brain: FDA 510k form Knowledge Base")
    
    initialize_session_state()
    config_options()

    # Create tabs
    tab1, tab2 = st.tabs(["Chat", "Generate Report"])

    with tab1:
        display_chat_interface()

    with tab2:
        display_report_generator()

def display_chat_interface():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input("How can I help you concerning FDA medical devices submissions?"):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            question = question.replace("'", "")
    
            with st.spinner(f"{st.session_state.model_name} thinking..."):
                response = complete_query(question)
                res_text = response['RESPONSE'].iloc[0]
                message_placeholder.markdown(res_text)
        
        st.session_state.messages.append({"role": "assistant", "content": res_text})

def display_report_generator():
    st.header("Generate Submission Report")
    
    # Input fields
    name = st.text_input("Product Name", max_chars=80)
    applicant_name = st.text_input("Applicant Name", max_chars=200)
    description = st.text_area("Device Description", max_chars=500)
    indication = st.text_area("Proposed Indications for Use", max_chars=200)
    usage_context = st.text_area("Usage Context", max_chars=300)
    algorithm_type = st.text_input("Algorithm Type", max_chars=100)
    training_dataset = st.text_area("Training Dataset Description", max_chars=500)

    if st.button("Generate Report"):
        report = generate_submission_report(
            name, applicant_name, description, indication, 
            usage_context, algorithm_type, training_dataset
        )
        st.markdown(report)

if __name__ == "__main__":
    main()