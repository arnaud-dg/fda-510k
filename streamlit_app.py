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
    
    # Pre-filled input fields
    name = st.text_input("Product Name", value="NeuroScan AI", max_chars=80)
    applicant_name = st.text_input("Applicant Name", value="MedInnovate Technologies", max_chars=200)
    description = st.text_area("Device Description", value="The NeuroScan AI is a revolutionary medical device designed to diagnose and monitor neurological disorders through an in-depth analysis of brain activity. This system relies on artificial intelligence to interpret electroencephalographic (EEG) data in real time, enabling the rapid and accurate detection of neuronal abnormalities.", max_chars=500)
    indication = st.text_area("Proposed Indications for Use", value="The NeuroScan AI is primarily used for the early detection of epilepsy, sleep disorders, and cognitive anomalies in patients with unexplained neurological symptoms. It is also useful for the continuous monitoring of patients at high risk of epileptic seizures.", max_chars=200)
    usage_context = st.text_area("Usage Context", value="The device is designed for use both in hospitals and at home. Once placed on the patient's head, it captures brain signals through a set of electrodes integrated into an ergonomic headset. The user experience is facilitated by an intuitive interface that allows the patient or medical team to start the analysis in just a few simple steps.", max_chars=300)
    algorithm_type = st.text_area("Algorithm Type", value="The NeuroScan AI primarily uses convolutional neural networks (CNNs) for analyzing EEG signals, combined with supervised and unsupervised learning algorithms. The CNNs are used to identify specific patterns related to brain abnormalities, while a clustering model is used to distinguish normal signals from pathological signals. A recurrent LSTM (Long Short-Term Memory) model is also integrated to analyze the temporal variations of EEG data, detecting events such as epileptic seizures.", max_chars=500)
    training_dataset = st.text_area("Training Dataset Description", value="The model was trained on a large dataset comprising over 50,000 EEG recordings collected from multiple specialized medical centers around the world. These recordings included data from patients with epilepsy, sleep disorders, as well as healthy subjects, allowing for the development of a robust and generalized model. The data were anonymized and preprocessed to remove non-neuronal artifacts (such as muscle movements or eye blinks). The training dataset was enriched using data augmentation techniques, simulating various scenarios to improve the model's generalization capability.", max_chars=500)

    if st.button("Generate Report"):
        report = generate_submission_report(
            name, applicant_name, description, indication, 
            usage_context, algorithm_type, training_dataset
        )
        st.markdown(report)

if __name__ == "__main__":
    main()