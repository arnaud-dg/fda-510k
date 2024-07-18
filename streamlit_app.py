import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
# from snowflake.snowpark.session import Session
# from snowflake.snowpark.context import get_active_session
import os

# Set options for pandas to display full text in columns
pd.set_option("max_colwidth", None)

####################     Snowflake connection     ######################
conn = st.connection("snowflake")

cursor = conn.cursor()

##########################     Constants     ###########################
NUM_CHUNKS = 3  # Number of chunks provided as context
SLIDE_WINDOW = 7  # Number of last conversations to remember

###########################  Main Functions  ###########################
def main():
    st.title(":brain: FDA 510k form Knowledge Base")
    
    # docs_available = conn.query("ls @FDA_510k_PDF_LIST").collect()
    # list_docs = [doc["name"] for doc in docs_available]
    # st.dataframe(list_docs)

    config_options()
    init_messages()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input("How can I help you concerning FDA medical devices submissions ?"):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            question = question.replace("'", "")
    
            with st.spinner(f"{st.session_state.model_name} thinking..."):
                response = complete(question)
                res_text = response[0].RESPONSE.replace("'", "")
                message_placeholder.markdown(res_text)
        
        st.session_state.messages.append({"role": "assistant", "content": res_text})

def config_options():
    st.sidebar.image("https://raw.githubusercontent.com/arnaud-dg/fda-510k/main/assets/510k.png", width=250, caption="Web app logo")
    st.sidebar.write("This web application is a personalized LLM chatbot. Unlike other general-purpose LLMs, it is specifically designed to help you explore FDA submission files for medical devices utilizing artificial intelligence.")
    st.sidebar.write("780 documents serve as the response base for the LLM.") 
    st.sidebar.selectbox('Select your LLM model:',(
        'snowflake-arctic',
        'llama3-8b',
        'mistral-7b',
        'gemma-7b'), key="model_name")
                                           
    st.sidebar.checkbox('Do you want that I remember the chat history?', key="use_chat_history", value=True)
    st.sidebar.checkbox('Debug: Click to see summary generated of previous conversation', key="debug", value=True)
    st.sidebar.button("Start Over", key="clear_conversation")
    st.sidebar.expander("Session State").write(st.session_state)

def init_messages():
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

def get_similar_chunks(question):
    cmd = """
        with results as (
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
    cursor.execute(cmd, params=[question, NUM_CHUNKS])
    df_chunks = cursor.fetch_pandas_all()
    similar_chunks = " ".join(df_chunks["CHUNK"].replace("'", ""))
    return similar_chunks

def get_chat_history():
    chat_history = []
    start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
    for i in range(start_index, len(st.session_state.messages) - 1):
        chat_history.append(st.session_state.messages[i])
    return chat_history

def summarize_question_with_history(chat_history, question):
    print(chat_history)
    print(question)

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
    
    cursor.execute(cmd, (st.session_state.model_name, prompt))
    df_response = cursor.fetch_pandas_all()
    summary = df_response#[0].RESPONSE.replace("'", "")

    if st.session_state.debug:
        st.sidebar.text("Summary to be used to find similar chunks in the docs:")
        st.sidebar.caption(summary)

    return summary

def create_prompt(myquestion):
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = summarize_question_with_history(chat_history, myquestion)
            prompt_context = get_similar_chunks(question_summary)
        else:
            prompt_context = get_similar_chunks(myquestion)
    else:
        prompt_context = get_similar_chunks(myquestion)
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
        {myquestion}
        </question>
        Answer: 
    """
    return prompt

def complete(myquestion):
    prompt = create_prompt(myquestion)
    cmd = """
        SELECT snowflake.cortex.complete(?, ?) as response
    """

    conn = st.connection("snowflake")

    cursor = conn.cursor()

    cursor.execute(cmd, params=[st.session_state.model_name, prompt])
    df_response = cursor.fetch_pandas_all()
    return df_response

if __name__ == "__main__":
    main()