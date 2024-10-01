[Streamlit Weblink](https://fda-510k-dkpqqdmyxeshkpspqez74e.streamlit.app/)

# FDA 510k form Knowledge Base (Project on-going)

The goal of this project is to build a knowledge base using a LLM (Large Language Model), enriched with the RAG (Retrieval-Augmented Generation) technology, for medical device submissions to the FDA.
The FDA 510(k) form is a premarket submission made to the U.S. Food and Drug Administration (FDA) to demonstrate that a new medical device is at least as safe and effective as a legally marketed device that is not subject to Premarket Approval (PMA). This process is based on the concept of "substantial equivalence" and is a critical step for many medical devices to enter the U.S. market.
It is very interesting to analyse the state-of-the-art of the submission to enrich and foster the risk analysis.
By using existing 510(k) forms as references, companies can streamline the submission process, reduce costs, and accelerate the time it takes to bring new, innovative medical devices to patients and healthcare providers.

In the end, we obtain a real knowledge base that saves a lot of time when making new submissions. It is very difficult to manually browse the various documents and search for medical devices similar to ours. With this kind of tool, we can, for example, ask for the registration numbers of medical devices that are similar and quickly access their files. It is also possible to ask what verification and validation tests have been done for a given type of technology.

# Data Source
Data used to build this LLM-RAG agent is provided by the FDA on its open-source website [FDA](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices)

# Methodology
1. A python script has been desgined to scrap the data from the FDA website. Metadata are collected about each medical devices submissions and PDF files are downloaded ; the PDF files contain a detailed description of the medical device and described the action performed to asses and control risks.
2. PDF files are stored in an AWS S3 bucket. They are accessible via Snowflake, which serves as the data foundation
3. Classical steps to build a RAG are performed in Snwoflake with native snowflake functions:
- PDF scraping
- text splitting in chunks
- embedding with [SNOWFLAKE.CORTEX.EMBED_TEXT_768 function](https://docs.snowflake.com/en/sql-reference/functions/embed_text-snowflake-cortex). We obtain a vector of 768 dimensions specific of the context and concpets described for each chunks
- storage of the chunks, metadata and vecor in a snowflake table
5. Finally, the chatbot is built through the Streamlit interface. Streamlit is a very simple interface that allows for quick interaction and prototype construction. Each question asked by the user is "vectorized by the LLM", which constructs responses from the closest vectors with a cosine similarity analysis. The streamlit webapp allow the user to change the nature of the LLM used to answer and also to clear the cache of the chatbot agent (whihc keep in memory in a standard way the 7 last questions) 

An important step is the design of prompts, becauses these prompts will guide the quality, repetability and veracity of the answers provided by the LLM. The LLM is specifically instructed not to provide a response if the question posed is not in the RAG database, thus limiting the hallucination phenomenon. 

This project is still under development !
Points to improve:
- RAG database diversity
- text splitting optimization
- prompt engineering
- LLM models benchmark
- modulate the nature of the answer (plain text VS SQL query results as dataframe)

# Example of question to ask 
- What kind of medical devices are using CNN technologies or neural network for image analysis ?
- How to validate a medical device using CNN ?
- What are the risk associated to tomoraphy medical devices ?
- ... etc
