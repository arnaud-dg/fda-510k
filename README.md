[Streamlit Weblink](https://fda-510k-dkpqqdmyxeshkpspqez74e.streamlit.app/)

# FDA 510k Knowledge Base Project
## Description
This project is a Streamlit-based web application that leverages a Large Language Model (LLM) with Retrieval-Augmented Generation (RAG) capabilities, powered by Snowflake. The application serves as a knowledge base for FDA 510k submissions, allowing users to chat with an AI assistant and generate submission reports.
Features

Chat interface for querying about FDA medical device submissions
Report generator for creating detailed FDA 510(k) submission reports
Integration with Snowflake for data storage and retrieval
Utilization of LLM-RAG for enhanced query responses

## Installation

Clone the repository:
Copygit clone https://github.com/your-username/fda-510k-knowledge-base.git
cd fda-510k-knowledge-base

Install the required dependencies:
Copypip install -r requirements.txt

Set up your Snowflake connection:

Ensure you have a Snowflake account and the necessary credentials
Configure your Snowflake connection in the Streamlit secrets management. 
/!\ Unfortunately RAG vector database is hosted in my snowflake account, so this project can not be runned without my credentials.


## Usage

Run the Streamlit app:
Copystreamlit run streamlit_app.py

Open your web browser and navigate to the provided local URL (usually http://localhost:8501)
Use the chat interface to ask questions about FDA 510k submissions
Generate submission reports using the provided form in the "Generate Report" tab

## Project Structure

streamlit_app.py: Main application file containing the Streamlit interface
helper.py: Contains helper functions for LLM interactions and report generation
requirements.txt: List of Python dependencies
assets/: Directory containing additional resources (e.g., images, documents)

## Dependencies

Streamlit
Snowflake Snowpark
Pandas
Other dependencies as listed in requirements.txt

## Configuration

Snowflake connection: Configure in Streamlit's secrets management
Model selection: Available in the sidebar of the application
Debug mode: Toggle in the sidebar for additional information

## Notes

This application uses vectorized PDF documents as a knowledge base
The LLM-RAG system is built on top of Snowflake's infrastructure
Ensure proper handling of sensitive information in FDA submissions

##License
This project is licensed under the MIT License.
