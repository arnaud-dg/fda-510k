/*
* FDA 510k Document Processing Pipeline
* ====================================
*
* This script sets up a processing pipeline for FDA 510k documents that:
* 1. Configures database and permissions
* 2. Creates a function for chunking PDF text
* 3. Sets up document storage and processing
* 4. Generates vector embeddings for semantic search
*
* Dependencies:
* - Snowflake with Cortex enabled
* - Python 3.9
* - Packages: snowflake-snowpark-python, PyPDF2, langchain
*
* Author: [Your Name]
* Date: 2024-10-25
*/

-- Database setup
CREATE DATABASE IF NOT EXISTS fda_document_db;
CREATE SCHEMA IF NOT EXISTS document_processing;
USE ROLE ACCOUNTADMIN;

-- Role and permissions configuration
CREATE ROLE IF NOT EXISTS document_processor_role;
--GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE document_processor_role;
--GRANT ROLE document_processor_role TO USER PZPNYMB.IX42431;

-- Check available functions
SHOW USER FUNCTIONS LIKE 'EMBED_TEXT_768' IN SCHEMA SNOWFLAKE.CORTEX;

-- PDF chunking function
CREATE OR REPLACE FUNCTION document_text_chunker(file_url STRING)
RETURNS TABLE (chunk VARCHAR)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.9'
HANDLER = 'PDFTextChunker'
PACKAGES = ('snowflake-snowpark-python','PyPDF2', 'langchain')
AS
$$
from snowflake.snowpark.types import StringType, StructField, StructType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from snowflake.snowpark.files import SnowflakeFile
import PyPDF2
import io
import logging
import pandas as pd

class PDFTextChunker:
    """
    Class for chunking PDF documents into text segments.
    
    Attributes:
        CHUNK_SIZE (int): Target size for each text chunk
        CHUNK_OVERLAP (int): Overlap between chunks to maintain context
    """
    
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    
    def read_pdf(self, file_url: str) -> str:
        """
        Reads and extracts text from a PDF file.
        
        Args:
            file_url (str): URL of the PDF file to process
            
        Returns:
            str: Extracted text from the PDF
        """
        logger = logging.getLogger("udf_logger")
        logger.info(f"Processing file {file_url}")
    
        with SnowflakeFile.open(file_url, 'rb') as pdf_file:
            pdf_buffer = io.BytesIO(pdf_file.readall())
            
        pdf_reader = PyPDF2.PdfReader(pdf_buffer)   
        extracted_text = ""
        
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                cleaned_text = page_text.replace('\n', ' ').replace('\0', ' ')
                extracted_text += cleaned_text
            except Exception as e:
                logger.warning(f"Extraction error - file: {file_url}, page: {page}, error: {str(e)}")
                return "Extraction failed"
        
        return extracted_text

    def process(self, file_url: str):
        """
        Processes a PDF and splits it into chunks.
        
        Args:
            file_url (str): URL of the file to process
            
        Yields:
            tuple: Extracted text chunks
        """
        document_text = self.read_pdf(file_url)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            length_function=len
        )
    
        text_chunks = text_splitter.split_text(document_text)
        chunks_df = pd.DataFrame(text_chunks, columns=['chunks'])
        
        yield from chunks_df.itertuples(index=False, name=None)
$$;

-- Storage configuration
CREATE OR REPLACE STAGE document_storage 
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE') 
    DIRECTORY = (ENABLE = true);

-- Content verification
LIST @fda_510k_pdf_list;

-- Chunks and vectors storage table
CREATE OR REPLACE TABLE document_chunks (
    document_path VARCHAR(16777216),
    document_size NUMBER(38,0),
    document_url VARCHAR(16777216),
    document_scoped_url VARCHAR(16777216),
    text_chunk VARCHAR(16777216),
    chunk_embedding VECTOR(FLOAT, 768)
);

-- Document processing and embedding generation
INSERT INTO document_chunks (
    document_path, 
    document_size,
    document_url, 
    document_scoped_url,
    text_chunk,
    chunk_embedding
)
SELECT 
    relative_path, 
    size,
    file_url, 
    build_scoped_file_url(@fda_510k_pdf_list, relative_path),
    chunk,
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', chunk)
FROM 
    directory(@fda_510k_pdf_list),
    TABLE(document_text_chunker(
        build_scoped_file_url(@fda_510k_pdf_list, relative_path)
    ));

-- Analysis queries
-- Check chunks and embeddings
SELECT text_chunk, chunk_embedding 
FROM document_chunks;

-- Analyze chunk distribution per document
SELECT 
    document_path,
    COUNT(*) as chunks_count
FROM document_chunks
GROUP BY document_path;

-- Completion model configuration
SELECT snowflake.cortex.complete(
    'snowflake-arctic',
    '
    Based on the chat history below and the question,
    generate a query that extends the question with the provided history.
    The query should be in natural language.
    Answer with only the query, no explanation.
    
    <chat_history>
    {chat_history}
    </chat_history>
    <question>
    {question}
    </question>
    '
) as completion_response;