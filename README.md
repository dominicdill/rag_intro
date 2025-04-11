# RAG Implementation for Document Processing

This repository contains code for implementing a Retrieval-Augmented Generation (RAG) system for document processing, with and without using the Langchain framework. It includes scripts for creating a vector database, processing documents, and performing question answering. Document processing is performed using Docling and the vector database is created using the PGVector extension for PostgreSQL.

## Project Structure

The project structure is as follows:

-   `.env`: Stores environment variables for database connection and other configurations.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `create_vector_db_without_langchain.py`: Script to create the vector database without Langchain.
-   `failed_files_log.csv`: Log file for files that failed during processing.
-   `rag_db_without_langchain_100_rows.csv`: Example output of vector db when created without Langchain.
-   `testing_langchain_rag.py`: Script to create vector db with Langchain.
-   `annotated_rag_documents/`: Directory containing annotated PDF documents.
-   `test_rag_documents/`: Directory containing documents used for testing.

## Key Components

-   **Configuration:** The [`Config`](create_vector_db_without_langchain.py) class in `create_vector_db_without_langchain.py` handles configuration parameters loaded from environment variables.
-   **Document Processing:** The [`DocumentProcessor`](testing_langchain_rag.py) class in `testing_langchain_rag.py` handles loading models, processing files, chunking, embedding, and storing documents in the vector store.
-   **Vector Database:** The project uses PGVector, a PostgreSQL extension for vector similarity search.


## Usage

1.  **Create the vector database:**

    ```bash
    python create_vector_db_without_langchain.py
    ```

    This script reads documents from the directory specified by the `DOCUMENT_DIRECTORY` environment variable (default: `test_rag_documents`), processes them, and stores the embeddings in the PGVector database.

2.  **Test the RAG implementation:**

    ```bash
    python testing_langchain_rag.py
    ```

    This script provides functionality to test the RAG system, query the database, and evaluate the results.

## Environment Variables

The following environment variables are used to configure the application:

-   `DB_HOST`: Hostname of the PostgreSQL database server (default: `localhost`).
-   `DB_PORT`: Port number of the PostgreSQL database server (default: `5432`).
-   `DB_NAME`: Name of the PostgreSQL database (default: `rag_lyme_docs`).
-   `DB_USER`: Username for connecting to the PostgreSQL database (default: `postgres`).
-   `DB_PASSWORD`: Password for connecting to the PostgreSQL database.
-   `DOCUMENT_DIRECTORY`: Directory containing the documents to be processed (default: `rag_documents`).
-   `DOCUMENT_IMAGE_DIRECTORY`: Directory to store document images (default: `rag_document_images`).
-   `MODEL_ID`: Identifier of the embedding model to use (default: `intfloat/multilingual-e5-large-instruct`).
-   `ENCODING_BATCH_SIZE`: Batch size for encoding documents (default: `32`).
-   `DB_INSERT_BATCH_SIZE`: Batch size for inserting data into the database (default: 100).
-   `FAILED_FILES_LOG`: Log file for failed files (default: `failed_files_log.csv`).

