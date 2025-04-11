import json
from pathlib import Path
import os
import time
import hashlib
import csv
import logging
from dataclasses import dataclass, field
from typing import Optional, Iterator, List # Added typing

from tqdm import tqdm
import torch
from dotenv import load_dotenv

# Langchain Imports
from langchain_core.documents import Document
from langchain_postgres import PGVector # Langchain PGVector integration
from langchain_huggingface import HuggingFaceEmbeddings # Langchain Embeddings
from langchain_docling import DoclingLoader # Langchain Docling Loader

# Docling Imports (Chunker is still needed to pass to DoclingLoader)
from docling.chunking import HybridChunker

# Transformers needed for tokenizer for chunker
from transformers import AutoTokenizer


# --- Basic Logging Setup ---
# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration Class ---
@dataclass
class Config:
    """Holds configuration parameters loaded from environment variables."""
    # Removed vector_dim and max_tokens as they are determined dynamically
    db_host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    db_port: str = field(default_factory=lambda: os.getenv('DB_PORT', '5432'))
    db_name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'rag_db')) # Generic name, collection differentiates
    db_user: str = field(default_factory=lambda: os.getenv('DB_USER', 'postgres'))
    db_password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD'))
    collection_name: str = field(default_factory=lambda: os.getenv('COLLECTION_NAME', 'lyme_docs')) # Collection name for PGVector
    document_directory: str = field(default_factory=lambda: os.getenv('DOCUMENT_DIRECTORY', 'rag_documents'))
    model_id: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL_ID', 'intfloat/multilingual-e5-large-instruct'))
    failed_files_log: str = field(default_factory=lambda: os.getenv('FAILED_FILES_LOG', 'failed_files_log.csv'))

# --- Document Processor Class --- (Refactored)
class DocumentProcessor:
    """Handles loading models, processing files, chunking, embedding, and storing using Langchain."""

    def __init__(self, config: Config):
        """Initializes the DocumentProcessor."""
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_function: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[PGVector] = None
        self.tokenizer = None # Keep tokenizer for chunker
        self.chunker: Optional[HybridChunker] = None
        self.connection_string = ""

        # --- Statistics ---
        self.total_documents_added = 0
        self.files_skipped = 0
        self.files_processed = 0
        self.files_failed = 0

    def _log_failed_file(self, filename: str, file_hash: Optional[str], error_message: str):
        """Appends details of a failed file to the CSV log."""
        log_filename = self.config.failed_files_log
        file_exists = os.path.exists(log_filename)
        try:
            # Ensure directory exists for the log file
            os.makedirs(os.path.dirname(log_filename) or '.', exist_ok=True)
            with open(log_filename, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'file_hash', 'error_message', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists or os.path.getsize(log_filename) == 0:
                    writer.writeheader() # Write header only if file is new or empty

                writer.writerow({
                    'filename': filename,
                    'file_hash': file_hash if file_hash else "HashingFailed/NotAvailable",
                    'error_message': str(error_message), # Ensure error is string
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
        except IOError as e:
            logging.critical(f"Failed to write to log file {log_filename}: {e}")
            logging.critical(f"Failed file details: Name={filename}, Hash={file_hash}, Error={error_message}")


    def _calculate_sha256(self, filepath: Path, buffer_size: int = 65536) -> Optional[str]:
        """Calculates the SHA-256 hash of a file efficiently."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    sha256_hash.update(data)
            return sha256_hash.hexdigest()
        except IOError as e:
            logging.error(f"Error reading file {filepath} for hashing: {e}")
            return None

    def load_models_and_components(self):
        """Loads the embedding model, tokenizer and chunker."""
        logging.info(f"Using device: {self.device}")
        if self.device == 'cuda':
            logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

        try:
            logging.info(f"Loading embedding model: {self.config.model_id}")
            # Use Langchain's HuggingFaceEmbeddings
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=self.config.model_id,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True} # Normalize for cosine similarity
            )
            logging.info("Embedding function loaded.") # Removed logging of dimension

            # Load tokenizer needed for the chunker
            logging.info(f"Loading Tokenizer for Chunker: {self.config.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

            logging.info("Initializing Chunker...")
            self.chunker = HybridChunker(
                tokenizer=self.tokenizer, # Pass the loaded tokenizer object
                merge_peers=True,
            )
            logging.info("Chunker initialized.")

        except Exception as e:
            logging.error(f"Failed to load models or components: {e}", exc_info=True)
            raise # Propagate the error

    def initialize_vector_store(self):
        """Initializes the PGVector store."""
        if not self.embedding_function:
             raise RuntimeError("Embedding function not loaded. Call load_models_and_components first.")
        # Removed pre-check for self.embedding_dim == 0

        # Construct the connection string using PGVector's helper for clarity
        self.connection_string = PGVector.connection_string_from_db_params(
            driver="psycopg", # Make sure psycopg (v3) or psycopg2-binary is installed
            host=self.config.db_host,
            port=int(self.config.db_port),
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
        )

        logging.info(f"Initializing PGVector for collection '{self.config.collection_name}'...")
        try:
            # Initialize the vector store. PGVector handles table creation and
            # determines embedding dimension internally from the embedding_function.
            self.vector_store = PGVector(
                connection=self.connection_string,
                embeddings=self.embedding_function,
                collection_name=self.config.collection_name,
                use_jsonb=True # Recommended for storing metadata efficiently
            )
            logging.info("PGVector store initialized successfully.")
            # Optional: Consider creating an index after adding documents for better performance
            # Example: self.vector_store.create_index(index_type="HNSW", distance_strategy="COSINE")
            # Check PGVector documentation for index creation options and timing.
        except Exception as e:
            logging.error(f"Failed to initialize PGVector: {e}", exc_info=True)
            raise

    def _check_if_hash_exists(self, file_hash: str) -> bool:
        """Checks if any document with the given file_hash exists in the vector store."""
        if not self.vector_store:
            logging.error("Vector store is not initialized.")
            return False # Or raise error

        try:
            # Use similarity search with a filter and k=1 as an existence check.
            # Note: Performance depends on DB indexing of the metadata field.
            # For JSONB metadata, a GIN index might be needed on the 'file_hash' key.
            # Example SQL (run manually or via migration tool):
            # CREATE INDEX idx_gin_metadata_file_hash ON langchain_pg_embedding USING gin ((metadata -> 'file_hash'));
            results = self.vector_store.similarity_search_with_score(
                query="existence check", # Dummy query, content doesn't matter
                k=1,
                filter={"file_hash": file_hash} # Filter on the metadata field
            )
            return len(results) > 0
        except Exception as e:
            # This might fail if filtering on metadata isn't properly supported or indexed
            logging.warning(f"Could not perform filtered search to check hash '{file_hash}': {e}. Assuming file does not exist.")
            # Fallback: Assume not exists on error to avoid skipping files due to check failure.
            # Consider more robust error handling if this check is critical.
            return False


    def process_file(self, file_path: Path):
        """Processes a single file using DoclingLoader and PGVector."""
        filename = file_path.name
        current_file_hash = None
        if not self.vector_store or not self.chunker:
             logging.error(f"Processor not fully initialized for file {filename}. Skipping.")
             self.files_failed += 1
             self._log_failed_file(filename, None, "Processor not initialized")
             return

        try:
            # --- Calculate File Hash ---
            start_hash_time = time.time()
            current_file_hash = self._calculate_sha256(file_path)
            hash_time = time.time() - start_hash_time
            if not current_file_hash:
                # Log failure and skip file if hashing fails
                raise IOError(f"File hashing returned None for {filename}")
            logging.debug(f"Hashed '{filename}' in {hash_time:.2f}s (Hash: {current_file_hash[:8]}...)")

            # --- Check if Hash Already Exists in Vector Store ---
            start_check_time = time.time()
            hash_exists = self._check_if_hash_exists(current_file_hash)
            check_time = time.time() - start_check_time
            logging.debug(f"Checked hash existence for '{filename}' in {check_time:.2f}s")

            if hash_exists:
                logging.info(f"Skipping '{filename}' (Hash: {current_file_hash[:8]}...) - Already processed.")
                self.files_skipped += 1
                return

            logging.info(f"Processing '{filename}' (Hash: {current_file_hash[:8]}...)")
            start_time_file = time.time()

            # --- 1. Load and Chunk Document (Lazily) ---
            loader = DoclingLoader(
                 file_path=str(file_path), # DoclingLoader usually expects string path
                 chunker=self.chunker,
                 # export_type=... # Check DoclingLoader defaults or specify if needed
            )

            # Use lazy_load() for memory efficiency
            doc_iterator: Iterator[Document] = loader.lazy_load()
            documents_to_add: List[Document] = []
            file_docs_processed = 0

            # --- 2. Iterate through chunks/docs, prepare for insertion ---
            for i, doc in enumerate(doc_iterator): # Add index for unique ID generation
                # Basic validation of yielded object
                if not isinstance(doc, Document):
                     logging.warning(f"Unexpected item type from DoclingLoader for {filename}: {type(doc)}. Skipping item.")
                     continue

                # Ensure metadata dictionary exists
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                     doc.metadata = {}

                # Add custom metadata not automatically included by Docling/Chunker
                doc.metadata['file_hash'] = current_file_hash
                doc.metadata['source_filename'] = filename
                # Add timestamp of processing
                doc.metadata['processing_timestamp_utc'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                # Add chunk index within the file
                doc.metadata['chunk_index'] = i

                # Skip if page content is empty or whitespace only
                if not doc.page_content or not doc.page_content.strip():
                     logging.debug(f"Skipping empty chunk {i} from {filename}")
                     continue

                # (Optional) Extract/flatten metadata from Docling if needed
                # e.g., doc.metadata['page'] = doc.metadata.get('source_details', {}).get('page_number')

                documents_to_add.append(doc)
                file_docs_processed += 1

            # --- 3. Add processed documents to Vector Store ---
            if documents_to_add:
                logging.info(f"Adding {len(documents_to_add)} documents from '{filename}' to collection '{self.config.collection_name}'...")
                start_insert_time = time.time()
                try:
                    # add_documents handles embedding via self.embedding_function and DB insertion
                    # Provide unique IDs based on hash + chunk index for potential idempotency/updates
                    doc_ids = [f"{current_file_hash}_{doc.metadata.get('chunk_index', idx)}" for idx, doc in enumerate(documents_to_add)]
                    added_ids = self.vector_store.add_documents(documents_to_add, ids=doc_ids)

                    insert_time = time.time() - start_insert_time
                    if added_ids:
                         self.total_documents_added += len(added_ids)
                         logging.info(f"  Successfully added {len(added_ids)} documents from '{filename}' in {insert_time:.2f}s.")
                    else:
                         # This might happen if IDs already existed and overwrite/ignore occurred
                         logging.warning(f"  add_documents for '{filename}' returned no added IDs (potentially skipped duplicates based on ID). Check DB logs if unexpected.")

                except Exception as insert_error:
                    insert_time = time.time() - start_insert_time
                    logging.error(f"Error adding documents from {filename} to vector store after {insert_time:.2f}s: {insert_error}", exc_info=True)
                    # Re-raise the error to mark the file as failed in the outer try-except
                    raise insert_error
            else:
                 logging.warning(f"No valid documents generated from '{filename}'.")


            # --- File Successfully Processed ---
            self.files_processed += 1
            processing_time = time.time() - start_time_file
            logging.info(f"Successfully processed '{filename}' ({file_docs_processed} documents generated) in {processing_time:.2f}s.")

        except Exception as e: # Catch general exceptions during file processing
            # --- File Failed Processing ---
            self.files_failed += 1
            error_type = type(e).__name__
            # Log error with traceback info if needed (exc_info=True)
            logging.error(f"ERROR processing file {filename} (Hash: {current_file_hash[:8] if current_file_hash else 'N/A'}): {error_type} - {e}", exc_info=False)
            self._log_failed_file(filename, current_file_hash, f"{error_type}: {e}")


    def process_directory(self):
        """Processes all PDF files in the configured directory."""
        doc_dir = Path(self.config.document_directory)
        if not doc_dir.is_dir():
            logging.error(f"Document directory '{self.config.document_directory}' not found or is not a directory.")
            return

        try:
            # Use glob for potentially simpler pattern matching if needed later
            all_files = sorted([f for f in doc_dir.glob('*.pdf') if f.is_file()])
        except OSError as e:
            logging.error(f"Error listing files in '{doc_dir}': {e}")
            return

        logging.info(f"Found {len(all_files)} PDF files in '{doc_dir}'.")

        # Prepare the log file (ensure header etc.)
        self._prepare_log_file()

        start_time_all_files = time.time()

        # Process files using tqdm for progress bar
        file_progress = tqdm(all_files, desc="Processing Files", unit="file", dynamic_ncols=True)
        for file_path in file_progress:
            # Update progress bar description with current file
            file_progress.set_postfix_str(f"Current: {file_path.name}", refresh=True)
            # Call the instance method to process the file
            self.process_file(file_path)

        # --- Final Summary ---
        end_time_all_files = time.time()
        total_time_secs = end_time_all_files - start_time_all_files
        total_time_mins = total_time_secs / 60
        logging.info("--- Processing Complete ---")
        logging.info(f"Total files found: {len(all_files)}")
        logging.info(f"Files processed successfully: {self.files_processed}")
        logging.info(f"Files skipped (already processed): {self.files_skipped}")
        logging.info(f"Files failed: {self.files_failed} (See {self.config.failed_files_log} for details)")
        logging.info(f"Total documents added to vector store: {self.total_documents_added}")
        logging.info(f"Total execution time: {total_time_secs:.2f} seconds ({total_time_mins:.2f} minutes)")

    def _prepare_log_file(self):
        """Ensures the log file exists and has a header if it's new or empty."""
        log_filename = self.config.failed_files_log
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_filename) or '.', exist_ok=True)
            # Check if file needs a header
            needs_header = not os.path.exists(log_filename) or os.path.getsize(log_filename) == 0
            if needs_header:
                with open(log_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['filename', 'file_hash', 'error_message', 'timestamp']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
            logging.info(f"Logging failed files to: {log_filename}")
        except IOError as e:
            logging.critical(f"Could not prepare log file {log_filename}: {e}")
            # Decide if execution should stop if logging isn't possible


# --- Main Execution --- (Refactored)
def main():
    """Main function to set up and run the document processing."""
    config = Config()
    processor = DocumentProcessor(config)

    try:
        # 1. Load models and necessary components (tokenizer, chunker)
        processor.load_models_and_components()

        # 2. Initialize the vector store connection (PGVector)
        processor.initialize_vector_store()

        # 3. Process the documents in the specified directory
        processor.process_directory()

    except Exception as e:
        # Catch errors during model loading, vector store init, or processing directory listing
        logging.critical(f"A critical error occurred during setup or directory processing: {e}", exc_info=True)
        # Exit or handle appropriately if setup fails
        return # Stop execution if setup fails

if __name__ == "__main__":
    main()
