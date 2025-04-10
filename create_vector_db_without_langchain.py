import json
from pathlib import Path
import os
import time
import hashlib
import csv
import logging
from dataclasses import dataclass, field

from tqdm import tqdm
import psycopg2
from psycopg2 import sql, Error
from psycopg2.extras import execute_values, Json
import torch

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Basic Logging Setup ---
# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration Class ---
@dataclass
class Config:
    """Holds configuration parameters loaded from environment variables."""
    db_host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    db_port: str = field(default_factory=lambda: os.getenv('DB_PORT', '5432'))
    db_name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'rag_lyme_docs'))
    db_user: str = field(default_factory=lambda: os.getenv('DB_USER', 'postgres'))
    db_password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD'))
    document_directory: str = field(default_factory=lambda: os.getenv('DOCUMENT_DIRECTORY', 'rag_documents'))
    document_image_directory: str = field(default_factory=lambda: os.getenv('DOCUMENT_IMAGE_DIRECTORY', 'rag_document_images'))
    model_id: str = field(default_factory=lambda: os.getenv('MODEL_ID', 'intfloat/multilingual-e5-large-instruct'))
    encoding_batch_size: int = field(default_factory=lambda: int(os.getenv('ENCODING_BATCH_SIZE', '32')))
    db_insert_batch_size: int = field(default_factory=lambda: int(os.getenv('DB_INSERT_BATCH_SIZE', '100')))
    failed_files_log: str = field(default_factory=lambda: os.getenv('FAILED_FILES_LOG', 'failed_files_log.csv'))
    vector_dim: int = 0 # Will be set after model loading

# --- Database Manager Class ---
class DatabaseManager:
    """Handles database connection and operations."""
    def __init__(self, config: Config):
        """
        Initializes the DatabaseManager with connection parameters.

        Args:
            config: The configuration object containing DB credentials.
        """
        self.config = config
        self.conn = None
        self.vector_dim = 0 # Store vector dimension for table creation

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            conn_params = {
                "host": self.config.db_host,
                "database": self.config.db_name,
                "user": self.config.db_user,
                "password": self.config.db_password
            }
            if self.config.db_port:
                conn_params["port"] = self.config.db_port
            self.conn = psycopg2.connect(**conn_params)
            logging.info(f"Connected to the database '{self.config.db_name}' successfully.")
        except Error as e:
            logging.error(f"Error connecting to PostgreSQL on {self.config.db_host}:{self.config.db_port} DB '{self.config.db_name}': {e}")
            raise  # Re-raise the exception to be handled by the caller

    def close(self):
        """Closes the database connection if it's open."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
            self.conn = None

    def create_table_if_not_exists(self, embedding_dim: int):
        """
        Creates the pgvector extension and the document_chunks table if they don't exist.

        Args:
            embedding_dim: The dimension of the vector embeddings.
        """
        if not self.conn:
            logging.error("Database connection is not established.")
            return

        self.vector_dim = embedding_dim # Store for potential later use if needed
        create_extension_query = "CREATE EXTENSION IF NOT EXISTS vector;"
        # Use sql.SQL and sql.Literal for safe dynamic table/column names and values
        create_table_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            pk SERIAL PRIMARY KEY,
            embedding vector({embedding_dim}),
            text TEXT,
            doc_name TEXT,
            heading TEXT[],
            doc_items JSONB,
            file_hash TEXT -- Stores SHA-256 hash of the source file
        );
        """).format(embedding_dim=sql.Literal(embedding_dim))
        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_doc_chunks_file_hash ON document_chunks (file_hash);
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_extension_query)
                cur.execute(create_table_query)
                cur.execute(create_index_query)
            self.conn.commit()
            logging.info("Database table 'document_chunks' checked/created (with file_hash column and index).")
        except Error as e:
            logging.error(f"Error during table creation/check: {e}")
            self.conn.rollback() # Rollback changes on error
            raise

    def check_if_hash_exists(self, file_hash: str) -> bool:
        """
        Checks if any chunk with the given file_hash exists in the database.

        Args:
            file_hash: The SHA-256 hash of the file.

        Returns:
            True if the hash exists, False otherwise.
        """
        if not self.conn:
            logging.error("Database connection is not established.")
            return False # Or raise an error

        query = "SELECT 1 FROM document_chunks WHERE file_hash = %s LIMIT 1;"
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (file_hash,))
                return cur.fetchone() is not None
        except Error as e:
            logging.error(f"Error checking hash '{file_hash}': {e}")
            # Depending on desired behavior, you might want to return False or raise
            return False # Assume it doesn't exist if check fails

    def insert_chunks_batch(self, rows_batch: list) -> int:
        """
        Inserts a batch of rows into the document_chunks table using execute_values.

        Args:
            rows_batch: A list of tuples, where each tuple represents a row to insert.

        Returns:
            The number of rows inserted.
        """
        if not self.conn:
            logging.error("Database connection is not established.")
            return 0
        if not rows_batch:
            return 0

        insert_query = """
        INSERT INTO document_chunks (embedding, text, doc_name, heading, doc_items, file_hash)
        VALUES %s;
        """
        try:
            with self.conn.cursor() as cur:
                # Use Json helper for JSONB columns
                formatted_batch = [
                    (row[0], row[1], row[2], row[3], Json(row[4]), row[5])
                    for row in rows_batch
                ]
                execute_values(cur, insert_query, formatted_batch, page_size=len(rows_batch))
            self.conn.commit()
            return len(rows_batch)
        except Error as e:
            logging.error(f"Error inserting batch: {e}")
            self.conn.rollback()
            # Consider how to handle partial batch failures or logging specific failed rows
            return 0 # Indicate failure or partial success

    def __enter__(self):
        """Context manager entry: establish connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close connection."""
        self.close()


# --- Document Processor Class ---
class DocumentProcessor:
    """Handles loading models, processing files, chunking, embedding, and storing."""

    def __init__(self, config: Config, db_manager: DatabaseManager):
        """
        Initializes the DocumentProcessor.

        Args:
            config: The configuration object.
            db_manager: The database manager instance.
        """
        self.config = config
        self.db_manager = db_manager
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.converter = None
        self.chunker = None
        self.embedding_dim = 0
        self.max_tokens = 0

        # --- Statistics ---
        self.total_chunks_processed = 0
        self.total_rows_inserted = 0
        self.files_skipped = 0
        self.files_processed = 0
        self.files_failed = 0

    def _log_failed_file(self, filename: str, file_hash: str | None, error_message: str):
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

    def _calculate_sha256(self, filepath: Path, buffer_size: int = 65536) -> str | None:
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

    def load_models(self):
        """Loads the embedding model, tokenizer, converter, and chunker."""
        logging.info(f"Using device: {self.device}")
        if self.device == 'cuda':
            logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

        try:
            logging.info(f"Loading embedding model: {self.config.model_id}")
            # Consider adding cache_folder argument if needed
            self.model = SentenceTransformer(self.config.model_id, device=self.device)
            self.max_tokens = self.model.max_seq_length
            # Some models might not expose max_seq_length directly, handle potential AttributeError
            if not self.max_tokens:
                 # Try getting from tokenizer config if model doesn't have it directly
                try:
                    self.max_tokens = self.model.tokenizer.model_max_length
                except AttributeError:
                    logging.warning("Could not automatically determine max_seq_length. Using default 512.")
                    self.max_tokens = 512 # Fallback

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.config.vector_dim = self.embedding_dim # Update config
            logging.info(f"Model loaded. Max sequence length: {self.max_tokens}, Embedding dimension: {self.embedding_dim}")

            logging.info("Loading Document Converter...")

            # pipeline_options = PdfPipelineOptions()
            # pipeline_options.images_scale = 2
            # pipeline_options.generate_page_images = True
            # pipeline_options.generate_picture_images = True

            # self.converter = DocumentConverter(
            #     format_options={
            #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            #     }
            # )

            self.converter = DocumentConverter()
            logging.info("Document Converter loaded.")

            logging.info("Initializing Chunker...")
            self.chunker = HybridChunker(
                tokenizer=self.tokenizer,
                max_tokens=self.max_tokens,
                merge_peers=True,
            )
            logging.info("Chunker initialized.")
        except Exception as e:
            logging.error(f"Failed to load models or components: {e}")
            raise # Propagate the error to stop execution if models can't load

    def process_file(self, file_path: Path):
        """Processes a single file: hash, check DB, convert, chunk, embed, insert."""
        filename = file_path.name
        current_file_hash = None # Initialize hash

        try:
            # --- Calculate File Hash ---
            current_file_hash = self._calculate_sha256(file_path)
            if not current_file_hash:
                raise IOError("File hashing returned None") # Treat hashing failure as an error

            # --- Check if Hash Already Exists in DB ---
            if self.db_manager.check_if_hash_exists(current_file_hash):
                logging.info(f"Skipping '{filename}' (Hash: {current_file_hash[:8]}...) - Already processed.")
                self.files_skipped += 1
                return # Skip to the next file

            logging.info(f"Processing '{filename}' (Hash: {current_file_hash[:8]}...)")
            start_time_file = time.time()

            # --- 1. Convert Document ---
            # Wrap conversion in try-except if DocumentConverter can raise specific errors
            document = self.converter.convert(file_path).document

            # # Save page images
            # for page_no, page in document.pages.items():
            #     page_no = page.page_no
            #     page_image_filename = Path(self.config.document_image_directory) / f"{file_path.stem}-{page_no}.png"
            #     with page_image_filename.open("wb") as fp:
            #         page.image.pil_image.save(fp, format="PNG")


            # --- 2. Chunk Document ---
            # Wrap chunking in try-except if Chunker can raise specific errors
            chunk_iter = self.chunker.chunk(dl_doc=document)

            # --- 3. Process Chunks ---
            chunks_for_encoding_batch = []
            rows_to_insert_batch = []
            file_chunks_processed = 0

            # Use enumerate for potential index tracking if needed later
            for chunk in chunk_iter: # No inner tqdm needed if outer tqdm updates filename
                chunk_text = self.chunker.serialize(chunk=chunk)
                if len(self.tokenizer.tokenize(chunk_text)) > self.max_tokens:
                    logging.warning(f"Chunk exceeds max tokens ({self.max_tokens}). Encoding will miss information")
                if not chunk_text or not chunk_text.strip():
                    continue # Skip empty chunks

                # Extract metadata safely using .get() with defaults
                chunk_metadata = chunk.meta.export_json_dict()
                metadata_for_db = {
                    'doc_name': chunk_metadata.get('origin', {}).get('filename', filename),
                    'heading': chunk_metadata.get('headings', []), # Default to empty list
                    # Ensure doc_items is valid JSON, default to empty list if missing/invalid
                    'doc_items': chunk_metadata.get('doc_items', []),
                    'file_hash': current_file_hash
                }

                chunks_for_encoding_batch.append((chunk_text, metadata_for_db))
                file_chunks_processed += 1

                # --- Encode Batch ---
                if len(chunks_for_encoding_batch) >= self.config.encoding_batch_size:
                    self._encode_and_prepare_insert(chunks_for_encoding_batch, rows_to_insert_batch)
                    chunks_for_encoding_batch.clear() # Clear after processing
                    logging.info(f"  Processed {file_chunks_processed} chunks for '{filename}'")

                # --- Insert Batch ---
                if len(rows_to_insert_batch) >= self.config.db_insert_batch_size:
                    inserted_count = self.db_manager.insert_chunks_batch(rows_to_insert_batch)
                    self.total_rows_inserted += inserted_count
                    rows_to_insert_batch.clear() # Clear after inserting

            # --- Process Remaining Chunks after Loop ---
            if chunks_for_encoding_batch:
                self._encode_and_prepare_insert(chunks_for_encoding_batch, rows_to_insert_batch)
                chunks_for_encoding_batch.clear()

            # --- Insert any Remaining Rows ---
            if rows_to_insert_batch:
                inserted_count = self.db_manager.insert_chunks_batch(rows_to_insert_batch)
                self.total_rows_inserted += inserted_count
                rows_to_insert_batch.clear()

            # --- File Successfully Processed ---
            self.total_chunks_processed += file_chunks_processed
            self.files_processed += 1
            processing_time = time.time() - start_time_file
            logging.info(f"Successfully processed '{filename}' ({file_chunks_processed} chunks) in {processing_time:.2f}s.")
            # Explicitly delete large objects to free memory sooner, if necessary
            del document
            del chunk_iter

        except (Error, Exception) as e: # Catch specific DB errors and general exceptions
            # --- File Failed Processing ---
            self.files_failed += 1
            error_type = type(e).__name__
            logging.error(f"ERROR processing file {filename} (Hash: {current_file_hash[:8] if current_file_hash else 'N/A'}): {error_type} - {e}", exc_info=False) # Set exc_info=True for full traceback if needed
            self._log_failed_file(filename, current_file_hash, f"{error_type}: {e}")
            # No need to continue here, loop will proceed

    def _encode_and_prepare_insert(self, chunks_for_encoding: list, rows_to_insert: list):
        """Encodes a batch of texts and prepares the rows for DB insertion."""
        if not chunks_for_encoding:
            return

        texts_to_encode = [text for text, meta in chunks_for_encoding]
        try:
            embeddings = self.model.encode(
                texts_to_encode,
                batch_size=self.config.encoding_batch_size, # Use configured batch size
                show_progress_bar=False, # Already have file progress bar
                convert_to_numpy=True, # Often required for DB insertion libraries
                normalize_embeddings=True # Normalize for cosine similarity
            )

            for i, embedding in enumerate(embeddings):
                text, meta = chunks_for_encoding[i]
                # Ensure embedding is a list for psycopg2/pgvector
                embedding_list = embedding.tolist()
                row = (
                    embedding_list, text, meta['doc_name'],
                    meta['heading'], meta['doc_items'], meta['file_hash']
                )
                rows_to_insert.append(row)
        except Exception as e:
            # This error might affect a whole batch. Log appropriately.
            # Consider logging which texts failed if possible.
            logging.error(f"Failed to encode batch: {e}", exc_info=True)
            # Decide how to handle: skip batch, log individual failures, etc.
            # For now, we'll log the error and potentially miss inserting this batch.
            # You might want to add more robust error handling here, e.g., logging failed texts.


    def process_directory(self):
        """Processes all PDF files in the configured directory."""
        doc_dir = Path(self.config.document_directory)
        if not doc_dir.is_dir():
            logging.error(f"Document directory '{self.config.document_directory}' not found or is not a directory.")
            return

        # List files safely
        try:
            all_files = [f for f in doc_dir.iterdir() if f.is_file() and f.suffix.lower() == '.pdf']
        except OSError as e:
            logging.error(f"Error listing files in '{doc_dir}': {e}")
            return

        logging.info(f"Found {len(all_files)} PDF files in '{doc_dir}'.")

        # Prepare log file (ensure header if new/empty)
        self._prepare_log_file()

        start_time_all_files = time.time()

        # Use tqdm for overall file progress
        file_progress = tqdm(all_files, desc="Processing Files", unit="file", dynamic_ncols=True)
        for file_path in file_progress:
            file_progress.set_postfix_str(f"Current: {file_path.name}", refresh=True)
            self.process_file(file_path) # Call the instance method to process each file

        # --- Final Summary ---
        end_time_all_files = time.time()
        logging.info("--- Processing Complete ---")
        logging.info(f"Total files found: {len(all_files)}")
        logging.info(f"Files processed successfully: {self.files_processed}")
        logging.info(f"Files skipped (already processed): {self.files_skipped}")
        logging.info(f"Files failed: {self.files_failed} (See {self.config.failed_files_log} for details)")
        logging.info(f"Total chunks generated (from processed files): {self.total_chunks_processed}")
        logging.info(f"Total rows inserted into DB: {self.total_rows_inserted}")
        logging.info(f"Total execution time: {end_time_all_files - start_time_all_files:.2f} seconds")

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


# --- Main Execution ---
def main():
    """Main function to set up and run the document processing."""
    config = Config()

    # Use DatabaseManager as a context manager for automatic connection handling
    try:
        with DatabaseManager(config) as db_manager:
            processor = DocumentProcessor(config, db_manager)

            # Load models and determine embedding dimension
            processor.load_models() # This might raise errors if loading fails

            # Ensure table exists *after* knowing the embedding dimension
            db_manager.create_table_if_not_exists(processor.embedding_dim)

            # Process the documents
            processor.process_directory()

    except Error as db_error:
        # Catch DB connection or table creation errors specifically
        logging.critical(f"Database setup failed: {db_error}", exc_info=True)
    except Exception as e:
        # Catch other potential errors during setup or processing
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
