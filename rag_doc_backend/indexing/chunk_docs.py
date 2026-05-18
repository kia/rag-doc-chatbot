import logging
logger = logging.getLogger("uvicorn.error")
def create_chunks(documents, text_splitter):
    logger.debug("✂️  Step 3: Splitting text into chunks...")
    chunks = text_splitter.split_documents(documents)
    logger.debug(f"Creating {len(chunks)} text chunks from {len(documents)} documents.")
    return chunks