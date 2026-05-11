def create_chunks(documents, text_splitter):
    print("✂️  Step 3: Splitting text into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Creating {len(chunks)} text chunks from {len(documents)} documents.")
    return chunks