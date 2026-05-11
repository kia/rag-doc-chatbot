from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from indexing.chunk_docs import create_chunks


def collect_documents_state(docs_folder: Path) -> list[dict]:
    pdfs = sorted(docs_folder.glob("**/*.pdf"), key=lambda p: str(p).lower())
    state: list[dict] = []
    for pdf in pdfs:
        stat = pdf.stat()
        state.append(
            {
                "path": str(pdf.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return state


def load_and_split_documents(docs_folder: Path, text_splitter):
    print("🔍 Step 1: Checking documents folder...")
    if not docs_folder.exists():
        docs_folder.mkdir(parents=True)
        raise FileNotFoundError(f"Folder {docs_folder} not found.")

    print("📄 Step 2: Loading PDF documents...")
    loader = DirectoryLoader(
        docs_folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    print(f"   ✅ {len(documents)} documents loaded")

    if not documents:
        raise ValueError("No PDF documents found.")

    return create_chunks(documents, text_splitter)