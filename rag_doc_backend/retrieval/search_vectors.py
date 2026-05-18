from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

class ScoreRetriever(BaseRetriever):
    vectorstore: Chroma
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        results = []
        for doc, score in docs_and_scores:
            doc.metadata["score"] = score
            results.append(doc)
        return results

def build_vector_search_chain(llm, vectorstore, k=5) -> RetrievalQA:
    retriever = ScoreRetriever(vectorstore=vectorstore, k=k)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
