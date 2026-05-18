from langchain_classic.chains import RetrievalQA


def build_vector_search_chain(llm, vectorstore, k=5) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
    )
