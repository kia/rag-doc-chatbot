import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from rag_doc_backend.vectorestore_creation.pipeline import VectorstoreCreationPipeline

@pytest.fixture
def mock_engine_params():
    return {
        'has_index': True,
        'collection_name': 'test_collection',
        'persist_dir': Path('/tmp/test_persist'),
        'use_openai_api': False,
        'ollama_ingest_batch_size': 2,
        'embeddings': MagicMock()
    }

@pytest.fixture
def mock_chunks():
    return [MagicMock(page_content=f'chunk {i}') for i in range(5)]

@patch('rag_doc_backend.vectorestore_creation.pipeline.Chroma')
def test_create_vectorstore_pipeline_openai(mock_chroma, mock_engine_params, mock_chunks):
    mock_engine_params['use_openai_api'] = True
    pipeline = VectorstoreCreationPipeline(**mock_engine_params)
    
    mock_vectorstore = MagicMock()
    mock_chroma.return_value = mock_vectorstore
    
    result = pipeline.create_vectorstore_pipeline(mock_chunks)
    
    assert result == mock_vectorstore
    mock_vectorstore.add_documents.assert_called_once_with(mock_chunks)
    mock_chroma.assert_called_with(
        collection_name=mock_engine_params['collection_name'],
        persist_directory=str(mock_engine_params['persist_dir']),
        embedding_function=mock_engine_params['embeddings']
    )

@patch('rag_doc_backend.vectorestore_creation.pipeline.Chroma')
@patch('rag_doc_backend.vectorestore_creation.pipeline.time')
def test_create_vectorstore_pipeline_ollama_batching(mock_time, mock_chroma, mock_engine_params, mock_chunks):
    pipeline = VectorstoreCreationPipeline(**mock_engine_params)
    mock_vectorstore = MagicMock()
    mock_chroma.return_value = mock_vectorstore
    
    mock_time.time.side_effect = range(100, 200)
    mock_time.strftime.return_value = '00:00:01'
    mock_time.gmtime.return_value = None
    mock_time.localtime.return_value = None

    result = pipeline.create_vectorstore_pipeline(mock_chunks)
    
    assert result == mock_vectorstore
    assert mock_vectorstore.add_documents.call_count == 3

@patch('rag_doc_backend.vectorestore_creation.pipeline.Chroma')
def test_create_vectorstore_pipeline_clear_existing(mock_chroma, mock_engine_params, mock_chunks):
    pipeline = VectorstoreCreationPipeline(**mock_engine_params)
    mock_vectorstore = MagicMock()
    mock_chroma.side_effect = [mock_vectorstore, mock_vectorstore]
    
    pipeline.create_vectorstore_pipeline(mock_chunks, clear_existing=True)
    
    mock_vectorstore.delete_collection.assert_called_once()
    assert mock_chroma.call_count == 2

@patch('rag_doc_backend.vectorestore_creation.pipeline.Chroma')
def test_create_vectorstore_pipeline_dimension_mismatch(mock_chroma, mock_engine_params, mock_chunks):
    pipeline = VectorstoreCreationPipeline(**mock_engine_params)
    mock_vectorstore = MagicMock()
    
    # First call raises dimension error, second call (after clear) succeeds
    mock_chroma.side_effect = [
        Exception("InvalidArgumentError: Collection expecting embedding with dimension of 384, got 896"),
        mock_vectorstore
    ]
    
    with patch('shutil.rmtree') as mock_rmtree:
        result = pipeline.create_vectorstore_pipeline(mock_chunks)
        
        assert result == mock_vectorstore
        mock_rmtree.assert_called_once_with(mock_engine_params['persist_dir'], ignore_errors=True)
        assert mock_chroma.call_count == 2
