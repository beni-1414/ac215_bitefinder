from unittest.mock import patch, MagicMock
import api.services.pinecone_adapter as pinecone_adapter


# Test get_secret with mock
@patch('api.services.pinecone_adapter.secretmanager.SecretManagerServiceClient')
def test_get_secret(mock_client):
    mock_instance = mock_client.return_value
    mock_instance.access_secret_version.return_value.payload.data.decode.return_value = 'fake_secret'
    with patch('os.getenv', return_value='test-project'):
        secret = pinecone_adapter.get_secret('FAKE_ID')
    assert secret == 'fake_secret'


# Test _index with mock
@patch('api.services.pinecone_adapter.get_secret', return_value='pinecone_key')
@patch('api.services.pinecone_adapter.Pinecone')
def test_index(mock_pinecone, mock_get_secret):
    mock_pc = MagicMock()
    mock_pinecone.return_value = mock_pc
    with patch.dict('os.environ', {'PINECONE_API_KEY': 'pinecone_key', 'PINECONE_INDEX': 'test_index'}):
        _ = pinecone_adapter._index()
    mock_pinecone.assert_called_with(api_key='pinecone_key')
    mock_pc.Index.assert_called_with('test_index')


# Test upsert_embeddings with mock index
@patch('api.services.pinecone_adapter._index')
def test_upsert_embeddings(mock_index):
    mock_idx = MagicMock()
    mock_index.return_value = mock_idx
    pinecone_adapter.upsert_embeddings(ids=['id1'], texts=['text1'], metadatas=[{'meta': 'data'}], vectors=[[0.1, 0.2]])
    mock_idx.upsert.assert_called()


# Test query_by_vector with mock index
@patch('api.services.pinecone_adapter._index')
def test_query_by_vector(mock_index):
    mock_idx = MagicMock()
    mock_match = MagicMock()
    mock_match.metadata = {'text': 'foo'}
    mock_match.id = 'id1'
    mock_idx.query.return_value.matches = [mock_match]
    mock_index.return_value = mock_idx
    result = pinecone_adapter.query_by_vector([0.1, 0.2], top_k=1)
    assert 'ids' in result and 'documents' in result and 'metadatas' in result
