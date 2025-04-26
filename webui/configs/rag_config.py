import os

# TODO: @wubingheng111 需要完善RAG配置

KNOWLEDGE_BASE_CONFIG = {
    'embedding_model': os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002'),
    'top_k': int(os.environ.get('TOP_K', 5)),
    'similarity_threshold': float(os.environ.get('SIMILARITY_THRESHOLD', 0.75)),
    
}