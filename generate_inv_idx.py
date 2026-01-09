from retriever import BM25Retriever
from config import QDRANT_API_KEY, QDRANT_URL, QDRANT_COLLECTION_NAME
if __name__ == '__main__':
    retriever = BM25Retriever(index_path='data/inverted_index/bm25_index_2.json', type='qdrant', qdrant_api=QDRANT_API_KEY, qdrant_url=QDRANT_URL, dbCollection=QDRANT_COLLECTION_NAME, segmenter_path='/Users/nnam/Documents/Workspace/university/seminar/vncorenlp')
    candidates = retriever.search('Alzheimer', limit=3)
    retriever.save(candidates=candidates, path_dir='demobm25')