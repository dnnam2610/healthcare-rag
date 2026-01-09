from retriever import TopKRetriever, BM25Retriever, HybridRetriever
from config import VECTOR_SIZE, QDRANT_API_KEY, QDRANT_URL

if __name__ == '__main__':
    raw_retriever = BM25Retriever(
        type='qdrant',
        index_path='/Users/nnam/Documents/Workspace/university/seminar/data/inverted_index/bm25_index_2.json',
        segmenter_path='/Users/nnam/Documents/Workspace/university/seminar/vncorenlp',
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL
    )
    vector_retriver = TopKRetriever(
        type='qdrant',               
        embeddingName="BAAI/bge-m3",
        vector_size=VECTOR_SIZE,
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL
    )
    
    hybrid_retriever = HybridRetriever(
        raw_retriever=raw_retriever,
        vector_retriever=vector_retriver
    )
    
    
    query = 'Tôi mới treo sa trễ ngực và đặt túi độn ngực được 2 tháng. Sau khi tiêm vaccine Covid-19 về, tôi cảm thấy bị đau, nhức ngực. Đây có phải do phản ứng sau tiêm của vaccine hay không?'
    results = hybrid_retriever.search(query=query, limit=5)
    hybrid_retriever.save(candidates=results, path_dir='_final')
