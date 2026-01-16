from agent import Agent, loop
from retriever import HybridRetriever, TopKRetriever, BM25Retriever
from reranker import Reranker
from llms import LLMs
from reflection import Reflection
from senmatic_router import SemanticRouter
from embedders import SentenceTransformerEmbedding, EmbeddingConfig
from config import VECTOR_SIZE, QDRANT_API_KEY, QDRANT_URL, GROQ_API_KEY, EMBEDDING_MODEL_NAME
from prompts import AGENT_PROMPT, ANSWER_WITH_RETRIVAL, ANSWER_WITHOUT_RETRIVAL, ANSWER_WITH_UNSUFFICIENT_RETRIVAL_INFORMATION
from typing import List, Dict

def init_components():
    reflection_llm = LLMs(
        type="online",
        model_name="chatgroq",
        api_key=GROQ_API_KEY,
        model_version="openai/gpt-oss-20b",
        base_url="https://api.groq.com"
    )
    reflector = Reflection(llm=reflection_llm)

    router = SemanticRouter(
        embedding=SentenceTransformerEmbedding(config=EmbeddingConfig(name=EMBEDDING_MODEL_NAME, device='mps')),
        save_path='data/router/routingEmbeddings/bgem3_routing_embedding_2000.json'
    )

    vector_retriever = TopKRetriever(
        type='qdrant',               
        embeddingName="BAAI/bge-m3",
        vector_size=VECTOR_SIZE,
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL
    )
    
    raw_retriever = BM25Retriever(
        type='qdrant',
        index_path= '/Users/nnam/Documents/Workspace/university/seminar/data/inverted_index/bm25_index_2.json',
        segmenter_path='/Users/nnam/Documents/Workspace/university/seminar/vncorenlp',
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL    
    )
    
    retriever = HybridRetriever(vector_retriever=vector_retriever, raw_retriever=raw_retriever)


    client = LLMs(
            type="online",
            model_name="chatgroq",
            api_key=GROQ_API_KEY,
            model_version="llama-3.3-70b-versatile",
            base_url="https://api.groq.com"
        )
    agent = Agent(
        client=client,
        system_prompt=AGENT_PROMPT
    )
    reranker = Reranker(device='mps')

    llm = LLMs(
        type="online",
        model_name="chatgroq",
        api_key=GROQ_API_KEY,
        model_version="qwen/qwen3-32b",
        base_url="https://api.groq.com"
    )
    return reflector, router, retriever, reranker, agent, llm

reflector, router, retriever, reranker, agent, llm = init_components()

def pipeline(query:List[Dict], use_reranker=True):
    
    print(query)
    
    reflection_text = reflector(query)
    print(f"Reflection text: {reflection_text}")
    
    decision = router.guide(str(reflection_text))
    print(f"Decision: {decision}")
    
    retrieved_docs = []
    if decision[1] == "medical":
        initial_candidates = retriever.search(reflection_text, limit=3)
        
        print("=== INITIAL RETRIEVAL ===")
        for c in initial_candidates:
            print(f"[{c.id}] {c.content[:80]}...")
        print("=========================")
        
        final_candidates, is_sufficient = loop(
            agent=agent,
            query=query,
            initial_candidates=initial_candidates,
            max_iterations=2
        )
        
        retriever.save(query=query, path_dir="_loop", candidates=final_candidates)
        
        print(f"Is sufficient: {is_sufficient}")
        if is_sufficient:
            retrieved_docs = [str(doc.content) for doc in final_candidates]
            
            # Rerank
            if use_reranker:
                reranked_results = reranker.rerank(reflection_text, retrieved_docs)
                _, final_docs = zip(*reranked_results)
            else:
                final_docs = retrieved_docs
                
            context_text = "\n".join([doc for doc in final_docs])
            prompt = [{"role": "system", "content": ANSWER_WITH_RETRIVAL.format(knowledge=context_text)}]
            
        else:
            prompt = [{"role": "system", "content": ANSWER_WITH_UNSUFFICIENT_RETRIVAL_INFORMATION}]
           
    else: 
        prompt = [{"role": "system", "content": ANSWER_WITHOUT_RETRIVAL}]
     
    # Append reflection text
    prompt.append({"role": "user", "content": reflection_text})
    # Answer
    answer = llm.generate_content(prompt)   
    
    return answer
    