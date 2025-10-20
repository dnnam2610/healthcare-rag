from embedders.base import APIBaseEmbedding, APIEmbeddingConfig
import google.generativeai as genai

class GeminiEmbedding(APIBaseEmbedding):
    def __init__(self, config: APIEmbeddingConfig):
        super().__init__(config.name, apiKey=config.apiKey)
        genai.configure(api_key=self.apiKey)

    def encode(self, text):
        return genai.embed_content(
            model=self.name,
            content=[text]
        )['embedding']
    
# if __name__ == '__main__':
#     config = APIEmbeddingConfig(
#         name='gemini-embedding-001',
#         apiKey='AIzaSyAYOLcSrYq1CNJHNo42u0x6Decf7g3QB_s'
#     )
#     embedder = GeminiEmbedding(
#         config=config
#     )

#     print(embedder.encode('xin chào buổi sáng'))

