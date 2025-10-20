import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FlagEmbedding import FlagReranker
from config import RERANKER_MODEL_NAME
import torch


class Reranker():
    def __init__(self, model_name=RERANKER_MODEL_NAME, device=None):
        """
        Initialize Reranker with automatic device detection and error handling.
        
        Args:
            model_name (str, optional): Model name/path. Defaults to RERANKER_MODEL_NAME from config.
            device (str, optional): Device to use ('cuda', 'cpu', 'mps'). Auto-detects if None.
        """
        self.model_name = model_name
        self.device = device or self._auto_detect_device()
        self.model = None
        
        # Load model with error handling
        self._load_viranker()

    def _auto_detect_device(self):
        """
        Automatically detect the best available device.
        
        Returns:
            str: Device name ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print("MPS (Apple Silicon) detected")
        else:
            device = 'cpu'
            print("Using CPU")
        
        return device

    def _load_viranker(self):
        """
        Load ViRanker model for reranking with comprehensive error handling.
        """
        try:
            # Kiểm tra model name
            if not self.model_name:
                raise ValueError("Model name cannot be empty")
            
            print(f"🔵 Loading ViRanker model: {self.model_name}")
            print(f"📍 Target device: {self.device}")
            
            # Xác định use_fp16 dựa trên device
            use_fp16 = self.device == 'cuda'
            
            # Load model
            self.model = FlagReranker(
                self.model_name, 
                use_fp16=use_fp16, 
                device=self.device
            )
            
            print(f"🟢 ViRanker loaded successfully on {self.device}")
            
        except FileNotFoundError as e:
            print(f"❌ Model file not found: {self.model_name}")
            print(f"   Error: {e}")
            print("💡 Please check if the model path is correct or download the model first")
            raise
            
        except RuntimeError as e:
            # Xử lý lỗi CUDA/memory
            if 'CUDA' in str(e) or 'out of memory' in str(e).lower():
                print(f"❌ CUDA/Memory error on {self.device}: {e}")
                print("💡 Trying to fallback to CPU...")
                try:
                    self.device = 'cpu'
                    self.model = FlagReranker(
                        self.model_name, 
                        use_fp16=False, 
                        device='cpu'
                    )
                    print(f"🟢 Successfully loaded on CPU as fallback")
                except Exception as fallback_error:
                    print(f"❌ Fallback to CPU failed: {fallback_error}")
                    raise
            else:
                print(f"❌ Runtime error: {e}")
                raise
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Please install required packages: pip install FlagEmbedding")
            raise
            
        except Exception as e:
            print(f"❌ Unexpected error loading ViRanker: {e}")
            print(f"    Model: {self.model_name}")
            print(f"    Device: {self.device}")
            raise

    def rerank(self, query, documents, top_k=None, normalize=True):
        """
        Rerank documents based on query relevance.
        
        Args:
            query (str): Search query
            documents (list): List of documents to rerank
            top_k (int, optional): Return top k results. Returns all if None.
            
        Returns:
            list: Reranked documents with scores
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform reranking.")
        
        if not query or not documents:
            print("⚠️ Empty query or documents provided")
            return []
        
        try:
            # Tạo pairs cho reranking
            pairs = [[query, doc] for doc in documents]
            
            # Compute scores
            scores = self.model.compute_score(pairs, normalize=normalize, batch_size=5)
            
            ranked_pairs = sorted(zip(scores, documents), reverse=True)
            
            # Trả về top_k nếu được chỉ định
            if top_k:
                ranked_pairs = ranked_pairs[:top_k]
            
            return ranked_pairs
            
        except Exception as e:
            print(f"❌ Error during reranking: {e}")
            raise

    def is_loaded(self):
        """Check if model is loaded successfully."""
        return self.model is not None

    def get_device(self):
        """Get current device."""
        return self.device

    def get_model_name(self):
        """Get current model name."""
        return self.model_name
    
# if __name__ == '__main__':
#     reranker = Reranker()
