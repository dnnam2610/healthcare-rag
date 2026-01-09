from FlagEmbedding import FlagModel
from embedders.base import BaseEmbedding, EmbeddingConfig
from typing import List, Union
import torch


class FlagBaseEmbedding(BaseEmbedding):
    def __init__(self, 
                 config: EmbeddingConfig, 
                 use_fp16: bool = True,
                 device: str = None):
        super().__init__(config.name)
        device = device or self._auto_detect_device()
        self.model = FlagModel(config.name, use_fp16=use_fp16, device=device)

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        """
        Encode a single string or list of strings into embeddings.
        """
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(text)
        return embeddings
    
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