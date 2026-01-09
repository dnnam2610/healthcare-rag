from embedders.base import BaseEmbedding, EmbeddingConfig
from sentence_transformers import SentenceTransformer
import torch

class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config.name)
        self.config = config

        # ===== 1. RESOLVE DEVICE =====
        resolved_device = self._resolve_device(self.config.device)

        # ===== 2. LOAD MODEL =====
        self.embedding_model = SentenceTransformer(
            self.config.name,
            trust_remote_code=True,
            device=resolved_device
        )

        # ===== 3. VERIFY REAL DEVICE =====
        real_device = str(self.embedding_model.device)

        print(
            f"🟢 SentenceTransformer loaded\n"
            f"   ├─ Model           : {self.config.name}\n"
            f"   ├─ Requested device: {self.config.device}\n"
            f"   ├─ Resolved device : {resolved_device}\n"
            f"   └─ Actual device   : {real_device}"
        )

        if real_device != resolved_device:
            print(
                f"⚠️ WARNING: Device mismatch! "
                f"Model is running on '{real_device}'"
            )

    def _resolve_device(self, device: str) -> str:
        """
        Check if device is actually usable, otherwise fallback safely
        """
        device = device.lower()

        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("⚠️ CUDA requested but not available → fallback to CPU")
            return "cpu"

        if device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            print("⚠️ MPS requested but not available → fallback to CPU")
            return "cpu"

        # default CPU
        return "cpu"

    def encode(self, text: str):
        return self.embedding_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
