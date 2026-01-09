from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json

class BaseVectorDB(ABC):
    """
    Abstract base class cho các vector database implementations.
    Định nghĩa interface chung cho tất cả vector databases.
    """
    
    def __init__(self):
        """
        Initialize base vector database.
        
        Args:
        """
        self.client = None
    
    @abstractmethod
    def create_collection(self, vector_size: int, **kwargs):
        """
        Tạo collection/index mới.
        
        Args:
            vector_size (int): Kích thước vector embedding
            **kwargs: Các tham số khác tùy theo implementation
        """
        pass
    
    @abstractmethod
    def upsert_points(self, points: List[Dict[str, Any]], **kwargs):
        """
        Insert hoặc update points vào database.
        
        Args:
            points (List[Dict]): Danh sách points cần upsert
            **kwargs: Các tham số khác (batch_size, max_workers, etc.)
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], limit: int = 5, **kwargs) -> List[Any]:
        """
        Tìm kiếm vectors tương tự.
        
        Args:
            query_vector (List[float]): Vector query
            limit (int): Số lượng kết quả trả về
            **kwargs: Các tham số filter khác
            
        Returns:
            List[Any]: Danh sách kết quả tìm kiếm
        """
        pass
    