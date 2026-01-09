import google.generativeai as genai
import json
from typing import List, Dict
import pandas as pd
from datetime import datetime
import os

# Cấu hình API key
genai.configure(api_key="AIzaSyAYOLcSrYq1CNJHNo42u0x6Decf7g3QB_s")

class RAGMetrics:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model_name)
    
    def calculate_context_precision(
        self, 
        question: str, 
        contexts: List[str], 
        ground_truth: str
    ) -> float:
        """
        Context Precision: Đo lường độ chính xác của các context được retrieve.
        Tính tỷ lệ các context có liên quan trong tổng số context được retrieve.
        
        Args:
            question: Câu hỏi của user
            contexts: Danh sách các context được retrieve (theo thứ tự rank)
            ground_truth: Câu trả lời đúng/ground truth
        
        Returns:
            Context precision score (0-1)
        """
        relevance_scores = []
        
        for i, context in enumerate(contexts):
            prompt = f"""
Đánh giá xem context sau có liên quan và hữu ích để trả lời câu hỏi không.

Câu hỏi: {question}
Ground Truth: {ground_truth}
Context: {context}

Trả lời chỉ bằng JSON với format:
{{"relevant": true/false, "reason": "lý do ngắn gọn"}}
"""
            
            try:
                response = self.model.generate_content(prompt)
                result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
                relevance_scores.append(1 if result['relevant'] else 0)
            except Exception as e:
                print(f"Error processing context {i}: {e}")
                relevance_scores.append(0)
        
        if not relevance_scores:
            return 0.0
        
        # Context Precision = (số context relevant) / (tổng số context)
        precision = sum(relevance_scores) / len(relevance_scores)
        return precision
    
    def calculate_context_recall(
        self, 
        ground_truth: str, 
        contexts: List[str]
    ) -> float:
        """
        Context Recall: Đo lường khả năng retrieve được tất cả thông tin cần thiết.
        Tính tỷ lệ các thông tin trong ground truth có xuất hiện trong contexts.
        
        Args:
            ground_truth: Câu trả lời đúng/ground truth
            contexts: Danh sách các context được retrieve
        
        Returns:
            Context recall score (0-1)
        """
        combined_contexts = "\n\n".join(contexts)
        
        prompt = f"""
Phân tích xem các context đã retrieve có chứa đủ thông tin để tạo ra ground truth không.

Ground Truth: {ground_truth}

Contexts đã retrieve:
{combined_contexts}

Hãy:
1. Chia ground truth thành các statement/thông tin riêng biệt
2. Đánh giá xem mỗi statement có được hỗ trợ bởi contexts không

Trả lời bằng JSON với format:
{{
    "statements": [
        {{"statement": "thông tin 1", "supported": true/false}},
        {{"statement": "thông tin 2", "supported": true/false}}
    ],
    "recall": 0.0-1.0
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            return result['recall']
        except Exception as e:
            print(f"Error calculating recall: {e}")
            
            # Fallback: tính bằng keyword matching đơn giản
            gt_words = set(ground_truth.lower().split())
            context_words = set(combined_contexts.lower().split())
            overlap = len(gt_words.intersection(context_words))
            return overlap / len(gt_words) if gt_words else 0.0


class MetricsLogger:
    """Class để lưu trữ và quản lý kết quả test"""
    
    def __init__(self, csv_path='rag_metrics_results.csv'):
        self.csv_path = csv_path
        self.results = []
        
    def add_result(
        self, 
        sample_id: str,
        question: str,
        ground_truth: str,
        precision: float,
        recall: float,
        num_contexts: int,
        retrieved_doc_ids: List[str] = None,
        retriever_type: str = None,
        metadata: Dict = None
    ):
        """Thêm một kết quả test"""
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_id': sample_id,
            'question': question[:100],  # Giới hạn độ dài
            'ground_truth': ground_truth[:100],
            'precision': precision,
            'recall': recall,
            'num_contexts': num_contexts,
            'retrieved_doc_ids': ','.join(map(str, retrieved_doc_ids)) if retrieved_doc_ids else '',
            'retriever_type': retriever_type or 'unknown'
        }
        
        # Thêm metadata nếu có
        if metadata:
            result.update(metadata)
            
        self.results.append(result)
    
    def save_to_csv(self, mode='append', auto_save=False):
        """
        Lưu kết quả vào CSV
        
        Args:
            mode: 'append' để thêm vào file cũ, 'overwrite' để ghi đè
            auto_save: Nếu True, chỉ lưu kết quả mới nhất và xóa khỏi memory
        """
        if not self.results:
            return
            
        if auto_save:
            # Chỉ lưu kết quả cuối cùng
            df_new = pd.DataFrame([self.results[-1]])
        else:
            df_new = pd.DataFrame(self.results)
        
        if mode == 'append' and os.path.exists(self.csv_path):
            # Đọc file cũ và append
            df_old = pd.read_csv(self.csv_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_csv(self.csv_path, index=False)
            if not auto_save:
                print(f"✅ Đã thêm {len(df_new)} kết quả vào {self.csv_path}")
        else:
            # Tạo file mới
            df_new.to_csv(self.csv_path, index=False)
            if not auto_save:
                print(f"✅ Đã tạo file mới {self.csv_path} với {len(df_new)} kết quả")
        
        # Nếu auto_save, xóa kết quả đã lưu khỏi memory
        if auto_save and len(self.results) > 0:
            self.results.pop()
    
    def get_summary(self):
        """Lấy tóm tắt kết quả"""
        if not self.results:
            return "Chưa có kết quả nào"
        
        df = pd.DataFrame(self.results)
        summary = {
            'total_samples': len(df),
            'avg_precision': df['precision'].mean(),
            'avg_recall': df['recall'].mean(),
            'std_precision': df['precision'].std(),
            'std_recall': df['recall'].std(),
            'min_precision': df['precision'].min(),
            'max_precision': df['precision'].max(),
            'min_recall': df['recall'].min(),
            'max_recall': df['recall'].max()
        }
        return summary
    
    def save_summary_to_txt(self, txt_path='rag_metrics_summary.txt', mode='append'):
        """
        Lưu tóm tắt kết quả vào file txt
        
        Args:
            txt_path: Đường dẫn file txt
            mode: 'append' để thêm vào file cũ, 'overwrite' để ghi đè
        """
        summary = self.get_summary()
        
        if isinstance(summary, str):
            print(summary)
            return
        
        # Tạo nội dung
        content = []
        content.append("=" * 60)
        content.append(f"📊 RAG METRICS SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("=" * 60)
        content.append("")
        
        # Thông tin chung
        content.append("📈 GENERAL INFO:")
        content.append(f"   Total Samples: {summary['total_samples']}")
        content.append("")
        
        # Precision metrics
        content.append("🎯 PRECISION METRICS:")
        content.append(f"   Average:  {summary['avg_precision']:.4f}")
        content.append(f"   Std Dev:  {summary['std_precision']:.4f}")
        content.append(f"   Min:      {summary['min_precision']:.4f}")
        content.append(f"   Max:      {summary['max_precision']:.4f}")
        content.append("")
        
        # Recall metrics
        content.append("🔍 RECALL METRICS:")
        content.append(f"   Average:  {summary['avg_recall']:.4f}")
        content.append(f"   Std Dev:  {summary['std_recall']:.4f}")
        content.append(f"   Min:      {summary['min_recall']:.4f}")
        content.append(f"   Max:      {summary['max_recall']:.4f}")
        content.append("")
        
        # Thêm thông tin về retriever nếu có
        if self.results:
            df = pd.DataFrame(self.results)
            if 'retriever_type' in df.columns:
                content.append("⚙️  CONFIGURATION:")
                content.append(f"   Retriever Type: {df['retriever_type'].iloc[0]}")
                if 'embedding_model' in df.columns:
                    content.append(f"   Embedding Model: {df['embedding_model'].iloc[0]}")
                if 'limit' in df.columns:
                    content.append(f"   Retrieval Limit: {df['limit'].iloc[0]}")
                content.append("")
        
        content.append("=" * 60)
        content.append("")
        
        # Ghi vào file
        write_mode = 'a' if mode == 'append' else 'w'
        with open(txt_path, write_mode, encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"✅ Đã lưu summary vào {txt_path}")
        
        return summary
    
    def clear(self):
        """Xóa kết quả tạm trong memory"""
        self.results = []


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from retriever import MMRRetriever, TopKRetriever, KneedleRetriever
    from datasets import load_from_disk
    import numpy as np
    from config import VECTOR_SIZE, QDRANT_API_KEY, QDRANT_URL

    # Khởi tạo retriever
    retriever = TopKRetriever(
        type='qdrant',               
        embeddingName="BAAI/bge-m3",
        vector_size=VECTOR_SIZE,
        qdrant_api=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL
    )
    
    # Khởi tạo metrics và logger
    metrics = RAGMetrics()
    logger = MetricsLogger(csv_path='rag_metrics_results.csv')
    
    # Load data
    subset_path = "data/ViHealthQA_test_200"

    if os.path.exists(subset_path):
        print("📁 Loading 200-sample subset from disk...")
        samples = load_from_disk(subset_path)
    else:
        print("🆕 Creating 200-sample subset...")
        data = load_from_disk('data/ViHealthQA_test')
        samples = data.select(range(200))
        samples.save_to_disk(subset_path)
        print("💾 Saved subset to disk!")

    print("➡️ Number of samples:", len(samples))
    avg_precision = []
    avg_recall = []
    
    print("🚀 Bắt đầu đánh giá...")
    print("="*50)
    
    for i, sample in enumerate(samples, 1):
        try:
            idx = sample['id']
            question = sample['question']
            answer = sample['answer']

            # Retrieve contexts
            candidates = retriever.vector_search(user_query=question, limit=5)
            retrived_texts = [c.content for c in candidates]
            retrieved_ids = [c.id for c in candidates]

            # Tính metrics
            precision = metrics.calculate_context_precision(
                question=question, 
                ground_truth=answer, 
                contexts=retrived_texts
            )
            recall = metrics.calculate_context_recall(
                ground_truth=answer, 
                contexts=retrived_texts
            )
            
            avg_precision.append(precision)
            avg_recall.append(recall)
            
            # Lưu kết quả vào logger
            logger.add_result(
                sample_id=idx,
                question=question,
                ground_truth=answer,
                precision=precision,
                recall=recall,
                num_contexts=len(retrived_texts),
                retrieved_doc_ids=retrieved_ids,
                retriever_type='TopKRetriever',
                metadata={
                    'embedding_model': 'BAAI/bge-m3',
                    'limit': 5
                }
            )
            
            # 💾 LƯU NGAY SAU MỖI SAMPLE
            logger.save_to_csv(mode='append', auto_save=True)

            # In kết quả
            print(f'\n📝 Sample {i}/{len(samples)} - ID: {idx}')
            print(f'❓ Question: {question[:80]}...')
            print(f'✅ Truth: {answer[:80]}...')
            print(f'📄 Retrieved IDs: {", ".join(map(str, retrieved_ids))}')
            print(f'📊 Precision: {precision:.3f}')
            print(f'📊 Recall: {recall:.3f}')
            print(f'💾 Đã lưu vào CSV')
            print('-'*50)
            
        except Exception as e:
            print(f'\n❌ Lỗi khi xử lý sample {i}: {e}')
            print(f'   Bỏ qua và tiếp tục...')
            print('-'*50)
            continue
    
    # Tính trung bình
    if avg_precision:
        avg_precision = np.mean(avg_precision)
        avg_recall = np.mean(avg_recall)
        
        print("\n" + "="*50)
        print("📈 KẾT QUẢ TỔNG HỢP:")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print("="*50)
        
        # Đọc lại toàn bộ kết quả từ CSV để tính summary
        if os.path.exists(logger.csv_path):
            df_all = pd.read_csv(logger.csv_path)
            logger.results = df_all.to_dict('records')
            
            # Lưu summary vào txt
            logger.save_summary_to_txt(txt_path='rag_metrics_summary.txt', mode='append')
            
            # In summary
            summary = logger.get_summary()
            print("\n📊 SUMMARY:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
    else:
        print("\n⚠️ Không có kết quả nào được xử lý thành công")