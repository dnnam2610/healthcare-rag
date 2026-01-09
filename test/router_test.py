import json
import csv
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
from datetime import datetime


class RoutingTester:
    """
    This class is used to eval router in RAG system.
    We compute F1, Accuracy, Precision, Recall, and Confusion matrix
    """
    
    def __init__(self):
        """
        Initialize the RoutingTester.
        
        Args:
            batch_size: Number of samples to process in each batch
        """
        self.results = {}
        self.incorrect_samples = []
        
    def _process_batch(self, router, batch: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples.
        
        Args:
            router: The router object with a predict/route method
            batch: List of (query, true_label) tuples
            
        Returns:
            List of prediction results
        """
        predictions = []
        
        queries, true_labels = zip(*batch)
        queries = list(queries)
        true_labels = list(true_labels)

        try:
            # Assuming router has a predict or route method
            _, pred_labels = zip(*router.batch_guide(queries))
            for query, pred_label, true_label in zip(queries, pred_labels, true_labels):
                predictions.append({
                    'query': query,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'correct': pred_label == true_label
                })
        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            for query, true_label in zip(queries, true_labels):
                predictions.append({
                    "query": query,
                    "true_label": true_label,
                    "predicted_label": None,
                    "correct": False,
                    "error": error_msg
                })
            
        return predictions
    
    def _compute_metrics(self, all_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Args:
            all_predictions: List of all prediction results
            
        Returns:
            Dictionary containing all metrics
        """
        # Extract labels
        true_labels = [p['true_label'] for p in all_predictions]
        pred_labels = [p['predicted_label'] for p in all_predictions]
        
        # Get unique labels
        unique_labels = sorted(set(true_labels + [l for l in pred_labels if l is not None]))
        
        # Compute per-class metrics with TP, FP, TN, FN
        per_class_metrics = {}
        for label in unique_labels:
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)
            tn = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[label] = {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': int(tp + fn)
            }
        
        # Compute overall metrics
        total_correct = sum(p['correct'] for p in all_predictions)
        total_samples = len(all_predictions)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Macro-averaged metrics
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
        
        # Weighted-averaged metrics
        total_support = sum(m['support'] for m in per_class_metrics.values())
        weighted_precision = sum(m['precision'] * m['support'] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0
        weighted_recall = sum(m['recall'] * m['support'] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0
        weighted_f1 = sum(m['f1'] * m['support'] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0
        
        return {
            'accuracy': round(accuracy, 4),
            'macro_avg': {
                'precision': round(macro_precision, 4),
                'recall': round(macro_recall, 4),
                'f1': round(macro_f1, 4)
            },
            'weighted_avg': {
                'precision': round(weighted_precision, 4),
                'recall': round(weighted_recall, 4),
                'f1': round(weighted_f1, 4)
            },
            'per_class': per_class_metrics,
            'total_samples': total_samples,
            'correct_predictions': total_correct,
            'incorrect_predictions': total_samples - total_correct
        }
    
    def eval(self, 
             router, 
             test_data: List[Tuple[str, str]], 
             output_dir:str = "./router_results",
             batch_size:int = 32
            ) -> Dict[str, Any]:
        """
        Evaluate the router on test data.
        
        Args:
            router: Router object to evaluate
            test_data: List of (query, true_label) tuples
            output_file: Path to save evaluation results (JSON)
            predictions_csv: Path to save all predictions (CSV)
            incorrect_file: Path to save incorrect predictions (JSON)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"Starting evaluation on {len(test_data)} samples...")
        print(f"Batch size: {batch_size}")
        
        # Create batches
        batches = [test_data[i:i + batch_size] 
                   for i in range(0, len(test_data), batch_size)]
        
        self.all_predictions = []
        
        # Sequential processing with progress bar
        for batch in tqdm(batches, desc="Processing batches"):
            batch_predictions = self._process_batch(router, batch)
            self.all_predictions.extend(batch_predictions)
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = self._compute_metrics(self.all_predictions)
        
        # Find incorrect predictions
        self.incorrect_samples = [p for p in self.all_predictions if not p['correct']]
        
        # Prepare results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'configuration': {
                'batch_size': batch_size
            }
        }

        self._save_results(output_dir=output_dir)
        
        # Print summary
        self._print_summary(metrics)
        
        return self.results
    

    def _save_results(self, output_dir):
        """
        Lưu kết quả ra JSON, CSV, và JSON riêng cho các mẫu sai.
        """
        # === Tạo thư mục đầu ra nếu chưa tồn tại ===
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created output directory: {output_dir}")
        else:
            print(f"📁 Using existing directory: {output_dir}")

        # === Tạo tên file theo thời gian ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"results_{timestamp}.json")
        predictions_csv = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        incorrect_file = os.path.join(output_dir, f"incorrect_{timestamp}.json")

        # === Lưu kết quả chính ===
        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # === Lưu toàn bộ dự đoán ===
        print(f"📄 Saving all predictions to {predictions_csv}...")
        with open(predictions_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query', 'true_label', 'predicted_label'])
            for p in self.all_predictions:   # đảm bảo bạn có self.all_predictions
                writer.writerow([p['query'], p['true_label'], p['predicted_label']])

        # === Lưu mẫu sai ===
        if hasattr(self, 'incorrect_samples') and self.incorrect_samples:
            print(f"⚠️ Saving {len(self.incorrect_samples)} incorrect predictions to {incorrect_file}...")
            incorrect_data = {
                'timestamp': datetime.now().isoformat(),
                'total_incorrect': len(self.incorrect_samples),
                'samples': self.incorrect_samples
            }
            with open(incorrect_file, 'w', encoding='utf-8') as f:
                json.dump(incorrect_data, f, indent=2, ensure_ascii=False)

        print("✅ All results saved successfully.")
        return
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nMacro Average:")
        print(f"  Precision: {metrics['macro_avg']['precision']:.4f}")
        print(f"  Recall:    {metrics['macro_avg']['recall']:.4f}")
        print(f"  F1 Score:  {metrics['macro_avg']['f1']:.4f}")
        print(f"\nWeighted Average:")
        print(f"  Precision: {metrics['weighted_avg']['precision']:.4f}")
        print(f"  Recall:    {metrics['weighted_avg']['recall']:.4f}")
        print(f"  F1 Score:  {metrics['weighted_avg']['f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for label, values in metrics['per_class'].items():
            print(f"\n  {label}:")
            print(f"    TP: {values['tp']}, FP: {values['fp']}, TN: {values['tn']}, FN: {values['fn']}")
            print(f"    Precision: {values['precision']:.4f}")
            print(f"    Recall:    {values['recall']:.4f}")
            print(f"    F1 Score:  {values['f1']:.4f}")
            print(f"    Support:   {values['support']}")
        
        print(f"\nTotal Samples: {metrics['total_samples']}")
        print(f"Correct: {metrics['correct_predictions']}")
        print(f"Incorrect: {metrics['incorrect_predictions']}")
        print("="*60)

# Example usage:
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from embedders import FlagBaseEmbedding, EmbeddingConfig, GeminiEmbedding, APIEmbeddingConfig, SentenceTransformerEmbedding
    from senmatic_router import SemanticRouter
    from config import EMBEDDING_MODEL_NAME
    # Test data
    MED_ROUTE_NAME = 'medical'
    NON_MED_ROUTE_NAME = 'non_medical'
    med_path = 'data/router/testData/test_med.json'
    non_med_path = 'data/router/testData/test_non_med.json'

    with open(med_path, 'r') as f:
        med_data = json.load(f)
    with open(non_med_path, 'r') as f:
        non_med_data = json.load(f)[:2000]

    test_samples = [(q, MED_ROUTE_NAME) for q in med_data]
    test_samples.extend(
        [(q, NON_MED_ROUTE_NAME) for q in non_med_data]
    )

    # ============= API =============
    # config = APIEmbeddingConfig(
    #     name='gemini-embedding-001',
    #     apiKey='AIzaSyAYOLcSrYq1CNJHNo42u0x6Decf7g3QB_s'
    # )
    # gemini = GeminiEmbedding(
    #     config=config
    # )
    # router = SemanticRouter(
    #     embedding=gemini,
    #     save_path='data/router/routingEmbedddings/visbert_routing_embedding_1000.json'
    # )

    # ============= ST =============
    config = EmbeddingConfig(
        name='keepitreal/vietnamese-sbert',
    )
    st = SentenceTransformerEmbedding(
        config=config
    )
    router = SemanticRouter(
        embedding=st,
        save_path='data/router/routingEmbedddings/visbert_routing_embedding_1000.json'
    )
    # Run evaluation
    tester = RoutingTester()
    results = tester.eval(
        router=router,
        test_data=test_samples,
        output_dir='test/routerResults/visbert_1000',
        batch_size=8
    )