import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv
from video_embeddings import VideoEmbeddingManager
from sacrebleu import BLEU
from sacrebleu.metrics import BLEU, CHRF, TER

# Load environment variables
load_dotenv()

# Mock metrics for evaluation
class MockAnswerRelevancyMetric:
    def score(self, query, answer):
        # Simple mock implementation that returns a score between 0.7 and 1.0
        # based on keyword matching
        keywords = query.lower().split()
        answer_lower = answer.lower()
        matches = sum(1 for keyword in keywords if keyword in answer_lower)
        base_score = 0.7 + (matches / len(keywords)) * 0.3
        return min(1.0, base_score)

class MockFaithfulnessMetric:
    def score(self, query, answer, sources):
        # Simple mock implementation that returns a score between 0.7 and 1.0
        # based on whether the answer appears in the sources
        if not sources:
            return 0.7
        
        source_texts = [s.get("text", "") for s in sources]
        answer_lower = answer.lower()
        
        # Check if the answer is contained in any of the sources
        for source in source_texts:
            if answer_lower in source.lower():
                return 0.9
        
        # If not found exactly, check for partial matches
        for source in source_texts:
            if any(word in source.lower() for word in answer_lower.split()[:5]):
                return 0.8
        
        return 0.7

class MockContextualRelevancyMetric:
    def score(self, query, sources, answer):
        # Simple mock implementation that returns a score between 0.7 and 1.0
        # based on keyword matching between query and sources
        if not sources:
            return 0.7
        
        keywords = query.lower().split()
        source_texts = [s.get("text", "") for s in sources]
        
        # Check if keywords appear in sources
        matches = 0
        for keyword in keywords:
            for source in source_texts:
                if keyword in source.lower():
                    matches += 1
                    break
        
        base_score = 0.7 + (matches / len(keywords)) * 0.3
        return min(1.0, base_score)

class SystemEvaluator:
    """Class to evaluate the performance of the VideoEmbeddingManager system."""
    
    def __init__(self, video_manager: VideoEmbeddingManager):
        """Initialize the evaluator with a VideoEmbeddingManager instance."""
        self.video_manager = video_manager
        self.results = {}
    
    def evaluate_transcript_extraction(self, video_urls: List[str]) -> Dict[str, Any]:
        """Evaluate the transcript extraction functionality."""
        results = {
            "total_videos": len(video_urls),
            "successful_extractions": 0,
            "failed_extractions": 0,
            "extraction_times": [],
            "errors": []
        }
        
        for url in tqdm(video_urls, desc="Evaluating transcript extraction"):
            start_time = time.time()
            video_id = self.video_manager.extract_video_id(url)
            
            if not video_id:
                results["failed_extractions"] += 1
                results["errors"].append(f"Failed to extract video ID from {url}")
                continue
                
            transcript = self.video_manager.extract_transcript(video_id)
            extraction_time = time.time() - start_time
            results["extraction_times"].append(extraction_time)
            
            if transcript:
                results["successful_extractions"] += 1
            else:
                results["failed_extractions"] += 1
                results["errors"].append(f"Failed to extract transcript for video ID {video_id}")
        
        # Calculate statistics
        results["success_rate"] = results["successful_extractions"] / results["total_videos"] if results["total_videos"] > 0 else 0
        results["avg_extraction_time"] = np.mean(results["extraction_times"]) if results["extraction_times"] else 0
        
        self.results["transcript_extraction"] = results
        return results
    
    def evaluate_text_processing(self, sample_texts: List[str]) -> Dict[str, Any]:
        """Evaluate the text processing functionality."""
        results = {
            "total_samples": len(sample_texts),
            "processing_times": [],
            "token_counts": [],
            "named_entities": [],
            "pos_tag_distributions": []
        }
        
        for text in tqdm(sample_texts, desc="Evaluating text processing"):
            start_time = time.time()
            processed_text, metadata = self.video_manager.preprocess_text(text)
            processing_time = time.time() - start_time
            
            results["processing_times"].append(processing_time)
            results["token_counts"].append(len(processed_text.split()))
            results["named_entities"].append(metadata["named_entities"])
            results["pos_tag_distributions"].append(metadata["top_pos_tags"])
        
        # Calculate statistics
        results["avg_processing_time"] = np.mean(results["processing_times"]) if results["processing_times"] else 0
        results["avg_token_count"] = np.mean(results["token_counts"]) if results["token_counts"] else 0
        
        self.results["text_processing"] = results
        return results
    
    def evaluate_embedding_quality(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Evaluate the quality of embeddings."""
        results = {
            "embedding_dimension": embeddings.shape[1] if embeddings is not None else 0,
            "num_embeddings": len(embeddings) if embeddings is not None else 0,
            "embedding_diversity": 0,
            "clustering_quality": 0
        }
        
        if embeddings is not None and len(embeddings) > 0:
            # Calculate embedding diversity using cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
            avg_similarity = np.mean(similarity_matrix)
            results["embedding_diversity"] = 1 - avg_similarity  # Higher is better
            
            # Evaluate clustering quality if we have enough embeddings
            if len(embeddings) > 10:
                try:
                    # Use K-means to cluster embeddings
                    n_clusters = min(5, len(embeddings) // 2)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    # Calculate silhouette score (higher is better)
                    silhouette_avg = silhouette_score(embeddings, cluster_labels)
                    results["clustering_quality"] = silhouette_avg
                except Exception as e:
                    results["clustering_quality"] = 0
                    print(f"Error in clustering evaluation: {e}")
        
        self.results["embedding_quality"] = results
        return results
    
    def evaluate_retrieval_performance(self, query_texts: List[str], 
                                      ground_truth: List[List[int]], 
                                      k: int = 5) -> Dict[str, Any]:
        """Evaluate retrieval performance using precision@k, recall@k, and MRR."""
        results = {
            "precision_at_k": [],
            "recall_at_k": [],
            "mrr": [],
            "retrieval_times": []
        }
        
        for i, (query, relevant_docs) in enumerate(tqdm(zip(query_texts, ground_truth), 
                                                      desc="Evaluating retrieval performance")):
            start_time = time.time()
            
            # Process the query
            processed_query, _ = self.video_manager.preprocess_text(query)
            query_embedding = self.video_manager.get_embedding(processed_query)
            
            # Search in Pinecone
            search_results = self.video_manager.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            retrieval_time = time.time() - start_time
            results["retrieval_times"].append(retrieval_time)
            
            # Extract retrieved document IDs
            retrieved_docs = []
            for match in search_results.matches:
                # Extract chunk_id from the ID (format: video_id_chunk_id)
                chunk_id = int(match.id.split('_')[-1])
                retrieved_docs.append(chunk_id)
            
            # Calculate precision@k
            relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
            precision = len(relevant_retrieved) / k if k > 0 else 0
            results["precision_at_k"].append(precision)
            
            # Calculate recall@k
            recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
            results["recall_at_k"].append(recall)
            
            # Calculate MRR
            for rank, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    results["mrr"].append(1.0 / rank)
                    break
            else:
                results["mrr"].append(0.0)
        
        # Calculate average metrics
        results["avg_precision_at_k"] = np.mean(results["precision_at_k"]) if results["precision_at_k"] else 0
        results["avg_recall_at_k"] = np.mean(results["recall_at_k"]) if results["recall_at_k"] else 0
        results["avg_mrr"] = np.mean(results["mrr"]) if results["mrr"] else 0
        results["avg_retrieval_time"] = np.mean(results["retrieval_times"]) if results["retrieval_times"] else 0
        
        self.results["retrieval_performance"] = results
        return results
    
    def evaluate_system_performance(self, video_urls: List[str]) -> Dict[str, Any]:
        """Evaluate overall system performance including processing time and storage efficiency."""
        results = {
            "total_videos": len(video_urls),
            "processing_times": [],
            "embedding_generation_times": [],
            "storage_sizes": []
        }
        
        for url in tqdm(video_urls, desc="Evaluating system performance"):
            start_time = time.time()
            
            # Process video
            success = self.video_manager.process_video(url)
            
            if success:
                processing_time = time.time() - start_time
                results["processing_times"].append(processing_time)
                
                # Get video ID for storage size estimation
                video_id = self.video_manager.extract_video_id(url)
                if video_id:
                    # Query Pinecone to get the number of vectors for this video
                    query_vector = np.random.rand(768).tolist()  # Dummy query
                    search_results = self.video_manager.index.query(
                        vector=query_vector,
                        top_k=1000,
                        filter={"video_id": video_id},
                        include_metadata=False
                    )
                    
                    # Estimate storage size (each embedding is 768 dimensions * 4 bytes)
                    num_vectors = len(search_results.matches)
                    storage_size = num_vectors * 768 * 4  # in bytes
                    results["storage_sizes"].append(storage_size)
        
        # Calculate statistics
        results["avg_processing_time"] = np.mean(results["processing_times"]) if results["processing_times"] else 0
        results["avg_storage_size"] = np.mean(results["storage_sizes"]) if results["storage_sizes"] else 0
        
        self.results["system_performance"] = results
        return results
    
    def generate_evaluation_report(self, output_file: str = "evaluation_report.json") -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        report = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "model_name": self.video_manager.model.get_sentence_embedding_dimension(),
                "device": self.video_manager.device,
                "pinecone_index": os.getenv('PINECONE_INDEX_NAME', 'embeddings')
            },
            "results": self.results,
            "summary": {
                "transcript_extraction_success_rate": self.results.get("transcript_extraction", {}).get("success_rate", 0),
                "avg_text_processing_time": self.results.get("text_processing", {}).get("avg_processing_time", 0),
                "embedding_diversity": self.results.get("embedding_quality", {}).get("embedding_diversity", 0),
                "clustering_quality": self.results.get("embedding_quality", {}).get("clustering_quality", 0),
                "avg_precision_at_k": self.results.get("retrieval_performance", {}).get("avg_precision_at_k", 0),
                "avg_recall_at_k": self.results.get("retrieval_performance", {}).get("avg_recall_at_k", 0),
                "avg_mrr": self.results.get("retrieval_performance", {}).get("avg_mrr", 0),
                "avg_system_processing_time": self.results.get("system_performance", {}).get("avg_processing_time", 0)
            }
        }
        
        # Save report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def run_evaluation(video_urls: List[str], query_texts: List[str] = None, 
                  ground_truth: List[List[int]] = None, output_file: str = "evaluation_report.json"):
    """Run a complete evaluation of the system."""
    # Initialize video manager
    video_manager = VideoEmbeddingManager()
    evaluator = SystemEvaluator(video_manager)
    
    # Run evaluations
    evaluator.evaluate_transcript_extraction(video_urls)
    
    # Sample some texts for text processing evaluation
    sample_texts = []
    for url in video_urls[:min(5, len(video_urls))]:  # Use up to 5 videos
        video_id = video_manager.extract_video_id(url)
        if video_id:
            transcript = video_manager.extract_transcript(video_id)
            if transcript:
                chunks = video_manager.chunk_text(transcript, max_chunks=5)
                sample_texts.extend(chunks)
    
    evaluator.evaluate_text_processing(sample_texts)
    
    # Generate embeddings for a sample video for embedding quality evaluation
    if video_urls:
        video_id = video_manager.extract_video_id(video_urls[0])
        if video_id:
            embeddings, _, _ = video_manager.generate_embeddings(video_id)
            evaluator.evaluate_embedding_quality(embeddings)
    
    # Evaluate retrieval performance if query texts and ground truth are provided
    if query_texts and ground_truth:
        evaluator.evaluate_retrieval_performance(query_texts, ground_truth)
    
    # Evaluate system performance
    evaluator.evaluate_system_performance(video_urls)
    
    # Generate and return the evaluation report
    return evaluator.generate_evaluation_report(output_file)

def evaluate_rag_performance(system, ground_truth_dict):
    """Evaluate RAG system performance using mock metrics."""
    results = []
    
    # Use mock metrics instead of DeepEval metrics
    relevance_metric = MockAnswerRelevancyMetric()
    faithfulness_metric = MockFaithfulnessMetric()
    contextual_relevance_metric = MockContextualRelevancyMetric()
    
    # Iterate through the ground truth dictionary
    for query, expected_answer in ground_truth_dict.items():
        # Generate answer using the system
        answer, sources, _ = system.process_query(query)
        
        # Calculate scores using mock metrics
        relevance_score = relevance_metric.score(query, answer)
        faithfulness_score = faithfulness_metric.score(query, answer, sources)
        contextual_relevance_score = contextual_relevance_metric.score(query, sources, answer)
        
        results.append({
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "relevance_score": relevance_score,
            "faithfulness_score": faithfulness_score,
            "contextual_relevance_score": contextual_relevance_score
        })
    
    return results

def evaluate_text_quality(system, ground_truth_dict):
    """Evaluate text generation quality using SacreBLEU metrics."""
    results = []
    
    # Initialize metrics
    bleu = BLEU()
    chrf = CHRF()
    ter = TER()
    
    # Iterate through the ground truth dictionary
    for query, expected_answer in ground_truth_dict.items():
        # Generate answer using the system
        answer, _, _ = system.process_query(query)
        
        # Calculate BLEU score
        bleu_score = bleu.corpus_score([answer], [[expected_answer]])
        
        # Calculate CHRF score (character n-gram F-score)
        chrf_score = chrf.corpus_score([answer], [[expected_answer]])
        
        # Calculate TER score (Translation Edit Rate)
        ter_score = ter.corpus_score([answer], [[expected_answer]])
        
        results.append({
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "bleu_score": bleu_score.score,
            "chrf_score": chrf_score.score,
            "ter_score": ter_score.score
        })
    
    return results

def comprehensive_evaluation(system, test_queries_list, ground_truth_dict):
    """Comprehensive evaluation of the RAG system."""
    # Evaluate RAG performance - Pass the ground_truth dictionary
    rag_results = evaluate_rag_performance(system, ground_truth_dict)
    
    # Evaluate text quality - Pass the ground_truth dictionary
    text_quality_results = evaluate_text_quality(system, ground_truth_dict)
    
    # Combine results
    combined_results = []
    # Ensure the lengths match before combining
    if len(rag_results) != len(text_quality_results):
        print("Warning: Mismatch in length between RAG results and text quality results.")
        # Handle mismatch appropriately, e.g., log error or use the shorter length
        min_len = min(len(rag_results), len(text_quality_results))
    else:
        min_len = len(rag_results)

    for i in range(min_len):
        rag_result = rag_results[i]
        text_result = text_quality_results[i]
        # ... rest of the combining logic ...
        combined_result = {
            "query": rag_result["query"],
            "answer": rag_result["answer"],
            "expected_answer": rag_result["expected_answer"],
            "relevance_score": rag_result["relevance_score"],
            "faithfulness_score": rag_result["faithfulness_score"],
            "contextual_relevance_score": rag_result["contextual_relevance_score"],
            "bleu_score": text_result["bleu_score"],
            "chrf_score": text_result["chrf_score"],
            "ter_score": text_result["ter_score"]
        }
        combined_results.append(combined_result)
    
    # Calculate aggregate metrics
    # Check if combined_results is empty before calculating averages
    if not combined_results:
        print("Warning: No results to aggregate.")
        aggregate_metrics = {
            "avg_relevance": 0,
            "avg_faithfulness": 0,
            "avg_contextual_relevance": 0,
            "avg_bleu": 0,
            "avg_chrf": 0,
            "avg_ter": 0
        }
    else:
        aggregate_metrics = {
            "avg_relevance": sum(r["relevance_score"] for r in combined_results) / len(combined_results),
            "avg_faithfulness": sum(r["faithfulness_score"] for r in combined_results) / len(combined_results),
            "avg_contextual_relevance": sum(r["contextual_relevance_score"] for r in combined_results) / len(combined_results),
            "avg_bleu": sum(r["bleu_score"] for r in combined_results) / len(combined_results),
            "avg_chrf": sum(r["chrf_score"] for r in combined_results) / len(combined_results),
            "avg_ter": sum(r["ter_score"] for r in combined_results) / len(combined_results)
        }
    
    return {
        "detailed_results": combined_results,
        "aggregate_metrics": aggregate_metrics
    }

if __name__ == "__main__":
    # Example usage
    test_videos = [
        "https://www.youtube.com/watch?v=NUy_wOxOM8E",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ]
    
    # Example queries and ground truth for retrieval evaluation
    test_queries = {
        "What is machine learning?": "Machine learning is a subset of artificial intelligence that involves training models to make predictions or decisions based on data.",
        "How does neural networks work?": "Neural networks are a set of algorithms modeled after the human brain, designed to recognize patterns."
    }
    
    # Ground truth: list of relevant chunk IDs for each query
    # This would typically come from human annotation
    test_ground_truth = {
        "What is machine learning?": [0, 2, 5],  # Relevant chunk IDs for first query
        "How does neural networks work?": [1, 3, 4]   # Relevant chunk IDs for second query
    }
    
    report = run_evaluation(test_videos, test_queries, test_ground_truth)
    print(f"Evaluation complete. Report saved to evaluation_report.json")