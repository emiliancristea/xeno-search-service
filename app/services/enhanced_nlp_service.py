import asyncio
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import functools
import time
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    pipeline, set_seed, BartTokenizer, BartForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import structlog
from langdetect import detect
import logging

from app.core.config import get_config
from app.core.monitoring import track_operation, get_monitor
from app.core.cache import get_cache_manager
from app.models.search_models import ContentCategory

logger = structlog.get_logger(__name__)

# Suppress some transformer warnings
logging.getLogger("transformers").setLevel(logging.WARNING)


@dataclass
class SummarizationResult:
    """Result of summarization operation"""
    summary: str
    confidence_score: float
    processing_time_ms: int
    model_used: str
    original_length: int
    summary_length: int
    compression_ratio: float


@dataclass
class ClassificationResult:
    """Result of content classification"""
    category: ContentCategory
    confidence: float
    all_scores: Dict[ContentCategory, float]


class EnhancedNLPService:
    """
    Enhanced NLP service with multiple models and advanced capabilities
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.monitor = get_monitor()
        
        # Model instances (lazy loaded)
        self.summarization_model = None
        self.summarization_tokenizer = None
        self.classification_model = None
        self.embedding_model = None
        self.spacy_nlp = None
        
        # Model configurations
        self.nlp_config = self.config.get_nlp_config()
        self.device = self._get_device()
        
        # Cache for expensive operations
        self.model_cache = {}
        
        # TF-IDF for content analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        logger.info("Enhanced NLP Service initialized", device=self.device)
    
    def _get_device(self) -> str:
        """Determine the best device for model inference"""
        device_config = self.nlp_config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
            else:
                return 'cpu'
        else:
            return device_config
    
    async def _load_summarization_model(self):
        """Lazily load summarization model"""
        if self.summarization_model is not None:
            return
        
        try:
            model_name = self.nlp_config['summarization_model']
            logger.info("Loading summarization model", model=model_name, device=self.device)
            
            # Load in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def load_model():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                if self.device != 'cpu':
                    model = model.to(self.device)
                
                return tokenizer, model
            
            self.summarization_tokenizer, self.summarization_model = await loop.run_in_executor(
                None, load_model
            )
            
            logger.info("Summarization model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load summarization model", error=str(e))
            # Fallback to pipeline
            self.summarization_model = "pipeline_fallback"
    
    async def _load_embedding_model(self):
        """Lazily load sentence transformer model"""
        if self.embedding_model is not None:
            return
        
        try:
            model_name = self.nlp_config['embedding_model']
            logger.info("Loading embedding model", model=model_name)
            
            loop = asyncio.get_event_loop()
            
            def load_model():
                return SentenceTransformer(model_name)
            
            self.embedding_model = await loop.run_in_executor(None, load_model)
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            self.embedding_model = None
    
    async def _load_classification_model(self):
        """Load content classification model"""
        if self.classification_model is not None:
            return
        
        try:
            # Using a general text classification model
            # In production, this would be fine-tuned for content categories
            model_name = "microsoft/DialoGPT-medium"  # Placeholder
            
            logger.info("Loading classification model")
            
            # For now, we'll use a simpler rule-based classification
            # In production, this would be a proper ML model
            self.classification_model = "rule_based"
            
        except Exception as e:
            logger.error("Failed to load classification model", error=str(e))
            self.classification_model = None
    
    async def _load_spacy_model(self):
        """Load spaCy model for NER and advanced text processing"""
        if self.spacy_nlp is not None:
            return
        
        try:
            loop = asyncio.get_event_loop()
            
            def load_spacy():
                try:
                    return spacy.load("en_core_web_sm")
                except OSError:
                    # Fallback to blank model if language model not installed
                    logger.warning("spaCy language model not found, using blank model")
                    return spacy.blank("en")
            
            self.spacy_nlp = await loop.run_in_executor(None, load_spacy)
            logger.info("spaCy model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load spaCy model", error=str(e))
            self.spacy_nlp = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better model performance"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.!?]{3,}', '...', text)
        
        # Clean up
        text = text.strip()
        
        return text
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text"""
        try:
            if len(text) < 50:  # Too short for reliable detection
                return None
            return detect(text)
        except:
            return None
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            # Create a simple corpus with the text
            corpus = [text]
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get scores for the document
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score and get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw for kw, score in keyword_scores[:max_keywords] if score > 0]
            
        except Exception as e:
            logger.warning("Keyword extraction failed", error=str(e))
            return []
    
    @track_operation("advanced_summarization")
    async def summarize_text_advanced(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        strategy: str = "balanced"  # "extractive", "abstractive", "balanced"
    ) -> Optional[SummarizationResult]:
        """
        Advanced text summarization with multiple strategies
        """
        if not text or len(text.strip()) < 100:
            logger.warning("Text too short for summarization")
            return None
        
        start_time = time.time()
        
        # Load model if needed
        await self._load_summarization_model()
        
        if self.summarization_model is None:
            logger.error("Summarization model not available")
            return None
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            original_length = len(processed_text)
            
            # Set default lengths based on text length
            if max_length is None:
                max_length = min(self.nlp_config['max_length'], max(50, original_length // 4))
            if min_length is None:
                min_length = min(self.nlp_config['min_length'], max_length // 3)
            
            # Check cache first
            cache_manager = await get_cache_manager()
            content_hash = hashlib.sha256(processed_text.encode()).hexdigest()
            cache_key = f"summary_{strategy}_{max_length}_{min_length}_{content_hash[:16]}"
            
            cached_result = await cache_manager.get_summary(cache_key)
            if cached_result:
                return cached_result
            
            summary = None
            confidence_score = 0.0
            model_used = "unknown"
            
            if self.summarization_model == "pipeline_fallback":
                # Use HuggingFace pipeline as fallback
                summary, confidence_score, model_used = await self._summarize_with_pipeline(
                    processed_text, max_length, min_length
                )
            else:
                # Use loaded model
                summary, confidence_score, model_used = await self._summarize_with_model(
                    processed_text, max_length, min_length, strategy
                )
            
            if not summary:
                return None
            
            # Calculate metrics
            processing_time_ms = int((time.time() - start_time) * 1000)
            summary_length = len(summary)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            result = SummarizationResult(
                summary=summary,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                model_used=model_used,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio
            )
            
            # Cache result
            await cache_manager.set_summary(cache_key, result)
            
            logger.info("Advanced summarization completed",
                       original_length=original_length,
                       summary_length=summary_length,
                       compression_ratio=compression_ratio,
                       processing_time_ms=processing_time_ms)
            
            return result
            
        except Exception as e:
            logger.error("Advanced summarization failed", error=str(e))
            self.monitor.record_error("advanced_summarization", str(e))
            return None
    
    async def _summarize_with_model(
        self,
        text: str,
        max_length: int,
        min_length: int,
        strategy: str
    ) -> Tuple[Optional[str], float, str]:
        """Summarize using the loaded transformer model"""
        try:
            loop = asyncio.get_event_loop()
            
            def generate_summary():
                # Tokenize input
                inputs = self.summarization_tokenizer(
                    text,
                    max_length=1024,  # Model's max input length
                    truncation=True,
                    return_tensors="pt"
                )
                
                if self.device != 'cpu':
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate summary
                with torch.no_grad():
                    summary_ids = self.summarization_model.generate(
                        inputs["input_ids"],
                        max_length=max_length + len(inputs["input_ids"][0]),
                        min_length=min_length + len(inputs["input_ids"][0]),
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                
                # Decode summary
                summary = self.summarization_tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                return summary
            
            summary = await loop.run_in_executor(None, generate_summary)
            
            if summary:
                # Calculate confidence based on length and quality heuristics
                confidence = min(1.0, len(summary) / max_length)
                model_name = self.nlp_config['summarization_model']
                return summary, confidence, model_name
            
            return None, 0.0, "failed"
            
        except Exception as e:
            logger.error("Model-based summarization failed", error=str(e))
            return None, 0.0, "error"
    
    async def _summarize_with_pipeline(
        self,
        text: str,
        max_length: int,
        min_length: int
    ) -> Tuple[Optional[str], float, str]:
        """Fallback summarization using HuggingFace pipeline"""
        try:
            loop = asyncio.get_event_loop()
            
            def create_pipeline_and_summarize():
                summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if self.device == 'cuda' else -1
                )
                
                result = summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                
                return result[0]['summary_text'] if result else None
            
            summary = await loop.run_in_executor(None, create_pipeline_and_summarize)
            
            if summary:
                confidence = 0.8  # Pipeline generally reliable
                return summary, confidence, "pipeline_bart"
            
            return None, 0.0, "pipeline_failed"
            
        except Exception as e:
            logger.error("Pipeline summarization failed", error=str(e))
            return None, 0.0, "pipeline_error"
    
    @track_operation("content_classification")
    async def classify_content(self, text: str, title: str = "") -> Optional[ClassificationResult]:
        """
        Classify content into categories
        """
        if not text:
            return None
        
        await self._load_classification_model()
        
        try:
            # For now, use rule-based classification
            # In production, this would use a trained ML model
            combined_text = f"{title} {text}".lower()
            
            # Define keywords for each category
            category_keywords = {
                ContentCategory.NEWS: ['breaking', 'news', 'report', 'journalist', 'press', 'article'],
                ContentCategory.ACADEMIC: ['research', 'study', 'university', 'journal', 'academic', 'paper', 'thesis'],
                ContentCategory.BLOG: ['blog', 'opinion', 'thoughts', 'personal', 'diary'],
                ContentCategory.DOCUMENTATION: ['documentation', 'manual', 'guide', 'tutorial', 'howto', 'api'],
                ContentCategory.FORUM: ['forum', 'discussion', 'thread', 'reply', 'post', 'community'],
                ContentCategory.SOCIAL: ['twitter', 'facebook', 'instagram', 'social', 'share'],
                ContentCategory.COMMERCIAL: ['buy', 'sell', 'price', 'product', 'shop', 'store', 'commerce'],
                ContentCategory.GOVERNMENT: ['government', 'official', 'policy', 'law', 'regulation', '.gov'],
                ContentCategory.REFERENCE: ['wikipedia', 'encyclopedia', 'reference', 'definition']
            }
            
            scores = {}
            
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in combined_text)
                scores[category] = score / len(keywords)  # Normalize
            
            # Find best category
            if scores:
                best_category = max(scores, key=scores.get)
                best_score = scores[best_category]
                
                if best_score > 0:
                    return ClassificationResult(
                        category=best_category,
                        confidence=min(1.0, best_score * 2),  # Scale confidence
                        all_scores=scores
                    )
            
            # Default to OTHER if no clear category
            return ClassificationResult(
                category=ContentCategory.OTHER,
                confidence=0.5,
                all_scores=scores
            )
            
        except Exception as e:
            logger.error("Content classification failed", error=str(e))
            return None
    
    @track_operation("semantic_similarity")
    async def calculate_semantic_similarity(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """
        Calculate semantic similarity between query and texts
        """
        if not query or not texts:
            return []
        
        await self._load_embedding_model()
        
        if not self.embedding_model:
            logger.warning("Embedding model not available for semantic similarity")
            return [0.0] * len(texts)
        
        try:
            loop = asyncio.get_event_loop()
            
            def calculate_similarities():
                # Encode query and texts
                query_embedding = self.embedding_model.encode([query])
                text_embeddings = self.embedding_model.encode(texts)
                
                # Calculate cosine similarities
                similarities = cosine_similarity(query_embedding, text_embeddings)[0]
                
                return similarities.tolist()
            
            similarities = await loop.run_in_executor(None, calculate_similarities)
            
            logger.info("Semantic similarity calculated",
                       query_length=len(query),
                       num_texts=len(texts))
            
            return similarities
            
        except Exception as e:
            logger.error("Semantic similarity calculation failed", error=str(e))
            return [0.0] * len(texts)
    
    @track_operation("extract_entities")
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        """
        if not text:
            return {}
        
        await self._load_spacy_model()
        
        if not self.spacy_nlp:
            return {}
        
        try:
            loop = asyncio.get_event_loop()
            
            def extract():
                doc = self.spacy_nlp(text[:10000])  # Limit text length for performance
                
                entities = {}
                for ent in doc.ents:
                    label = ent.label_
                    if label not in entities:
                        entities[label] = []
                    entities[label].append(ent.text)
                
                # Deduplicate
                for label in entities:
                    entities[label] = list(set(entities[label]))
                
                return entities
            
            entities = await loop.run_in_executor(None, extract)
            
            logger.info("Entity extraction completed",
                       num_entity_types=len(entities),
                       total_entities=sum(len(ents) for ents in entities.values()))
            
            return entities
            
        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return {}
    
    async def analyze_text_comprehensively(
        self,
        text: str,
        title: str = "",
        include_summary: bool = True,
        include_classification: bool = True,
        include_entities: bool = True,
        include_keywords: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis
        """
        results = {
            'text_length': len(text),
            'language': self._detect_language(text),
            'processing_timestamp': time.time()
        }
        
        # Run analyses concurrently
        tasks = []
        
        if include_summary and len(text) > 200:
            tasks.append(('summary', self.summarize_text_advanced(text)))
        
        if include_classification:
            tasks.append(('classification', self.classify_content(text, title)))
        
        if include_entities:
            tasks.append(('entities', self.extract_entities(text)))
        
        # Execute tasks
        for task_name, task in tasks:
            try:
                result = await task
                results[task_name] = result
            except Exception as e:
                logger.error(f"{task_name} analysis failed", error=str(e))
                results[task_name] = None
        
        # Extract keywords (synchronous)
        if include_keywords:
            try:
                results['keywords'] = self._extract_keywords(text)
            except Exception as e:
                logger.error("Keyword extraction failed", error=str(e))
                results['keywords'] = []
        
        return results


# Global instance
enhanced_nlp_service = EnhancedNLPService()


async def get_enhanced_nlp_service() -> EnhancedNLPService:
    """Get global enhanced NLP service instance"""
    return enhanced_nlp_service


# Convenience functions for backward compatibility
async def summarize_text_async(
    text: str,
    max_length: int = 150,
    min_length: int = 30,
    do_sample: bool = False
) -> Optional[str]:
    """Legacy function for backward compatibility"""
    service = await get_enhanced_nlp_service()
    result = await service.summarize_text_advanced(text, max_length, min_length)
    return result.summary if result else None


async def analyze_content_semantically(
    query: str,
    contents: List[str]
) -> List[float]:
    """Calculate semantic similarity scores"""
    service = await get_enhanced_nlp_service()
    return await service.calculate_semantic_similarity(query, contents)