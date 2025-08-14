import pandas as pd
import numpy as np
import re
import requests
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ReviewAnalysis:
    """Structured result for review analysis"""
    prediction: str
    confidence: float
    reasoning: str
    key_indicators: List[str]
    processing_time: float
    similarity_score: float = 0.0

class GroqClient:
    """
    Optimized Groq API Client with improved error handling and caching
    Free tier: 14,400 tokens/min | 30 requests/min
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._get_api_key()
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "FakeReviewDetector/1.0"
        }
        
        # Optimized model selection
        self.models = {
            "fast": "llama-3.3-70b-versatile",
            "smart": "llama-3.3-70b-versatile", 
            "balanced": "mixtral-8x7b-32768",
            "gemma": "gemma2-9b-it"
        }
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.requests_count = 0
        self.reset_time = time.time() + 60
        
        # Simple response cache
        self.response_cache = {}
        self.cache_max_size = 100

    def _get_api_key(self) -> str:
        """Get API key with better user experience"""
        import os
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("🔑 Groq API key not found in environment variables.")
            api_key = input("Please enter your Groq API key from https://console.groq.com: ").strip()
        return api_key

    def _get_cache_key(self, messages: List[Dict], model: str, temperature: float) -> str:
        """Generate cache key for request"""
        content = json.dumps(messages, sort_keys=True) + model + str(temperature)
        return hashlib.md5(content.encode()).hexdigest()

    def _handle_rate_limit(self):
        """Smart rate limiting"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time > self.reset_time:
            self.requests_count = 0
            self.reset_time = current_time + 60
        
        # Ensure minimum time between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 2:  # Minimum 2 seconds between requests
            time.sleep(2 - time_since_last)
        
        self.last_request_time = time.time()
        self.requests_count += 1

    def make_request(self, messages: List[Dict], model: str = "fast", 
                    max_tokens: int = 500, temperature: float = 0.3, 
                    retries: int = 3, use_cache: bool = True) -> Optional[Dict]:
        """Enhanced request with caching and better error handling"""
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(messages, model, temperature)
            if cache_key in self.response_cache:
                logger.info("Using cached response")
                return self.response_cache[cache_key]

        # Rate limiting
        self._handle_rate_limit()

        payload = {
            "model": self.models.get(model, self.models["fast"]),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1,
            "stream": False
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    self.base_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30  # Add timeout
                )
                response.raise_for_status()
                
                data = response.json()
                result = {
                    "choices": [{
                        "message": {
                            "content": data["choices"][0]["message"]["content"]
                        }
                    }],
                    "usage": data.get("usage", {}),
                    "headers": dict(response.headers)
                }
                
                # Cache successful response
                if use_cache and len(self.response_cache) < self.cache_max_size:
                    cache_key = self._get_cache_key(messages, model, temperature)
                    self.response_cache[cache_key] = result
                
                return result
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = self._extract_retry_after(e.response)
                    logger.warning(f"Rate limit hit. Waiting {retry_after}s (attempt {attempt + 1}/{retries})")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error(f"HTTP error: {e}")
                    break
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                break

        logger.error(f"Request failed after {retries} attempts")
        return None

    def _extract_retry_after(self, response) -> float:
        """Extract retry-after time from error response"""
        try:
            error_data = response.json().get('error', {})
            message = error_data.get('message', '')
            if 'Please try again in' in message:
                return float(message.split('Please try again in ')[1].split('s')[0])
        except:
            pass
        return 2.0

    def test_connection(self) -> bool:
        """Test API connection with detailed feedback"""
        logger.info("Testing Groq API connection...")
        
        test_messages = [
            {"role": "user", "content": "Respond with 'Connection successful' only."}
        ]
        
        result = self.make_request(test_messages, model="gemma", max_tokens=10, use_cache=False)
        
        if result and 'choices' in result:
            response_text = result['choices'][0]['message']['content'].strip().lower()
            if 'connection successful' in response_text:
                logger.info("✅ Groq API connection successful!")
                return True
            else:
                logger.warning(f"Unexpected response: {response_text}")
                
        logger.error("❌ Groq API connection failed")
        return False

def clean_text(text: str) -> str:
    """Optimized text cleaning with better preprocessing"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Convert to lowercase and basic cleaning
    text = text.lower().strip()
    
    # Remove URLs, HTML tags, extra whitespace
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Keep more characters for better analysis
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    text = re.sub(r'([.,;:]){2,}', r'\1', text)

    return text.strip()

class ReviewKnowledgeBase:
    """Enhanced knowledge base with better pattern extraction and caching"""

    def __init__(self, reviews_df: pd.DataFrame):
        self.reviews_df = reviews_df.copy()
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased for better representation
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.8
        )
        self.review_vectors = None
        self.patterns = {'fake': [], 'real': []}
        self.feature_stats = {}
        self.build_knowledge_base()

    def build_knowledge_base(self):
        """Build comprehensive knowledge base"""
        logger.info("Building knowledge base...")
        
        # Build TF-IDF vectors
        texts = self.reviews_df['cleaned_text'].fillna('')
        self.review_vectors = self.vectorizer.fit_transform(texts)
        
        # Extract patterns for both classes
        fake_reviews = self.reviews_df[self.reviews_df['label'] == 0]
        real_reviews = self.reviews_df[self.reviews_df['label'] == 1]
        
        self.patterns['fake'] = self._extract_advanced_patterns(fake_reviews, 'fake')
        self.patterns['real'] = self._extract_advanced_patterns(real_reviews, 'real')
        
        # Calculate feature statistics
        self._calculate_feature_stats()
        
        logger.info(f"Knowledge base built: {len(self.patterns['fake'])} fake patterns, "
                   f"{len(self.patterns['real'])} real patterns")

    def _extract_advanced_patterns(self, reviews_df: pd.DataFrame, label_type: str) -> List[Dict]:
        """Extract comprehensive patterns from reviews"""
        patterns = []
        sample_size = min(200, len(reviews_df))  # Increased sample size
        sample_reviews = reviews_df.sample(n=sample_size, random_state=42)
        
        for _, row in sample_reviews.iterrows():
            text = row['cleaned_text']
            original_text = row['text_']
            
            if len(text.split()) < 3:  # Skip very short reviews
                continue
                
            pattern = {
                'text': text,
                'original': original_text,
                'type': label_type,
                'length': len(text.split()),
                'char_length': len(text),
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'caps_ratio': sum(1 for c in original_text if c.isupper()) / len(original_text) if original_text else 0,
                'punctuation_density': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text),
                'unique_words': len(set(text.split())),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'sentiment_indicators': self._count_sentiment_words(text)
            }
            patterns.append(pattern)
        
        return patterns

    def _count_sentiment_words(self, text: str) -> Dict[str, int]:
        """Count sentiment-indicating words"""
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love', 'best', 'awesome']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'bad', 'poor', 'disappointing']
        extreme_words = ['extremely', 'absolutely', 'completely', 'totally', 'definitely', 'never', 'always']
        
        return {
            'positive': sum(1 for word in positive_words if word in text),
            'negative': sum(1 for word in negative_words if word in text),
            'extreme': sum(1 for word in extreme_words if word in text)
        }

    def _calculate_feature_stats(self):
        """Calculate statistical features for comparison"""
        fake_patterns = self.patterns['fake']
        real_patterns = self.patterns['real']
        
        def calc_stats(patterns, feature):
            values = [p[feature] for p in patterns if feature in p]
            if not values:
                return {'mean': 0, 'std': 0}
            return {'mean': np.mean(values), 'std': np.std(values)}
        
        features = ['length', 'caps_ratio', 'punctuation_density', 'exclamation_count']
        
        self.feature_stats = {
            'fake': {feature: calc_stats(fake_patterns, feature) for feature in features},
            'real': {feature: calc_stats(real_patterns, feature) for feature in features}
        }

    def retrieve_similar_reviews(self, query_text: str, k: int = 5) -> List[Dict]:
        """Enhanced similarity search with scoring"""
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.review_vectors).flatten()
        
        # Get top k similar reviews
        top_indices = similarities.argsort()[-k:][::-1]
        similar_reviews = []
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Only include meaningful similarities
                similar_reviews.append({
                    'text': self.reviews_df.iloc[idx]['cleaned_text'],
                    'original': self.reviews_df.iloc[idx]['text_'],
                    'label': self.reviews_df.iloc[idx]['label'],
                    'similarity': float(similarities[idx]),
                    'label_text': 'Fake' if self.reviews_df.iloc[idx]['label'] == 0 else 'Real'
                })
        
        return similar_reviews

    def get_enhanced_context(self, query_text: str) -> Dict[str, any]:
        """Get comprehensive context for RAG"""
        similar_reviews = self.retrieve_similar_reviews(query_text, k=5)
        
        # Analyze query features
        query_features = {
            'length': len(query_text.split()),
            'caps_ratio': sum(1 for c in query_text if c.isupper()) / len(query_text) if query_text else 0,
            'exclamation_count': query_text.count('!'),
            'punctuation_density': sum(1 for c in query_text if not c.isalnum() and not c.isspace()) / len(query_text)
        }
        
        return {
            'similar_reviews': similar_reviews,
            'query_features': query_features,
            'feature_stats': self.feature_stats,
            'top_fake_patterns': self.patterns['fake'][:3],
            'top_real_patterns': self.patterns['real'][:3]
        }

    def save(self, path: str):
        """Save knowledge base with compression"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Knowledge base saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load knowledge base"""
        try:
            with open(path, 'rb') as f:
                kb = pickle.load(f)
            logger.info(f"Knowledge base loaded from {path}")
            return kb
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            raise

class RAGDetector:
    """Enhanced fake review detector with improved prompting and analysis"""

    def __init__(self, groq_client: GroqClient, knowledge_base: ReviewKnowledgeBase):
        self.client = groq_client
        self.kb = knowledge_base
        
        self.system_prompt = """You are an expert AI system specialized in detecting fake reviews. You analyze reviews using linguistic patterns, sentiment analysis, and comparison with known examples.

Key indicators of fake reviews:
- Excessive use of superlatives and extreme language
- Generic or vague descriptions
- Unusual grammar patterns or awkward phrasing
- Inconsistent sentiment or details
- Over-emphasis on specific features
- Unnatural repetition of keywords

Key indicators of real reviews:
- Specific details and personal experiences
- Balanced sentiment with both positives and negatives
- Natural language flow and authentic tone
- Contextual information and realistic scenarios
- Appropriate length and detail level"""

        self.analysis_prompt = """Analyze this review for authenticity using the provided context:

REVIEW: "{review_text}"

CONTEXT DATA:
Similar Reviews:
{similar_reviews}

Query Analysis:
- Word count: {word_count}
- Capital letters ratio: {caps_ratio:.2%}
- Exclamation marks: {exclamation_count}
- Punctuation density: {punctuation_density:.3f}

Statistical Benchmarks:
Fake reviews avg: length={fake_length_mean:.1f}, caps={fake_caps_mean:.2%}, exclamations={fake_excl_mean:.1f}
Real reviews avg: length={real_length_mean:.1f}, caps={real_caps_mean:.2%}, exclamations={real_excl_mean:.1f}

INSTRUCTIONS:
1. Compare the review against similar examples and statistical patterns
2. Identify specific linguistic and stylistic indicators
3. Provide confidence score (0-100) based on evidence strength
4. Give detailed reasoning for your classification

Respond in JSON format:
{{
    "prediction": "fake" or "real",
    "confidence": confidence_score_0_to_100,
    "reasoning": "detailed explanation with specific evidence",
    "key_indicators": ["list of specific indicators found"],
    "similarity_analysis": "analysis of similar reviews context"
}}"""

    def predict_single(self, review_text: str) -> ReviewAnalysis:
        """Enhanced single review prediction with detailed analysis"""
        start_time = time.time()
        
        try:
            # Get comprehensive context
            context = self.kb.get_enhanced_context(review_text)
            
            # Format similar reviews for prompt
            similar_reviews_text = ""
            max_similarity = 0
            for i, review in enumerate(context['similar_reviews'][:3], 1):
                similar_reviews_text += f"{i}. [{review['label_text']}] (sim: {review['similarity']:.3f})\n"
                similar_reviews_text += f"   {review['text'][:100]}...\n"
                max_similarity = max(max_similarity, review['similarity'])
            
            # Build comprehensive prompt
            prompt = self.analysis_prompt.format(
                review_text=review_text,
                similar_reviews=similar_reviews_text,
                word_count=context['query_features']['length'],
                caps_ratio=context['query_features']['caps_ratio'],
                exclamation_count=context['query_features']['exclamation_count'],
                punctuation_density=context['query_features']['punctuation_density'],
                fake_length_mean=context['feature_stats']['fake']['length']['mean'],
                fake_caps_mean=context['feature_stats']['fake']['caps_ratio']['mean'],
                fake_excl_mean=context['feature_stats']['fake']['exclamation_count']['mean'],
                real_length_mean=context['feature_stats']['real']['length']['mean'],
                real_caps_mean=context['feature_stats']['real']['caps_ratio']['mean'],
                real_excl_mean=context['feature_stats']['real']['exclamation_count']['mean']
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.make_request(
                messages, 
                model="smart", 
                max_tokens=600, 
                temperature=0.1
            )
            
            processing_time = time.time() - start_time
            
            if response and 'choices' in response:
                return self._parse_response(
                    response['choices'][0]['message']['content'], 
                    processing_time, 
                    max_similarity
                )
            else:
                return ReviewAnalysis(
                    prediction="unknown",
                    confidence=0.0,
                    reasoning="Failed to get API response",
                    key_indicators=[],
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return ReviewAnalysis(
                prediction="error",
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                key_indicators=[],
                processing_time=time.time() - start_time
            )

    def _parse_response(self, content: str, processing_time: float, similarity_score: float) -> ReviewAnalysis:
        """Enhanced response parsing with fallback"""
        try:
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                return ReviewAnalysis(
                    prediction=result.get('prediction', 'unknown'),
                    confidence=float(result.get('confidence', 0)),
                    reasoning=result.get('reasoning', 'No reasoning provided'),
                    key_indicators=result.get('key_indicators', []),
                    processing_time=processing_time,
                    similarity_score=similarity_score
                )
            else:
                # Fallback parsing
                return self._fallback_parse(content, processing_time, similarity_score)
                
        except json.JSONDecodeError:
            return self._fallback_parse(content, processing_time, similarity_score)

    def _fallback_parse(self, content: str, processing_time: float, similarity_score: float) -> ReviewAnalysis:
        """Fallback parsing when JSON fails"""
        logger.warning("Using fallback parsing")
        
        lower_content = content.lower()
        
        # Determine prediction
        fake_score = sum([
            lower_content.count('fake') * 2,
            lower_content.count('artificial'),
            lower_content.count('generated'),
            lower_content.count('suspicious')
        ])
        
        real_score = sum([
            lower_content.count('real') * 2,
            lower_content.count('authentic'),
            lower_content.count('genuine'),
            lower_content.count('legitimate')
        ])
        
        prediction = "fake" if fake_score > real_score else "real"
        
        # Extract confidence
        confidence_match = re.search(r'confidence[:\s]*(\d+)', lower_content)
        confidence = float(confidence_match.group(1)) if confidence_match else 50.0
        
        # Extract key indicators
        indicators = re.findall(r'[-•]\s*(.+)', content)
        
        return ReviewAnalysis(
            prediction=prediction,
            confidence=confidence,
            reasoning=content.strip(),
            key_indicators=indicators[:5],
            processing_time=processing_time,
            similarity_score=similarity_score
        )

    def predict_batch(self, review_texts: List[str], max_reviews: int = 5) -> List[ReviewAnalysis]:
        """Batch prediction with rate limiting"""
        results = []
        reviews_to_process = review_texts[:max_reviews]
        
        logger.info(f"Processing {len(reviews_to_process)} reviews")
        
        for i, text in enumerate(reviews_to_process):
            logger.info(f"Processing review {i+1}/{len(reviews_to_process)}")
            result = self.predict_single(text)
            results.append(result)
            
            # Rate limiting between requests
            if i < len(reviews_to_process) - 1:
                time.sleep(2)
        
        return results

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced data preprocessing with validation"""
    logger.info(f"Preprocessing data: {df.shape}")
    
    # Validate required columns
    required_cols = ['text_', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean and filter data
    df = df.dropna(subset=['text_']).copy()
    df['cleaned_text'] = df['text_'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() >= 10]  # Minimum length filter
    
    # Map labels
    label_mapping = {'CG': 0, 'OR': 1}
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map(label_mapping)
    
    # Remove any rows with unmapped labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    logger.info(f"Preprocessed data: {df.shape}, Labels: {df['label'].value_counts().to_dict()}")
    return df

def main():
    """Main execution function"""
    print("🚀 Initializing Fake Review Detection System")
    
    # Initialize Groq client
    groq_client = GroqClient()
    if not groq_client.test_connection():
        print("❌ Failed to connect to Groq API. Please check your API key.")
        return
    
    # Load and preprocess data
    try:
        file_path = "fake reviews dataset.csv"  # Update path as needed
        if not Path(file_path).exists():
            print(f"❌ Dataset file not found: {file_path}")
            print("Please ensure the dataset file is in the current directory.")
            return
            
        df = pd.read_csv(file_path)
        df = preprocess_data(df)
        
        # Build or load knowledge base
        kb_path = "knowledge_base.pkl"
        if Path(kb_path).exists():
            print("📚 Loading existing knowledge base...")
            kb = ReviewKnowledgeBase.load(kb_path)
        else:
            print("🧠 Building new knowledge base...")
            kb = ReviewKnowledgeBase(df)
            kb.save(kb_path)
        
        # Initialize detector
        detector = RAGDetector(groq_client, kb)
        
        # Example predictions
        sample_reviews = [
            "This product is absolutely terrible! Worst purchase ever made, completely useless and waste of money!!!",
            "I bought this for my daughter and she really enjoys using it. The quality seems decent for the price, though the instructions could be clearer."
        ]
        
        print("🔍 Analyzing sample reviews...")
        results = detector.predict_batch(sample_reviews)
        
        # Display results
        for i, (review, result) in enumerate(zip(sample_reviews, results)):
            print(f"\n{'='*60}")
            print(f"Review {i+1}: {review[:80]}...")
            print(f"Prediction: {result.prediction.upper()}")
            print(f"Confidence: {result.confidence:.1f}%")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Key indicators: {', '.join(result.key_indicators[:3])}")
            print(f"Reasoning: {result.reasoning[:200]}...")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()