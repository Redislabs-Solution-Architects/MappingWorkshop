from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from threading import Thread, Event
import math
import redis
import json
import random
import time
import threading
import pprint
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import redis.commands.search
import redis.commands.search.aggregation as aggregations
import redis.commands.search.reducers as reducers
from redis.commands.json.path import Path
from redis.commands.search import Search
from redis.commands.search.field import (
    GeoField,
    GeoShapeField,
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import GeoFilter, NumericFilter, Query
from redis.commands.search.result import Result
from redis.commands.search.suggestion import Suggestion
import openai
# from sentence_transformers import SentenceTransformer
import re
from typing import Dict, List, Tuple, Optional
import spacy


# Import for advanced image processing
try:
    from sentence_transformers import SentenceTransformer
    import openai
    import clip
    import torch
    from torchvision import transforms
    EMBEDDINGS_AVAILABLE = True
    
    # Try to load CLIP model
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
        CLIP_AVAILABLE = True
        print("CLIP model loaded successfully")
    except Exception as e:
        print(f"CLIP not available: {e}")
        CLIP_AVAILABLE = False
        
except ImportError:
    print("Warning: Some AI libraries not available. Installing with pip install sentence-transformers openai-clip-torch")
    EMBEDDINGS_AVAILABLE = False
    CLIP_AVAILABLE = False

# # Initialize enhanced AI models
# try:
#     # Load a better sentence transformer model
#     sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
#     # Optional: Load spaCy for advanced NLP
#     # nlp = spacy.load("en_core_web_sm")
    

#     print("Enhanced AI models loaded successfully")
#     ENHANCED_AI_AVAILABLE = True
# except Exception as e:
#     print(f"Using fallback AI models: {e}")
#     ENHANCED_AI_AVAILABLE = False


# # Import for embeddings and AI capabilities
# try:
#     from sentence_transformers import SentenceTransformer    
#     import openai
#     import clip
#     import torch
#     EMBEDDINGS_AVAILABLE = True
# except ImportError:
#     print("Warning: Some AI libraries not available. Installing with pip install sentence-transformers openai-clip-torch")
#     EMBEDDINGS_AVAILABLE = False

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
r = redis.Redis(decode_responses=True)
r.flushdb()

ELEMENT_TYPES = ['airplane', 'motorcycle', 'bike', 'bird', 'person']
ELEMENT_VECTORS={}
TOTAL_ELEMENTS = 1000
movement_speed = 2  # default speed in seconds

# Initialize AI models if available
if EMBEDDINGS_AVAILABLE:
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # For simplicity, we'll use mock CLIP functionality
        # In production, use: clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
        print("AI models loaded successfully")
        ENHANCED_AI_AVAILABLE = True
    except Exception as e:
        print(f"Error loading AI models: {e}")
        EMBEDDINGS_AVAILABLE = False
        ENHANCED_AI_AVAILABLE = False

# def initialize_reference_embeddings():
#     """Pre-compute reference embeddings for each element type using CLIP"""
#     global REFERENCE_IMAGE_EMBEDDINGS
    
#     if not CLIP_AVAILABLE:
#         print("CLIP not available, using fallback image analysis")
#         return
    
#     # Text descriptions for each element type (CLIP can encode text too)
#     element_descriptions = {
#         'airplane': ["airplane", "aircraft", "plane", "jet", "flying machine", "commercial airliner", "fighter jet"],
#         'bird': ["bird", "flying bird", "eagle", "seagull", "pigeon", "sparrow", "flying animal"],
#         'motorcycle': ["motorcycle", "motorbike", "bike", "scooter", "two-wheeler", "motor vehicle"],
#         'bike': ["bicycle", "bike", "cycling", "pedal bike", "mountain bike", "road bike"],
#         'person': ["person", "human", "pedestrian", "people walking", "individual", "man", "woman"]
#     }
    
#     try:
#         with torch.no_grad():
#             for element_type, descriptions in element_descriptions.items():
#                 # Create embeddings from text descriptions
#                 text_tokens = clip.tokenize(descriptions).to("cpu")
#                 text_embeddings = clip_model.encode_text(text_tokens)
                
#                 # Average the embeddings for more robust matching
#                 avg_embedding = text_embeddings.mean(dim=0).cpu().numpy()
#                 REFERENCE_IMAGE_EMBEDDINGS[element_type] = avg_embedding
                
#         print("Reference embeddings initialized successfully")
#     except Exception as e:
#         print(f"Error initializing reference embeddings: {e}")

# def analyze_image_with_clip(image):
#     """Advanced image analysis using CLIP model"""
#     if not CLIP_AVAILABLE:
#         return analyze_image_simple(image)
    
#     try:
#         # Preprocess image for CLIP
#         image_tensor = clip_preprocess(image).unsqueeze(0).to("cpu")
        
#         with torch.no_grad():
#             # Get image embedding
#             image_embedding = clip_model.encode_image(image_tensor)
#             image_embedding = image_embedding.cpu().numpy().flatten()
            
#             # Compare with reference embeddings
#             similarities = {}
#             for element_type, ref_embedding in REFERENCE_IMAGE_EMBEDDINGS.items():
#                 # Calculate cosine similarity
#                 similarity = np.dot(image_embedding, ref_embedding) / (
#                     np.linalg.norm(image_embedding) * np.linalg.norm(ref_embedding)
#                 )
#                 similarities[element_type] = float(similarity)
            
#             # Find best match
#             best_match = max(similarities.items(), key=lambda x: x[1])
            
#             print(f"CLIP Analysis - Best match: {best_match[0]} (confidence: {best_match[1]:.3f})")
#             print(f"All similarities: {similarities}")
            
#             # Convert to visual features vector that matches our schema
#             visual_features = convert_clip_to_visual_features(image_embedding, similarities)
            
#             return visual_features, similarities
            
#     except Exception as e:
#         print(f"CLIP analysis error: {e}")
#         return analyze_image_simple(image), {}

# def convert_clip_to_visual_features(clip_embedding, similarities):
#     """Convert CLIP embedding to our 5D visual features format"""
#     # Our visual features: [wings, wheels, size, speed, altitude]
    
#     # Use similarities to infer features
#     wings = max(similarities.get('airplane', 0), similarities.get('bird', 0))
#     wheels = max(similarities.get('motorcycle', 0), similarities.get('bike', 0))
    
#     # Size estimation based on object type confidence
#     size = (
#         similarities.get('airplane', 0) * 1.0 +  # Large
#         similarities.get('motorcycle', 0) * 0.4 +  # Medium
#         similarities.get('bike', 0) * 0.3 +  # Small-medium
#         similarities.get('bird', 0) * 0.2 +  # Small
#         similarities.get('person', 0) * 0.3   # Small-medium
#     )
    
#     # Speed potential based on object type
#     speed = (
#         similarities.get('airplane', 0) * 1.0 +  # Very fast
#         similarities.get('bird', 0) * 0.6 +  # Fast
#         similarities.get('motorcycle', 0) * 0.7 +  # Fast
#         similarities.get('bike', 0) * 0.3 +  # Slow
#         similarities.get('person', 0) * 0.1   # Very slow
#     )
    
#     # Altitude preference
#     altitude = (
#         similarities.get('airplane', 0) * 1.0 +  # High
#         similarities.get('bird', 0) * 0.5 +  # Medium
#         similarities.get('motorcycle', 0) * 0.0 +  # Ground
#         similarities.get('bike', 0) * 0.0 +  # Ground
#         similarities.get('person', 0) * 0.0   # Ground
#     )
    
#     return [wings, wheels, size, speed, altitude]

# def analyze_image_advanced_fallback(image):
#     """Advanced fallback image analysis when CLIP is not available"""
#     import cv2
#     import numpy as np
#     from PIL import ImageFilter, ImageStat
    
#     try:
#         # Convert PIL to OpenCV format
#         image_array = np.array(image.convert('RGB'))
#         image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
#         # Extract more sophisticated features
#         features = {}
        
#         # 1. Edge density (higher for complex objects like bikes/motorcycles)
#         gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)
#         edge_density = np.sum(edges > 0) / edges.size
#         features['edge_density'] = edge_density
        
#         # 2. Color distribution analysis
#         hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        
#         # Sky blue detection (for airplanes/birds)
#         sky_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
#         sky_ratio = np.sum(sky_mask > 0) / sky_mask.size
#         features['sky_content'] = sky_ratio
        
#         # Green detection (for outdoor scenes with bikes/people)
#         green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
#         green_ratio = np.sum(green_mask > 0) / green_mask.size
#         features['green_content'] = green_ratio
        
#         # 3. Shape analysis using contours
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if contours:
#             # Find largest contour
#             largest_contour = max(contours, key=cv2.contourArea)
            
#             # Aspect ratio
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             aspect_ratio = w / h if h > 0 else 1
#             features['aspect_ratio'] = aspect_ratio
            
#             # Circularity (wheels detection)
#             area = cv2.contourArea(largest_contour)
#             perimeter = cv2.arcLength(largest_contour, True)
#             if perimeter > 0:
#                 circularity = 4 * np.pi * area / (perimeter * perimeter)
#                 features['circularity'] = circularity
#             else:
#                 features['circularity'] = 0
#         else:
#             features['aspect_ratio'] = 1.0
#             features['circularity'] = 0.0
        
#         # 4. Texture analysis
#         stat = ImageStat.Stat(image)
#         brightness = sum(stat.mean) / len(stat.mean)
#         contrast = sum(stat.stddev) / len(stat.stddev)
#         features['brightness'] = brightness / 255.0
#         features['contrast'] = contrast / 255.0
        
#         # Convert features to visual vector
#         visual_features = advanced_features_to_vector(features)
        
#         return visual_features
        
#     except Exception as e:
#         print(f"Advanced fallback analysis error: {e}")
#         return analyze_image_simple(image)

# def advanced_features_to_vector(features):
#     """Convert advanced features to visual vector"""
#     # Heuristic mapping based on typical characteristics
    
#     # Wings detection (high for flying objects)
#     wings = 0.0
#     if features['sky_content'] > 0.3:  # Likely flying object
#         wings = 0.7
#     if features['aspect_ratio'] > 2.0:  # Wing-like shape
#         wings = max(wings, 0.8)
    
#     # Wheels detection (based on circularity and edge density)
#     wheels = 0.0
#     if features['circularity'] > 0.5:  # Circular shapes
#         wheels = 0.6
#     if features['edge_density'] > 0.1 and features['aspect_ratio'] < 2.0:  # Complex ground vehicle
#         wheels = max(wheels, 0.7)
    
#     # Size estimation (based on edge complexity)
#     size = min(features['edge_density'] * 2, 1.0)
    
#     # Speed potential (flying objects and complex vehicles)
#     speed = wings * 0.8 + wheels * 0.6
    
#     # Altitude (sky content)
#     altitude = features['sky_content']
    
#     return [wings, wheels, size, speed, altitude]
# Pre-defined element type embeddings and characteristics
ELEMENT_CHARACTERISTICS = {
    'airplane': {
        'base_speed': 0.08,
        'speed_variance': 0.02,
        'altitude_preference': 'high',
        'visual_features': [1.0, 0.0, 0.0, 1.0, 0.8],  # [wings, wheels, size, speed, altitude]
        'behavior_pattern': [0.9, 0.1, 0.8, 0.7]  # [stability, predictability, speed_consistency, path_linearity]
    },
    'bird': {
        'base_speed': 0.05,
        'speed_variance': 0.03,
        'altitude_preference': 'medium',
        'visual_features': [1.0, 0.0, 0.3, 0.6, 0.5],
        'behavior_pattern': [0.4, 0.3, 0.5, 0.4]
    },
    'motorcycle': {
        'base_speed': 0.04,
        'speed_variance': 0.015,
        'altitude_preference': 'ground',
        'visual_features': [0.0, 1.0, 0.4, 0.7, 0.0],
        'behavior_pattern': [0.7, 0.8, 0.6, 0.8]
    },
    'bike': {
        'base_speed': 0.02,
        'speed_variance': 0.01,
        'altitude_preference': 'ground',
        'visual_features': [0.0, 1.0, 0.2, 0.3, 0.0],
        'behavior_pattern': [0.6, 0.7, 0.7, 0.9]
    },
    'person': {
        'base_speed': 0.005,
        'speed_variance': 0.003,
        'altitude_preference': 'ground',
        'visual_features': [0.0, 0.0, 0.1, 0.1, 0.0],
        'behavior_pattern': [0.3, 0.2, 0.4, 0.3]
    }
}

# Function to convert image to vector
def image_to_vector(image_path):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
    with torch.no_grad():
        vector = clip_model.encode_image(image)
        vector /= vector.norm(dim=-1, keepdim=True)
    return vector.cpu().numpy().astype(np.float32)[0].tolist()

def create_elements_vectors():
    for i in ELEMENT_TYPES:
        ELEMENT_VECTORS[i] = image_to_vector(f"backend/static/img/{i}.png")



class AIQueryProcessor:
    """Advanced AI-powered query processor for natural language search"""
    
    def __init__(self):
        self.element_types = ['airplane', 'motorcycle', 'bike', 'bird', 'person']
        self.speed_keywords = {
            'fast': ['fast', 'quick', 'rapid', 'speedy', 'swift', 'racing', 'zooming', 'rushing'],
            'slow': ['slow', 'sluggish', 'crawling', 'leisurely', 'gentle', 'dawdling', 'plodding'],
            'moderate': ['moderate', 'medium', 'average', 'normal', 'regular']
        }
        self.location_keywords = {
            'north': ['north', 'northern', 'arctic', 'up', 'top'],
            'south': ['south', 'southern', 'antarctic', 'down', 'bottom'],
            'east': ['east', 'eastern', 'right'],
            'west': ['west', 'western', 'left'],
            'europe': ['europe', 'european', 'eu'],
            'asia': ['asia', 'asian', 'orient'],
            'america': ['america', 'americas', 'usa', 'us'],
            'africa': ['africa', 'african'],
            'ocean': ['ocean', 'sea', 'water', 'marine']
        }
        self.behavior_keywords = {
            'erratic': ['erratic', 'chaotic', 'unpredictable', 'random', 'wild', 'crazy'],
            'stable': ['stable', 'steady', 'consistent', 'regular', 'predictable', 'smooth'],
            'unusual': ['unusual', 'strange', 'odd', 'weird', 'anomalous', 'abnormal']
        }

    def process_natural_query(self, query_text: str) -> Dict:
        """Process natural language query using AI and return structured search parameters"""
        query_lower = query_text.lower().strip()
        
        # Extract entities using multiple approaches
        entities = self._extract_entities(query_lower)
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(query_lower, entities)
        
        # Generate semantic embedding
        if ENHANCED_AI_AVAILABLE:
            semantic_vector = sentence_model.encode(query_text).tolist()
        else:
            semantic_vector = self._generate_fallback_vector(query_text)
        
        return {
            'query_text': query_text,
            'entities': entities,
            'search_strategy': search_strategy,
            'semantic_vector': semantic_vector,
            'confidence': entities.get('confidence', 0.7)
        }

    def _extract_entities(self, query: str) -> Dict:
        """Extract entities from the query using multiple NLP techniques"""
        entities = {
            'element_types': [],
            'speed_filters': [],
            'location_filters': [],
            'behavior_filters': [],
            'temporal_filters': [],
            'quantity_filters': [],
            'confidence': 0.0
        }
        
        confidence_scores = []
        
        # Extract element types with confidence scoring
        for element_type in self.element_types:
            patterns = [
                rf'\b{element_type}s?\b',
                rf'\b{element_type}[s]?\b'
            ]
            
            for pattern in patterns:
                if re.search(pattern, query):
                    entities['element_types'].append(element_type)
                    confidence_scores.append(0.9)
                    break
        
        # Enhanced speed detection
        for speed_type, keywords in self.speed_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    entities['speed_filters'].append(speed_type)
                    confidence_scores.append(0.8)
                    break
        
        # Enhanced location detection
        for location_type, keywords in self.location_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    entities['location_filters'].append(location_type)
                    confidence_scores.append(0.8)
                    break
        
        # Behavior pattern detection
        for behavior_type, keywords in self.behavior_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    entities['behavior_filters'].append(behavior_type)
                    confidence_scores.append(0.7)
                    break
        
        # Quantity detection
        quantity_patterns = [
            r'\b(\d+)\s+(?:of\s+)?(?:the\s+)?(?:most|top|fastest|slowest)\b',
            r'\bfind\s+(\d+)\b',
            r'\bshow\s+(?:me\s+)?(\d+)\b',
            r'\b(\d+)\s+elements?\b'
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, query)
            if match:
                entities['quantity_filters'].append(int(match.group(1)))
                confidence_scores.append(0.9)
                break
        
        # Temporal detection (near, around, recently, etc.)
        temporal_patterns = [
            r'\bnear\s+(\w+)\b',
            r'\baround\s+(\w+)\b',
            r'\brecently\b',
            r'\bnow\b',
            r'\bcurrently\b'
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, query):
                entities['temporal_filters'].append('recent')
                confidence_scores.append(0.6)
                break
        
        # Calculate overall confidence
        if confidence_scores:
            entities['confidence'] = sum(confidence_scores) / len(confidence_scores)
        else:
            entities['confidence'] = 0.3  # Low confidence for no matches
        
        return entities

    def _determine_search_strategy(self, query: str, entities: Dict) -> str:
        """Determine the best search strategy based on query analysis"""
        
        # Check for explicit similarity requests
        similarity_indicators = ['similar', 'like', 'resembling', 'comparable', 'alike']
        if any(indicator in query for indicator in similarity_indicators):
            return 'similarity_search'
        
        # Check for anomaly detection requests
        anomaly_indicators = ['anomaly', 'unusual', 'strange', 'odd', 'outlier', 'different']
        if any(indicator in query for indicator in anomaly_indicators):
            return 'anomaly_detection'
        
        # Check for spatial queries
        spatial_indicators = ['near', 'around', 'in', 'within', 'close to', 'nearby']
        if any(indicator in query for indicator in spatial_indicators):
            return 'spatial_search'
        
        # If we have specific entities, use traditional search
        if (entities['element_types'] or entities['speed_filters'] or 
            entities['location_filters'] or entities['behavior_filters']):
            return 'traditional_search'
        
        # Default to semantic search for complex natural language
        return 'semantic_search'

    def _generate_fallback_vector(self, query_text: str) -> List[float]:
        """Generate a fallback semantic vector when advanced models aren't available"""
        # Enhanced fallback using word embeddings simulation
        words = query_text.lower().split()
        
        # Create a more sophisticated feature vector
        feature_vector = [0.0] * 50  # Larger vector for better representation
        
        # Word-based features
        for i, word in enumerate(words[:10]):  # Use first 10 words
            word_hash = hash(word) % 1000000
            feature_vector[i % 20] += (word_hash / 1000000.0)
        
        # N-gram features
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            bigram_hash = hash(bigram) % 1000000
            feature_vector[20 + (i % 15)] += (bigram_hash / 1000000.0)
        
        # Normalize the vector
        magnitude = sum(x**2 for x in feature_vector) ** 0.5
        if magnitude > 0:
            feature_vector = [x / magnitude for x in feature_vector]
        
        return feature_vector


def execute_spatial_search(entities: Dict, user_id: str, k: int):
    """Perform spatial search using geographic filters (e.g., region, circle, polygon)"""
    query_parts = []

    # Use region filters if available
    for location in entities.get('location_filters', []):
        if location == 'north':
            query_parts.append("@lat:[30 90]")
        elif location == 'south':
            query_parts.append("@lat:[-90 -30]")
        elif location == 'europe':
            query_parts.extend(["@lat:[35 70]", "@lng:[-10 40]"])
        elif location == 'asia':
            query_parts.extend(["@lat:[10 70]", "@lng:[60 150]"])
        elif location == 'america':
            query_parts.extend(["@lat:[-55 70]", "@lng:[-130 -35]"])
        elif location == 'africa':
            query_parts.extend(["@lat:[-35 35]", "@lng:[-20 50]"])
        elif location == 'north_of_asia':
            query_parts.extend(["@lat:[60 90]", "@lng:[60 150]"])
        # Add more mappings as needed

    if entities['element_types']:
        types = [f"@type:{{{t}}}" for t in entities['element_types']]
        query_parts.append(f"({' | '.join(types)})")

    final_query = " ".join(query_parts) if query_parts else "*"

    try:
        q = Query(final_query).paging(0, min(k, 100))
        result = r.ft("elements_idx").search(q)

        elements_res = {}
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                element = json.loads(doc.json)
                elements_res[element_id] = {
                    'id': element_id,
                    'type': element['type'],
                    'lat': element['lat'],
                    'lng': element['lng'],
                    'search_type': 'spatial',
                    'matched_locations': entities.get('location_filters', []),
                }

        return jsonify({
            "elements": elements_res,
            "search_type": "spatial",
            "matched_locations": entities.get('location_filters', []),
            "query_used": final_query
        })

    except Exception as e:
        print(f"Spatial search error: {e}")
        return jsonify({"error": str(e)}), 500

# Enhanced natural language search endpoint
@app.route('/enhanced_natural_search', methods=['POST'])
def enhanced_natural_language_search():
    """Enhanced AI-powered natural language search"""
    data = request.json
    query_text = data.get('query_text', '')
    user_id = data.get('user_id')
    k = data.get('k', 20)
    
    if not query_text or not user_id:
        return jsonify({"error": "query_text and user_id required"}), 400
    
    try:
        # Initialize AI processor
        ai_processor = AIQueryProcessor()
        
        # Process the query
        processed_query = ai_processor.process_natural_query(query_text)
        
        # Execute the appropriate search based on strategy
        search_strategy = processed_query['search_strategy']
        entities = processed_query['entities']
        
        if search_strategy == 'semantic_search':
            return execute_semantic_search(processed_query, user_id, k)
        elif search_strategy == 'traditional_search':
            return execute_enhanced_traditional_search(entities, user_id, k)
        elif search_strategy == 'similarity_search':
            return execute_similarity_search(entities, user_id, k)
        elif search_strategy == 'anomaly_detection':
            return execute_anomaly_search(entities, user_id)
        elif search_strategy == 'spatial_search':
            return execute_spatial_search(entities, user_id, k)
        else:
            # Fallback to semantic search
            return execute_semantic_search(processed_query, user_id, k)
            
    except Exception as e:
        print(f"Enhanced natural language search error: {e}")
        return jsonify({"error": str(e), "fallback": True}), 500

def execute_semantic_search(processed_query: Dict, user_id: str, k: int):
    """Execute semantic vector search"""
    try:
        semantic_vector = processed_query['semantic_vector']
        
        # Perform vector search
        vector_query = f"*=>[KNN {k} @semantic_vector $vec AS score]"
        query_params = {"vec": np.array(semantic_vector, dtype=np.float32).tobytes()}
        
        q = Query(vector_query).return_fields("id", "type", "lat", "lng", "score").paging(0, k).dialect(2)
        result = r.ft("elements_idx").search(query=q, query_params=query_params)
        
        elements_res = {}
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                elements_res[element_id] = {
                    'id': element_id,
                    'type': doc.type,
                    'lat': float(doc.lat),
                    'lng': float(doc.lng),
                    'similarity_score': float(doc.score) if hasattr(doc, 'score') else 0.0,
                    'search_explanation': f"Semantic match for: {processed_query['query_text']}"
                }
        
        return jsonify({
            "elements": elements_res,
            "search_type": "enhanced_semantic",
            "query_analysis": processed_query,
            "total_results": len(elements_res)
        })
        
    except Exception as e:
        print(f"Semantic search error: {e}")
        return jsonify({"error": str(e)}), 500

def execute_enhanced_traditional_search(entities: Dict, user_id: str, k: int):
    """Execute enhanced traditional search based on extracted entities"""
    query_parts = []
    
    # Build query from entities
    if entities['element_types']:
        type_queries = [f"@type:{{{et}}}" for et in entities['element_types']]
        query_parts.append(f"({' | '.join(type_queries)})")
    
    if entities['speed_filters']:
        for speed_filter in entities['speed_filters']:
            if speed_filter == 'fast':
                query_parts.append("@speed:[0.05 +inf]")
            elif speed_filter == 'slow':
                query_parts.append("@speed:[-inf 0.02]")
            elif speed_filter == 'moderate':
                query_parts.append("@speed:[0.02 0.05]")
    
    if entities['location_filters']:
        # Add location-based queries
        for location in entities['location_filters']:
            if location == 'north':
                query_parts.append("@lat:[30 90]")
            elif location == 'south':
                query_parts.append("@lat:[-90 -30]")
            elif location == 'europe':
                query_parts.extend(["@lat:[35 70]", "@lng:[-10 40]"])
            elif location == 'asia':
                query_parts.extend(["@lat:[10 70]", "@lng:[60 150]"])
            # Add more location mappings as needed
    
    final_query = " ".join(query_parts) if query_parts else "*"
    
    try:
        # Limit results based on quantity filters if specified
        limit = entities['quantity_filters'][0] if entities['quantity_filters'] else k
        
        q = Query(final_query).paging(0, 10000)
        result = r.ft("elements_idx").search(q)
        
        elements_res = {}
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                element = json.loads(doc.json)
                elements_res[element_id] = {
                    'id': element_id,
                    'type': element['type'],
                    'lat': element['lat'],
                    'lng': element['lng'],
                    'entity_match': True,
                    'matched_entities': entities
                }
        
        return jsonify({
            "elements": elements_res,
            "search_type": "enhanced_traditional",
            "matched_entities": entities,
            "query_used": final_query
        })
        
    except Exception as e:
        print(f"Enhanced traditional search error: {e}")
        return jsonify({"error": str(e)}), 500

# Add this to your existing routes
@app.route('/ai_query_analysis', methods=['POST'])
def analyze_query():
    """Analyze a natural language query and return the interpretation"""
    data = request.json
    query_text = data.get('query_text', '')
    
    if not query_text:
        return jsonify({"error": "query_text required"}), 400
    
    try:
        ai_processor = AIQueryProcessor()
        analysis = ai_processor.process_natural_query(query_text)
        
        return jsonify({
            "query_analysis": analysis,
            "suggestions": generate_query_suggestions(analysis),
            "estimated_results": estimate_result_count(analysis)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_query_suggestions(analysis: Dict) -> List[str]:
    """Generate query suggestions based on analysis"""
    suggestions = []
    entities = analysis['entities']
    
    if entities['element_types']:
        suggestions.append(f"Try: 'Show me all {entities['element_types'][0]}s'")
    
    if entities['speed_filters']:
        suggestions.append(f"Try: 'Find {entities['speed_filters'][0]} moving elements'")
    
    if entities['location_filters']:
        suggestions.append(f"Try: 'Elements in {entities['location_filters'][0]}'")
    
    # Add more contextual suggestions
    suggestions.extend([
        "Try: 'Fast airplanes near Europe'",
        "Try: 'Slow bikes in Asia'",
        "Try: 'Find 5 unusual birds'",
        "Try: 'Show erratic motorcycles'"
    ])
    
    return suggestions[:5]  # Return top 5 suggestions

def estimate_result_count(analysis: Dict) -> int:
    """Estimate number of results based on query analysis"""
    # Simple estimation logic
    base_estimate = 50
    
    entities = analysis['entities']
    
    # Reduce estimate for specific filters
    if entities['element_types']:
        base_estimate *= 0.2  # Type filter is very specific
    
    if entities['speed_filters']:
        base_estimate *= 0.4
    
    if entities['location_filters']:
        base_estimate *= 0.3
    
    if entities['quantity_filters']:
        return min(entities['quantity_filters'][0], int(base_estimate))
    
    return max(5, int(base_estimate))

def create_index():
    schema = (
        TextField("$.id", as_name="id"),
        TagField("$.type", as_name="type"),
        NumericField("$.lat", as_name="lat"),
        NumericField("$.lng", as_name="lng"),
        NumericField("$.speed", as_name="speed"),
        NumericField("$.direction", as_name="direction"),
        GeoField("$.location", as_name="location"),
        GeoShapeField("$.polylocation", as_name="poly_loc", coord_system=GeoShapeField.SPHERICAL),
        # Vector fields for different types of similarity search
        VectorField("$.visual_vector", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": 5,
            "DISTANCE_METRIC": "COSINE"
        }, as_name="visual_vector"),
        VectorField("$.behavior_vector", "FLAT", {
            "TYPE": "FLOAT32", 
            "DIM": 4,
            "DISTANCE_METRIC": "COSINE"
        }, as_name="behavior_vector"),
        VectorField("$.semantic_vector", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": 384 if EMBEDDINGS_AVAILABLE else 10,
            "DISTANCE_METRIC": "COSINE"
        }, as_name="semantic_vector"),
        VectorField("$.image_vector", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": 512,
            "DISTANCE_METRIC": "COSINE"
        }, as_name="image_vector")
    )
    try:
        r.ft("elements_idx").create_index(schema, definition=IndexDefinition(
            prefix=["element:"], index_type=IndexType.JSON))
        print("Index created successfully")
    except Exception as e:
        print(f"Index creation error (may already exist): {e}")

def generate_semantic_vector(element_type, lat, lng, speed, direction):
    """Generate semantic vector for natural language queries"""
    if EMBEDDINGS_AVAILABLE:
        try:
            # Create rich descriptive text that captures the element's context
            speed_desc = "fast" if speed > 0.05 else "slow" if speed < 0.02 else "moderate"
            
            # Direction description
            direction_desc = ""
            if 0 <= direction < 45 or 315 <= direction < 360:
                direction_desc = "moving north"
            elif 45 <= direction < 135:
                direction_desc = "moving east"
            elif 135 <= direction < 225:
                direction_desc = "moving south"
            elif 225 <= direction < 315:
                direction_desc = "moving west"
            
            # Location context
            location_desc = ""
            if lat > 60:
                location_desc = "in northern regions"
            elif lat < -30:
                location_desc = "in southern regions"
            elif -30 <= lat <= 60:
                location_desc = "in temperate zones"
            
            if lng > 100:
                location_desc += " in Asia"
            elif 0 <= lng <= 100:
                location_desc += " in Europe or Africa"
            elif -100 <= lng < 0:
                location_desc += " in Americas"
            
            # Activity context based on element type
            activity_context = {
                'airplane': f"aircraft flying {speed_desc} {direction_desc} {location_desc}",
                'bird': f"bird {speed_desc} {direction_desc} {location_desc}",
                'motorcycle': f"motorcycle riding {speed_desc} {direction_desc} {location_desc}",
                'bike': f"bicycle pedaling {speed_desc} {direction_desc} {location_desc}",
                'person': f"person walking {speed_desc} {direction_desc} {location_desc}"
            }
            
            description = activity_context.get(element_type, f"{element_type} moving {speed_desc} {direction_desc} {location_desc}")
            
            # Generate embedding
            vector = sentence_model.encode(description).tolist()
            return vector
            
        except Exception as e:
            print(f"Error generating semantic vector: {e}")
            # Fall through to fallback
    
    # Enhanced fallback: create more meaningful semantic features
    # One-hot encoding for element type
    type_encoding = [1.0 if t == element_type else 0.0 for t in ELEMENT_TYPES]
    
    # Location features (normalized)
    location_features = [
        lat / 90.0,  # Normalized latitude
        lng / 180.0,  # Normalized longitude
        math.sin(math.radians(lat)),  # Seasonal/climate indicator
        math.cos(math.radians(lng))   # Time zone indicator
    ]
    
    # Movement features
    movement_features = [
        speed * 10,  # Scaled speed
        math.sin(math.radians(direction)),  # Direction X component
        math.cos(math.radians(direction)),  # Direction Y component
        speed * math.sin(math.radians(direction)),  # Velocity X
        speed * math.cos(math.radians(direction))   # Velocity Y
    ]
    
    # Combine all features
    semantic_vector = type_encoding + location_features + movement_features
    
    # Pad or truncate to desired length
    target_length = 384 if EMBEDDINGS_AVAILABLE else 14
    if len(semantic_vector) < target_length:
        # Pad with derived features
        while len(semantic_vector) < target_length:
            semantic_vector.append(random.uniform(-0.1, 0.1))
    elif len(semantic_vector) > target_length:
        semantic_vector = semantic_vector[:target_length]
    
    return semantic_vector

def calculate_behavior_vector(element):
    """Calculate behavior characteristics for similarity matching"""
    characteristics = ELEMENT_CHARACTERISTICS.get(element['type'], ELEMENT_CHARACTERISTICS['person'])
    
    # Add some randomness to make each element unique
    behavior_vector = [
        characteristics['behavior_pattern'][0] + random.uniform(-0.1, 0.1),  # stability
        characteristics['behavior_pattern'][1] + random.uniform(-0.1, 0.1),  # predictability  
        characteristics['behavior_pattern'][2] + random.uniform(-0.1, 0.1),  # speed_consistency
        characteristics['behavior_pattern'][3] + random.uniform(-0.1, 0.1),  # path_linearity
    ]
    
    # Normalize to [0, 1]
    return [max(0, min(1, x)) for x in behavior_vector]

def generate_elements():
    print("Generating elements with vector embeddings...")
    for i in range(TOTAL_ELEMENTS):
        element_id = f"{i}"
        element_type = random.choice(ELEMENT_TYPES)
        lat = random.uniform(-85, 85)  # Avoid poles
        lng = random.uniform(-170, 170)
        
        # Use type-specific characteristics for more realistic movement
        characteristics = ELEMENT_CHARACTERISTICS[element_type]
        speed = characteristics['base_speed'] + random.uniform(-characteristics['speed_variance'], characteristics['speed_variance'])
        speed = max(0.001, speed)  # Ensure positive speed
        direction = random.uniform(0, 360)
        
        # Generate vectors
        image_vector = ELEMENT_VECTORS[element_type]
        visual_vector = characteristics['visual_features'].copy()
        behavior_vector = calculate_behavior_vector({'type': element_type})
        semantic_vector = generate_semantic_vector(element_type, lat, lng, speed, direction)
        
        element = {
            'id': element_id,
            'type': element_type,
            'speed': speed,
            'direction': direction,
            'lat': lat,
            'lng': lng,
            'location': f"{lng},{lat}",
            'polylocation': f"POINT ({lng} {lat})",
            'visual_vector': visual_vector,
            'behavior_vector': behavior_vector,
            'semantic_vector': semantic_vector,
            'image_vector': image_vector,
            'last_update': time.time(),
            'path_history': [(lat, lng)]  # Track movement history
        }
        r.json().set(f"element:{element_id}", '.', element)
    print(f'Generated {TOTAL_ELEMENTS} elements with vectors')

def update_element_location_with_vectors(element):
    """Enhanced location update that also updates behavior vectors"""
    direction = math.radians(element['direction'])
    speed = element['speed']
    delta_lat = speed * math.cos(direction)
    delta_lng = speed * math.sin(direction)

    # Update position
    element['lat'] += delta_lat
    element['lng'] += delta_lng

    # Keep the element within bounds
    element['lat'] = max(min(element['lat'], 85), -85)
    element['lng'] = max(min(element['lng'], 170), -170)
    element['location'] = f"{element['lng']},{element['lat']}"
    element['polylocation'] = f"POINT ({element['lng']} {element['lat']})"

    # Update path history (keep last 10 positions)
    if 'path_history' not in element:
        element['path_history'] = []
    element['path_history'].append((element['lat'], element['lng']))
    if len(element['path_history']) > 10:
        element['path_history'] = element['path_history'][-10:]

    # Randomly change direction slightly to simulate natural movement
    element['direction'] += random.uniform(-10, 10)
    if element['direction'] >= 360 or element['direction'] < 0:
        element['direction'] = element['direction'] % 360

    # Update behavior vector based on recent movement
    if len(element['path_history']) > 3:
        element['behavior_vector'] = calculate_updated_behavior_vector(element)
    
    # Update semantic vector
    element['semantic_vector'] = generate_semantic_vector(
        element['type'], element['lat'], element['lng'], 
        element['speed'], element['direction']
    )
    
    element['last_update'] = time.time()
    return element

def calculate_updated_behavior_vector(element):
    """Calculate behavior vector based on recent movement patterns"""
    if len(element['path_history']) < 3:
        return element.get('behavior_vector', [0.5, 0.5, 0.5, 0.5])
    
    # Calculate movement stability
    recent_positions = element['path_history'][-5:]
    distances = []
    for i in range(1, len(recent_positions)):
        lat1, lng1 = recent_positions[i-1]
        lat2, lng2 = recent_positions[i]
        dist = math.sqrt((lat2-lat1)**2 + (lng2-lng1)**2)
        distances.append(dist)
    
    stability = 1.0 - (np.std(distances) if len(distances) > 1 else 0)
    predictability = 1.0 - abs(element['speed'] - ELEMENT_CHARACTERISTICS[element['type']]['base_speed']) * 10
    speed_consistency = 1.0 - (np.std([element['speed']] * 5) if 'speed_history' in element else 0)
    path_linearity = min(1.0, len(recent_positions) / 5.0)
    
    return [
        max(0, min(1, stability)),
        max(0, min(1, predictability)), 
        max(0, min(1, speed_consistency)),
        max(0, min(1, path_linearity))
    ]

def prepare_stream_data(element):
    """Prepare element data for Redis stream by converting complex types to strings"""
    stream_data = {}
    for key, value in element.items():
        if isinstance(value, list):
            # Convert lists to JSON strings
            stream_data[key] = json.dumps(value)
        elif isinstance(value, (dict, tuple)):
            # Convert complex types to JSON strings
            stream_data[key] = json.dumps(value)
        elif isinstance(value, (int, float, str)):
            # Keep simple types as is
            stream_data[key] = value
        else:
            # Convert everything else to string
            stream_data[key] = str(value)
    return stream_data

def update_elements():
    while True:
        for user_id in r.smembers('users'):
            user_elements_key = f"user:{user_id}:elements"
            user_stream_key = f"user:{user_id}:stream"
            user_elements = r.smembers(user_elements_key)
            for element_id in user_elements:
                element = r.json().get(f"{element_id}")
                if element:
                    updated_element = update_element_location_with_vectors(element)
                    r.json().set(f"{element_id}", '.', updated_element)
                    
                    # Prepare stream data - convert complex types to strings
                    stream_data = prepare_stream_data(updated_element)
                    r.xadd(user_stream_key, stream_data)
        time.sleep(movement_speed)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    user_id = request.json['user_id']
    num_elements = request.json.get('num_elements', 10)
    r.sadd('users', user_id)
    user_elements_key = f"user:{user_id}:elements"
    user_stream_key = f"user:{user_id}:stream"
    elements_ids = random.sample(range(1, min(TOTAL_ELEMENTS, num_elements*2)), num_elements)
    for key in elements_ids:
        r.sadd(user_elements_key, f'element:{key}')
        element = r.json().get(f'element:{key}')
        if element:
            # Prepare stream data - convert complex types to strings
            stream_data = prepare_stream_data(element)
            r.xadd(user_stream_key, stream_data)
    print(f'Registered user {user_id} with elements:', elements_ids)
    return jsonify({'status': 'registered', 'user_id': user_id})

@app.route('/query', methods=['POST'])
def query_elements():
    data = request.json
    query_type = data.get('query_type')
    element_type = data.get('element_type')
    user_id = data.get('user_id')
    region = data.get('region')

    elements_res = {}
    if query_type == 'type':
        query = f"@type:{{{element_type}}}"
    elif query_type == 'region':
        lat_min = region['lat_min']
        lat_max = region['lat_max']
        lng_min = region['lng_min']
        lng_max = region['lng_max']
        query = f"@location:[{lng_min} {lat_min} {lng_max} {lat_max}]"
    elif query_type == 'type_region':
        lat_min = region['lat_min']
        lat_max = region['lat_max']
        lng_min = region['lng_min']
        lng_max = region['lng_max']
        query = f"@type:{element_type} @location:[{lng_min} {lat_min} {lng_max} {lat_max}]"
    else:
        query = "*"
    
    q = Query(query).paging(0, 10000)
    result = r.ft("elements_idx").search(q)
    for doc in result.docs:
        element_id = doc.id
        if r.sismember(f'user:{user_id}:elements', element_id):
            element = json.loads(doc.json)
            elements_res[element_id] = {
                'id': element_id,
                'type': element['type'], 
                'lat': element['lat'], 
                'lng': element['lng']
            }

    return jsonify({"elements": elements_res})

@app.route('/query_polygon', methods=['POST'])
def query_polygon():
    data = request.json
    points = data['points']
    user_id = data.get('user_id')
    polygon_str = " ".join([f"{p[0]} {p[1]}," for p in points])

    elements_res = {}
    query = f"@poly_loc:[within $shape]"
    query_params = {"shape": f"POLYGON(({polygon_str}))"}
    q = Query(query).dialect(3).paging(0, 10000)
    result = r.ft("elements_idx").search(query=q, query_params=query_params)
    
    for doc in result.docs:
        element_id = doc.id
        if r.sismember(f'user:{user_id}:elements', element_id):
            element = json.loads(doc.json)
            elements_res[element_id] = {
                'id': element_id,
                'type': element[0]['type'],
                'lat': element[0]['lat'], 
                'lng': element[0]['lng']
            }

    return jsonify({"elements": elements_res})

@app.route('/query_circle', methods=['POST'])
def query_circle():
    data = request.json
    center = data['center']
    radius = data['radius'] / 1000
    user_id = data['user_id']
    
    elements_res = {}
    query = f"@location:[{center['lng']} {center['lat']} {radius} km]"
    q = Query(query).paging(0, 10000)
    result = r.ft("elements_idx").search(query=q)
    
    for doc in result.docs:
        element_id = doc.id
        if r.sismember(f'user:{user_id}:elements', element_id):
            element = json.loads(doc.json)
            elements_res[element_id] = {
                'id': element_id,
                'type': element['type'],
                'lat': element['lat'], 
                'lng': element['lng']
            }

    return jsonify({"elements": elements_res})

@app.route('/vector_search', methods=['POST'])
def vector_search():
    """Unified vector search endpoint for similarity queries"""
    data = request.json
    search_type = data.get('search_type')  # 'visual', 'behavior', 'semantic'
    user_id = data.get('user_id')
    k = data.get('k', 20)  # Number of results
    
    elements_res = {}
    
    try:
        if search_type == 'visual':
            # Visual similarity search
            target_vector = data.get('target_vector')
            if not target_vector:
                return jsonify({"error": "target_vector required for visual search"}), 400
            
            vector_query = f"*=>[KNN {k} @visual_vector $vec AS score]"
            query_params = {"vec": np.array(target_vector, dtype=np.float32).tobytes()}
            
        elif search_type == 'behavior':
            # Behavior similarity search
            target_element_id = data.get('target_element_id')
            if target_element_id:
                target_element = r.json().get(target_element_id)
                if not target_element:
                    return jsonify({"error": "target element not found"}), 404
                target_vector = target_element['behavior_vector']
            else:
                target_vector = data.get('target_vector')
            
            if not target_vector:
                return jsonify({"error": "target_vector or target_element_id required"}), 400
                
            vector_query = f"*=>[KNN {k} @behavior_vector $vec AS score]"
            query_params = {"vec": np.array(target_vector, dtype=np.float32).tobytes()}
            
        elif search_type == 'semantic':
            # Semantic search from natural language
            query_text = data.get('query_text')
            if not query_text:
                return jsonify({"error": "query_text required for semantic search"}), 400
            
            if EMBEDDINGS_AVAILABLE:
                try:
                    target_vector = sentence_model.encode(query_text).tolist()
                except:
                    return jsonify({"error": "Failed to generate semantic embedding"}), 500
            else:
                # Fallback: simple keyword matching converted to vector
                target_vector = generate_fallback_semantic_vector(query_text)
            
            vector_query = f"*=>[KNN {k} @semantic_vector $vec AS score]"
            query_params = {"vec": np.array(target_vector, dtype=np.float32).tobytes()}
            
        else:
            return jsonify({"error": "Invalid search_type"}), 400
        
        # Execute vector search
        q = Query(vector_query).return_fields("id", "type", "lat", "lng", "score").paging(0, k).dialect(2)
        result = r.ft("elements_idx").search(query=q, query_params=query_params)
        
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                elements_res[element_id] = {
                    'id': element_id,
                    'type': doc.type,
                    'lat': float(doc.lat),
                    'lng': float(doc.lng),
                    'similarity_score': float(doc.score) if hasattr(doc, 'score') else 0.0
                }
    
    except Exception as e:
        print(f"Vector search error: {e}")
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"elements": elements_res, "search_type": search_type})

def generate_fallback_semantic_vector(query_text):
    """Generate a simple semantic vector when sentence transformers isn't available"""
    query_lower = query_text.lower()
    
    # Simple keyword-based encoding
    keywords = {
        'fast': [1, 0, 0, 0, 0],
        'slow': [0, 1, 0, 0, 0], 
        'airplane': [0, 0, 1, 0, 0],
        'bird': [0, 0, 0, 1, 0],
        'motorcycle': [0, 0, 0, 0, 1],
        'bike': [0, 0, 0, 0, 1],
        'person': [0, 0, 0, 0, 1]
    }
    
    vector = [0.0] * 10
    for keyword, encoding in keywords.items():
        if keyword in query_lower:
            for i, val in enumerate(encoding):
                vector[i] += val
    
    # Add some randomness for the remaining dimensions
    vector.extend([random.random() for _ in range(10 - len(vector) % 10)])
    return vector[:10]

@app.route('/image_search', methods=['POST'])
def image_search():
    """Search for elements similar to uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image_file = request.files['image']
        user_id = request.form.get('user_id')
        k = int(request.form.get('k', 10))
        
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
        
        # Process the image
        image = Image.open(image_file.stream)
        input_image_vector = image_to_vector(image_file.stream)
        # Simple image analysis for demo purposes
        # In production, use CLIP or similar model
        visual_features = analyze_image_simple(image)
        
        # Perform vector search
        vector_query = f"*=>[KNN {k} @image_vector $vec AS score]"
        query_params = {"vec": np.array(input_image_vector, dtype=np.float32).tobytes()}
        
        q = Query(vector_query).return_fields("id", "type", "lat", "lng", "score").paging(0, k).dialect(2)
        result = r.ft("elements_idx").search(query=q, query_params=query_params)
        
        elements_res = {}
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                elements_res[element_id] = {
                    'id': element_id,
                    'type': doc.type,
                    'lat': float(doc.lat),
                    'lng': float(doc.lng),
                    'similarity_score': float(doc.score) if hasattr(doc, 'score') else 0.0
                }
        
        return jsonify({
            "elements": elements_res, 
            "search_type": "image",
            "detected_features": visual_features
        })
        
    except Exception as e:
        print(f"Image search error: {e}")
        return jsonify({"error": str(e)}), 500

def analyze_image_simple(image):
    """Simple image analysis for demo purposes"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get image statistics
    width, height = image.size
    aspect_ratio = width / height
    
    # Simple color analysis
    pixels = list(image.getdata())
    avg_r = sum(p[0] for p in pixels) / len(pixels) / 255.0
    avg_g = sum(p[1] for p in pixels) / len(pixels) / 255.0  
    avg_b = sum(p[2] for p in pixels) / len(pixels) / 255.0
    
    # Create a simple feature vector
    # [aspect_ratio, avg_red, avg_green, avg_blue, brightness]
    brightness = (avg_r + avg_g + avg_b) / 3.0
    
    return [
        min(aspect_ratio, 2.0) / 2.0,  # Normalize aspect ratio
        avg_r,
        avg_g, 
        avg_b,
        brightness
    ]

@app.route('/natural_language_search', methods=['POST'])
def natural_language_search():
    return enhanced_natural_language_search()
    """Search using natural language queries"""
    data = request.json
    query_text = data.get('query_text', '')
    user_id = data.get('user_id')
    k = data.get('k', 20)
    
    if not query_text or not user_id:
        return jsonify({"error": "query_text and user_id required"}), 400
    
    try:
        # Parse the natural language query
        parsed_query = parse_natural_language_query(query_text)
        
        # Execute the appropriate search
        if parsed_query['type'] == 'semantic':
            return vector_search_internal('semantic', query_text, user_id, k)
        elif parsed_query['type'] == 'traditional':
            return execute_traditional_query(parsed_query, user_id)
        else:
            return jsonify({"error": "Could not understand query"}), 400
            
    except Exception as e:
        print(f"Natural language search error: {e}")
        return jsonify({"error": str(e)}), 500

def parse_natural_language_query(query_text):
    """Parse natural language into structured query using AI"""
    query_lower = query_text.lower()
    
    # Always try semantic search first if embeddings are available
    if EMBEDDINGS_AVAILABLE:
        return {'type': 'semantic', 'intent': 'ai_search', 'query': query_text}
    
    # Fallback to enhanced keyword matching when AI is not available
    query_params = {}
    
    # Extract element types with synonyms
    element_synonyms = {
        'airplane': ['plane', 'aircraft', 'jet', 'flight'],
        'bird': ['birds', 'flying animal', 'winged creature'],
        'motorcycle': ['motorbike', 'bike with engine', 'motor'],
        'bike': ['bicycle', 'cycle', 'pedal'],
        'person': ['people', 'human', 'pedestrian', 'walker']
    }
    
    for element_type, synonyms in element_synonyms.items():
        if element_type in query_lower or any(syn in query_lower for syn in synonyms):
            query_params['element_type'] = element_type
            break
    
    # Extract speed indicators with more variations
    speed_fast = ['fast', 'quick', 'rapid', 'speedy', 'swift', 'high speed', 'racing']
    speed_slow = ['slow', 'sluggish', 'crawling', 'leisurely', 'low speed', 'gentle']
    
    if any(word in query_lower for word in speed_fast):
        query_params['speed_filter'] = 'high'
    elif any(word in query_lower for word in speed_slow):
        query_params['speed_filter'] = 'low'
    
    # Extract location indicators with more regions
    locations = {
        'europe': {'lat_min': 35, 'lat_max': 70, 'lng_min': -10, 'lng_max': 40},
        'asia': {'lat_min': 10, 'lat_max': 70, 'lng_min': 60, 'lng_max': 150},
        'america': {'lat_min': 15, 'lat_max': 70, 'lng_min': -130, 'lng_max': -60},
        'africa': {'lat_min': -35, 'lat_max': 35, 'lng_min': -20, 'lng_max': 50},
        'north america': {'lat_min': 25, 'lat_max': 70, 'lng_min': -130, 'lng_max': -60},
        'south america': {'lat_min': -55, 'lat_max': 15, 'lng_min': -80, 'lng_max': -35},
        'australia': {'lat_min': -45, 'lat_max': -10, 'lng_min': 110, 'lng_max': 155},
        'arctic': {'lat_min': 66, 'lat_max': 90, 'lng_min': -180, 'lng_max': 180},
        'antarctic': {'lat_min': -90, 'lat_max': -60, 'lng_min': -180, 'lng_max': 180}
    }
    
    for location, bounds in locations.items():
        if location in query_lower:
            query_params['region'] = bounds
            break
    
    # Check for behavioral patterns
    behavior_patterns = {
        'erratic': ['erratic', 'chaotic', 'unpredictable', 'random'],
        'stable': ['stable', 'steady', 'consistent', 'regular'],
        'unusual': ['unusual', 'strange', 'odd', 'weird', 'anomalous']
    }
    
    for pattern, keywords in behavior_patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            query_params['behavior_filter'] = pattern
            break
    
    if query_params:
        return {'type': 'traditional', 'params': query_params}
    else:
        # If no specific patterns found, try semantic search anyway
        return {'type': 'semantic', 'intent': 'general', 'query': query_text}

def vector_search_internal(search_type, query_text, user_id, k):
    """Internal vector search function for semantic queries"""
    if not EMBEDDINGS_AVAILABLE:
        # Fallback to enhanced keyword search
        return execute_enhanced_keyword_search(query_text, user_id)
    
    try:
        # Generate semantic embedding for the query
        query_vector = sentence_model.encode(query_text).tolist()
        
        # Perform vector search
        vector_query = f"*=>[KNN {k} @semantic_vector $vec AS score]"
        query_params = {"vec": np.array(query_vector, dtype=np.float32).tobytes()}
        
        q = Query(vector_query).return_fields("id", "type", "lat", "lng", "score").paging(0, k).dialect(2)
        result = r.ft("elements_idx").search(query=q, query_params=query_params)
        
        elements_res = {}
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                elements_res[element_id] = {
                    'id': element_id,
                    'type': doc.type,
                    'lat': float(doc.lat),
                    'lng': float(doc.lng),
                    'similarity_score': float(doc.score) if hasattr(doc, 'score') else 0.0
                }
        
        return jsonify({
            "elements": elements_res, 
            "search_type": "semantic_ai",
            "query": query_text,
            "embedding_used": True
        })
        
    except Exception as e:
        print(f"AI semantic search error: {e}")
        # Fallback to enhanced keyword search
        return execute_enhanced_keyword_search(query_text, user_id)

def execute_enhanced_keyword_search(query_text, user_id):
    """Enhanced keyword-based search when AI is not available"""
    query_lower = query_text.lower()
    
    # Build a more sophisticated query based on keywords
    query_parts = []
    
    # Element type detection with synonyms
    element_mappings = {
        'airplane': ['plane', 'aircraft', 'jet', 'flight', 'flying machine'],
        'bird': ['birds', 'flying animal', 'winged creature', 'avian'],
        'motorcycle': ['motorbike', 'bike with engine', 'motor', 'scooter'],
        'bike': ['bicycle', 'cycle', 'pedal bike', 'two wheeler'],
        'person': ['people', 'human', 'pedestrian', 'walker', 'individual']
    }
    
    detected_types = []
    for element_type, synonyms in element_mappings.items():
        if element_type in query_lower or any(syn in query_lower for syn in synonyms):
            detected_types.append(element_type)
    
    if detected_types:
        type_query = " | ".join([f"@type:{{{t}}}" for t in detected_types])
        query_parts.append(f"({type_query})")
    
    # Speed detection
    if any(word in query_lower for word in ['fast', 'quick', 'rapid', 'speedy', 'swift', 'racing']):
        query_parts.append("@speed:[0.04 +inf]")  # Fast elements
    elif any(word in query_lower for word in ['slow', 'sluggish', 'crawling', 'leisurely']):
        query_parts.append("@speed:[-inf 0.02]")  # Slow elements
    
    # Location detection (expanded)
    location_keywords = {
        'north': {'lat_min': 30, 'lat_max': 90, 'lng_min': -180, 'lng_max': 180},
        'south': {'lat_min': -90, 'lat_max': -30, 'lng_min': -180, 'lng_max': 180},
        'europe': {'lat_min': 35, 'lat_max': 70, 'lng_min': -10, 'lng_max': 40},
        'asia': {'lat_min': 10, 'lat_max': 70, 'lng_min': 60, 'lng_max': 150},
        'africa': {'lat_min': -35, 'lat_max': 35, 'lng_min': -20, 'lng_max': 50},
        'america': {'lat_min': -55, 'lat_max': 70, 'lng_min': -130, 'lng_max': -35}
    }
    
    for location, bounds in location_keywords.items():
        if location in query_lower:
            lat_query = f"@lat:[{bounds['lat_min']} {bounds['lat_max']}]"
            lng_query = f"@lng:[{bounds['lng_min']} {bounds['lng_max']}]"
            query_parts.extend([lat_query, lng_query])
            break
    
    # Combine query parts
    final_query = " ".join(query_parts) if query_parts else "*"
    
    try:
        q = Query(final_query).paging(0, 100)
        result = r.ft("elements_idx").search(q)
        
        elements_res = {}
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                element = json.loads(doc.json)
                elements_res[element_id] = {
                    'id': element_id,
                    'type': element['type'],
                    'lat': element['lat'], 
                    'lng': element['lng'],
                    'keyword_match': True
                }
        
        return jsonify({
            "elements": elements_res, 
            "search_type": "enhanced_keyword",
            "query": query_text,
            "embedding_used": False,
            "detected_types": detected_types
        })
        
    except Exception as e:
        print(f"Enhanced keyword search error: {e}")
        return jsonify({"error": "Search failed", "query": query_text}), 500

def execute_traditional_query(parsed_query, user_id):
    """Execute traditional Redis queries from parsed natural language"""
    params = parsed_query['params']
    query_parts = []
    
    # Build query string
    if 'element_type' in params:
        query_parts.append(f"@type:{{{params['element_type']}}}")
    
    if 'speed_filter' in params:
        if params['speed_filter'] == 'high':
            query_parts.append("@speed:[0.05 +inf]")
        elif params['speed_filter'] == 'low':
            query_parts.append("@speed:[-inf 0.02]")
    
    if 'region' in params:
        region = params['region']
        lat_query = f"@lat:[{region['lat_min']} {region['lat_max']}]"
        lng_query = f"@lng:[{region['lng_min']} {region['lng_max']}]"
        query_parts.extend([lat_query, lng_query])
    
    # Handle behavior filters (new feature)
    if 'behavior_filter' in params:
        behavior = params['behavior_filter']
        if behavior == 'erratic':
            # Look for elements with low stability (first component of behavior vector)
            # This is a simplified approach - in reality you'd want more sophisticated behavior analysis
            pass  # We'll handle this differently since we can't easily query vector components
        elif behavior == 'stable':
            pass  # Similar handling needed
    
    query = " ".join(query_parts) if query_parts else "*"
    
    # Execute query
    try:
        q = Query(query).paging(0, 100)
        result = r.ft("elements_idx").search(q)
        
        elements_res = {}
        for doc in result.docs:
            element_id = doc.id
            if r.sismember(f'user:{user_id}:elements', element_id):
                element = json.loads(doc.json)
                elements_res[element_id] = {
                    'id': element_id,
                    'type': element['type'],
                    'lat': element['lat'], 
                    'lng': element['lng']
                }
        
        return jsonify({
            "elements": elements_res, 
            "search_type": "traditional_nlp",
            "parsed_params": params
        })
        
    except Exception as e:
        print(f"Traditional query error: {e}")
        return jsonify({"error": "Query execution failed"}), 500

@app.route('/find_similar', methods=['POST'])
def find_similar():
    """Find elements similar to a specific element"""
    data = request.json
    target_element_id = data.get('target_element_id')
    user_id = data.get('user_id')
    similarity_type = data.get('similarity_type', 'behavior')  # 'visual', 'behavior', 'semantic'
    k = data.get('k', 10)
    
    if not target_element_id or not user_id:
        return jsonify({"error": "target_element_id and user_id required"}), 400
    
    # Use the vector search endpoint
    request_data = {
        'search_type': similarity_type,
        'target_element_id': target_element_id,
        'user_id': user_id,
        'k': k
    }
    
    with app.test_request_context('/vector_search', json=request_data):
        return vector_search()

@app.route('/anomaly_detection', methods=['POST'])
def anomaly_detection():
    """Detect anomalous behavior patterns"""
    data = request.json
    user_id = data.get('user_id')
    threshold = data.get('threshold', 0.3)  # Anomaly threshold
    
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    
    try:
        user_elements_key = f"user:{user_id}:elements"
        anomalous_elements = {}
        
        for element_id in r.smembers(user_elements_key):
            element = r.json().get(element_id)
            if element:
                # Calculate anomaly score
                anomaly_score = calculate_anomaly_score(element)
                
                if anomaly_score > threshold:
                    anomalous_elements[element_id] = {
                        'id': element_id,
                        'type': element['type'],
                        'lat': element['lat'],
                        'lng': element['lng'],
                        'anomaly_score': anomaly_score,
                        'anomaly_reasons': get_anomaly_reasons(element, anomaly_score)
                    }
        
        return jsonify({
            "anomalous_elements": anomalous_elements,
            "threshold": threshold,
            "total_checked": len(list(r.smembers(user_elements_key)))
        })
        
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return jsonify({"error": str(e)}), 500

def calculate_anomaly_score(element):
    """Calculate how anomalous an element's behavior is"""
    element_type = element['type']
    characteristics = ELEMENT_CHARACTERISTICS[element_type]
    
    anomaly_factors = []
    
    # Speed anomaly
    expected_speed = characteristics['base_speed']
    speed_diff = abs(element['speed'] - expected_speed) / expected_speed
    anomaly_factors.append(min(speed_diff, 1.0))
    
    # Behavior vector anomaly (compare to type average)
    expected_behavior = characteristics['behavior_pattern']
    current_behavior = element.get('behavior_vector', expected_behavior)
    
    behavior_diff = sum(abs(a - b) for a, b in zip(current_behavior, expected_behavior)) / len(expected_behavior)
    anomaly_factors.append(behavior_diff)
    
    # Location-based anomaly (elements shouldn't be at extreme coordinates too often)
    lat_anomaly = 1.0 if abs(element['lat']) > 80 else 0.0
    lng_anomaly = 1.0 if abs(element['lng']) > 170 else 0.0
    anomaly_factors.extend([lat_anomaly, lng_anomaly])
    
    # Movement pattern anomaly
    if 'path_history' in element and len(element['path_history']) > 3:
        positions = element['path_history'][-5:]
        movements = []
        for i in range(1, len(positions)):
            lat1, lng1 = positions[i-1]
            lat2, lng2 = positions[i]
            movement = math.sqrt((lat2-lat1)**2 + (lng2-lng1)**2)
            movements.append(movement)
        
        if len(movements) > 1:
            movement_variance = np.var(movements) if len(movements) > 1 else 0
            anomaly_factors.append(min(movement_variance * 100, 1.0))
    
    return sum(anomaly_factors) / len(anomaly_factors)

def get_anomaly_reasons(element, score):
    """Get human-readable reasons for anomaly detection"""
    reasons = []
    element_type = element['type']
    characteristics = ELEMENT_CHARACTERISTICS[element_type]
    
    # Speed check
    expected_speed = characteristics['base_speed']
    if abs(element['speed'] - expected_speed) / expected_speed > 0.5:
        if element['speed'] > expected_speed:
            reasons.append(f"Moving unusually fast for {element_type}")
        else:
            reasons.append(f"Moving unusually slow for {element_type}")
    
    # Location check
    if abs(element['lat']) > 80:
        reasons.append("Located in extreme latitude (near poles)")
    if abs(element['lng']) > 170:
        reasons.append("Located in extreme longitude")
    
    # Behavior check
    behavior_vector = element.get('behavior_vector', [])
    expected_behavior = characteristics['behavior_pattern']
    
    if len(behavior_vector) >= 4 and len(expected_behavior) >= 4:
        if abs(behavior_vector[0] - expected_behavior[0]) > 0.3:
            reasons.append("Unusual movement stability")
        if abs(behavior_vector[1] - expected_behavior[1]) > 0.3:
            reasons.append("Unpredictable movement pattern")
    
    return reasons

@socketio.on('set_speed')
def handle_set_speed(data):
    global movement_speed
    movement_speed = data['speed']
    print(f'Set movement speed to {movement_speed} seconds')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('subscribe')
def handle_subscribe(data):
    user_id = data['user_id']
    stream_key = f"user:{user_id}:stream"
    last_id_key = f"user:{user_id}:last_id"

    last_id = r.get(last_id_key) or '0'
    print(f'Subscribed to {stream_key} starting from {last_id}')

    while True:
        messages = r.xread(streams={stream_key: last_id}, count=100, block=1000)
        if messages:
            for message in messages[0][1]:
                last_id = message[0]
                msg = {k: v for k, v in message[1].items()}
                msg['id'] = f"element:{msg.get('id')}"
                emit('update', msg)
                r.set(last_id_key, last_id)
        else:
            time.sleep(1)

@app.route('/map_data/<user_id>')
def get_map_data(user_id):
    user_elements_key = f"user:{user_id}:elements"
    elements = {}
    for element_id in r.smembers(user_elements_key):
        element = r.json().get(f"{element_id}")
        if element:
            elements[element_id] = element
    return jsonify({"elements": elements})

@app.route('/get_element_details/<element_id>')
def get_element_details(element_id):
    """Get detailed information about a specific element"""
    element = r.json().get(element_id)
    if element:
        return jsonify(element)
    else:
        return jsonify({"error": "Element not found"}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "total_elements": TOTAL_ELEMENTS,
        "embeddings_available": EMBEDDINGS_AVAILABLE,
        "redis_connected": r.ping(),
        "ai_models": {
            "sentence_transformers": EMBEDDINGS_AVAILABLE,
            "model_name": "all-MiniLM-L6-v2" if EMBEDDINGS_AVAILABLE else None
        }
    })

@app.route('/ai_status')
def ai_status():
    """Get AI capabilities status"""
    return jsonify({
        "embeddings_available": EMBEDDINGS_AVAILABLE,
        "fallback_mode": not EMBEDDINGS_AVAILABLE,
        "capabilities": {
            "semantic_search": EMBEDDINGS_AVAILABLE,
            "image_analysis": True,  # Basic image analysis always available
            "enhanced_nlp": True,    # Enhanced keyword matching always available
            "vector_similarity": EMBEDDINGS_AVAILABLE
        }
    })


print('Starting enhanced Redis workshop application...')
create_index()
create_elements_vectors()
generate_elements()

# Start the update thread
update_thread = Thread(target=update_elements)
update_thread.daemon = True
update_thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)