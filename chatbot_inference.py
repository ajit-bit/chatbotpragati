import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import nltk
from pymongo import MongoClient
from bson.objectid import ObjectId
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta
import logging
from groq import Groq 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt_tab', quiet=True)
warnings.filterwarnings('ignore')

class LLMEnhancer:  
    """Groq LLaMA API integration for database enhancement"""
    
    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = "deepseek-r1-distill-llama-70b"
        
        self.collection_schemas = {
            'industries': {
                'fields': ['name', 'market_size_billion', 'growth_rate', 'description'],
                'keywords': ['industry', 'market size', 'market', 'sector', 'growth rate']
            },
            'funding_data': {
                'fields': ['startup_name', 'amount_million', 'funding_round', 'valuation_million', 'investors'],
                'keywords': ['funding', 'investment', 'raised', 'investors', 'valuation', 'round', 'startup']
            },
            'competitors': {
                'fields': ['company_name', 'revenue_million', 'founded_year', 'employees', 'funding_raised_million', 'website'],
                'keywords': ['company', 'startup', 'business', 'revenue', 'employees', 'founded', 'competitor']
            },
            'market_trends': {
                'fields': ['trend_name', 'impact_score', 'description', 'date_identified'],
                'keywords': ['trend', 'trending', 'emerging', 'growth', 'pattern', 'technology']
            },
            'customer_segments': {
                'fields': ['segment_name', 'size_million', 'demographics', 'avg_spending', 'pain_points'],
                'keywords': ['customer', 'segment', 'demographics', 'users', 'audience', 'target market']
            }
        }
    
    def determine_data_type(self, query):
        query_lower = query.lower()
        scores = {}
        
        for collection, schema in self.collection_schemas.items():
            score = sum(1 for keyword in schema['keywords'] if keyword in query_lower)
            if score > 0:
                scores[collection] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        if any(word in query_lower for word in ['company', 'startup', 'business']):
            return 'competitors'
        elif any(word in query_lower for word in ['funding', 'investment', 'raised']):
            return 'funding_data'
        elif any(word in query_lower for word in ['market', 'industry', 'size']):
            return 'industries'
        elif any(word in query_lower for word in ['trend', 'emerging']):
            return 'market_trends'
        else:
            return 'customer_segments'
    
    def create_extraction_prompt(self, query, collection_type):
        schema = self.collection_schemas[collection_type]
        fields = schema['fields']
        
        prompt = f"""
                You are a market research data analyst. Based on the user query: "{query}"

                Please provide accurate, factual information that can be used to populate a {collection_type} database record.

                Return your response as a JSON object with the following fields:
                {json.dumps(fields, indent=2)}

                Guidelines:
                - Only include factual, verifiable information
                - Use realistic numbers for financial data (in millions for revenue/funding)
                - For market_size_billion, provide the value in billions
                - For growth_rate, provide annual percentage (e.g., 15.5 for 15.5%)
                - For impact_score, use a scale of 1-10
                - For dates, use YYYY-MM-DD format
                - If you don't have specific information for a field, use null
                - Be conservative with estimates and clearly indicate if data is approximate

                Example format:
                {{
                    "{fields[0]}": "example_value",
                    "{fields[1] if len(fields) > 1 else 'field2'}": "example_value",
                    ...
                }}

                Query: "{query}"

                Please provide the JSON response:
                """
        return prompt
    
    def extract_and_structure_data(self, query, collection_type):
        try:
            prompt = self.create_extraction_prompt(query, collection_type)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise market research assistant. Provide accurate, structured JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            response_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    data = json.loads(json_str)
                    return self.clean_and_validate_data(data, collection_type)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLaMA response: {json_str}")
                    return None
            else:
                logger.error(f"No JSON found in LLaMA response: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return None
    
    def clean_and_validate_data(self, data, collection_type):
        schema = self.collection_schemas[collection_type]
        cleaned_data = {}
        
        for field in schema['fields']:
            value = data.get(field)
            
            if value is None or value == "null":
                continue
                
            if field.endswith('_million') or field.endswith('_billion'):
                try:
                    if isinstance(value, str):
                        numeric_str = re.sub(r'[^\d.]', '', value)
                        if numeric_str:
                            cleaned_data[field] = float(numeric_str)
                    elif isinstance(value, (int, float)):
                        cleaned_data[field] = float(value)
                except (ValueError, TypeError):
                    continue
            
            elif field == 'growth_rate':
                try:
                    if isinstance(value, str):
                        numeric_str = re.sub(r'[^\d.]', '', value)
                        if numeric_str:
                            cleaned_data[field] = float(numeric_str)
                    elif isinstance(value, (int, float)):
                        cleaned_data[field] = float(value)
                except (ValueError, TypeError):
                    continue
            
            elif field == 'founded_year':
                try:
                    year = int(re.search(r'\d{4}', str(value)).group())
                    if 1800 <= year <= datetime.now().year:
                        cleaned_data[field] = year
                except (ValueError, TypeError, AttributeError):
                    continue
            
            elif field == 'employees':
                try:
                    if isinstance(value, str):
                        numeric_str = re.sub(r'[^\d]', '', value)
                        if numeric_str:
                            cleaned_data[field] = int(numeric_str)
                    elif isinstance(value, (int, float)):
                        cleaned_data[field] = int(value)
                except (ValueError, TypeError):
                    continue
            
            elif field == 'impact_score':
                try:
                    score = float(value)
                    if 1 <= score <= 10:
                        cleaned_data[field] = score
                except (ValueError, TypeError):
                    continue
            
            elif field in ['date_identified']:
                if isinstance(value, str) and value.strip():
                    try:
                        datetime.strptime(value, '%Y-%m-%d')
                        cleaned_data[field] = value
                    except ValueError:
                        cleaned_data[field] = datetime.now().strftime('%Y-%m-%d')
            
            else:
                if isinstance(value, str) and value.strip():
                    cleaned_data[field] = value.strip()
        
        cleaned_data['_source'] = 'groq_api' 
        cleaned_data['_created_at'] = datetime.now()
        cleaned_data['_query_context'] = collection_type
        
        return cleaned_data if cleaned_data else None
    
    def generate_response_from_data(self, data, collection_type, original_query):
        if not data:
            return None
        
        try:
            if collection_type == 'industries':
                response = f"{data.get('name', 'This industry')} has a market size of ${data.get('market_size_billion', 0)} billion with a growth rate of {data.get('growth_rate', 0)}%. {data.get('description', '')}"
            
            elif collection_type == 'funding_data':
                response = f"{data.get('startup_name', 'This company')} raised ${data.get('amount_million', 0)} million in {data.get('funding_round', 'funding')}. The company is valued at ${data.get('valuation_million', 0)} million. Investors include {data.get('investors', 'various investors')}."
            
            elif collection_type == 'competitors':
                response = f"{data.get('company_name', 'This company')} generates ${data.get('revenue_million', 0)} million in revenue annually. The company was founded in {data.get('founded_year', 'N/A')} and has {data.get('employees', 0)} employees. They have raised ${data.get('funding_raised_million', 0)} million in funding."
            
            elif collection_type == 'market_trends':
                response = f"{data.get('trend_name', 'This trend')} is a market trend with an impact score of {data.get('impact_score', 0)}/10. {data.get('description', '')} Identified on {data.get('date_identified', 'recently')}."
            
            elif collection_type == 'customer_segments':
                response = f"{data.get('segment_name', 'This customer segment')} is a customer segment of {data.get('size_million', 0)} million people. Demographics: {data.get('demographics', 'N/A')}. Average spending: ${data.get('avg_spending', 0)}. Pain points: {data.get('pain_points', 'N/A')}"
            
            else:
                response = "I found some relevant information, but couldn't format it properly."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response from data: {e}")
            return None


class TextPreprocessor:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text):
        tokens = word_tokenize(text)
        return [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 1]

    def preprocess(self, text):
        return self.tokenize_text(self.clean_text(text))


class RealTimeMongoSearcher:
    def __init__(self, db):
        self.db = db
        self.collection_mappings = {
            'industries': ['market size', 'industry', 'market', 'sector', 'growth rate'],
            'funding_data': ['funding', 'investment', 'raised', 'investors', 'valuation', 'round'],
            'competitors': ['company', 'startup', 'business', 'revenue', 'employees', 'founded'],
            'market_trends': ['trend', 'trending', 'emerging', 'growth', 'pattern'],
            'customer_segments': ['customer', 'segment', 'demographics', 'users', 'audience']
        }
        
    def extract_entities_from_query(self, query):
        query_lower = query.lower()
        potential_entities = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        quoted_entities = re.findall(r'"([^"]*)"', query)
        all_entities = potential_entities + quoted_entities
        common_words = {'What', 'How', 'Tell', 'Show', 'Give', 'Find', 'Search', 'About', 'The', 'This', 'That'}
        entities = [entity for entity in all_entities if entity not in common_words]
        return entities
    
    def determine_query_type(self, query):
        query_lower = query.lower()
        relevant_collections = []
        
        for collection, keywords in self.collection_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_collections.append(collection)
        
        if not relevant_collections:
            relevant_collections = list(self.collection_mappings.keys())
            
        return relevant_collections
    
    def calculate_relevance_score(self, query, text_fields):
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        total_score = 0
        total_words = 0
        
        for field_value in text_fields:
            if field_value:
                field_words = set(re.findall(r'\b\w+\b', str(field_value).lower()))
                matches = len(query_words.intersection(field_words))
                total_score += matches
                total_words += len(field_words)
        
        if total_words > 0:
            return total_score / len(query_words) if len(query_words) > 0 else 0
        return 0
    
    def search_industries(self, query, entities):
        try:
            query_lower = query.lower()
            search_conditions = []
            
            for entity in entities:
                search_conditions.extend([
                    {"name": {"$regex": entity, "$options": "i"}},
                    {"description": {"$regex": entity, "$options": "i"}}
                ])
            
            keywords = ['fintech', 'tech', 'healthcare', 'finance', 'retail', 'e-commerce', 'saas']
            for keyword in keywords:
                if keyword in query_lower:
                    search_conditions.extend([
                        {"name": {"$regex": keyword, "$options": "i"}},
                        {"description": {"$regex": keyword, "$options": "i"}}
                    ])
            
            if search_conditions:
                industries = list(self.db.industries.find({"$or": search_conditions}))
            else:
                industries = list(self.db.industries.find().limit(5))
            
            scored_results = []
            for industry in industries:
                text_fields = [
                    industry.get('name', ''),
                    industry.get('description', '')
                ]
                score = self.calculate_relevance_score(query, text_fields)
                
                result = {
                    'text': (
                        f"The {industry.get('name', '')} market size is ${industry.get('market_size_billion', 0)} billion "
                        f"with a growth rate of {industry.get('growth_rate', 0)}%. "
                        f"{industry.get('description', '')}"
                    ),
                    'score': score
                }
                scored_results.append(result)
            
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[0]['text'] if scored_results and scored_results[0]['score'] > 0 else None
            
        except Exception as e:
            logger.error(f"Error searching industries: {e}")
            return None
    
    def search_funding_data(self, query, entities):
        try:
            query_lower = query.lower()
            search_conditions = []
            
            for entity in entities:
                search_conditions.extend([
                    {"startup_name": {"$regex": entity, "$options": "i"}},
                    {"investors": {"$regex": entity, "$options": "i"}}
                ])
            
            words = re.findall(r'\b\w+\b', query_lower)
            for word in words:
                if len(word) > 2:
                    search_conditions.extend([
                        {"startup_name": {"$regex": word, "$options": "i"}},
                        {"investors": {"$regex": word, "$options": "i"}}
                    ])
            
            if search_conditions:
                funding_records = list(self.db.funding_data.find({"$or": search_conditions}))
            else:
                funding_records = list(self.db.funding_data.find().limit(5))
            
            scored_results = []
            for funding in funding_records:
                text_fields = [
                    funding.get('startup_name', ''),
                    funding.get('investors', ''),
                    funding.get('funding_round', '')
                ]
                score = self.calculate_relevance_score(query, text_fields)
                
                result = {
                    'text': (
                        f"{funding.get('startup_name', '')} raised ${funding.get('amount_million', 0)} million "
                        f"in {funding.get('funding_round', '')} round. The company is valued at "
                        f"${funding.get('valuation_million', 0)} million. "
                        f"Investors include {funding.get('investors', '')}."
                    ),
                    'score': score
                }
                scored_results.append(result)
            
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[0]['text'] if scored_results and scored_results[0]['score'] > 0 else None
            
        except Exception as e:
            logger.error(f"Error searching funding data: {e}")
            return None
    
    def search_competitors(self, query, entities):
        try:
            query_lower = query.lower()
            search_conditions = []
            
            for entity in entities:
                search_conditions.append({"company_name": {"$regex": entity, "$options": "i"}})
            
            words = re.findall(r'\b\w+\b', query_lower)
            for word in words:
                if len(word) > 2:
                    search_conditions.append({"company_name": {"$regex": word, "$options": "i"}})
            
            if search_conditions:
                competitors = list(self.db.competitors.find({"$or": search_conditions}))
            else:
                competitors = list(self.db.competitors.find().limit(5))
            
            scored_results = []
            for competitor in competitors:
                text_fields = [
                    competitor.get('company_name', ''),
                    str(competitor.get('founded_year', '')),
                    str(competitor.get('revenue_million', ''))
                ]
                score = self.calculate_relevance_score(query, text_fields)
                
                result = {
                    'text': (
                        f"{competitor.get('company_name', '')} generates ${competitor.get('revenue_million', 0)} million "
                        f"in revenue annually. The company was founded in {competitor.get('founded_year', 0)} "
                        f"and has {competitor.get('employees', 0)} employees. "
                        f"They have raised ${competitor.get('funding_raised_million', 0)} million in funding. "
                        f"Website: {competitor.get('website', '')}"
                    ),
                    'score': score
                }
                scored_results.append(result)
            
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[0]['text'] if scored_results and scored_results[0]['score'] > 0 else None
            
        except Exception as e:
            logger.error(f"Error searching competitors: {e}")
            return None
    
    def search_market_trends(self, query, entities):
        try:
            query_lower = query.lower()
            search_conditions = []
            
            for entity in entities:
                search_conditions.extend([
                    {"trend_name": {"$regex": entity, "$options": "i"}},
                    {"description": {"$regex": entity, "$options": "i"}}
                ])
            
            trend_keywords = ['ai', 'artificial intelligence', 'blockchain', 'fintech', 'digital', 'mobile', 'cloud']
            for keyword in trend_keywords:
                if keyword in query_lower:
                    search_conditions.extend([
                        {"trend_name": {"$regex": keyword, "$options": "i"}},
                        {"description": {"$regex": keyword, "$options": "i"}}
                    ])
            
            if search_conditions:
                trends = list(self.db.market_trends.find({"$or": search_conditions}))
            else:
                trends = list(self.db.market_trends.find().limit(5))
            
            scored_results = []
            for trend in trends:
                text_fields = [
                    trend.get('trend_name', ''),
                    trend.get('description', '')
                ]
                score = self.calculate_relevance_score(query, text_fields)
                
                result = {
                    'text': (
                        f"{trend.get('trend_name', '')} is a market trend with an impact score of "
                        f"{trend.get('impact_score', 0)}/10. {trend.get('description', '')} "
                        f"Identified on {trend.get('date_identified', '')}."
                    ),
                    'score': score
                }
                scored_results.append(result)
            
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[0]['text'] if scored_results and scored_results[0]['score'] > 0 else None
            
        except Exception as e:
            logger.error(f"Error searching market trends: {e}")
            return None
    
    def search_customer_segments(self, query, entities):
        try:
            query_lower = query.lower()
            search_conditions = []
            
            for entity in entities:
                search_conditions.extend([
                    {"segment_name": {"$regex": entity, "$options": "i"}},
                    {"demographics": {"$regex": entity, "$options": "i"}}
                ])
            
            segment_keywords = ['business', 'consumer', 'enterprise', 'small', 'large', 'millennial', 'gen z']
            for keyword in segment_keywords:
                if keyword in query_lower:
                    search_conditions.extend([
                        {"segment_name": {"$regex": keyword, "$options": "i"}},
                        {"demographics": {"$regex": keyword, "$options": "i"}}
                    ])
            
            if search_conditions:
                segments = list(self.db.customer_segments.find({"$or": search_conditions}))
            else:
                segments = list(self.db.customer_segments.find().limit(5))
            
            scored_results = []
            for segment in segments:
                text_fields = [
                    segment.get('segment_name', ''),
                    segment.get('demographics', ''),
                    segment.get('pain_points', '')
                ]
                score = self.calculate_relevance_score(query, text_fields)
                
                result = {
                    'text': (
                        f"{segment.get('segment_name', '')} is a customer segment of "
                        f"{segment.get('size_million', 0)} million people. "
                        f"Demographics: {segment.get('demographics', '')}. "
                        f"Average spending: ${segment.get('avg_spending', 0)}. "
                        f"Pain points: {segment.get('pain_points', '')}"
                    ),
                    'score': score
                }
                scored_results.append(result)
            
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            return scored_results[0]['text'] if scored_results and scored_results[0]['score'] > 0 else None
            
        except Exception as e:
            logger.error(f"Error searching customer segments: {e}")
            return None
    
    def search_all_collections(self, query):
        try:
            entities = self.extract_entities_from_query(query)
            relevant_collections = self.determine_query_type(query)
            
            best_result = None
            best_score = 0
            
            for collection in relevant_collections:
                result = None
                if collection == 'industries':
                    result = self.search_industries(query, entities)
                elif collection == 'funding_data':
                    result = self.search_funding_data(query, entities)
                elif collection == 'competitors':
                    result = self.search_competitors(query, entities)
                elif collection == 'market_trends':
                    result = self.search_market_trends(query, entities)
                elif collection == 'customer_segments':
                    result = self.search_customer_segments(query, entities)
                
                if result:
                    return result
            return None
            
        except Exception as e:
            logger.error(f"Error in search_all_collections: {e}")
            return None


class ChatbotInference:
    def __init__(self, models_path="./models/"):
        self.models_path = models_path
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = set()
        self.questions_processed = []
        self.df = pd.DataFrame()
        self.max_question_length = 20
        self.max_answer_length = 20
        self.db = None
        self.preprocessor = TextPreprocessor()
        self.mongo_searcher = None
        self.mongodb_connected = False
        
        try:
            self.gemini_enhancer = LLMEnhancer()
            logger.info("Groq LLaMA API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLaMA API: {e}")
            self.gemini_enhancer = None
        
        self.query_cache = {}
        self.cache_expiry = timedelta(minutes=10)
        
        self.mongodb_connected = self.connect_to_mongodb()
        self.load_components()
        
        if self.mongodb_connected and self.db is not None:
            self.mongo_searcher = RealTimeMongoSearcher(self.db)

    def connect_to_mongodb(self):
        max_retries = 3
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            logger.error("MONGO_URI not found in environment variables")
            return False
        
        for attempt in range(max_retries):
            try:
                client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=5000
                )
                client.admin.command('ping')
                self.db = client['market_research_mongo']
                logger.info("MongoDB connected successfully")
                return True
            except Exception as e:
                logger.error(f"MongoDB connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error("All MongoDB connection attempts failed")
                    self.db = None
                    return False

    def load_components(self):
        try:
            logger.info("Loading components...")
            components_file = os.path.join(self.models_path, "PragatiTSP_components.pkl")
            
            if not os.path.exists(components_file):
                raise FileNotFoundError(f"Components file not found at {components_file}")
            
            with open(components_file, "rb") as f:
                components = pickle.load(f)

            self.vocab = components['vocab']
            self.word_to_idx = components['word_to_idx']
            self.idx_to_word = components['idx_to_word']
            self.questions_processed = components['questions_processed']
            self.df = components['df']
            self.max_question_length = components['max_question_length']
            self.max_answer_length = components['max_answer_length']

            logger.info(f"Vocab loaded | Vocab size: {len(self.vocab)} | Dataset: {len(self.df)}")

        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise e

    def store_gemini_data(self, data, collection_type):
        try:
            if not data or not self.mongodb_connected or self.db is None:
                logger.error("Cannot store data: Invalid data or no MongoDB connection")
                return False
            
            collection = self.db[collection_type]
            result = collection.insert_one(data)
            
            if result.inserted_id:
                logger.info(f"Successfully stored data in {collection_type} collection: {result.inserted_id}")
                return True
            else:
                logger.error(f"Failed to store data in {collection_type} collection")
                return False
                
        except Exception as e:
            logger.error(f"Error storing Groq data in MongoDB: {e}")
            return False

    def check_cache(self, query):
        try:
            if query in self.query_cache:
                cached_result, timestamp = self.query_cache[query]
                if datetime.now() - timestamp < self.cache_expiry:
                    return cached_result
                else:
                    del self.query_cache[query]
            return None
        except Exception as e:
            logger.error(f"Cache check error: {e}")
            return None

    def cache_result(self, query, result):
        try:
            self.query_cache[query] = (result, datetime.now())
            if len(self.query_cache) > 100:
                oldest_queries = sorted(self.query_cache.keys(), 
                                      key=lambda x: self.query_cache[x][1])[:20]
                for old_query in oldest_queries:
                    del self.query_cache[old_query]
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    def search_real_time_data(self, query):
        try:
            if not self.mongo_searcher:
                return None
            
            cached_result = self.check_cache(query)
            if cached_result:
                return cached_result
            
            result = self.mongo_searcher.search_all_collections(query)
            
            if not result and self.gemini_enhancer:
                try:
                    collection_type = self.gemini_enhancer.determine_data_type(query)
                    gemini_data = self.gemini_enhancer.extract_and_structure_data(query, collection_type)
                    if gemini_data:
                        self.store_gemini_data(gemini_data, collection_type)
                        result = self.mongo_searcher.search_all_collections(query)
                except Exception as e:
                    logger.error(f"Groq API fallback error: {e}")
            
            if result:
                self.cache_result(query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time search error: {e}")
            return None

    def similarity_search(self, query_tokens, threshold=0.15):
        best_idx, best_score = -1, 0
        query_set = set(query_tokens)

        for i, tokens in enumerate(self.questions_processed):
            token_set = set(tokens)
            overlap = len(query_set.intersection(token_set))
            union = len(query_set.union(token_set))
            
            if union > 0:
                score = overlap / union
                if score > best_score and score >= threshold:
                    best_score, best_idx = score, i

        return best_idx, best_score

    def detect_company_names(self, query):
        query_lower = query.lower()
        potential_companies = []
        
        known_companies = {
            'amazon', 'google', 'microsoft', 'apple', 'meta', 'facebook', 'tesla', 
            'netflix', 'uber', 'airbnb', 'spotify', 'twitter', 'linkedin', 'snapchat',
            'stripe', 'paypal', 'square', 'robinhood', 'coinbase', 'binance',
            'anthropic', 'openai', 'nvidia', 'intel', 'amd', 'ibm', 'oracle',
            'salesforce', 'adobe', 'zoom', 'slack', 'discord', 'pinterest',
            'tiktok', 'bytedance', 'alibaba', 'tencent', 'baidu', 'samsung',
            'sony', 'nintendo', 'activision', 'electronic arts', 'epic games',
            'shopify', 'etsy', 'ebay', 'walmart', 'target', 'costco',
            'starbucks', 'mcdonalds', 'coca cola', 'pepsi', 'nike', 'adidas',
            'boeing', 'airbus', 'general electric', 'ford', 'general motors',
            'jp morgan', 'goldman sachs', 'morgan stanley', 'blackrock',
            'berkshire hathaway', 'johnson & johnson', 'pfizer', 'moderna'
        }
        
        for company in known_companies:
            if company in query_lower:
                potential_companies.append(company.title())
        
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', query)
        common_words = {'What', 'How', 'Tell', 'Show', 'Give', 'Find', 'Search', 
                       'About', 'The', 'This', 'That', 'Where', 'When', 'Why',
                       'Company', 'Information', 'Details', 'Me', 'I', 'You'}
        
        for word in capitalized_words:
            if word not in common_words and len(word) > 2:
                potential_companies.append(word)
        
        if self.mongodb_connected and self.db is not None:
            try:
                query_words = re.findall(r'\b\w+\b', query_lower)
                for word in query_words:
                    if len(word) > 2:
                        company_matches = list(self.db.competitors.find(
                            {"company_name": {"$regex": word, "$options": "i"}}, 
                            {"company_name": 1}
                        ).limit(5))
                        
                        for match in company_matches:
                            company_name = match.get('company_name', '')
                            if company_name:
                                potential_companies.append(company_name)
                        
                        funding_matches = list(self.db.funding_data.find(
                            {"startup_name": {"$regex": word, "$options": "i"}}, 
                            {"startup_name": 1}
                        ).limit(5))
                        
                        for match in funding_matches:
                            startup_name = match.get('startup_name', '')
                            if startup_name:
                                potential_companies.append(startup_name)
            except Exception as e:
                logger.error(f"Error checking database for companies: {e}")
        
        return list(set(potential_companies))

    def is_domain_relevant(self, query):
        query_lower = query.lower()
        
        domain_keywords = [
            'market', 'industry', 'business', 'company', 'startup', 'funding', 'investment',
            'revenue', 'employees', 'trend', 'customer', 'segment', 'growth', 'valuation',
            'fintech', 'healthcare', 'e-commerce', 'saas', 'technology', 'market size',
            'competitor', 'analysis', 'strategy', 'profit', 'sales', 'enterprise'
        ]
        
        if any(keyword in query_lower for keyword in domain_keywords):
            return True
        
        potential_companies = self.detect_company_names(query)
        if potential_companies:
            logger.info(f"Detected potential companies: {potential_companies}")
            return True
        
        business_patterns = [
            r'\b\w+\s+(raised|funding|investment|revenue|valuation|employees|founded)\b',
            r'\b(ceo|founder|executive|leadership)\s+of\s+\w+\b',
            r'\b\w+\s+(market|industry|sector|business)\b',
            r'\binformation\s+about\s+[A-Z]\w+\b',
            r'\btell\s+me\s+about\s+[A-Z]\w+\b',
            r'\b[A-Z]\w+\s+(company|corp|inc|ltd)\b',
            r'\bhow\s+much\s+.*(worth|valued|revenue|profit)\b'
        ]
        
        for pattern in business_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        financial_patterns = [
            'how much', 'what is the', 'tell me about', 'information about',
            'details about', 'profile of', 'overview of', 'statistics'
        ]
        
        for pattern in financial_patterns:
            if pattern in query_lower:
                if re.search(r'\b[A-Z][a-zA-Z]+\b', query):
                    return True
        
        return False

    def enhance_query_with_business_context(self, query):
        potential_companies = self.detect_company_names(query)
        
        if potential_companies:
            query_lower = query.lower()
            business_words = ['company', 'business', 'information', 'details', 'profile']
            
            if not any(word in query_lower for word in business_words):
                enhanced_query = f"{query} company business information"
                logger.info(f"Enhanced query: {enhanced_query}")
                return enhanced_query
        
        return query

    def get_response(self, user_input):
        try:
            if not self.is_domain_relevant(user_input):
                return {
                    "response": "I can only answer questions related to market research, companies, or business trends. Please ask a relevant question.",
                    "confidence": 0.0,
                    "method": "domain_fallback"
                }

            enhanced_input = self.enhance_query_with_business_context(user_input)
            
            user_tokens = self.preprocessor.preprocess(enhanced_input)
            if not user_tokens:
                return {
                    "response": "Please provide a more specific question about market research, companies, or business trends.", 
                    "confidence": 0.0, 
                    "method": "fallback"
                }

            idx, sim_score = self.similarity_search(user_tokens, threshold=0.15)
            
            if idx != -1 and sim_score >= 0.4: 
                return {
                    "response": self.df.iloc[idx]['Answer'], 
                    "confidence": sim_score, 
                    "method": "similarity_search_high_confidence"
                }

            if self.mongodb_connected and self.mongo_searcher:
                try:
                    mongodb_result = self.search_real_time_data(enhanced_input)
                    if mongodb_result:
                        return {
                            "response": mongodb_result, 
                            "confidence": 0.85, 
                            "method": "mongodb_search"
                        }
                except Exception as e:
                    logger.error(f"MongoDB search error: {e}")

            potential_companies = self.detect_company_names(user_input)
            if potential_companies and self.gemini_enhancer:
                try:
                    company_query = f"company information about {potential_companies[0]} including revenue, employees, funding, founding year"
                    collection_type = 'competitors'
                    
                    gemini_data = self.gemini_enhancer.extract_and_structure_data(company_query, collection_type)
                    if gemini_data:
                        groq_response = self.gemini_enhancer.generate_response_from_data(
                            gemini_data, collection_type, user_input
                        )
                        if groq_response:
                            self.store_gemini_data(gemini_data, collection_type)
                            return {
                                "response": groq_response,
                                "confidence": 0.8,
                                "method": "groq_company_info"
                            }
                except Exception as e:
                    logger.error(f"Groq company info error: {e}")

            if idx != -1:
                return {
                    "response": self.df.iloc[idx]['Answer'], 
                    "confidence": sim_score, 
                    "method": "fallback_similarity"
                }

            return {
                "response": self.get_helpful_fallback_response(user_input),
                "confidence": 0.3,
                "method": "enhanced_fallback"
            }

        except Exception as e:
            logger.error(f"get_response error: {e}")
            return {
                "response": "I encountered an error processing your request. Please try rephrasing your question.", 
                "confidence": 0.0, 
                "method": "error"
            }

    def get_helpful_fallback_response(self, user_input):
        query_lower = user_input.lower()
        
        if any(word in query_lower for word in ['funding', 'investment', 'raised']):
            return ("I don't have specific information about that funding query. "
                   "Try asking about specific companies like 'How much funding did Stripe raise?' "
                   "or 'What is the latest funding in fintech?'")
        
        elif any(word in query_lower for word in ['market', 'industry', 'size']):
            return ("I don't have specific market data for that query. "
                   "Try asking about specific industries like 'What is the FinTech market size?' "
                   "or 'How big is the healthcare market?'")
        
        elif any(word in query_lower for word in ['company', 'startup', 'business']):
            return ("I don't have information about that specific company. "
                   "Try asking about well-known companies or check if the company name is in our database.")
        
        elif any(word in query_lower for word in ['trend', 'trending']):
            return ("I don't have information about that specific trend. "
                   "Try asking about 'latest market trends' or 'emerging technologies'.")
        
        else:
            return ("I couldn't find relevant information for your query. "
                   "Try asking about market research, company funding, industry trends, "
                   "or specific business topics. You can also ask questions like: "
                   "'What is market research?', 'How to validate a business idea?', "
                   "or 'What are key startup metrics?'")

    def get_chat_suggestions(self):
        base_suggestions = [
            "What is the market size of FinTech?",
            "How much funding did Stripe raise?",
            "Tell me about Open Banking trends",
            "Who are Small Business Owners?",
            "What are the key startup metrics?",
            "How to validate a business idea?"
        ]
        
        if self.mongodb_connected and self.db is not None:
            try:
                latest_companies = list(self.db.funding_data.find().limit(2))
                for company in latest_companies:
                    company_name = company.get('startup_name', '')
                    if company_name:
                        base_suggestions.append(f"Tell me about {company_name}")
                
                latest_trends = list(self.db.market_trends.find().limit(1))
                for trend in latest_trends:
                    trend_name = trend.get('trend_name', '')
                    if trend_name:
                        base_suggestions.append(f"What is {trend_name}?")
                        
            except Exception as e:
                logger.error(f"Error getting dynamic suggestions: {e}")
        
        return base_suggestions[:6]

    def get_database_stats(self):
        if not self.mongodb_connected or self.db is None:
            return {"status": "disconnected"}
        
        try:
            stats = {
                "status": "connected",
                "collections": {}
            }
            
            collections = ['industries', 'funding_data', 'competitors', 'market_trends', 'customer_segments']
            
            for collection_name in collections:
                try:
                    count = self.db[collection_name].count_documents({})
                    stats["collections"][collection_name] = count
                except Exception as e:
                    stats["collections"][collection_name] = f"Error: {e}"
            
            return stats
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def health_check(self):
        try:
            test_response = self.get_response("What is market research?")
            db_stats = self.get_database_stats()
            
            return {
                "status": "healthy",
                "mongodb_status": db_stats["status"],
                "database_collections": db_stats.get("collections", {}),
                "vocab_size": len(self.vocab),
                "training_data_size": len(self.df),
                "cache_size": len(self.query_cache),
                "gemini_available": self.gemini_enhancer is not None,
                "sample_response": test_response["response"][:100] + "...",
                "real_time_search_available": self.mongo_searcher is not None,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def clear_cache(self):
        self.query_cache.clear()
        return {"status": "cache_cleared", "timestamp": datetime.now().isoformat()}

    def get_query_analytics(self):
        return {
            "total_cached_queries": len(self.query_cache),
            "cache_hit_potential": len(self.query_cache) > 0,
            "cached_queries_sample": list(self.query_cache.keys())[:5]
        }

if __name__ == "__main__":
    logger.info("Initializing Enhanced Chatbot Inference with Groq LLaMA and MongoDB Integration...") 
    bot = ChatbotInference()
    health_status = bot.health_check()
    logger.info("Health Check Results:")
    for key, value in health_status.items():
        logger.info(f"   {key}: {value}")
    
    sample_queries = [
        "What is the market size of FinTech?",  
        "How much funding did Klarna raise?", 
        "Tell me about Stripe", 
        "What are the latest market trends?",
        "What is market research?", 
        "How to validate a business idea?",
        "Tell me about Tesla",
        "What is the weather today?",
        "Who is Small Business Owners?" 
    ]

    logger.info("\nTesting sample queries:")
    for q in sample_queries:
        logger.info(f"\nQuery: {q}")
        res = bot.get_response(q)
        logger.info(f"Response: {res['response'][:100]}...")
        logger.info(f"Confidence: {res['confidence']:.2f} | Method: {res['method']}")