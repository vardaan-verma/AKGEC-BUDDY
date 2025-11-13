from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import hashlib
import json
import os
from datetime import datetime
import logging     
from typing import Optional, Dict, List, Tuple
import uvicorn
import re
from dotenv import load_dotenv
load_dotenv()


# Google GenAI imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

app = FastAPI(title="College Chatbot API", version="1.0.0")
# Serve frontend build files
# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    USE_GOOGLE_GENAI = True
    GOOGLE_MODEL = "gemini-1.5-flash"  # or gemini-1.5-pro
    
    # Add your Google API key here or set as environment variable
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

config = Config()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    language: Optional[str] = None  # Auto-detect if not provided

class ChatResponse(BaseModel):
    response: str
    confidence: float
    response_time: float
    cached: bool
    detected_language: str
    detected_intent: str
    sources: Optional[List[str]] = []

# Simple In-Memory Cache
class SimpleCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str):
        return self.cache.get(key)
    
    def set(self, key: str, value: dict):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def generate_key(self, message: str, language: str = "auto"):
        return hashlib.md5(f"{message.lower().strip()}_{language}".encode()).hexdigest()

# Global cache instance
cache = SimpleCache()

# College Knowledge Base
COLLEGE_KNOWLEDGE = {
    "admissions": { 
        "en": [
            "Admission process starts in June every year.",
            "Required documents: 10th, 12th marksheets, entrance exam scores.",
            "Application deadline is typically end of July.",
            "Entrance exams accepted: JEE Main, JEE Advanced, state CET."
        ],
        "hi": [
            "प्रवेश प्रक्रिया हर साल जून में शुरू होती है।",
            "आवश्यक दस्तावेज: 10वीं, 12वीं की मार्कशीट, प्रवेश परीक्षा स्कोर।",
            "आवेदन की अंतिम तिथि आमतौर पर जुलाई के अंत में होती है।"
        ]
    },
    "courses": {
        "en": [
            "We offer BTech in Computer Science, Mechanical, Electrical, Civil Engineering.",
            "MTech programs available in all engineering branches.",
            "Duration: BTech - 4 years, MTech - 2 years.",
            "Specializations available in AI/ML, Data Science, Robotics."
        ],
        "hi": [
            "हम कंप्यूटर साइंस, मैकेनिकल, इलेक्ट्रिकल, सिविल इंजीनियरिंग में बीटेक कराते हैं।",
            "सभी इंजीनियरिंग शाखाओं में एमटेक कार्यक्रम उपलब्ध हैं।"
        ]
    },
    "fees": {
        "en": [
            "BTech fee structure: Rs. 1,50,000 per year.",
            "Scholarship available for meritorious students.",
            "Hostel fees: Rs. 1,20,000 per year including meals.",
            "Payment can not be made in installments."
        ],
        "hi": [
            "बीटेक फीस संरचना: प्रति वर्ष 1,50,000 रुपये।",
            "मेधावी छात्रों के लिए छात्रवृत्ति उपलब्ध है।",
            "हॉस्टल फीस: भोजन सहित प्रति वर्ष 80,000 रुपये।"
        ]
    },
    "facilities": {
        "en": [
            "24/7 library with digital resources.",
            "Modern labs for all engineering branches.",
            "Sports complex with gym, basketball, cricket ground.",
            "Separate hostels for boys and girls.",
            "High-speed Wi-Fi across campus."
        ],
        "hi": [
            "डिजिटल संसाधनों के साथ 24/7 पुस्तकालय।",
            "सभी इंजीनियरिंग शाखाओं के लिए आधुनिक प्रयोगशालाएं।",
            "जिम, बास्केटबॉल, क्रिकेट मैदान के साथ खेल परिसर।"
        ]
    },
    "placement": {
        "en": [
            "Average placement package: Rs. 8-12 LPA.",
            "Top recruiters: Google, Microsoft, Amazon, TCS, Infosys.",
            "Placement rate: 95% for eligible students.",
            "Career guidance and training programs available."
        ],
        "hi": [
            "औसत प्लेसमेंट पैकेज: 8-12 लाख प्रति वर्ष।",
            "शीर्ष भर्तीकर्ता: गूगल, माइक्रोसॉफ्ट, अमेज़न, टीसीएस, इन्फोसिस।"
        ]
    }
}

# Fixed LLM Service - Single class using Google GenAI
class LLMService:
    def __init__(self):
        self.model_name = config.GOOGLE_MODEL
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.2, 
                max_output_tokens=512,
            )
            logging.info(f"Initialized Google GenAI with model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Google GenAI: {e}")
            self.llm = None

    async def call_llm(self, prompt: str) -> str:
        """Call Google Generative AI (Gemini) via LangChain"""
        if not self.llm:
            logging.error("LLM not properly initialized")
            return ""
            
        try:
            # Add more detailed logging
            logging.info(f"Calling Google GenAI with prompt length: {len(prompt)}")
            result = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response = result.content.strip()
            logging.info(f"Google GenAI response length: {len(response)}")
            return response
        except Exception as e:
            logging.error(f"Google GenAI API error: {e}")
            return ""

# Language Detection Service with improved logic
class LanguageDetectionService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
        # Expanded Hinglish keywords and patterns
        self.hinglish_patterns = {
            # Common Hindi words written in Latin script
            "hindi_in_latin": [
                "kya", "hai", "hain", "kaise", "kese", "kesi", "kesa", "kaisa", "kaisi",
                "milti", "milta", "milte", "hota", "hoti", "hote", "karta", "karti", "karte",
                "college", "admission", "fees", "kitna", "kitne", "kitni", "padhai", "padhna",
                "placement", "job", "career", "sapna", "chahiye", "karna", "kar", "kar",
                "mein", "me", "main", "mai", "tum", "tere", "tumhe", "tumko", "mujhe", "mujhko",
                "acha", "accha", "achha", "bura", "bhi", "bhe", "waha", "yaha", "yahan", "wahan",
                "koi", "kuch", "kaam", "kam", "zyada", "jyada", "thoda", "bahut", "bohot",
                "hostel", "library", "lab", "sports", "facility", "infrastructure",
                "scholarship", "cost", "payment", "process", "procedure", "eligibility",
                "branch", "course", "program", "degree", "semester", "year", "marks",
                "exam", "test", "interview", "selection", "merit", "rank", "score",
                "company", "package", "salary", "internship", "training", "skill",
                "help", "madad", "batao", "bolo", "samjhao", "explain", "detail", "info",
                "time", "date", "schedule", "timing", "duration", "period",
                "available", "mil", "nahi", "nahin", "ha", "han", "haa", "ji", "sir", "madam"
            ],
            
            # Common Hindi question words
            "question_words": [
                "kya", "kaise", "kab", "kaha", "kahan", "kaun", "kyun", "kyu", "kitna", "kitne"
            ],
            
            # Hindi sentence patterns (transliterated)
            "sentence_patterns": [
                r"\bkya\b.*\bhai\b", r"\bkaise\b.*\bmilti\b", r"\bkitna\b.*\bcost\b",
                r"\bmain\b.*\bkarna\b", r"\bmujhe\b.*\bchahiye\b", r"\btum\b.*\bbata\b"
            ]
        }
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """Enhanced language detection with better Hinglish support"""
        
        # Step 1: Check for Devanagari script (pure Hindi)
        devanagari_score = self._check_devanagari_script(text)
        if devanagari_score > 0.3:  # Lowered threshold
            return "hi", min(0.5 + devanagari_score, 0.9)
        
        # Step 2: Enhanced Hinglish detection
        hinglish_score = self._detect_hinglish_patterns(text)
        if hinglish_score > 0.4:  # Lowered threshold for better detection
            confidence = min(0.6 + hinglish_score * 0.3, 0.85)
            return "hi", confidence
        
        # Step 3: Context-based detection using common word combinations
        context_score = self._detect_context_patterns(text)
        if context_score > 0.3:
            return "hi", min(0.7 + context_score * 0.2, 0.8)
        
        # Step 4: Use LLM for ambiguous cases
        if hinglish_score > 0.2 or context_score > 0.1:
            try:
                llm_lang, llm_conf = await self._llm_language_detection(text)
                if llm_lang == "hi" and llm_conf > 0.6:
                    return "hi", llm_conf
            except Exception as e:
                logging.error(f"LLM language detection failed: {e}")
        
        # Step 5: Default to English for pure English or unknown
        english_score = self._check_english_patterns(text)
        return "en", max(english_score, 0.6)
    
    def _check_devanagari_script(self, text: str) -> float:
        """Check for Devanagari Unicode characters"""
        devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return 0
        
        return devanagari_chars / total_chars
    
    def _detect_hinglish_patterns(self, text: str) -> float:
        """Detect Hinglish based on vocabulary and patterns"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return 0
        
        # Count Hindi words written in Latin script
        hindi_word_count = 0
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word in self.hinglish_patterns["hindi_in_latin"]:
                hindi_word_count += 1
        
        # Check for question patterns
        question_bonus = 0
        for q_word in self.hinglish_patterns["question_words"]:
            if q_word in text_lower:
                question_bonus += 0.1
        
        # Check for sentence patterns using regex
        import re
        pattern_bonus = 0
        for pattern in self.hinglish_patterns["sentence_patterns"]:
            if re.search(pattern, text_lower):
                pattern_bonus += 0.15
        
        # Calculate score
        vocab_score = hindi_word_count / total_words
        total_score = vocab_score + question_bonus + pattern_bonus
        
        return min(total_score, 1.0)
    
    def _detect_context_patterns(self, text: str) -> float:
        """Detect common Hinglish conversation patterns"""
        text_lower = text.lower().strip()
        
        # Common Hinglish phrases and combinations
        hinglish_phrases = [
            "job kaise milti", "kya hai", "kitna cost", "kaise karna",
            "mujhe chahiye", "batao na", "help karo", "samjhao please",
            "college mein", "admission ke liye", "fees kitni", "placement kaisi",
            "hostel facility", "library timing", "course details", "eligibility criteria"
        ]
        
        phrase_matches = sum(1 for phrase in hinglish_phrases if phrase in text_lower)
        
        # Look for mixed language patterns (English + Hindi transliterated words)
        mixed_patterns = [
            (r'\b(college|university|admission)\b.*\b(kaise|kya|kitna)\b', 0.3),
            (r'\b(job|career|placement)\b.*\b(milti|hota|kaisa)\b', 0.3),
            (r'\b(fees|cost|payment)\b.*\b(kitna|kitni|hai)\b', 0.3),
            (r'\b(help|please|sir)\b.*\b(karo|batao|samjhao)\b', 0.2),
        ]
        
        import re
        pattern_score = 0
        for pattern, weight in mixed_patterns:
            if re.search(pattern, text_lower):
                pattern_score += weight
        
        return min((phrase_matches * 0.2) + pattern_score, 1.0)
    
    def _check_english_patterns(self, text: str) -> float:
        """Check if text is primarily English"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        if not words:
            return 0.5
        
        # Common English words that are unlikely to be in Hinglish
        english_indicators = [
            "the", "and", "or", "but", "with", "from", "about", "into", "through",
            "during", "before", "after", "above", "below", "up", "down", "out", "off",
            "over", "under", "again", "further", "then", "once", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "only", "own", "same", "so", "than", "too", "very"
        ]
        
        english_word_count = sum(1 for word in words if word in english_indicators)
        english_score = english_word_count / len(words)
        
        # Bonus for purely English technical terms
        technical_english = [
            "engineering", "computer", "science", "mechanical", "electrical",
            "information", "technology", "management", "bachelor", "master"
        ]
        
        tech_bonus = sum(0.1 for term in technical_english if term in text_lower)
        
        return min(english_score + tech_bonus, 0.9)
    
    async def _llm_language_detection(self, text: str) -> Tuple[str, float]:
        """Use LLM for complex language detection"""
        prompt = f"""
Analyze this text and determine if it's primarily:
1. English (en) - Pure English text
2. Hindi/Hinglish (hi) - Hindi words written in Latin script, mixed Hindi-English

Text: "{text}"

Consider:
- "job kaise milti h?" = Hinglish (Hindi question structure with English words)
- "kya hai admission process?" = Hinglish 
- "What is the admission process?" = English
- "How to apply for courses?" = English

Respond ONLY with:
Language: [en/hi]
Confidence: [0.5-1.0]
Reason: [brief explanation]
"""
        
        try:
            response = await self.llm_service.call_llm(prompt)
            
            if response:
                import re
                lang_match = re.search(r'Language:\s*([a-z]+)', response.lower())
                conf_match = re.search(r'Confidence:\s*([0-9.]+)', response)
                
                if lang_match and conf_match:
                    language = lang_match.group(1)
                    confidence = float(conf_match.group(1))
                    logging.info(f"LLM detected language: {language} (confidence: {confidence})")
                    return language, confidence
        except Exception as e:
            logging.error(f"LLM language detection error: {e}")
        
        return "en", 0.5

# Intent Classification Service with improved prompting
class IntentClassificationService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def classify_intent(self, text: str, language: str = "en") -> Tuple[str, float]:
        """Classify user intent using LLM with better prompting"""
        
        intent_examples = {
            "admissions": ["admission process", "how to apply", "entrance exam", "eligibility", "admission requirements"],
            "courses": ["what courses", "available programs", "BTech", "MTech", "branches", "specializations"],
            "fees": ["fee structure", "how much cost", "scholarship", "payment", "tuition fees"],
            "facilities": ["hostel", "library", "labs", "sports", "campus facilities", "infrastructure"],
            "placement": ["placement", "job", "recruiters", "package", "career", "companies"],
            "general": ["hello", "hi", "thanks", "help", "information"]
        }
        
        examples_text = "\n".join([
            f"- {intent}: {', '.join(examples[:3])}" 
            for intent, examples in intent_examples.items()
        ])
        
        prompt = f"""
Classify this user message into ONE of these intents with HIGH confidence:

INTENTS & EXAMPLES:
{examples_text}

User Message: "{text}"
Language: {language}

Think step by step:
1. What is the user asking about?
2. Which intent category fits best?
3. How confident are you?

Respond EXACTLY as:
Intent: [intent_name]
Confidence: [0.6-1.0]
Reason: [one sentence]

Be confident - most queries clearly fit an intent!
"""
        
        try:
            response = await self.llm_service.call_llm(prompt)
            
            if response:
                logging.info(f"Intent classification response: {response}")
                
                # Parse response with better regex
                intent_match = re.search(r'Intent:\s*([a-zA-Z_]+)', response)
                conf_match = re.search(r'Confidence:\s*([0-9.]+)', response)
                
                if intent_match:
                    intent = intent_match.group(1).lower()
                    confidence = float(conf_match.group(1)) if conf_match else 0.7
                    
                    # Validate intent
                    valid_intents = list(intent_examples.keys())
                    if intent in valid_intents:
                        # Boost confidence for valid intents
                        confidence = max(confidence, 0.7)
                        logging.info(f"Intent classified: {intent} (confidence: {confidence})")
                        return intent, confidence
            
        except Exception as e:
            logging.error(f"Intent classification error: {e}")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        
        # Fallback with keyword matching
        return self._fallback_intent_detection(text)
    
    def _fallback_intent_detection(self, text: str) -> Tuple[str, float]:
        """Fallback intent detection using keywords"""
        intent_keywords = {
            "admissions": ["admission", "apply", "entrance", "eligibility"],
            "courses": ["course", "branch", "program", "btech", "mtech"],
            "fees": ["fee", "cost", "scholarship", "payment"],
            "facilities": ["hostel", "library", "lab", "sports", "facility"],
            "placement": ["placement", "job", "career", "package", "recruiter"]
        }
        
        text_lower = text.lower()
        best_intent = "general"
        best_score = 0
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        confidence = min(0.6 + (best_score * 0.1), 0.8) if best_score > 0 else 0.4
        return best_intent, confidence

class RAGProcessor:
    def __init__(self):
        self.llm_service = LLMService()
        self.language_detector = LanguageDetectionService(self.llm_service)
        self.intent_classifier = IntentClassificationService(self.llm_service)
    
    async def process_query(self, message: str, provided_language: str = None):
        start_time = datetime.now()
        
        logging.info(f"Processing query: {message}")
        
        # Step 1: Parallel processing of language detection and intent classification
        if provided_language:
            lang_task = asyncio.create_task(self._mock_lang_result(provided_language))
        else:
            lang_task = asyncio.create_task(self.language_detector.detect_language(message))
        
        intent_task = asyncio.create_task(self.intent_classifier.classify_intent(message))
        
        # Wait for both tasks to complete
        (detected_lang, lang_confidence), (intent, intent_confidence) = await asyncio.gather(
            lang_task, intent_task
        )
        
        logging.info(f"Language: {detected_lang} ({lang_confidence}), Intent: {intent} ({intent_confidence})")
        
        # Step 2: Confidence-based routing with lower thresholds
        if intent_confidence > 0.7:  # Lowered from 0.8
            response, sources = await self.quick_response(intent, message, detected_lang)
            final_confidence = intent_confidence
        elif intent_confidence > 0.4:  # Lowered from 0.5
            response, sources = await self.enhanced_processing(intent, message, detected_lang)
            final_confidence = intent_confidence * 0.9
        else:
            response, sources = await self.comprehensive_search_with_llm(message, detected_lang)
            final_confidence = 0.5  # Increased from 0.4
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        return response, final_confidence, response_time, sources, detected_lang, intent
    
    async def _mock_lang_result(self, lang: str):
        """Mock language detection result when language is provided"""
        return lang, 1.0
    
    async def quick_response(self, intent: str, message: str, language: str):
        """Fast response for high-confidence queries"""
        logging.info(f"Using quick response for intent: {intent}")
        
        if intent in COLLEGE_KNOWLEDGE and language in COLLEGE_KNOWLEDGE[intent]:
            relevant_info = COLLEGE_KNOWLEDGE[intent][language]
            
            if language == "hi":
                response = f"{intent} के बारे में जानकारी:\n\n" + "\n".join(relevant_info[:3])  # Show more info
            else:
                response = f"Here's information about {intent}:\n\n" + "\n".join(relevant_info[:3])
            
            return response, [f"college_db_{intent}"]
        
        # Fallback to English if language not available
        elif intent in COLLEGE_KNOWLEDGE and "en" in COLLEGE_KNOWLEDGE[intent]:
            relevant_info = COLLEGE_KNOWLEDGE[intent]["en"]
            response = f"Here's information about {intent}:\n\n" + "\n".join(relevant_info[:3])
            return response, [f"college_db_{intent}"]
        
        fallback_msg = "मैं कॉलेज की जानकारी में आपकी मदद कर सकता हूं।" if language == "hi" else "I can help you with college information. What would you like to know?"
        return fallback_msg, []
    
    async def enhanced_processing(self, intent: str, message: str, language: str):
        """Medium confidence processing with LLM enhancement"""
        logging.info(f"Using enhanced processing for intent: {intent}")
        
        if intent in COLLEGE_KNOWLEDGE:
            # Get relevant knowledge
            relevant_info = []
            if language in COLLEGE_KNOWLEDGE[intent]:
                relevant_info = COLLEGE_KNOWLEDGE[intent][language]
            elif "en" in COLLEGE_KNOWLEDGE[intent]:
                relevant_info = COLLEGE_KNOWLEDGE[intent]["en"]
            
            if relevant_info:
                # Use LLM to create a more contextual response
                context = "\n".join(relevant_info)
                prompt = f"""
You are a helpful college chatbot. Answer the user's question using this information:

COLLEGE INFORMATION ABOUT {intent.upper()}:
{context}

USER QUESTION: "{message}"
LANGUAGE: {language}

Instructions:
- Give a direct, helpful answer
- Use the provided information
- {"Answer in Hindi/Hinglish" if language == "hi" else "Answer in English"}
- Be conversational and informative
- Keep response under 150 words

Answer:
"""
                
                llm_response = await self.llm_service.call_llm(prompt)
                if llm_response:
                    logging.info(f"Enhanced response generated: {len(llm_response)} chars")
                    return llm_response, [f"college_db_{intent}_llm"]
        
        return await self.quick_response(intent, message, language)
    
    async def comprehensive_search_with_llm(self, message: str, language: str):
        """Comprehensive search using LLM for low confidence queries"""
        logging.info("Using comprehensive search")
        
        # Gather all relevant knowledge
        all_knowledge = {}
        for category, lang_data in COLLEGE_KNOWLEDGE.items():
            if language in lang_data:
                all_knowledge[category] = lang_data[language]
            elif "en" in lang_data:
                all_knowledge[category] = lang_data["en"]
        
        # Create context for LLM
        context_parts = []
        for category, info_list in all_knowledge.items():
            context_parts.append(f"{category.upper()}:\n" + "\n".join(info_list))
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
You are a helpful college chatbot. Use this college information to answer the user's question:

COLLEGE INFORMATION:
{context}

USER QUESTION: "{message}"
LANGUAGE: {language}

Instructions:
- Answer based on the available information
- {"Respond in Hindi/Hinglish" if language == "hi" else "Respond in English"}
- If information is not available, suggest contacting admissions office
- Be helpful and conversational
- Keep response under 200 words

Answer:
"""
        
        try:
            llm_response = await self.llm_service.call_llm(prompt)
            if llm_response:
                logging.info(f"Comprehensive response generated: {len(llm_response)} chars")
                return llm_response, ["college_db_comprehensive_llm"]
        except Exception as e:
            logging.error(f"Comprehensive search LLM error: {e}")
        
        # Fallback response
        fallback_msg = (
            "मुझे खुशी होगी अगर मैं आपकी मदद कर सकूं। कृपया प्रवेश कार्यालय से संपर्क करें।" 
            if language == "hi" 
            else "I'd be happy to help you. Please contact our admissions office for detailed assistance."
        )
        return fallback_msg, []

# Initialize RAG processor
rag_processor = RAGProcessor()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        start_time = datetime.now()
        
        # Generate cache key
        cache_key = cache.generate_key(request.message, request.language or "auto")
        
        # Check cache first
        cached_response = cache.get(cache_key)
        if cached_response:
            cached_response["cached"] = True
            cached_response["response_time"] = (datetime.now() - start_time).total_seconds()
            return ChatResponse(**cached_response)
        
        # Process with RAG
        response, confidence, processing_time, sources, detected_lang, intent = await rag_processor.process_query(
            request.message, request.language
        )
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        chat_response = {
            "response": response,
            "confidence": confidence,
            "response_time": total_time,
            "cached": False,
            "detected_language": detected_lang,
            "detected_intent": intent,
            "sources": sources
        }
        
        # Cache the response for future use (lowered threshold)
        if confidence > 0.5:  # Lowered from 0.6
            cache.set(cache_key, {
                "response": response,
                "confidence": confidence,
                "detected_language": detected_lang,
                "detected_intent": intent,
                "sources": sources
            })
        
        return ChatResponse(**chat_response)
        
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "cache_size": len(cache.cache),
        "llm_service": "google-genai",
        "model": config.GOOGLE_MODEL
    }

@app.get("/")
async def root():
    return {"message": "College Chatbot API with Google GenAI is running!"}

@app.post("/cache/clear")
async def clear_cache():
    cache.cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/cache/stats")
async def cache_stats():
    return {
        "cache_size": len(cache.cache),
        "max_size": cache.max_size
    }

# Test endpoint for language detection
@app.post("/test/language")
async def test_language_detection(request: dict):
    message = request.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    lang_detector = LanguageDetectionService(LLMService())
    language, confidence = await lang_detector.detect_language(message)
    
    return {
        "message": message,
        "detected_language": language,
        "confidence": confidence
    }

# Test endpoint for intent classification
@app.post("/test/intent")
async def test_intent_classification(request: dict):
    message = request.get("message", "")
    language = request.get("language", "en")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    intent_classifier = IntentClassificationService(LLMService())
    intent, confidence = await intent_classifier.classify_intent(message, language)
    
    return {
        "message": message,
        "language": language,
        "detected_intent": intent,
        "confidence": confidence
    }

# Enable logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8081, reload=True)
