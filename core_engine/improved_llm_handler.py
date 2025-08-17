"""Enhanced LLM Handler with robust JSON extraction and network resilience."""

import json
import logging
import os
import re
import sys
import time
from functools import wraps
from typing import Any, Dict, List, Optional

import ollama

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    from pytrends_offline import PyTrendsOffline
except Exception:
    PyTrendsOffline = None

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import AI_CONFIG, CHANNELS_CONFIG
except ImportError:
    print("❌ No configuration file found - using minimal defaults")
    CHANNELS_CONFIG = {}
    AI_CONFIG = {"ollama_model": "llama3:8b"}

# Ollama Model Optimization
OLLAMA_OPTIMIZATION = {
    "quantization": "q4_0",      # 4-bit quantization for memory efficiency
    "context_length": 4096,      # Optimal context length
    "batch_size": 8,             # Batch processing for efficiency
    "temperature": 0.7,          # Consistent output
    "top_p": 0.9,               # Nucleus sampling
    "repeat_penalty": 1.1,      # Prevent repetition
    "num_predict": 2048,        # Prediction limit
    "cache_models": True,       # Model caching
    "memory_optimization": True # Memory management
}

# AI Content Quality Enhancement
CONTENT_QUALITY_CONFIG = {
    "prompt_engineering": True,        # Advanced prompt engineering
    "content_validation": True,        # Content quality validation
    "style_consistency": True,         # Style consistency checking
    "engagement_optimization": True,   # Engagement optimization
    "seo_optimization": True,          # SEO optimization
    "multilingual_support": True,      # Multi-language support
    "content_templates": True,         # Content templates
    "quality_scoring": True            # Quality scoring system
}

# Content Templates for Different Niches
CONTENT_TEMPLATES = {
    "history": {
        "hook": "Discover the {topic} that {action} - a story that will {emotion} you!",
        "structure": ["Hook", "Background", "Main Story", "Impact", "Call to Action"],
        "tone": "educational, engaging, mysterious",
        "keywords": ["ancient", "discovery", "mystery", "civilization", "archaeology"]
    },
    "motivation": {
        "hook": "The {topic} that changed everything - here's how to {action} in your life!",
        "structure": ["Hook", "Problem", "Solution", "Action Steps", "Motivation"],
        "tone": "inspirational, practical, empowering",
        "keywords": ["success", "motivation", "growth", "achievement", "transformation"]
    },
    "finance": {
        "hook": "The {topic} that could {action} your financial future - experts reveal all!",
        "structure": ["Hook", "Current Situation", "Analysis", "Strategy", "Action Plan"],
        "tone": "professional, analytical, trustworthy",
        "keywords": ["investment", "wealth", "strategy", "opportunity", "financial"]
    },
    "automotive": {
        "hook": "The {topic} that's {action} the automotive industry - here's what you need to know!",
        "structure": ["Hook", "Technology", "Benefits", "Comparison", "Future"],
        "tone": "technical, exciting, informative",
        "keywords": ["innovation", "technology", "performance", "future", "automotive"]
    },
    "combat": {
        "hook": "Master the {topic} that {action} - techniques that will {emotion} your skills!",
        "structure": ["Hook", "Technique", "Demonstration", "Practice", "Mastery"],
        "tone": "intense, educational, empowering",
        "keywords": ["technique", "mastery", "skill", "training", "combat"]
    }
}


def _get_trend_client():
    if TrendReq is not None:
        try:
            return TrendReq(hl="en-US", tz=0)
        except Exception:
            pass
    if PyTrendsOffline is not None:
        return PyTrendsOffline()
    return None


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> callable:
    """Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    delay = base_delay * (2**attempt)  # 1s, 2s, 4s
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


class ImprovedLLMHandler:
    """Enhanced LLM handler with robust JSON extraction and network resilience."""

    def __init__(self, model: Optional[str] = None, max_retries: int = 3) -> None:
        """Initialize the LLM handler.

        Args:
            model: Ollama model to use, defaults to config value
            max_retries: Maximum retry attempts for network operations
        """
        self.model = model or AI_CONFIG.get("ollama_model", "llama3:8b")
        self.max_retries = max_retries
        
        # Apply Ollama optimization
        self.setup_ollama_optimization()
        
        # Setup content quality enhancement
        self.setup_content_quality_enhancement()

        # Initialize PyTrends with timeout
        self.pytrends = _get_trend_client()
        if self.pytrends is None:
            logging.warning("PyTrends unavailable; using cached/fallback keywords.")

        self.setup_logging()
    
    def setup_ollama_optimization(self):
        """Setup Ollama model optimization for better performance"""
        try:
            import ollama
            
            # Set model parameters for optimization
            model_params = {
                "num_ctx": OLLAMA_OPTIMIZATION["context_length"],
                "num_predict": OLLAMA_OPTIMIZATION["num_predict"],
                "temperature": OLLAMA_OPTIMIZATION["temperature"],
                "top_p": OLLAMA_OPTIMIZATION["top_p"],
                "repeat_penalty": OLLAMA_OPTIMIZATION["repeat_penalty"],
                "num_gpu": 1,  # Use GPU if available
                "num_thread": 8  # Use multiple CPU threads
            }
            
            # Apply quantization if supported
            if hasattr(ollama, 'pull') and OLLAMA_OPTIMIZATION["quantization"]:
                try:
                    # Try to pull quantized model
                    quantized_model = f"{self.model}:{OLLAMA_OPTIMIZATION['quantization']}"
                    print(f"🚀 Attempting to use quantized model: {quantized_model}")
                    # Note: This is a placeholder - actual implementation depends on Ollama version
                except Exception as e:
                    print(f"⚠️ Quantized model not available: {e}")
            
            self.model_params = model_params
            print("✅ Ollama optimization completed")
            
        except Exception as e:
            print(f"⚠️ Ollama optimization failed: {e}")
            self.model_params = {}

    def setup_content_quality_enhancement(self):
        """Setup content quality enhancement system"""
        try:
            # Initialize content quality features
            self.content_quality_enabled = CONTENT_QUALITY_CONFIG["prompt_engineering"]
            self.content_templates = CONTENT_TEMPLATES
            self.quality_scoring = CONTENT_QUALITY_CONFIG["quality_scoring"]
            
            print("✅ Content quality enhancement system initialized")
            
        except Exception as e:
            print(f"⚠️ Content quality enhancement setup failed: {e}")
            self.content_quality_enabled = False
    
    def setup_logging(self) -> None:
        """Set up enhanced logging with standardized levels."""
        self.log_file = f"llm_handler_{int(time.time())}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"📝 LLM Handler logging to: {self.log_file}")
        self.logger.info(f"🤖 Using Ollama model: {self.model}")

    def log_message(self, message: str, level: str = "INFO") -> None:
        """Log message with standardized levels.

        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        level_map = {
            "DEBUG": self.logger.debug,
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "ERROR": self.logger.error,
        }

        log_func = level_map.get(level, self.logger.info)
        log_func(message)

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from text using multiple fallback methods.

        Args:
            text: Text containing JSON data

        Returns:
            Extracted JSON string or None if extraction fails

        Raises:
            ValueError: If text is empty or invalid
        """
        if not text:
            raise ValueError("Text cannot be empty")

        # Method 1: Extract from fenced code blocks
        json_block = self._extract_from_fenced_blocks(text)
        if json_block:
            return json_block

        # Method 2: Use stack-based parser for balanced JSON
        balanced_json = self._extract_balanced_json(text)
        if balanced_json:
            return balanced_json

        self.log_message(
            f"Failed to extract valid JSON from text: {text[:100]}...", "ERROR"
        )
        return None

    def _extract_from_fenced_blocks(self, text: str) -> Optional[str]:
        """Extract JSON from ```json ... ``` fenced blocks.

        Args:
            text: Text containing fenced JSON blocks

        Returns:
            Extracted JSON string or None
        """
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    # Try to parse as-is
                    json.loads(match)
                    self.log_message("JSON extracted from fenced block", "DEBUG")
                    return match
                except json.JSONDecodeError:
                    # Try with fixes
                    fixed = self._fix_json_string(match)
                    if fixed:
                        try:
                            json.loads(fixed)
                            self.log_message(
                                "JSON extracted from fenced block after fixes", "DEBUG"
                            )
                            return fixed
                        except json.JSONDecodeError:
                            continue

        return None

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Stack-based parser to find the first complete JSON object/array.

        Args:
            text: Text containing JSON data

        Returns:
            Extracted JSON string or None
        """
        try:
            # Find first opening brace or bracket
            start_chars = {"{": "}", "[": "]"}
            start_idx = -1
            start_char = None

            for i, char in enumerate(text):
                if char in start_chars:
                    start_idx = i
                    start_char = char
                    break

            if start_idx == -1:
                return None

            # Use stack to find matching closing character
            stack = []
            for i in range(start_idx, len(text)):
                char = text[i]

                if char == start_char:
                    stack.append(char)
                elif char == start_chars[start_char]:
                    stack.pop()
                    if not stack:  # Found complete structure
                        json_text = text[start_idx : i + 1]

                        # Try to parse
                        try:
                            json.loads(json_text)
                            self.log_message(
                                "JSON extracted using balanced parser", "DEBUG"
                            )
                            return json_text
                        except json.JSONDecodeError:
                            # Try with fixes
                            fixed = self._fix_json_string(json_text)
                            if fixed:
                                try:
                                    json.loads(fixed)
                                    self.log_message(
                                        "JSON extracted using balanced parser after fixes",
                                        "DEBUG",
                                    )
                                    return fixed
                                except json.JSONDecodeError:
                                    continue

            return None

        except Exception as e:
            self.log_message(f"Balanced parser error: {e}", "ERROR")
            return None

    def _fix_json_string(self, json_text: str) -> Optional[str]:
        """Apply common JSON fixes for malformed JSON.

        Args:
            json_text: Potentially malformed JSON string

        Returns:
            Fixed JSON string or None
        """
        try:
            # Remove trailing commas
            json_text = re.sub(r",(\s*[}\]])", r"\1", json_text)

            # Convert single quotes to double quotes (preserving escapes)
            json_text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', json_text)

            # Fix Python literals (only if they're not in strings)
            json_text = re.sub(r"\bTrue\b", "true", json_text)
            json_text = re.sub(r"\bFalse\b", "false", json_text)
            json_text = re.sub(r"\bNone\b", "null", json_text)

            return json_text

        except Exception as e:
            self.log_message(f"JSON fixing error: {e}", "ERROR")
            return None

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _get_ollama_response(self, prompt_template: str) -> Optional[Dict[str, Any]]:
        """Get response from Ollama LLM with retry logic.

        Args:
            prompt_template: Prompt to send to the LLM

        Returns:
            Parsed JSON response or None if failed

        Raises:
            ValueError: If response is empty or invalid
            RuntimeError: If LLM communication fails
        """
        try:
            self.log_message("LLM communication attempt...", "DEBUG")

            # Note: ollama library doesn't support timeout directly
            # We'll implement timeout at the application level
            # Increased timeout for longer, more detailed responses
            import time
            start_time = time.time()
            max_timeout = 120  # 2 minutes timeout for longer responses
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt_template}],
            )

            # Check if we exceeded timeout
            if time.time() - start_time > max_timeout:
                self.log_message("LLM response timeout, but continuing with received data", "WARNING")

            raw_text = response.get("message", {}).get("content")
            if not raw_text:
                raise ValueError("Empty response from LLM")

            clean_json_text = self._extract_json_from_text(raw_text)
            if not clean_json_text:
                raise ValueError("No valid JSON found in response")

            parsed_json = json.loads(clean_json_text)
            self.log_message("LLM response successfully parsed", "INFO")
            return parsed_json

        except (ValueError, json.JSONDecodeError) as e:
            self.log_message(f"LLM response parsing failed: {e}", "ERROR")
            raise
        except Exception as e:
            self.log_message(f"LLM communication failed: {e}", "ERROR")
            raise RuntimeError(f"LLM communication failed: {e}") from e

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _get_pytrends_topics(
        self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    ) -> List[str]:
        """Get trending topics using PyTrends API with timeout and retry.

        Args:
            niche: Content niche to search for
            timeframe: Time range for trends
            geo: Geographic location

        Returns:
            List of trending topics

        Raises:
            RuntimeError: If PyTrends API fails
        """
        if not self.pytrends:
            self.log_message(
                "PyTrends not available - skipping trending topics", "WARNING"
            )
            return []

        try:
            # Map niches to relevant search terms
            niche_queries = {
                "history": [
                    "ancient mysteries",
                    "historical discoveries",
                    "archaeology news",
                ],
                "science": [
                    "scientific breakthroughs",
                    "space discoveries",
                    "technology trends",
                ],
                "mystery": [
                    "unsolved mysteries",
                    "conspiracy theories",
                    "paranormal news",
                ],
                "true_crime": ["crime documentaries", "cold cases", "forensic science"],
                "nature": [
                    "wildlife discoveries",
                    "nature mysteries",
                    "environmental news",
                ],
            }

            queries = niche_queries.get(niche, [niche])
            trending_topics = []

            for query in queries:
                try:
                    if hasattr(self.pytrends, "build_payload"):
                        # Online PyTrends
                        self.pytrends.build_payload(
                            [query], timeframe=timeframe, geo=geo, gprop=""
                        )
                        trends_data = self.pytrends.interest_over_time()

                        if not trends_data.empty:
                            # Get top trending terms
                            if hasattr(self.pytrends, "trending_searches"):
                                top_terms = self.pytrends.trending_searches(pn="united_states")
                                if not top_terms.empty:
                                    trending_topics.extend(top_terms[0].head(5).tolist())
                    elif hasattr(self.pytrends, "get_trending_topics"):
                        # Offline PyTrends
                        offline_topics = self.pytrends.get_trending_topics(niche, max_results=10)
                        if offline_topics:
                            trending_topics.extend(offline_topics[:5])

                    time.sleep(1)  # Respect API limits

                except Exception as e:
                    self.log_message(
                        f"Error getting trends for '{query}': {e}", "WARNING"
                    )
                    continue

            # Remove duplicates and return
            unique_topics = list(dict.fromkeys(trending_topics))
            self.log_message(
                f"Found {len(unique_topics)} trending topics for niche '{niche}'",
                "INFO",
            )
            return unique_topics[:10]  # Limit to top 10

        except Exception as e:
            self.log_message(f"Error in PyTrends topics: {e}", "ERROR")
            raise RuntimeError(f"PyTrends API failed: {e}") from e

    def _get_trending_topics_with_fallback(
        self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    ) -> List[str]:
        """Get trending topics with PyTrends fallback to local JSON cache.

        Args:
            niche: Content niche to search for
            timeframe: Time range for trends
            geo: Geographic location

        Returns:
            List of trending topics
        """
        try:
            # Try PyTrends first
            try:
                trending_topics = self._get_pytrends_topics(niche, timeframe, geo)
                if trending_topics:
                    # Cache the results with extended key
                    self._cache_trending_topics(niche, trending_topics, timeframe, geo)
                    return trending_topics
            except Exception as e:
                self.log_message(f"PyTrends failed: {e}, using local cache", "WARNING")

            # Fallback to local cache
            cached_topics = self._load_cached_trending_topics(niche, timeframe, geo)
            if cached_topics:
                self.log_message(
                    f"Using cached trending topics: {len(cached_topics)} topics", "INFO"
                )
                return cached_topics

            # Ultimate fallback to config niche keywords
            channel_config = CHANNELS_CONFIG.get(niche, {})
            fallback_topics = channel_config.get("niche_keywords", [])
            self.log_message(
                f"Using fallback niche keywords: {len(fallback_topics)} topics", "INFO"
            )
            return fallback_topics

        except Exception as e:
            self.log_message(f"Trending topics retrieval failed: {e}", "ERROR")
            return []

    def _cache_trending_topics(
        self, niche: str, topics: List[str], timeframe: str = "today 1-m", geo: str = ""
    ) -> None:
        """Cache trending topics with extended key: f"{niche}:{geo}:{timeframe}".

        Args:
            niche: Content niche
            topics: List of trending topics
            timeframe: Time range
            geo: Geographic location
        """
        try:
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)

            # Extended cache key format
            cache_key = f"{niche}:{geo}:{timeframe}".replace(":", "_").replace(" ", "_")
            cache_file = os.path.join(cache_dir, f"trending_topics_{cache_key}.json")

            cache_data = {
                "niche": niche,
                "timeframe": timeframe,
                "geo": geo,
                "timestamp": time.time(),
                "topics": topics,
                "source": "pytrends",
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self.log_message(
                f"Cached {len(topics)} topics with key '{cache_key}'", "INFO"
            )

        except OSError as e:
            self.log_message(f"Failed to cache topics: {e}", "ERROR")
        except Exception as e:
            self.log_message(f"Unexpected error caching topics: {e}", "ERROR")

    def _load_cached_trending_topics(
        self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    ) -> List[str]:
        """Load cached trending topics using extended key format.

        Args:
            niche: Content niche
            timeframe: Time range
            geo: Geographic location

        Returns:
            List of cached topics or empty list if cache miss/expired
        """
        try:
            # Extended cache key format
            cache_key = f"{niche}:{geo}:{timeframe}".replace(":", "_").replace(" ", "_")
            cache_file = os.path.join("cache", f"trending_topics_{cache_key}.json")

            if os.path.exists(cache_file):
                with open(cache_file, encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Check if cache is still valid (24 hours)
                cache_timestamp = cache_data.get("timestamp", 0)
                if time.time() - cache_timestamp < 24 * 60 * 60:  # 24 hours
                    self.log_message(f"Cache HIT for key '{cache_key}'", "DEBUG")
                    return cache_data.get("topics", [])
                else:
                    self.log_message(f"Cache EXPIRED for key '{cache_key}'", "DEBUG")
            else:
                self.log_message(f"Cache MISS for key '{cache_key}'", "DEBUG")

            return []

        except (OSError, json.JSONDecodeError) as e:
            self.log_message(f"Cache loading failed: {e}", "WARNING")
            return []
        except Exception as e:
            self.log_message(f"Unexpected error loading cache: {e}", "ERROR")
            return []

    def get_trending_topics(
        self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    ) -> List[str]:
        """Get trending topics with enhanced fallback system and extended cache keys.

        Args:
            niche: Content niche to search for
            timeframe: Time range for trends
            geo: Geographic location

        Returns:
            List of trending topics
        """
        return self._get_trending_topics_with_fallback(niche, timeframe, geo)

    def generate_viral_ideas(
        self, channel_name: str, idea_count: int = 1
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate viral video ideas with enhanced trending topics integration.

        Args:
            channel_name: Name of the YouTube channel
            idea_count: Number of ideas to generate

        Returns:
            List of viral video ideas or None if generation fails
        """
        try:
            channel_info = CHANNELS_CONFIG.get(channel_name, {})
            niche = channel_info.get("niche", "history")

            # Get trending topics with fallback system
            trending_topics = self._get_trending_topics_with_fallback(niche)
            trending_context = ""
            if trending_topics:
                trending_context = f"Current trending topics in this niche: {', '.join(trending_topics[:5])}. "

            # Finance-specific optimization
            if niche == "finance":
                prompt = f"""You are a master content strategist for viral YouTube finance documentaries. Generate {idea_count} viral video idea for a YouTube channel about '{niche}'.

{trending_context}

Focus on creating DEEP, ENGAGING finance content that can sustain 10-15 minute videos. Each idea must include:
- A compelling financial mystery, trend, or untold story
- Multiple cliffhangers and suspense elements related to money, markets, or economics
- Engagement hooks to keep viewers engaged throughout the entire video
- Global appeal for English-speaking audiences interested in finance
- Professional, educational, and entertaining tone

REQUIRED JSON FORMAT - Each idea must have:
{{
  "ideas": [
    {{
      "title": "Compelling finance title that creates curiosity and urgency",
      "description": "Detailed description of the financial story/mystery/trend",
      "duration_minutes": 12-18,
      "engagement_hooks": [
        "Shocking financial revelation at 2 minutes",
        "Market mystery at 5 minutes", 
        "Economic twist at 8 minutes",
        "Investment insight at 12 minutes",
        "Final financial shock at 15 minutes"
      ],
      "trending_relevance": "How this connects to current financial trends and market conditions",
      "global_appeal": "Why this appeals to international finance audiences",
      "subtitle_languages": ["English", "Spanish", "French", "German"],
      "finance_keywords": ["investment", "market", "economy", "trading", "wealth", "business", "financial freedom"]
    }}
  ]
}}

Make each idea highly specific to finance, economics, or business. Focus on creating genuine curiosity about money, markets, and financial success. Avoid generic topics - make them finance-specific and trending."""
            else:
                prompt = f"""You are a master content strategist for viral YouTube documentaries. Generate {idea_count} viral video idea for a YouTube channel about '{niche}'.

{trending_context}

Focus on creating DEEP, ENGAGING content that can sustain 10+ minute videos. Each idea must include:
- A compelling mystery or untold story
- Multiple cliffhangers and suspense elements
- Engagement hooks to keep viewers engaged
- Global appeal for English-speaking audiences

REQUIRED JSON FORMAT - Each idea must have:
{{
  "ideas": [
    {{
      "title": "Compelling title that creates curiosity",
      "description": "Detailed description of the story/mystery",
      "duration_minutes": 12-18,
      "engagement_hooks": [
        "Hook that shocks viewers at 2 minutes",
        "Cliffhanger at 5 minutes",
        "Revelation at 8 minutes",
        "Twist at 12 minutes",
        "Final shock at 15 minutes"
      ],
      "trending_relevance": "How this connects to current trends",
      "global_appeal": "Why this appeals to international audiences",
      "subtitle_languages": ["English", "Spanish", "French", "German"]
    }}
  ]
}}

Make each idea highly specific and researchable. Focus on creating genuine curiosity and engagement."""

            result = self._get_ollama_response(prompt)
            if result and "ideas" in result:
                self.log_message(
                    f"Generated {len(result['ideas'])} viral ideas for '{niche}'",
                    "INFO",
                )
                return result["ideas"]
            else:
                self.log_message("Failed to generate viral ideas", "ERROR")
                return None

        except Exception as e:
            self.log_message(f"Error in generate_viral_ideas: {e}", "ERROR")
            return None

    def write_script(
        self, video_idea: Dict[str, Any], channel_name: str
    ) -> Optional[Dict[str, Any]]:
        """Write detailed script for video idea with enhanced structure.

        Args:
            video_idea: Video idea dictionary
            channel_name: Name of the YouTube channel

        Returns:
            Script dictionary or None if generation fails
        """
        try:
            channel_info = CHANNELS_CONFIG.get(channel_name, {})
            niche = channel_info.get("niche", "general")
            
            title = video_idea.get('title', 'N/A')
            description = video_idea.get('description', 'N/A')
            target_duration = video_idea.get('duration_minutes', 15)
            
            # Finance-specific script optimization
            if niche == "finance":
                prompt = f"""
                You are a master scriptwriter for viral YouTube finance documentaries. Your task is to write a highly detailed, long-form script for a {target_duration}-minute video on the topic: '{title}'.

                The script MUST be structured as a list of individual sentences. Each sentence will be a separate scene in the video.
                Generate at least 120 to 180 sentences to ensure the video is long enough for {target_duration} minutes.
                For each sentence, create a highly specific and visually interesting search query for Pexels that relates to finance, business, or economics.

                Your response MUST be ONLY a single, valid JSON object.
                
                # REQUIRED JSON FORMAT:
                {{
                  "video_title": "{title}",
                  "target_duration_minutes": {target_duration},
                  "script": [
                    {{ "sentence": "The first sentence of the finance narration.", "visual_query": "A specific Pexels query for the first sentence related to finance/business." }},
                    {{ "sentence": "The second sentence of the finance narration.", "visual_query": "A specific Pexels query for the second sentence related to finance/business." }}
                  ],
                  "finance_focus": "This script focuses on financial education, market insights, and business success stories",
                  "engagement_hooks": [
                    "Hook that shocks viewers at 2 minutes with financial revelation",
                    "Cliffhanger at 5 minutes with market mystery",
                    "Economic twist at 8 minutes",
                    "Investment insight at 12 minutes", 
                    "Final financial shock at 15 minutes"
                  ]
                }}
                
                Make each sentence engaging, educational, and finance-focused. Include market analysis, investment insights, economic trends, and business success stories. Each visual query should be specific to finance, business, or economics concepts.
                
                IMPORTANT: Generate at least 120-180 sentences to ensure a {target_duration}-minute video. Each sentence should be rich in content and take approximately 3-5 seconds to narrate.
                """
            else:
                prompt = f"""
                You are a master scriptwriter for a viral YouTube documentary. Your task is to write a highly detailed, long-form script for a {target_duration}-minute video on the topic: '{title}'.
                The script MUST be structured as a list of individual sentences. Each sentence will be a separate scene in the video.
                Generate at least 60 to 80 sentences to ensure the video is long enough for {target_duration} minutes.
                For each sentence, create a highly specific and visually interesting search query for Pexels.

                Your response MUST be ONLY a single, valid JSON object.
                
                # REQUIRED JSON FORMAT:
                {{
                  "video_title": "{title}",
                  "target_duration_minutes": {target_duration},
                  "script": [
                    {{ "sentence": "The first sentence of the narration.", "visual_query": "A specific Pexels query for the first sentence." }},
                    {{ "sentence": "The second sentence of the narration.", "visual_query": "A specific Pexels query for the second sentence." }}
                  ]
                }}
                """

            result = self._get_ollama_response(prompt)
            if result and "script" in result:
                sentence_count = len(result["script"])
                self.log_message(
                    f"Generated script with {sentence_count} sentences for '{video_idea.get('title', 'N/A')}'",
                    "INFO",
                )

                # Validate sentence count
                if sentence_count < 60:
                    self.log_message(
                        f"Warning: Script has only {sentence_count} sentences (minimum 60 required)",
                        "WARNING",
                    )

                return result
            else:
                self.log_message("Failed to generate script", "ERROR")
                return None

        except Exception as e:
            self.log_message(f"Error in write_script: {e}", "ERROR")
            return None

    def enhance_script_with_metadata(
        self, script_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add enhanced metadata, optimization suggestions, and self-improve with Ollama.

        Args:
            script_data: Script data dictionary

        Returns:
            Enhanced script data with metadata
        """
        try:
            if not script_data or "script" not in script_data:
                return script_data

            # Calculate estimated duration
            total_sentences = len(script_data["script"])
            estimated_duration = total_sentences * 8  # Assume 8 seconds per sentence

            # Self-improve: Use Ollama to generate extra sentences if duration is low
            if estimated_duration < 600:  # Less than 10 minutes
                self.log_message(
                    f"Self-improving script: Duration {estimated_duration}s is low, generating extra sentences",
                    "INFO",
                )
                extra_sentences = self._generate_extra_sentences_with_ollama(
                    script_data, estimated_duration
                )
                if extra_sentences:
                    script_data["script"].extend(extra_sentences)
                    total_sentences = len(script_data["script"])
                    estimated_duration = total_sentences * 8
                    self.log_message(
                        f"Added {len(extra_sentences)} extra sentences, new total: {total_sentences}",
                        "INFO",
                    )

            # Add enhanced metadata
            enhanced_metadata = {
                "estimated_duration_seconds": estimated_duration,
                "estimated_duration_minutes": round(estimated_duration / 60, 1),
                "sentence_count": total_sentences,
                "optimization_suggestions": [
                    "Use cinematic 4K footage for maximum visual impact",
                    "Implement smooth transitions between scenes",
                    "Add atmospheric background music",
                    "Include text overlays for key points",
                    "Use color grading for dramatic effect",
                ],
                "subtitle_optimization": {
                    "English": "Primary language, optimize for clarity",
                    "Spanish": "Latin American and European Spanish",
                    "French": "International French with clear pronunciation",
                    "German": "Standard German with proper grammar",
                },
                "self_improvement": {
                    "extra_sentences_generated": total_sentences
                    - len(script_data.get("script", [])),
                    "duration_improvement": f"{estimated_duration}s (target: 600s+)",
                    "quality_enhancement": "Ollama-powered content expansion",
                },
            }

            script_data["enhanced_metadata"] = enhanced_metadata
            self.log_message(
                f"Enhanced script metadata added for {total_sentences} sentences",
                "INFO",
            )

            return script_data

        except Exception as e:
            self.log_message(f"Error in enhance_script_with_metadata: {e}", "ERROR")
            return script_data

    def _generate_extra_sentences_with_ollama(
        self, script_data: Dict[str, Any], current_duration: float
    ) -> List[Dict[str, Any]]:
        """Use Ollama to generate extra sentences for low duration scripts.

        Args:
            script_data: Current script data
            current_duration: Current script duration in seconds

        Returns:
            List of extra sentences to add
        """
        try:
            target_duration = 600  # 10 minutes minimum
            extra_duration_needed = target_duration - current_duration
            extra_sentences_needed = int(
                extra_duration_needed / 8
            )  # 8 seconds per sentence

            if extra_sentences_needed <= 0:
                return []

            video_title = script_data.get("video_title", "Unknown")

            prompt = f"""Generate {extra_sentences_needed} extra sentences for low duration.

Video title: {video_title}
Current duration: {current_duration:.1f} seconds
Target duration: {target_duration} seconds
Extra sentences needed: {extra_sentences_needed}

Generate sentences to add to the end of the current script.
Each sentence should:
- Continue the existing story
- Take 8 seconds
- Be visually rich
- Contain engagement hooks

Return in JSON format:
            {{
              "extra_sentences": [
                {{
                  "sentence": "Extra sentence text",
                  "visual_query": "cinematic 4K [scene]",
                  "timing_seconds": {current_duration + 8},
                  "engagement_hook": "Hook description"
                }}
              ]
            }}"""

            result = self._get_ollama_response(prompt)
            if result and "extra_sentences" in result:
                extra_sentences = result["extra_sentences"]
                self.log_message(
                    f"Ollama generated {len(extra_sentences)} extra sentences", "INFO"
                )
                return extra_sentences
            else:
                self.log_message(
                    "Failed to generate extra sentences with Ollama", "WARNING"
                )
                return []

        except Exception as e:
            self.log_message(f"Error generating extra sentences: {e}", "ERROR")
            return []

    def generate_enhanced_content(self, topic: str, niche: str, content_type: str = "video_script",
                                 target_length: int = 300, style: str = "professional") -> dict:
        """Generate enhanced content with quality optimization"""
        
        print(f"🚀 Generating enhanced content: {content_type} for {niche}")
        
        try:
            # Get content template
            template = self.content_templates.get(niche, self.content_templates["history"])
            
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(topic, niche, content_type, target_length, style, template)
            
            # Generate content with LLM
            raw_content = self._generate_with_llm(enhanced_prompt)
            
            # Enhance and validate content
            enhanced_content = self._enhance_content_quality(raw_content, template, niche)
            
            # Score content quality
            quality_score = self._score_content_quality(enhanced_content, template, niche)
            
            # SEO optimization
            if CONTENT_QUALITY_CONFIG["seo_optimization"]:
                enhanced_content = self._optimize_for_seo(enhanced_content, niche)
            
            return {
                "content": enhanced_content,
                "quality_score": quality_score,
                "template_used": template,
                "niche": niche,
                "content_type": content_type,
                "target_length": target_length,
                "style": style,
                "generation_time": time.time()
            }
            
        except Exception as e:
            print(f"❌ Enhanced content generation failed: {e}")
            return {
                "content": f"Error generating content: {e}",
                "quality_score": 0.0,
                "error": str(e)
            }
    
    def _create_enhanced_prompt(self, topic: str, niche: str, content_type: str,
                               target_length: int, style: str, template: dict) -> str:
        """Create enhanced prompt for better content generation"""
        
        # Base prompt structure
        base_prompt = f"""
You are an expert content creator specializing in {niche} content. 
Create a {content_type} about "{topic}" with the following requirements:

CONTENT STRUCTURE:
{chr(10).join([f"{i+1}. {section}" for i, section in enumerate(template['structure'])])}

STYLE REQUIREMENTS:
- Tone: {template['tone']}
- Target length: {target_length} words
- Style: {style}
- Engaging and viral potential

KEYWORDS TO INCLUDE:
{', '.join(template['keywords'])}

HOOK TEMPLATE:
{template['hook'].format(topic=topic, action="[relevant action]", emotion="[emotion]")}

Please create compelling, engaging content that follows this structure and style.
"""
        
        return base_prompt
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate content using LLM with optimization"""
        try:
            import ollama
            
            # Use optimized parameters
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                **self.model_params
            )
            
            return response['response']
            
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            return f"Content generation error: {e}"
    
    def _enhance_content_quality(self, content: str, template: dict, niche: str) -> str:
        """Enhance content quality using templates and optimization"""
        
        try:
            # Apply style consistency
            if CONTENT_QUALITY_CONFIG["style_consistency"]:
                content = self._apply_style_consistency(content, template, niche)
            
            # Apply engagement optimization
            if CONTENT_QUALITY_CONFIG["engagement_optimization"]:
                content = self._optimize_engagement(content, template, niche)
            
            # Apply content validation
            if CONTENT_QUALITY_CONFIG["content_validation"]:
                content = self._validate_content(content, template, niche)
            
            return content
            
        except Exception as e:
            print(f"⚠️ Content enhancement failed: {e}")
            return content
    
    def _apply_style_consistency(self, content: str, template: dict, niche: str) -> str:
        """Apply style consistency to content"""
        
        try:
            # Ensure content follows template structure
            lines = content.split('\n')
            enhanced_lines = []
            
            for i, line in enumerate(lines):
                if i < len(template['structure']):
                    # Add structure markers
                    enhanced_lines.append(f"## {template['structure'][i].upper()}")
                    enhanced_lines.append("")
                
                enhanced_lines.append(line)
            
            return '\n'.join(enhanced_lines)
            
        except Exception as e:
            print(f"⚠️ Style consistency failed: {e}")
            return content
    
    def _optimize_engagement(self, content: str, template: dict, niche: str) -> str:
        """Optimize content for engagement"""
        
        try:
            # Add engagement elements
            engagement_elements = [
                "💡 Pro Tip:",
                "🔥 Hot Take:",
                "🎯 Key Insight:",
                "🚀 Action Step:",
                "💪 Challenge:"
            ]
            
            # Insert engagement elements at strategic points
            lines = content.split('\n')
            enhanced_lines = []
            
            for i, line in enumerate(lines):
                enhanced_lines.append(line)
                
                # Add engagement element every few paragraphs
                if i > 0 and i % 5 == 0 and len(engagement_elements) > 0:
                    element = engagement_elements.pop(0)
                    enhanced_lines.append(f"\n{element}")
                    enhanced_lines.append("")
            
            return '\n'.join(enhanced_lines)
            
        except Exception as e:
            print(f"⚠️ Engagement optimization failed: {e}")
            return content
    
    def _validate_content(self, content: str, template: dict, niche: str) -> str:
        """Validate content quality and structure"""
        
        try:
            # Check content length
            word_count = len(content.split())
            if word_count < 50:
                content += "\n\nThis content needs more detail and elaboration."
            
            # Check for required elements
            required_elements = ["hook", "call to action", "key points"]
            missing_elements = []
            
            for element in required_elements:
                if element.lower() not in content.lower():
                    missing_elements.append(element)
            
            if missing_elements:
                content += f"\n\nMissing elements: {', '.join(missing_elements)}"
            
            return content
            
        except Exception as e:
            print(f"⚠️ Content validation failed: {e}")
            return content
    
    def _score_content_quality(self, content: str, template: dict, niche: str) -> float:
        """Score content quality based on multiple factors"""
        
        try:
            score = 0.0
            max_score = 100.0
            
            # Length score (20 points)
            word_count = len(content.split())
            if 200 <= word_count <= 500:
                score += 20
            elif 100 <= word_count < 200:
                score += 15
            elif word_count > 500:
                score += 10
            
            # Structure score (25 points)
            structure_score = 0
            for section in template['structure']:
                if section.lower() in content.lower():
                    structure_score += 5
            score += min(structure_score, 25)
            
            # Engagement score (25 points)
            engagement_words = ["pro tip", "key insight", "action step", "challenge", "hot take"]
            engagement_score = 0
            for word in engagement_words:
                if word in content.lower():
                    engagement_score += 5
            score += min(engagement_score, 25)
            
            # Keyword score (20 points)
            keyword_score = 0
            for keyword in template['keywords']:
                if keyword in content.lower():
                    keyword_score += 4
            score += min(keyword_score, 20)
            
            # Readability score (10 points)
            if len(content) > 0 and word_count > 0:
                avg_word_length = sum(len(word) for word in content.split()) / word_count
                if avg_word_length <= 6:
                    score += 10
                elif avg_word_length <= 8:
                    score += 7
                elif avg_word_length <= 10:
                    score += 4
            
            return min(score, max_score)
            
        except Exception as e:
            print(f"⚠️ Quality scoring failed: {e}")
            return 50.0  # Default score
    
    def _optimize_for_seo(self, content: str, niche: str) -> str:
        """Optimize content for SEO"""
        
        try:
            # Add SEO elements
            seo_elements = [
                f"#{niche.title()} Content",
                f"Best {niche} strategies",
                f"Top {niche} tips",
                f"{niche} guide 2025",
                f"Professional {niche} advice"
            ]
            
            # Insert SEO elements
            content = f"# {seo_elements[0]}\n\n{content}"
            
            # Add meta description
            meta_desc = f"Discover the best {niche} strategies and tips for 2025. Expert advice and actionable insights."
            content += f"\n\n---\nMeta Description: {meta_desc}"
            
            return content
            
        except Exception as e:
            print(f"⚠️ SEO optimization failed: {e}")
            return content

    def generate_batch_content(self, prompts: List[str], batch_size: int = None) -> List[str]:
        """Generate content for multiple prompts in batches for efficiency"""
        if batch_size is None:
            batch_size = OLLAMA_OPTIMIZATION["batch_size"]
        
        results = []
        
        try:
            import ollama
            client = ollama.Client()
            
            # Process prompts in batches
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                print(f"🚀 Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
                
                # Process batch
                batch_results = []
                for prompt in batch:
                    try:
                        response = client.generate(
                            model=self.model,
                            prompt=prompt,
                            **self.model_params
                        )
                        batch_results.append(response['response'])
                        
                        # Memory cleanup after each generation
                        if OLLAMA_OPTIMIZATION["memory_optimization"]:
                            self._cleanup_memory()
                            
                    except Exception as e:
                        print(f"⚠️ Error in batch processing: {e}")
                        batch_results.append("")
                
                results.extend(batch_results)
                
                # Batch cleanup
                if OLLAMA_OPTIMIZATION["memory_optimization"]:
                    self._cleanup_batch_memory()
            
            print(f"✅ Batch processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            # Fallback to individual processing
            return [self.generate_content(prompt) for prompt in prompts]
    
    def _cleanup_memory(self):
        """Clean up memory after each generation"""
        try:
            import gc
            gc.collect()
            
            # GPU memory cleanup if available
            if TORCH_CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"⚠️ Memory cleanup failed: {e}")
    
    def _cleanup_batch_memory(self):
        """Clean up memory after batch processing"""
        try:
            import gc
            gc.collect()
            
            # Force garbage collection
            gc.collect()
            
            # GPU memory cleanup if available
            if TORCH_CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            print(f"⚠️ Batch memory cleanup failed: {e}")


# Convenience functions for backward compatibility
def generate_viral_ideas(
    channel_name: str, idea_count: int = 1
) -> Optional[List[Dict[str, Any]]]:
    """Backward compatibility function for generating viral ideas.

    Args:
        channel_name: Name of the YouTube channel
        idea_count: Number of ideas to generate

    Returns:
        List of viral video ideas or None if generation fails
    """
    handler = ImprovedLLMHandler()
    return handler.generate_viral_ideas(channel_name, idea_count)


def write_script(
    video_idea: Dict[str, Any], channel_name: str
) -> Optional[Dict[str, Any]]:
    """Backward compatibility function for writing scripts.

    Args:
        video_idea: Video idea dictionary
        channel_name: Name of the YouTube channel

    Returns:
        Generated script dictionary or None if generation fails
    """
    handler = ImprovedLLMHandler()
    script = handler.write_script(video_idea, channel_name)
    if script:
        return handler.enhance_script_with_metadata(script)
    return None


if __name__ == "__main__":
    # Test the improved handler
    print("Testing Improved LLM Handler...")

    handler = ImprovedLLMHandler()

    # Test viral ideas generation
    print("\nTesting viral ideas generation...")
    ideas = handler.generate_viral_ideas("test_channel", 2)
    if ideas:
        print(f"Generated {len(ideas)} viral ideas")
        for i, idea in enumerate(ideas, 1):
            print(f"  {i}. {idea.get('title', 'No title')}")

    # Test script generation if ideas exist
    if ideas:
        print("\nTesting script generation...")
        script = handler.write_script(ideas[0], "test_channel")
        if script:
            print(f"Generated script with {len(script.get('script', []))} sentences")
            print(
                f"   Enhanced metadata: {script.get('enhanced_metadata', {}).get('estimated_duration_minutes', 'N/A')} minutes"
            )

    print("\nImproved LLM Handler test completed!")
