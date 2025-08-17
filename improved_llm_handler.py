"""Enhanced LLM Handler with robust JSON extraction and network resilience."""

import json
import logging
import os
import re
import sys
import time
import warnings
from functools import wraps
from typing import Any, Dict, List, Optional
import hashlib
from datetime import datetime, timedelta

import ollama

# Suppress FutureWarnings and add time budget helpers
import random
from contextlib import contextmanager
warnings.filterwarnings("ignore", category=FutureWarning)

# Import niche normalization and seed topics from config
from config import (
    NICHE_ALIASES, normalize_niche, TIER1_GEOS, TIER2_GEOS, 
    DEFAULT_TIMEFRAMES, MAX_TOPICS, SEED_TOPICS
)

def niche_from_channel(channel_name: str) -> str:
    """
    Helper function to automatically resolve niche from channel name.
    
    Args:
        channel_name: Channel name (e.g., "CKDrive", "cklegends", "CKIronWill")
        
    Returns:
        Normalized niche string (e.g., "automotive", "history", "motivation")
        
    Examples:
        >>> niche_from_channel("CKDrive")
        'automotive'
        >>> niche_from_channel("cklegends") 
        'history'
        >>> niche_from_channel("CKIronWill")
        'motivation'
        
    Usage in pipeline:
        # Instead of manually specifying niche:
        # topics = handler.get_topics_resilient("automotive", timeframe="today 1-m")
        
        # Use the helper for automatic resolution:
        niche = niche_from_channel(channel_name)  # "CKDrive" -> "automotive"
        topics = handler.get_topics_resilient(niche, timeframe="today 1-m", geo="US")
    """
    return normalize_niche(channel_name)

@contextmanager
def time_budget(seconds: float):
    start = time.monotonic()
    yield
    if (time.monotonic() - start) > seconds:
        raise TimeoutError(f"Time budget exceeded ({seconds}s)")

# --- Daily Cache + 7-day Dedupe Helpers ---
def _today_str():
    return datetime.utcnow().strftime("%Y-%m-%d")

def _cache_dir():
    d = os.path.join("data", "cache", "topics")
    os.makedirs(d, exist_ok=True)
    return d

def _cache_key(niche: str):
    base = f"{_today_str()}::{niche.strip().lower()}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:10]

def _cache_path(niche: str):
    return os.path.join(_cache_dir(), f"{_cache_key(niche)}.json")

def _load_recent_topics(niche: str, days: int = 7) -> list[str]:
    """Collect topics from last <days> cache files for dedupe."""
    recent = []
    root = _cache_dir()
    cutoff = datetime.utcnow() - timedelta(days=days)
    for fn in os.listdir(root):
        if not fn.endswith(".json"):
            continue
        try:
            date_part = fn.split("_")[0]  # backward-safe; ignore if missing
        except Exception:
            date_part = None
        fpath = os.path.join(root, fn)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            dt = datetime.utcfromtimestamp(data.get("ts", 0))
            if dt >= cutoff and data.get("niche") == niche.lower():
                recent.extend(data.get("topics", []))
        except Exception:
            continue
    # dedupe recent
    seen=set(); out=[]
    for t in recent:
        k=t.strip().lower()
        if k and k not in seen:
            seen.add(k); out.append(t.strip())
    return out

def _save_topics_cache(niche: str, topics: list[str]):
    payload = {"ts": int(time.time()), "date": _today_str(), "niche": niche.lower(), "topics": topics}
    with open(_cache_path(niche), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# --- Augment Cache Helpers ---
def _day_salt(niche: str) -> str:
    base = f"{_today_str()}::{niche.strip().lower()}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:6]

def _augment_cache_dir():
    d = os.path.join("data", "cache", "augment")
    os.makedirs(d, exist_ok=True)
    return d

def _augment_cache_path(niche: str):
    return os.path.join(_augment_cache_dir(), f"{_today_str()}_{niche.lower()}.json")

def _load_augment_cache(niche: str) -> list[str]:
    p = _augment_cache_path(niche)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("topics", [])
        except Exception:
            return []
    return []

def _save_augment_cache(niche: str, topics: list[str]):
    try:
        with open(_augment_cache_path(niche), "w", encoding="utf-8") as f:
            json.dump({"date": _today_str(), "niche": niche, "topics": topics}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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


def _get_trend_client():
    if TrendReq is not None:
        try:
            client = TrendReq(hl="en-US", tz=0)
            # Set timeout configuration for fast failure
            try:
                if hasattr(client, "timeout"):
                    client.timeout = (5, 10)  # (connect, read) timeout
                if hasattr(client, "requests_args") and isinstance(client.requests_args, dict):
                    client.requests_args.update({"allow_redirects": True})
            except Exception:
                pass  # Safe to ignore timeout config errors
            return client
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
    """Enhanced LLM Handler with robust JSON extraction and network resilience."""

    def __init__(self, ollama_model: str = "llama3:8b"):
        """Initialize the improved LLM handler.

        Args:
            ollama_model: Ollama model to use
        """
        self.ollama_model = ollama_model
        self.pytrends = None
        self.logger = None
        
        # Performance optimizations
        self._response_cache = {}  # Simple in-memory cache
        self._cache_max_size = 100  # Maximum cache size
        self._cache_ttl = 300  # Cache TTL in seconds (5 minutes)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize PyTrends
        self._init_pytrends()
        
        self.log_message("Improved LLM Handler initialized", "INFO")

    def _init_pytrends(self):
        """Initialize PyTrends client."""
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='en-US', tz=360)
            self.log_message("PyTrends initialized successfully", "INFO")
        except ImportError:
            self.log_message("PyTrends not available - using offline fallback", "WARNING")
            try:
                from pytrends_offline import OfflinePyTrends
                self.pytrends = OfflinePyTrends()
                self.log_message("Offline PyTrends initialized", "INFO")
            except ImportError:
                self.log_message("No PyTrends available - trending features disabled", "WARNING")
                self.pytrends = None

    def _setup_logging(self) -> None:
        """Set up enhanced logging with standardized levels."""
        self.log_file = f"llm_handler_{int(time.time())}.log"
        self.logger = logging.getLogger(f"llm_handler_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_message("Logging initialized", "DEBUG")

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
        """Extract valid JSON from text using multiple extraction methods.

        Args:
            text: Text containing JSON data

        Returns:
            Extracted JSON string or None if extraction fails

        Raises:
            ValueError: If text is empty or invalid
        """
        if not text:
            raise ValueError("Text cannot be empty")

        # Quick check: if text starts and ends with braces/brackets, it might already be valid JSON
        text = text.strip()
        if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
            try:
                json.loads(text)
                self.log_message("Text is already valid JSON", "DEBUG")
                return text
            except json.JSONDecodeError:
                pass  # Continue with extraction methods

        # Method 1: Extract from fenced code blocks (fastest)
        json_block = self._extract_from_fenced_blocks(text)
        if json_block:
            return json_block

        # Method 2: Use stack-based parser for balanced JSON
        balanced_json = self._extract_balanced_json(text)
        if balanced_json:
            return balanced_json

        # Method 3: Try to find JSON after common prefixes
        prefix_json = self._extract_after_prefixes(text)
        if prefix_json:
            return prefix_json

        # Method 4: Aggressive JSON search with multiple patterns
        aggressive_json = self._extract_aggressive_json(text)
        if aggressive_json:
            return aggressive_json

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
                    # Check if stack is not empty before popping
                    if stack:
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
                    else:
                        # Stack is empty but we found a closing character
                        # This means the JSON is malformed, skip this character
                        continue

            return None

        except Exception as e:
            self.log_message(f"Balanced parser error: {e}", "ERROR")
            return None

    def _extract_after_prefixes(self, text: str) -> Optional[str]:
        """Extract JSON that appears after common explanatory prefixes.
        
        Args:
            text: Text that may start with explanatory text before JSON
            
        Returns:
            Extracted JSON string or None
        """
        # Common prefixes that LLMs use before JSON responses
        prefixes = [
            r"Here is the script:?\s*",
            r"Here's the script:?\s*",
            r"Here is the response:?\s*",
            r"Here's the response:?\s*",
            r"Here is the answer:?\s*",
            r"Here's the answer:?\s*",
            r"Here is the result:?\s*",
            r"Here's the result:?\s*",
            r"Here is the output:?\s*",
            r"Here's the output:?\s*",
            r"Here is the data:?\s*",
            r"Here's the data:?\s*",
            r"Here is the JSON:?\s*",
            r"Here's the JSON:?\s*",
            r"The script is:?\s*",
            r"The response is:?\s*",
            r"The answer is:?\s*",
            r"The result is:?\s*",
            r"The output is:?\s*",
            r"The data is:?\s*",
            r"The JSON is:?\s*",
        ]
        
        for prefix in prefixes:
            match = re.search(prefix, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Get text after the prefix
                after_prefix = text[match.end():].strip()
                if after_prefix:
                    # Try to extract JSON from the remaining text
                    json_text = self._extract_balanced_json(after_prefix)
                    if json_text:
                        self.log_message("JSON extracted after prefix removal", "DEBUG")
                        return json_text
                    
                    # If balanced extraction fails, try to find any JSON-like structure
                    json_like = self._find_json_like_structure(after_prefix)
                    if json_like:
                        try:
                            json.loads(json_like)
                            self.log_message("JSON extracted after prefix removal (json-like)", "DEBUG")
                            return json_like
                        except json.JSONDecodeError:
                            # Try with fixes
                            fixed = self._fix_json_string(json_like)
                            if fixed:
                                try:
                                    json.loads(fixed)
                                    self.log_message("JSON extracted after prefix removal (fixed)", "DEBUG")
                                    return fixed
                                except json.JSONDecodeError:
                                    continue
                    
                    # If no JSON found, check if the text after prefix contains any JSON-like content
                    if any(char in after_prefix for char in '{['):
                        # There might be partial JSON, try to extract it
                        partial_json = self._extract_partial_json(after_prefix)
                        if partial_json:
                            return partial_json
        
        return None

    def _extract_partial_json(self, text: str) -> Optional[str]:
        """Extract partial JSON that might be incomplete but fixable.
        
        Args:
            text: Text that might contain partial JSON
            
        Returns:
            Fixed JSON string or None
        """
        # Look for the start of JSON structures
        start_chars = {'{': '}', '[': ']'}
        
        for start_char, end_char in start_chars.items():
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Found a potential JSON start, try to complete it
                partial = text[start_idx:]
                
                # Count opening and closing characters
                open_count = partial.count(start_char)
                close_count = partial.count(end_char)
                
                # If we have more opening than closing, add missing closers
                if open_count > close_count:
                    missing = open_count - close_count
                    partial += end_char * missing
                    
                    # Try to parse the completed JSON
                    try:
                        json.loads(partial)
                        self.log_message("Partial JSON completed and parsed", "DEBUG")
                        return partial
                    except json.JSONDecodeError:
                        # Try with additional fixes
                        fixed = self._fix_json_string(partial)
                        if fixed:
                            try:
                                json.loads(fixed)
                                self.log_message("Partial JSON completed, fixed, and parsed", "DEBUG")
                                return fixed
                            except json.JSONDecodeError:
                                continue
        
        return None

    def _extract_aggressive_json(self, text: str) -> Optional[str]:
        """Aggressively search for JSON patterns in the text.
        
        Args:
            text: Text to search for JSON patterns
            
        Returns:
            Extracted JSON string or None
        """
        # Look for JSON object patterns with more flexible matching
        patterns = [
            # Look for { ... } patterns
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            # Look for [ ... ] patterns  
            r'\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]',
            # Look for key-value patterns that might be JSON
            r'\{[^}]*"[^"]*"\s*:\s*[^}]*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Clean up the match
                cleaned = match.strip()
                if len(cleaned) < 10:  # Too short to be meaningful JSON
                    continue
                    
                try:
                    json.loads(cleaned)
                    self.log_message("JSON extracted using aggressive pattern matching", "DEBUG")
                    return cleaned
                except json.JSONDecodeError:
                    # Try with fixes
                    fixed = self._fix_json_string(cleaned)
                    if fixed:
                        try:
                            json.loads(fixed)
                            self.log_message("JSON extracted using aggressive pattern matching (fixed)", "DEBUG")
                            return fixed
                        except json.JSONDecodeError:
                            continue
        
        return None

    def _find_json_like_structure(self, text: str) -> Optional[str]:
        """Find structures that look like JSON but might be incomplete.
        
        Args:
            text: Text to search for JSON-like structures
            
        Returns:
            JSON-like string or None
        """
        # Look for the longest valid JSON-like structure
        best_match = None
        best_length = 0
        
        # Find all potential JSON start positions
        start_positions = []
        for i, char in enumerate(text):
            if char in '{[':
                start_positions.append(i)
        
        for start_pos in start_positions:
            # Try to find a complete structure from this position
            json_candidate = self._extract_balanced_json(text[start_pos:])
            if json_candidate and len(json_candidate) > best_length:
                best_match = json_candidate
                best_length = len(json_candidate)
        
        return best_match

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

            # Fix common LLM formatting issues
            # Remove any leading/trailing text that's not JSON
            json_text = re.sub(r'^[^{\[\]]*', '', json_text)
            json_text = re.sub(r'[^{\[\]]*$', '', json_text)
            
            # Fix missing quotes around keys (more robust pattern)
            # Look for unquoted keys followed by colons
            json_text = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_text)
            
            # Fix missing quotes around string values
            # Look for values that are not quoted, not numbers, not booleans, not null
            json_text = re.sub(r':\s*([^"][^,}\]]*[^"\s,}\]])', r': "\1"', json_text)
            
            # Fix trailing commas in arrays and objects
            json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
            
            # Fix missing commas between array elements
            json_text = re.sub(r'(\])\s*(\[)', r'\1,\2', json_text)
            json_text = re.sub(r'(\})\s*(\{)', r'\1,\2', json_text)
            
            # Fix missing quotes around array/object values
            json_text = re.sub(r':\s*([^"][^,}\]]*[^"\s,}\]])', r': "\1"', json_text)
            
            # Remove any non-printable characters
            json_text = ''.join(char for char in json_text if char.isprintable() or char in '\n\r\t')
            
            # Try to balance braces and brackets if they're mismatched
            open_braces = json_text.count('{')
            close_braces = json_text.count('}')
            open_brackets = json_text.count('[')
            close_brackets = json_text.count(']')
            
            # Add missing closing braces/brackets
            if open_braces > close_braces:
                json_text += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                json_text += ']' * (open_brackets - close_brackets)

            # Final cleanup: remove any remaining unquoted keys
            # This is a more aggressive fix for cases like {"key": value} -> {"key": "value"}
            json_text = re.sub(r':\s*([^"][^,}\]]*[^"\s,}\]])', r': "\1"', json_text)

            return json_text

        except Exception as e:
            self.log_message(f"JSON fixing error: {e}", "ERROR")
            return None

    def _create_fallback_response(self, prompt: str, raw_text: str) -> Dict[str, Any]:
        """Create a fallback response when JSON parsing completely fails.
        
        Args:
            prompt: The original prompt that was sent
            raw_text: The raw text response from the LLM
            
        Returns:
            A basic response structure that allows the system to continue
        """
        self.log_message("Creating fallback response due to JSON parsing failure", "WARNING")
        
        # Try to extract some meaningful content from the raw text
        fallback_content = raw_text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here is the script:", "Here's the script:", "Here is the response:", 
            "Here's the response:", "Here is the answer:", "Here's the answer:",
            "Here is the result:", "Here's the result:", "Here is the output:", 
            "Here's the output:", "Here is the data:", "Here's the data:",
            "Here is the JSON:", "Here's the JSON:", "The script is:", 
            "The response is:", "The answer is:", "The result is:", 
            "The output is:", "The data is:", "The JSON is:"
        ]
        
        for prefix in prefixes_to_remove:
            if fallback_content.startswith(prefix):
                fallback_content = fallback_content[len(prefix):].strip()
                break
        
        # SANITIZE content to prevent broken filenames
        def sanitize_text(text: str) -> str:
            """Remove or replace characters that break filenames"""
            import re
            # Remove or replace problematic characters
            text = re.sub(r'[<>:"/\\|?*\[\]{}]', '', text)
            # Remove quotes and other problematic characters
            text = text.replace('"', '').replace("'", '').replace('`', '')
            # Remove multiple spaces and newlines
            text = re.sub(r'\s+', ' ', text)
            # Limit length
            return text.strip()[:80]
        
        sanitized_content = sanitize_text(fallback_content)
        
        # Create a basic response structure based on the prompt content
        if "viral video idea" in prompt.lower() or "ideas" in prompt.lower():
            # Create meaningful fallback content - SHORT AND CLEAN
            fallback_title = "Tech Discovery"
            fallback_description = "Latest technology breakthroughs and innovations."
            
            return {
                "ideas": [
                    {
                        "title": fallback_title,
                        "description": fallback_description,
                        "duration_minutes": 15,
                        "engagement_hooks": [
                            "Shocking revelation at 2 minutes",
                            "Mystery unveiled at 5 minutes", 
                            "Unexpected twist at 8 minutes",
                            "Future prediction at 12 minutes",
                            "Final surprise at 15 minutes"
                        ],
                        "trending_relevance": "Technology trends and innovations",
                        "global_appeal": "Universal interest in technology",
                        "subtitle_languages": ["English"]
                    }
                ],
                "fallback_response": True
            }
        elif "script" in prompt.lower() or "write" in prompt.lower():
            # Create meaningful fallback script - SHORT AND CLEAN
            fallback_script = [
                "Welcome to technology world.",
                "Exploring amazing developments.",
                "AI and energy innovations.",
                "Discovering breakthroughs.",
                "Future is here now.",
                "Technology reshaping world.",
                "Stay tuned for more."
            ]
            
            return {
                "script": fallback_script,
                "title": "Tech Discovery",
                "duration_minutes": 3.5,  # 7 sentences * 0.5 seconds
                "enhanced_metadata": {
                    "estimated_duration_minutes": 3.5,
                    "sentence_count": 7,
                    "fallback_response": True
                },
                "fallback_response": True
            }
        else:
            # Generic fallback with meaningful content - SHORT AND CLEAN
            meaningful_content = "Technology innovations and breakthroughs."
            return {
                "content": meaningful_content,
                "fallback_response": True,
                "raw_text_length": len(raw_text)
            }

    @retry_with_backoff(max_retries=2, base_delay=1.0)  # Reduce retries to prevent long hangs
    def _get_ollama_response(self, prompt_template: str) -> Optional[Dict[str, Any]]:
        """Get response from Ollama LLM with retry logic and caching.

        Args:
            prompt_template: Prompt to send to the LLM

        Returns:
            Parsed JSON response or None if failed

        Raises:
            ValueError: If response is empty or invalid
            RuntimeError: If LLM communication fails
        """
        try:
            # Check cache first
            cached_response = self._get_cached_response(prompt_template)
            if cached_response:
                return cached_response
            
            self.log_message("LLM communication attempt...", "DEBUG")

            # Add aggressive timeout protection (Windows-compatible)
            import threading
            import queue
            import time
            
            response_queue = queue.Queue()
            timeout_occurred = False
            start_time = time.time()
            
            def ollama_request():
                try:
                    # Add internal timeout check
                    if time.time() - start_time > 30:  # 30 second internal timeout
                        response_queue.put(TimeoutError("Internal timeout"))
                        return
                    
                    # Try direct HTTP request first (more reliable)
                    try:
                        import requests
                        data = {
                            "model": self.ollama_model,
                            "prompt": prompt_template,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "top_k": 40
                            }
                        }
                        
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json=data,
                            timeout=25,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if response.status_code == 200:
                            response_queue.put(response.json())
                        else:
                            # Try ollama library as fallback
                            response = ollama.chat(
                                model=self.ollama_model,
                                messages=[{"role": "user", "content": prompt_template}],
                            )
                            response_queue.put(response)
                            
                    except Exception as http_error:
                        # Fallback to ollama library
                        response = ollama.chat(
                            model=self.ollama_model,
                            messages=[{"role": "user", "content": prompt_template}],
                        )
                        response_queue.put(response)
                        
                except Exception as e:
                    response_queue.put(e)
            
            # Start request in separate thread
            request_thread = threading.Thread(target=ollama_request)
            request_thread.daemon = True
            request_thread.start()
            
            # Wait for response with shorter timeout
            try:
                response = response_queue.get(timeout=30)  # Reduced to 30 seconds
                if isinstance(response, Exception):
                    raise response
            except queue.Empty:
                timeout_occurred = True
                self.log_message("LLM request timed out after 30 seconds", "ERROR")
                # Return fallback response on timeout
                fallback_response = self._create_fallback_response(prompt_template, "Request timed out")
                if fallback_response:
                    self.log_message("Using fallback response due to timeout", "WARNING")
                    return fallback_response
                raise RuntimeError("LLM request timed out after 30 seconds")

            # Check if response took too long
            if time.time() - start_time > 45:  # Total timeout check
                self.log_message("LLM request took too long, using fallback", "WARNING")
                fallback_response = self._create_fallback_response(prompt_template, "Request too slow")
                if fallback_response:
                    return fallback_response

            raw_text = response.get("message", {}).get("content")
            if not raw_text:
                raise ValueError("Empty response from LLM")

            # Log the raw response for debugging
            self.log_message(f"Raw LLM response (first 200 chars): {raw_text[:200]}...", "DEBUG")

            clean_json_text = self._extract_json_from_text(raw_text)
            if not clean_json_text:
                # Log more details about the failed extraction
                self.log_message(f"JSON extraction failed. Raw response length: {len(raw_text)}", "ERROR")
                self.log_message(f"Raw response preview: {raw_text[:500]}...", "ERROR")
                
                # Try to create a fallback response instead of failing completely
                fallback_response = self._create_fallback_response(prompt_template, raw_text)
                if fallback_response:
                    self.log_message("Using fallback response due to JSON parsing failure", "WARNING")
                    # Cache the fallback response
                    self._cache_response(prompt_template, fallback_response)
                    return fallback_response
                
                raise ValueError("No valid JSON found in response")

            # Log the extracted JSON for debugging
            self.log_message(f"Extracted JSON (first 200 chars): {clean_json_text[:200]}...", "DEBUG")

            parsed_json = json.loads(clean_json_text)
            self.log_message("LLM response successfully parsed", "INFO")
            
            # Cache the successful response
            self._cache_response(prompt_template, parsed_json)
            
            return parsed_json

        except (ValueError, json.JSONDecodeError) as e:
            self.log_message(f"LLM response parsing failed: {e}", "ERROR")
            # Log the full raw response for debugging
            if 'raw_text' in locals():
                self.log_message(f"Full raw response that failed to parse: {raw_text}", "ERROR")
            raise
        except Exception as e:
            self.log_message(f"LLM communication failed: {e}", "ERROR")
            # Always try to return fallback on any error
            try:
                fallback_response = self._create_fallback_response(prompt_template, f"Error: {str(e)}")
                if fallback_response:
                    self.log_message("Using fallback response due to error", "WARNING")
                    return fallback_response
            except:
                pass
            raise RuntimeError(f"LLM communication failed: {e}") from e

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a cache key for a prompt."""
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()[:16]
    
    def _get_cached_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        cache_key = self._get_cache_key(prompt)
        if cache_key in self._response_cache:
            cached_data = self._response_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self._cache_ttl:
                self.log_message("Using cached response", "DEBUG")
                return cached_data['response']
            else:
                # Remove expired cache entry
                del self._response_cache[cache_key]
        return None
    
    def _cache_response(self, prompt: str, response: Dict[str, Any]) -> None:
        """Cache a response for future use."""
        cache_key = self._get_cache_key(prompt)
        
        # Implement LRU cache eviction if needed
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = min(self._response_cache.keys(), 
                           key=lambda k: self._response_cache[k]['timestamp'])
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        self.log_message("Response cached", "DEBUG")

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

            # Try only the first query to avoid long loops - fast fail approach
            query = queries[0] if queries else niche
            try:
                if hasattr(self.pytrends, "build_payload"):
                    # Online PyTrends - single attempt
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

            except Exception as e:
                self.log_message(
                    f"PyTrends query failed for '{query}': {e}", "WARNING"
                )
                # Fast fail - don't continue with more queries
                pass

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

    # OLD FALLBACK FUNCTION - DISABLED (using get_topics_resilient instead)
    # def _get_trending_topics_with_fallback(
    #     self, niche: str, timeframe: str = "today 1-m", geo: str = ""
    # ) -> List[str]:
    #     """Get trending topics with PyTrends fallback to local JSON cache.
    #
    #     Args:
    #         niche: Content niche to search for
    #         timeframe: Time range for trends
    #         geo: Geographic location
    #
    #     Returns:
    #         List of trending topics
    #     """
    #     topics = []
    #     try:
    #         # Try online PyTrends first
    #         topics = self._get_pytrends_topics(niche=niche, timeframe=timeframe, geo=geo)
    #     except Exception as e:
    #         self.log_message(f"PyTrends online failed: {e}; switching to offline.", "WARNING")
    #         topics = []
    #
    #     # If empty, try offline fallback
    #     if not topics:
    #         try:
    #             if hasattr(self.pytrends, "get_trending_topics"):
    #                 topics = self.pytrends.get_trending_topics(niche, max_results=20)
    #         except Exception as e:
    #             self.log_message(f"Offline trends failed: {e}", "WARNING")
    #         topics = []
    #
    #     # Seed fallback: if still empty, provide default set based on niche
    #     if not topics:
    #         SEED = {
    #             "history": [
    #                 "ancient civilizations","lost cities","roman empire","greek mythology",
    #                 "egyptian pharaohs","archaeology discoveries","medieval knights",
    #                 "viking history","mysteries of history","ancient inventions"
    #             ],
    #         "motivation": [
    #                 "discipline tips","productivity habits","mental toughness","morning routine",
    #                 "goal setting","sports motivation","habit building","focus techniques",
    #                 "mindset shift","success stories"
    #         ],
    #         "science": [
    #             "space discoveries","quantum physics","evolution mysteries","climate science",
    #                 "medical breakthroughs","technology trends","scientific controversies",
    #                 "unexplained phenomena","research findings","future predictions"
    #         ],
    #         "mystery": [
    #             "unsolved mysteries","conspiracy theories","paranormal events",
    #                 "cryptid sightings","ancient artifacts","lost treasures","urban legends",
    #                 "supernatural stories","mysterious disappearances","occult history"
    #         ],
    #         "true_crime": [
    #             "cold cases","forensic breakthroughs","criminal psychology",
    #                 "unsolved murders","mysterious deaths","criminal investigations",
    #                 "justice system","crime prevention","victim stories","detective work"
    #         ],
    #         "nature": [
    #             "wildlife discoveries","environmental mysteries","natural phenomena",
    #                 "animal behavior","plant adaptations","ecosystem changes",
    #                 "climate effects","biodiversity","natural disasters","conservation"
    #         ]
    #     }
    #         topics = SEED.get(niche.lower(), SEED["history"])
    #         self.log_message(f"PyTrends online failed: {e}; switching to offline.", "WARNING")
    #
    #     return topics

    def get_topics_by_channel(self, channel_name: str, timeframe: str | None = None, geo: str | None = None) -> list[str]:
        """
        Convenience method to get topics by channel name with automatic niche resolution.
        
        Args:
            channel_name: Channel name (e.g., "CKDrive", "cklegends", "CKIronWill")
            timeframe: Time range for trends (e.g., "today 1-m", "now 7-d")
            geo: Geographic location (e.g., "US", "GB", "CA")
            
        Returns:
            List of trending topics for the channel's niche
            
        Examples:
            >>> handler = ImprovedLLMHandler()
            >>> topics = handler.get_topics_by_channel("CKDrive", geo="US")
            >>> # Automatically resolves "CKDrive" -> "automotive" niche
        """
        niche = niche_from_channel(channel_name)
        return self.get_topics_resilient(niche, timeframe, geo)

    def get_topics_resilient(self, niche: str, timeframe: str | None = None, geo: str | None = None) -> list[str]:
        """Get topics with resilient fallback: online → offline → seed"""
        # Niche normalization
        niche = normalize_niche(niche)

        topics: list[str] = []
        online_warned = False

        # ---- ONLINE (tek atış, 6sn time budget) ----
        geos_to_try = [geo] if geo else (TIER1_GEOS + TIER2_GEOS)
        frames_to_try = [timeframe] if timeframe else DEFAULT_TIMEFRAMES
        try:
            with time_budget(6.0):
                for g in geos_to_try:
                    for tf in frames_to_try:
                        try:
                            # Direct PyTrends call without going through old function
                            if hasattr(self.pytrends, "build_payload"):
                                self.pytrends.build_payload([niche], timeframe=tf, geo=g, gprop="")
                                trends_data = self.pytrends.interest_over_time()
                                if not trends_data.empty and hasattr(self.pytrends, "trending_searches"):
                                    top_terms = self.pytrends.trending_searches(pn="united_states")
                                    if not top_terms.empty:
                                        topics = top_terms[0].head(5).tolist()
                                        if topics:
                                            self.log_message(f"PyTrends online OK: {len(topics)} topics (geo={g}, tf='{tf}')", "INFO")
                                            raise StopIteration
                        except Exception as e:
                            if not online_warned:
                                self.log_message(f"PyTrends online fail → fast fallback ({g}, {tf}): {e}", "WARNING")
                                online_warned = True
                            # ilk hata aldığımızda online'ı tamamen bırak
                            raise StopIteration
        except StopIteration:
            pass
        except TimeoutError as e:
            if not online_warned:
                self.log_message(f"PyTrends online timeout → fast fallback: {e}", "WARNING")
            topics = []
        except Exception as e:
            if not online_warned:
                self.log_message(f"PyTrends online unexpected → fast fallback: {e}", "WARNING")
            topics = []

        # ---- OFFLINE ----
        if not topics and hasattr(self, "pytrends") and hasattr(self.pytrends, "get_trending_topics"):
            try:
                topics = self.pytrends.get_trending_topics(niche, max_results=MAX_TOPICS)
                if topics:
                    self.log_message(f"Offline trends used: {len(topics)} topics", "INFO")
            except Exception as e:
                self.log_message(f"Offline trends failed: {e}", "WARNING")
                topics = []

        # ---- SEED ----
        if not topics:
            topics = (SEED_TOPICS.get(niche.lower()) or SEED_TOPICS.get("history", []))[:MAX_TOPICS]
            self.log_message(f"Using seed fallback for '{niche}': {len(topics)} topics", "WARNING")

        # ---- DEDUPE + SHUFFLE + TRIM ----
        try:
            seen = set(); deduped=[]
            for t in topics:
                k = t.strip().lower()
                if k and k not in seen:
                    seen.add(k); deduped.append(t.strip())
            random.shuffle(deduped)
            topics = deduped[:MAX_TOPICS]
        except Exception:
            pass

        # --- 7 günlük dedupe + augment + backfill policy ---
        try:
            previous = _load_recent_topics(niche, days=7)
            prevset = {p.strip().lower() for p in previous} if previous else set()

            # 1) önce seed/online/offline'dan gelen listeyi dedupe et
            base = topics[:]  # 24 civarı
            novel = [t for t in base if t.strip().lower() not in prevset]

            # 2) yeterince yeni yoksa (örn. 0), günlük augment ile yeni varyant üret
            MIN_NEW, MAX_OUT = 12, MAX_TOPICS
            if len(novel) < MIN_NEW:
                aug = self.augment_seed_topics(niche, base, want=16)
                # augment'leri de 7g'e karşı dedupe et
                aug_novel = [t for t in aug if t.strip().lower() not in prevset]
                novel.extend([t for t in aug_novel if t.strip().lower() not in {x.strip().lower() for x in novel}])

            # 3) final listeyi oluştur (önce novel, sonra backfill)
            final = novel[:MAX_OUT]

            # eksikse backfill: önce base içinden, sonra previous içinden
            if len(final) < MAX_OUT:
                pool = [t for t in base if t not in final] + [t for t in previous if t not in final]
                random.shuffle(pool)
                seen = {x.strip().lower() for x in final}
                for t in pool:
                    if len(final) >= MAX_OUT: break
                    k = t.strip().lower()
                    if k and k not in seen:
                        seen.add(k); final.append(t)

            new_count = len([t for t in final if t.strip().lower() not in prevset])
            if new_count < MIN_NEW:
                self.logger.warning(f"Dedupe+augment produced {new_count} new topics; backfilled to {len(final)}.")

            topics = final
        except Exception as e:
            self.logger.warning(f"Dedupe/augment step failed: {e}")
            # topics olduğu gibi kalsın
        
        # --- cache yaz ---
        try:
            _save_topics_cache(niche, topics)
            self.logger.info(f"Cached {len(topics)} topics for {niche} (7d dedupe + augment + backfill applied)")
        except Exception as e:
            self.logger.warning(f"Topic cache write failed: {e}")
        return topics

    def score_topics_with_llm(self, niche: str, topics: list[str], top_k: int = 8) -> list[tuple[str, float]]:
        """
        Returns list of (topic, score) sorted desc by score. Falls back to simple heuristics.
        """
        if not topics:
            return []
        prompt = (
            "You are a YouTube growth strategist. Score each topic from 0.0 to 1.0 for potential CTR and retention. "
            "Consider curiosity gap, timeliness, evergreen appeal for the niche.\n"
            f"Niche: {niche}\n"
            "CRITICAL: Return ONLY valid JSON array of objects: [{\"topic\": str, \"score\": float}]\n"
            "No explanatory text before or after the JSON.\n"
            f"Topics: {topics[:24]}\n\n"
            "IMPORTANT: Start your response with [ and end with ]. Do not include any text before or after the JSON."
        )
        scored = []
        try:
            # Use your existing LLM call helper if you have one; else a minimal call:
            resp = self._get_ollama_response(prompt)  # your safe JSON extractor
            for item in resp:
                t = str(item.get("topic","")).strip()
                s = float(item.get("score", 0))
                if t:
                    scored.append((t, max(0.0, min(1.0, s))))
        except Exception:
            # Heuristic fallback: prefer shorter, high-curiosity tokens
            def heuristic(t):
                base = 0.5
                if any(k in t.lower() for k in ["mystery","unknown","secret","revealed","ancient","lost","why","how"]):
                    base += 0.2
                base += max(0, (40 - len(t))) / 100.0  # shorter titles slightly higher
                return min(1.0, base)
            scored = [(t, heuristic(t)) for t in topics]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def augment_seed_topics(self, niche: str, topics: list[str], want: int = 16) -> list[str]:
        """Parafraz + varyant üretir. LLM yoksa heuristikle üretir. Günlük cache vardır."""
        cached = _load_augment_cache(niche)
        if cached:
            return cached[:want]

        # LLM dene (kısa, hızlı yanıt)
        prompt = (
            "Rewrite each topic into 1 new catchy variant for YouTube titles.\n"
            "Keep meaning, change phrasing. Avoid clickbait.\n"
            "CRITICAL: Return ONLY valid JSON array of strings with the same order and length as input.\n"
            "No explanatory text before or after the JSON.\n"
            f"Niche: {niche}\nTopics: {topics[:want]}\n\n"
            "IMPORTANT: Start your response with [ and end with ]."
        )
        variants = []
        try:
            resp = self._get_ollama_response(prompt)  # your safe JSON extractor
            if isinstance(resp, list):
                variants = [str(x).strip() for x in resp if str(x).strip()]
        except Exception:
            variants = []

        # Heuristik fallback (güvenli ve hızlı)
        if not variants or len(variants) < max(8, want//2):
            suffixes = ["explained", "revealed", "in 5 minutes", "you should know", "that changed history",
                        "the untold story", "debunked", "guide", "timeline", "top facts"]
            out = []
            for t in topics:
                s = random.choice(suffixes)
                out.append(f"{t} — {s}")
                if len(out) >= want:
                    break
            variants = variants or out

        # dedupe + trim
        seen=set(); uniq=[]
        for v in variants:
            k=v.lower().strip()
            if k and k not in seen:
                seen.add(k); uniq.append(v.strip())
        uniq = uniq[:want]

        _save_augment_cache(niche, uniq)
        return uniq

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
            # Use the helper function for automatic niche resolution
            niche = niche_from_channel(channel_name)
            
            # Get trending topics with resilient fallback system
            trending_topics = self.get_topics_resilient(niche=niche)
            self.log_message(f"Selected topics for '{niche}': {len(trending_topics)}", "INFO")
            
            # Score and select top 8 topics
            topics_scored = self.score_topics_with_llm(niche, trending_topics, top_k=8)
            best_topics = [t for t,_ in topics_scored]
            self.logger.info(f"Selected top {len(best_topics)} topics: {best_topics}")
            
            trending_context = ""
            if best_topics:
                trending_context = f"Current trending topics in this niche: {', '.join(best_topics[:5])}. "

            prompt = f"""You are a master content strategist for viral YouTube documentaries. Generate {idea_count} viral video idea for a YouTube channel about '{niche}'.

{trending_context}

Focus on creating DEEP, ENGAGING content that can sustain 10+ minute videos. Each idea must include:
- A compelling mystery or untold story
- Multiple cliffhangers and suspense elements
- Engagement hooks to keep viewers engaged
- Global appeal for English-speaking audiences

CRITICAL: You must respond with ONLY valid JSON. No explanatory text before or after the JSON.

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

Make each idea highly specific and researchable. Focus on creating genuine curiosity and engagement.

IMPORTANT: Start your response with {{ and end with }}. Do not include any text before or after the JSON."""

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
        """Generate detailed script with exact niche match for visual prevention.

        Args:
            video_idea: Video idea dictionary
            channel_name: Name of the YouTube channel

        Returns:
            Generated script dictionary or None if generation fails
        """
        try:
            # Use the helper function for automatic niche resolution
            niche = niche_from_channel(channel_name)

            prompt = f"""You are a master scriptwriter for viral YouTube documentaries. Write a highly detailed, long-form script for a 15-20 minute video on: '{video_idea.get('title', 'N/A')}'.

CRITICAL REQUIREMENTS:
- Generate EXACTLY 60-80 sentences (no less, no more)
- Each sentence must be a complete, engaging thought
- Include multiple cliffhangers and suspense elements
- Create engagement hooks at specific time intervals
- Optimize Pexels queries for cinematic, high-quality visuals
- EXACT NICHE MATCH: Use precise, specific visual queries that match the exact niche '{niche}' to prevent irrelevant visuals

CRITICAL: You must respond with ONLY valid JSON. No explanatory text before or after the JSON.

REQUIRED JSON FORMAT:
{{
  "video_title": "{video_idea.get('title', 'N/A')}",
  "target_duration_minutes": 15-20,
  "script": [
    {{
      "sentence": "First sentence with rich narration",
      "visual_query": "cinematic 4K [exact {niche} scene] with dramatic lighting",
      "timing_seconds": 0,
      "engagement_hook": "Opening hook to grab attention"
    }},
    {{
      "sentence": "Second sentence building suspense",
      "visual_query": "cinematic 4K [exact {niche} scene] atmospheric mood",
      "timing_seconds": 8,
      "engagement_hook": "Building curiosity"
    }}
  ],
  "metadata": {{
    "subtitle_languages": ["English", "Spanish", "French", "German"],
    "target_audience": "Global English-speaking viewers",
    "engagement_strategy": "Multiple cliffhangers every 3-4 minutes",
    "visual_prevention": "Exact niche matching for '{niche}' to prevent irrelevant visuals"
  }}
}}

Focus on creating genuine suspense and curiosity. Each sentence should advance the story while maintaining viewer engagement. Use EXACT niche matching in visual queries.

IMPORTANT: Start your response with {{ and end with }}. Do not include any text before or after the JSON."""

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

    def generate_content(self, prompt: str) -> str:
        """Generate content using Ollama LLM.
        
        Args:
            prompt: The prompt to generate content from
            
        Returns:
            Generated content as string
        """
        try:
            result = self._get_ollama_response(prompt)
            if result and isinstance(result, dict):
                # Try to extract content from structured response
                if "content" in result:
                    return result["content"]
                elif "text" in result:
                    return result["text"]
                elif "response" in result:
                    return result["response"]
                else:
                    # Return the whole response as JSON string
                    return json.dumps(result, ensure_ascii=False, indent=2)
            elif result and isinstance(result, str):
                return result
            else:
                # Fallback: create a simple response
                return f"Generated content based on prompt: {prompt[:100]}..."
                
        except Exception as e:
            self.log_message(f"Error generating content: {e}", "ERROR")
            return f"Content generation failed: {str(e)}"


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

    # Test JSON extraction with problematic responses
    print("\nTesting JSON extraction improvements...")
    
    test_cases = [
        # Case 1: Response with prefix
        'Here is the script: {"title": "Test", "content": "Test content"}',
        
        # Case 2: Response with prefix and newlines
        '''Here is the script:

{"title": "Test", "content": "Test content"}''',
        
        # Case 3: Malformed JSON
        'Here is the script: {"title": "Test", content: "Test content"}',
        
        # Case 4: No JSON
        'Here is the script: This is just text without JSON',
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Input: {test_case[:50]}...")
        
        try:
            result = handler._extract_json_from_text(test_case)
            if result:
                print(f"✅ Successfully extracted: {result[:50]}...")
            else:
                print("❌ No JSON extracted")
        except Exception as e:
            print(f"❌ Error: {e}")

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
