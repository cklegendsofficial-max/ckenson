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
import warnings, time, random
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
    """Enhanced LLM handler with robust JSON extraction and network resilience."""

    def __init__(self, model: Optional[str] = None, max_retries: int = 3) -> None:
        """Initialize the LLM handler.

        Args:
            model: Ollama model to use, defaults to config value
            max_retries: Maximum retry attempts for network operations
        """
        self.model = model or AI_CONFIG.get("ollama_model", "llama3:8b")
        self.max_retries = max_retries

        # Initialize PyTrends with timeout
        self.pytrends = _get_trend_client()
        if self.pytrends is None:
            logging.warning("PyTrends unavailable; using cached/fallback keywords.")

        self.setup_logging()

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
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt_template}],
            )

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
        # Import pandas and configure to suppress silent downcasting warnings
        import pandas as pd
        pd.set_option('future.no_silent_downcasting', True)
        
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
            "Return JSON array of objects: [{\"topic\": str, \"score\": float}]\n"
            f"Topics: {topics[:24]}"
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
            "Return ONLY JSON array of strings with the same order and length as input.\n"
            f"Niche: {niche}\nTopics: {topics[:want]}"
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

Focus on creating genuine suspense and curiosity. Each sentence should advance the story while maintaining viewer engagement. Use EXACT niche matching in visual queries."""

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
