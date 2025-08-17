# config.py - Enhanced Configuration with Environment Variables and Validation

import os
import json
import time
from typing import Dict, List, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    print("✅ .env file loaded successfully")
except Exception as e:
    print(f"⚠️ Failed to load .env: {e}")

# Get Pexels API key from environment or use fallback
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "SkEG6SqXRKE6OzoUVqlA...")
if PEXELS_API_KEY:
    print(f"✅ Pexels API key loaded: {PEXELS_API_KEY[:20]}...")
else:
    print("❌ Pexels API key not found")

# AI Configuration
AI_CONFIG = {
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "llama3:8b"),
}

# === Multi-Channel niche normalization & rich seeds ===

# Kanal adları / niş isimleri eşlemesi (case-insensitive)
NICHE_ALIASES = {
    "cklegends": "history",
    "ckironwill": "motivation",
    "ckfinancecore": "finance",
    "ckdrive": "automotive",
    "ckcombat": "combat",
}

def normalize_niche(niche_or_channel: str) -> str:
    key = (niche_or_channel or "").strip().lower()
    # önce direkt niş adı olabilir
    if key in {"history","motivation","finance","automotive","combat"}:
        return key
    # kanal adı olabilir
    return NICHE_ALIASES.get(key, "history")

# Tier1/Tier2 geo ve timeframe setleri (varsa mevcutlarınızla birleştirin/tekilleştirin)
TIER1_GEOS = ["US","GB","CA","AU","DE","FR","IT","ES","NL","SE","NO","DK","CH"]
TIER2_GEOS = ["BR","MX","IN","PL","TR","RU","ID","MY","TH","ZA"]
DEFAULT_TIMEFRAMES = ["now 7-d","today 1-m","today 3-m"]
MAX_TOPICS = 24

# Tüm kanallar için zengin seed listeleri (en az 24'er)
SEED_TOPICS = {
    "history": [
        "ancient civilizations","lost cities","roman empire","greek mythology",
        "egyptian pharaohs","archaeology discoveries","medieval knights",
        "viking history","mysteries of history","ancient inventions",
        "silk road","mesopotamia","maya civilization","pompeii","stonehenge",
        "alexander the great","genghis khan","byzantine empire","ottoman history","world war myths",
        "ancient engineering","forgotten languages","temples and ruins","artifact mysteries",
        "rosetta stone","hittite empire","indus valley"
    ],
    "motivation": [
        "discipline tips","productivity habits","mental toughness","morning routine",
        "goal setting","sports motivation","habit building","focus techniques",
        "mindset shift","success stories","growth mindset","overcoming procrastination",
        "dopamine detox","confidence building","resilience training","habit stacking",
        "time management","cold showers","gym motivation","study motivation",
        "deep work","atomic habits","self improvement","stoic principles",
        "consistency challenge","no excuses mindset"
    ],
    "finance": [
        "inflation explained","interest rates impact","recession signals","dividend investing",
        "index funds vs ETFs","value vs growth stocks","real estate vs stocks","emergency fund tips",
        "compound interest power","financial freedom steps","budgeting frameworks","credit score hacks",
        "side hustles 2025","ai stocks outlook","crypto regulation watch","gold vs bitcoin",
        "market bubble signs","earnings season guide","dollar cost averaging","tax optimization basics",
        "retirement planning 101","FIRE movement","risk management rules","portfolio rebalancing",
        "behavioral finance biases","hedge against inflation"
    ],
    "automotive": [
        "ev vs hybrid comparison","battery tech breakthroughs","solid state batteries","fast charging myths",
        "self driving levels","best sports cars 2025","affordable performance cars","car maintenance hacks",
        "engine types explained","turbo vs supercharger","aerodynamics basics","track day essentials",
        "car detailing secrets","resale value tips","car insurance tricks","winter driving tips",
        "top road trip cars","classic car legends","racing history moments","motorsport tech transfer",
        "ev charging etiquette","range anxiety fixes","home charger setup","hydrogen vs electric",
        "otonom sürüş güvenliği","infotainment comparisons"
    ],
    "combat": [
        "mma striking basics","wrestling takedown chains","bjj submissions explained","boxing footwork drills",
        "muay thai knees and elbows","counter punching theory","southpaw vs orthodox tactics","defense fundamentals",
        "conditioning for fighters","injury prevention tips","fight IQ examples","legendary comebacks",
        "greatest rivalries","weight cutting science","octagon control","ground and pound efficiency",
        "clinch fighting secrets","kick checking techniques","karate in mma","sambo influence on grappling",
        "daily training routine","fight camp nutrition","mental preparation","corner advice breakdown",
        "scoring criteria myths","judging controversies"
    ],
}

class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass

class ConfigValidator:
    """Configuration validation and bounds checking"""
    
    @staticmethod
    def validate_quality_standards(config: Dict[str, Any]) -> None:
        """Validate quality standards configuration"""
        # 0-1 range validation
        for key in ['minimum_quality_score', 'scene_variety_threshold', 'engagement_score_threshold']:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise ConfigError(f"{key} must be between 0 and 1, got {value}")
        
        # Positive number validation
        for key in ['minimum_duration_minutes', 'target_fps']:
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ConfigError(f"{key} must be positive, got {value}")
        
        # Reasonable maximum validation
        if 'target_fps' in config and config['target_fps'] > 120:
            raise ConfigError(f"target_fps {config['target_fps']} is unreasonably high")
        
        if 'minimum_duration_minutes' in config and config['minimum_duration_minutes'] > 60:
            raise ConfigError(f"minimum_duration_minutes {config['minimum_duration_minutes']} is unreasonably high")
    
    @staticmethod
    def validate_ai_config(config: Dict[str, Any]) -> None:
        """Validate AI configuration"""
        # Learning rate bounds
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr < 0 or lr > 1:
                raise ConfigError(f"learning_rate must be between 0 and 1, got {lr}")
        
        # Max iterations bounds
        if 'max_iterations' in config:
            max_iter = config['max_iterations']
            if not isinstance(max_iter, int) or max_iter < 1 or max_iter > 100:
                raise ConfigError(f"max_iterations must be between 1 and 100, got {max_iter}")
    
    @staticmethod
    def validate_self_update_config(config: Dict[str, Any]) -> None:
        """Validate self-update configuration"""
        # Allowed update frequencies
        allowed_frequencies = ['after_each_video', 'daily', 'weekly', 'monthly', 'never']
        if 'update_frequency' in config:
            freq = config['update_frequency']
            if freq not in allowed_frequencies:
                raise ConfigError(f"update_frequency must be one of {allowed_frequencies}, got {freq}")
    
    @staticmethod
    def validate_model_names(config: Dict[str, Any]) -> None:
        """Validate model names and patterns"""
        # Allowed Ollama models
        allowed_models = ['llama3:8b', 'llama3:70b', 'mistral:7b', 'codellama:7b', 'llama2:7b']
        if 'ollama_model' in config:
            model = config['ollama_model']
            if model not in allowed_models:
                raise ConfigError(f"ollama_model must be one of {allowed_models}, got {model}")

# Environment variable configuration with graceful degradation
def get_env_var(key: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with logging and graceful degradation"""
    value = os.getenv(key, default)
    
    if value is None and required:
        print(f"⚠️ Required environment variable {key} not found")
        return None
    elif value is None:
        print(f"ℹ️ Optional environment variable {key} not found, using default: {default}")
        return default
    else:
        print(f"✅ Environment variable {key} loaded successfully")
        return value

# API Keys from environment variables
ELEVENLABS_API_KEY = get_env_var("ELEVENLABS_API_KEY", None, required=False)
ELEVENLABS_VOICE_ID = get_env_var("ELEVENLABS_VOICE_ID", None, required=False)
OLLAMA_BASE_URL = get_env_var("OLLAMA_BASE_URL", "http://localhost:11434", required=False)

# Enhanced CHANNELS_CONFIG with environment variable support
CHANNELS_CONFIG = {
    "CKLegends": {
        "name": "CKLegends",
        "niche": "history",
        "niche_keywords": ["ancient mysteries", "historical discoveries", "archaeology", "mythology", "legends"],
        "pexels_api_key": PEXELS_API_KEY,
        "self_improvement": True,
        "self_update": True,
        "target_duration_minutes": 15,
        "style_preference": "cinematic",
        "narrator_style": "morgan_freeman",
        "music_style": "epic_historical",
        "visual_style": "ancient_civilizations",
        "engagement_strategy": "mystery_cliffhangers",
        "subtitle_languages": ["English", "Spanish", "French", "German"],
        "quality_threshold": 0.8
    },
    "CKIronWill": {
        "name": "CKIronWill",
        "niche": "motivation",
        "niche_keywords": ["motivation", "willpower", "personal development", "success stories", "inspiration"],
        "pexels_api_key": PEXELS_API_KEY,
        "self_improvement": True,
        "self_update": True,
        "target_duration_minutes": 12,
        "style_preference": "inspirational",
        "narrator_style": "tony_robbins",
        "music_style": "uplifting_motivational",
        "visual_style": "achievement_success",
        "engagement_strategy": "emotional_peaks",
        "subtitle_languages": ["English", "Spanish", "French", "German"],
        "quality_threshold": 0.85
    },
    "CKFinanceCore": {
        "name": "CKFinanceCore",
        "niche": "finance",
        "niche_keywords": ["finance", "investing", "money", "economics", "trading", "wealth"],
        "pexels_api_key": PEXELS_API_KEY,
        "self_improvement": True,
        "self_update": True,
        "target_duration_minutes": 15,
        "style_preference": "professional",
        "narrator_style": "warren_buffett",
        "music_style": "corporate_ambient",
        "visual_style": "modern_finance",
        "engagement_strategy": "data_insights",
        "subtitle_languages": ["English", "Spanish", "French", "German"],
        "quality_threshold": 0.8
    },
    "CKDrive": {
        "name": "CKDrive",
        "niche": "automotive",
        "niche_keywords": ["cars", "automotive", "vehicles", "driving", "motorsport", "engineering"],
        "pexels_api_key": PEXELS_API_KEY,
        "self_improvement": True,
        "self_update": True,
        "target_duration_minutes": 12,
        "style_preference": "dynamic",
        "narrator_style": "jeremy_clarkson",
        "music_style": "high_energy",
        "visual_style": "automotive_glamour",
        "engagement_strategy": "speed_thrills",
        "subtitle_languages": ["English", "Spanish", "French", "German"],
        "quality_threshold": 0.8
    },
    "CKCombat": {
        "name": "CKCombat",
        "niche": "combat",
        "niche_keywords": ["martial arts", "fighting", "combat", "self defense", "sports", "training"],
        "pexels_api_key": PEXELS_API_KEY,
        "self_improvement": True,
        "self_update": True,
        "target_duration_minutes": 10,
        "style_preference": "intense",
        "narrator_style": "bruce_lee",
        "music_style": "epic_action",
        "visual_style": "combat_dynamic",
        "engagement_strategy": "adrenaline_rush",
        "subtitle_languages": ["English", "Spanish", "French", "German"],
        "quality_threshold": 0.8
    }
}

# Normalize channel keys (trim): avoid accidental whitespace/case drifts
CHANNELS_CONFIG = {k.strip(): v for k, v in CHANNELS_CONFIG.items()}

# AI Configuration with enhanced validation
AI_CONFIG = {
    "ollama_model": get_env_var("OLLAMA_MODEL", "llama3:8b", required=False),
    "ollama_base_url": OLLAMA_BASE_URL,
    "self_improvement_enabled": True,
    "code_generation_enabled": True,
    "config_update_enabled": True,
    "quality_analysis_enabled": True,
    "learning_rate": 0.1,
    "max_iterations": 5,
    "improvement_threshold": 0.1,
    # Enhanced AI module configuration
    "ai_modules": {
        "cinematic_director": {
            "enabled": True,
            "priority": "high",
            "fallback_enabled": True
        },
        "voice_acting": {
            "enabled": True,
            "priority": "high",
            "fallback_enabled": True
        },
        "visual_suite": {
            "enabled": True,
            "priority": "medium",
            "fallback_enabled": True
        },
        "audio_suite": {
            "enabled": True,
            "priority": "medium",
            "fallback_enabled": True
        },
        "content_suite": {
            "enabled": True,
            "priority": "high",
            "fallback_enabled": True
        },
        "video_suite": {
            "enabled": True,
            "priority": "high",
            "fallback_enabled": True
        },
        "analytics_suite": {
            "enabled": True,
            "priority": "low",
            "fallback_enabled": True
        },
        "realtime_director": {
            "enabled": True,
            "priority": "low",
            "fallback_enabled": True
        },
        "master_suite": {
            "enabled": True,
            "priority": "highest",
            "fallback_enabled": True,
            "premium_features": True
        }
    },
    # AI pipeline configuration
    "pipeline": {
        "max_parallel_channels": 3,
        "quality_threshold": 0.8,
        "retry_attempts": 2,
        "fallback_strategy": "graceful_degradation"
    }
}

# Quality Standards with validation
QUALITY_STANDARDS = {
    "minimum_duration_minutes": 10,
    "target_fps": 30,
    "target_resolution": "1920x1080",
    "target_codec": "libx264",
    "audio_codec": "aac",
    "minimum_quality_score": 0.7,
    "scene_variety_threshold": 0.6,
    "engagement_score_threshold": 0.75
}

# Pexels Configuration with graceful degradation
PEXELS_CONFIG = {
    "api_key": PEXELS_API_KEY,
    "base_url": "https://api.pexels.com",
    "search_endpoint": "/videos/search",
    "default_params": {
        "per_page": 1,
        "orientation": "landscape",
        "size": "large",
        "quality": "high"
    },
    "rate_limit": {
        "requests_per_hour": 200,
        "requests_per_day": 5000
    },
    "enabled": True
}

# Self-Update Configuration with validation
SELF_UPDATE_CONFIG = {
    "enabled": True,
    "update_frequency": "after_each_video",
    "learning_metrics": [
        "video_quality_score",
        "engagement_metrics",
        "scene_variety_score",
        "audio_quality_score",
        "visual_quality_score"
    ],
    "improvement_strategies": [
        "enhance_visual_effects",
        "optimize_audio_processing",
        "improve_scene_transitions",
        "enhance_narration_style",
        "optimize_music_selection"
    ]
}

# GPU Configuration for Local Optimization
GPU_CONFIG = {
    "enabled": True,
    "device_id": 0,
    "memory_fraction": 0.8,
    "compute_capability": "7.5",  # MX450 supports CUDA 7.5
    "auto_memory_management": True,
    "mixed_precision": True,  # FP16 for better performance
}

# Hardware Optimization Settings
HARDWARE_OPTIMIZATION = {
    "max_threads": 8,  # CPU thread optimization
    "use_gpu_encoding": True,  # NVIDIA GPU encoding
    "use_gpu_decoding": True,  # NVIDIA GPU decoding
    "memory_optimization": True,
    "cache_optimization": True,
}

# Storage Optimization for Local Usage
STORAGE_OPTIMIZATION = {
    "use_ramdisk": True,           # Use RAM disk for temporary files
    "temp_dir": "/tmp/chimera",    # Temporary directory
    "cache_size": "4GB",           # Cache size limit
    "auto_cleanup": True,          # Automatic cleanup
    "deduplication": True,         # File deduplication
    "compression": True,           # File compression
    "batch_operations": True,      # Batch file operations
    "parallel_io": True,           # Parallel I/O operations
}

# File Management Settings
FILE_MANAGEMENT = {
    "max_temp_files": 1000,        # Maximum temporary files
    "cleanup_interval": 3600,      # Cleanup interval (seconds)
    "file_retention": 86400,       # File retention (seconds)
    "compression_ratio": 0.7,      # Target compression ratio
    "dedupe_threshold": 0.95,      # Deduplication similarity threshold
}

# Memory Management Configuration
MEMORY_CONFIG = {
    "gpu_memory_limit_gb": 2.0,           # GPU memory limit in GB
    "cpu_fallback_threshold": 1.0,        # Use CPU if GPU memory below this threshold
    "sequential_initialization": True,     # Initialize modules sequentially
    "memory_cleanup_interval": 5,         # Cleanup memory every N operations
    "max_concurrent_models": 2,           # Maximum concurrent AI models
    "auto_memory_management": True,       # Automatic memory management
    "force_cpu_mode": False,              # Force CPU mode for all operations
    "memory_monitoring": True,            # Enable memory monitoring
    "cleanup_on_error": True,             # Cleanup memory on errors
}

# AI Model Configuration with Memory Management
AI_MODEL_CONFIG = {
    "sentiment_analysis": {
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "fallback_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "device_preference": "auto",      # auto, gpu, cpu
        "memory_efficient": True,
        "batch_size": 1
    },
    "text_generation": {
        "model": "gpt2",
        "device_preference": "auto",
        "memory_efficient": True,
        "max_length": 512
    },
    "image_processing": {
        "device_preference": "auto",
        "memory_efficient": True,
        "max_image_size": (1024, 1024)
    }
}

# Validate configurations
try:
    ConfigValidator.validate_quality_standards(QUALITY_STANDARDS)
    ConfigValidator.validate_ai_config(AI_CONFIG)
    ConfigValidator.validate_self_update_config(SELF_UPDATE_CONFIG)
    ConfigValidator.validate_model_names(AI_CONFIG)
    print("✅ All configuration validations passed")
except ConfigError as e:
    print(f"❌ Configuration validation failed: {e}")
    # Use safe defaults
    QUALITY_STANDARDS["minimum_quality_score"] = 0.5
    QUALITY_STANDARDS["target_fps"] = 25
    AI_CONFIG["ollama_model"] = "llama3:8b"
    print("⚠️ Using safe default values")

# Graceful degradation for missing API keys
if not PEXELS_API_KEY:
    print("⚠️ PEXELS_API_KEY not found - Pexels features will be disabled")
    PEXELS_CONFIG["enabled"] = False

if not ELEVENLABS_API_KEY:
    print("⚠️ ELEVENLABS_API_KEY not found - ElevenLabs features will be disabled")

# Ollama URL status (only show once)
if not OLLAMA_BASE_URL or OLLAMA_BASE_URL == "http://localhost:11434":
    # Only show this message once during startup
    pass

print(f"✅ Configuration loaded successfully")
print(f"   Pexels enabled: {PEXELS_CONFIG['enabled']}")
print(f"   ElevenLabs enabled: {ELEVENLABS_API_KEY is not None}")
print(f"   Ollama model: {AI_CONFIG['ollama_model']}")

# Export configurations
__all__ = [
    'CHANNELS_CONFIG',
    'AI_CONFIG',
    'QUALITY_STANDARDS',
    'PEXELS_CONFIG',
    'SELF_UPDATE_CONFIG',
    'ConfigError',
    'ConfigValidator',
    'NICHE_ALIASES',
    'normalize_niche',
    'TIER1_GEOS',
    'TIER2_GEOS',
    'DEFAULT_TIMEFRAMES',
    'MAX_TOPICS',
    'SEED_TOPICS'
]

class StorageManager:
    """Advanced storage management for local optimization"""
    
    def __init__(self):
        self.temp_dir = STORAGE_OPTIMIZATION["temp_dir"]
        self.cache_size = self._parse_size(STORAGE_OPTIMIZATION["cache_size"])
        self.auto_cleanup = STORAGE_OPTIMIZATION["auto_cleanup"]
        self.deduplication = STORAGE_OPTIMIZATION["deduplication"]
        self.compression = STORAGE_OPTIMIZATION["compression"]
        
        # Initialize storage
        self._setup_storage()
        self._start_cleanup_thread()
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024**3)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024**2)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        else:
            return int(size_str)
    
    def _setup_storage(self):
        """Setup optimized storage structure"""
        try:
            import os
            import tempfile
            
            # Create temp directory
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Set optimal permissions
            os.chmod(self.temp_dir, 0o755)
            
            print(f"✅ Storage initialized: {self.temp_dir}")
            
        except Exception as e:
            print(f"⚠️ Storage setup failed: {e}")
    
    def _start_cleanup_thread(self):
        """Start automatic cleanup thread"""
        if self.auto_cleanup:
            import threading
            import time
            
            def cleanup_worker():
                while True:
                    try:
                        self._cleanup_temp_files()
                        time.sleep(FILE_MANAGEMENT["cleanup_interval"])
                    except Exception as e:
                        print(f"⚠️ Cleanup worker error: {e}")
                        time.sleep(60)  # Wait 1 minute on error
            
            cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
            cleanup_thread.start()
            print("✅ Auto-cleanup thread started")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import os
            import time
            
            current_time = time.time()
            retention_time = FILE_MANAGEMENT["file_retention"]
            
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > retention_time:
                        os.remove(filepath)
                        print(f"🧹 Cleaned up old file: {filename}")
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
    
    def get_temp_path(self, filename: str) -> str:
        """Get optimized temporary file path"""
        return os.path.join(self.temp_dir, filename)
    
    def optimize_file_operations(self, file_paths: list):
        """Optimize batch file operations"""
        if STORAGE_OPTIMIZATION["batch_operations"]:
            # Implement batch optimization
            pass

# Initialize storage manager
storage_manager = StorageManager()
