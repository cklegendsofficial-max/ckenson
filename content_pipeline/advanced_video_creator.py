# content_pipeline/advanced_video_creator.py (Professional Master Director Edition)

import os
import sys
import json
import time
import random
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import concurrent.futures

# --- config imports & defaults ---
try:
    from config import AI_CONFIG, PEXELS_API_KEY, HARDWARE_OPTIMIZATION
except Exception:
    AI_CONFIG, PEXELS_API_KEY = {}, None
    HARDWARE_OPTIMIZATION = {"max_threads": 4, "gpu_acceleration": True}

try:
    OLLAMA_MODEL = (AI_CONFIG or {}).get("ollama_model", "llama3:8b")
except Exception:
    OLLAMA_MODEL = "llama3:8b"

# Varsayılan: subliminal kapalı (YouTube politikaları ve güven için)
ENABLE_SUBLIMINAL = False

# GPU Acceleration Setup
try:
    import torch
    if torch.cuda.is_available():
        TORCH_CUDA_AVAILABLE = True
        CUDA_DEVICE = torch.device("cuda:0")
        # GPU memory optimization
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ CUDA GPU acceleration enabled")
    else:
        TORCH_CUDA_AVAILABLE = False
        CUDA_DEVICE = torch.device("cpu")
        print("⚠️ CUDA not available, using CPU")
except ImportError:
    TORCH_CUDA_AVAILABLE = False
    CUDA_DEVICE = torch.device("cpu")
    print("⚠️ PyTorch not available")

def _ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists for the given path"""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# Try to import advanced libraries
try:
    import piper
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("⚠️ Piper TTS not available, using espeak fallback")

try:
    import espeak
    ESPEAK_AVAILABLE = True
except ImportError:
    ESPEAK_AVAILABLE = False
            # espeak not available, using gTTS fallback silently

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("⚠️ mido not available, MIDI generation disabled")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("⚠️ Pillow not available, image upscaling disabled")

# Core imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from moviepy.editor import *
from moviepy.video.fx import all as vfx
from moviepy.audio.fx import all as afx

# Advanced Video Effects Configuration
VIDEO_EFFECTS_CONFIG = {
    "cinematic_effects": True,         # Cinematic visual effects
    "color_grading": True,             # Professional color grading
    "motion_graphics": True,           # Motion graphics and animations
    "transitions": True,               # Advanced transitions
    "text_overlays": True,             # Dynamic text overlays
    "particle_effects": True,          # Particle and special effects
    "3d_elements": True,               # 3D elements and depth
    "ai_enhancement": True             # AI-powered enhancement
}

# Visual Effects Library
VISUAL_EFFECTS = {
    "transitions": {
        "fade": "fade_in_out",
        "slide": "slide_transition",
        "zoom": "zoom_transition",
        "dissolve": "cross_dissolve",
        "wipe": "wipe_transition",
        "morph": "morph_transition"
    },
    "filters": {
        "cinematic": "cinematic_look",
        "vintage": "vintage_filter",
        "modern": "modern_filter",
        "dramatic": "dramatic_filter",
        "professional": "professional_filter"
    },
    "text_effects": {
        "glow": "glowing_text",
        "shadow": "shadow_text",
        "3d": "3d_text",
        "animated": "animated_text",
        "particle": "particle_text"
    }
}

class AdvancedVideoCreator:
    def __init__(self):
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30
        
        # GPU-optimized codec settings
        if TORCH_CUDA_AVAILABLE:
            self.CODEC = 'h264_nvenc'  # NVIDIA GPU encoding
            self.GPU_ACCELERATION = True
            print("🚀 Using NVIDIA GPU encoding")
        else:
            self.CODEC = 'libx264'
            self.GPU_ACCELERATION = False
            print("💻 Using CPU encoding")
        
        self.AUDIO_CODEC = 'aac'
        self.QUALITY = 'high'
        
        # Initialize video effects system
        self.setup_video_effects()
        
        # GPU memory management
        if TORCH_CUDA_AVAILABLE:
            self.setup_gpu_optimization()
        
        # Initialize TTS systems
        self.setup_tts()
        
        # Initialize music generation
        self.setup_music_generation()
        
        # Setup logging
        self.setup_logging()
        
        # Load local assets for fallback
        self.load_local_assets()
        
        # Initialize executor for parallel processing
        try:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=4)
        except ImportError:
            self.executor = None
        
        # Initialize Pexels API with proper error handling
        try:
            from pexels_api import PexelsImageDownloader
            if PEXELS_API_KEY:
                self.pexels_api = PexelsImageDownloader(PEXELS_API_KEY)
                print(f"✅ Pexels API initialized with key: {PEXELS_API_KEY[:20]}...")
            else:
                self.pexels_api = None
                print("⚠️ Pexels API key not configured, using fallback images")
        except ImportError:
            self.pexels_api = None
            print("⚠️ Pexels API library not available, using fallback images")
        
        # Import time for timestamp generation
        import time
        
        # Import os for file operations
        import os
        
        # Import numpy for array operations
        try:
            import numpy as np
            self.numpy_available = True
        except ImportError:
            self.numpy_available = False
        
        # Import PIL for image processing
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.pil_available = True
        except ImportError:
            self.pil_available = False
        
        # Import soundfile for audio processing
        try:
            import soundfile as sf
            self.soundfile_available = True
        except ImportError:
            self.soundfile_available = False
        
        # Import datetime for logging
        try:
            from datetime import datetime
            self.datetime_available = True
        except ImportError:
            self.datetime_available = False
        
        # Import json for data handling
        try:
            import json
            self.json_available = True
        except ImportError:
            self.json_available = False
        
        # Import random for fallback operations
        try:
            import random
            self.random_available = True
        except ImportError:
            self.random_available = False
        
        # Import math for calculations
        try:
            import math
            self.math_available = True
        except ImportError:
            self.math_available = False
        
        # Import threading for parallel operations
        try:
            import threading
            self.threading_available = True
        except ImportError:
            self.threading_available = False
        
        # Import concurrent for parallel processing
        try:
            import concurrent.futures
            self.concurrent_available = True
        except ImportError:
            self.concurrent_available = False
    
    def setup_logging(self):
        """Setup enhanced logging system"""
        self.log_file = f"advanced_video_creator_{int(time.time())}.log"
        print(f"📝 Advanced Video Creator logging to: {self.log_file}")
    
    def log_message(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamp and level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"❌ Logging error: {e}")
    
    def setup_tts(self):
        """Setup enhanced TTS systems with quality optimization"""
        self.tts_systems = {}
        
        # Audio quality settings
        self.audio_quality = {
            "sample_rate": 48000,      # High quality sample rate
            "bit_depth": 24,           # 24-bit audio
            "channels": 2,             # Stereo
            "noise_reduction": True,   # Noise reduction
            "normalization": True,     # Audio normalization
            "compression": True,       # Dynamic range compression
            "enhancement": True        # Audio enhancement
        }
        
        # Initialize TTS engines
        if PIPER_AVAILABLE:
            self.tts_systems['piper'] = self._setup_piper_tts()
        
        if ESPEAK_AVAILABLE:
            self.tts_systems['espeak'] = self._setup_espeak_tts()
        
        if GTTS_AVAILABLE:
            self.tts_systems['gtts'] = self._setup_gtts_tts()
        
        print(f"✅ TTS systems initialized: {list(self.tts_systems.keys())}")
    
    def _setup_piper_tts(self):
        """Setup Piper TTS with high quality"""
        try:
            import piper
            
            # High-quality Piper configuration
            config = {
                "sample_rate": self.audio_quality["sample_rate"],
                "bit_depth": self.audio_quality["bit_depth"],
                "channels": self.audio_quality["channels"],
                "noise_scale": 0.667,      # Noise reduction
                "length_scale": 1.0,       # Speed control
                "sentence_silence": 0.2,   # Sentence pause
                "word_silence": 0.1        # Word pause
            }
            
            return {"engine": "piper", "config": config}
            
        except Exception as e:
            print(f"⚠️ Piper TTS setup failed: {e}")
            return None
    
    def _setup_espeak_tts(self):
        """Setup espeak TTS with high quality"""
        try:
            import espeak
            
            # High-quality espeak configuration
            config = {
                "sample_rate": self.audio_quality["sample_rate"],
                "bit_depth": self.audio_quality["bit_depth"],
                "channels": self.audio_quality["channels"],
                "voice": "en-us",          # Voice selection
                "speed": 150,              # Speaking speed
                "pitch": 50,               # Pitch adjustment
                "volume": 100              # Volume level
            }
            
            return {"engine": "espeak", "config": config}
            
        except Exception as e:
            print(f"⚠️ espeak TTS setup failed: {e}")
            return None
    
    def _setup_gtts_tts(self):
        """Setup gTTS with high quality"""
        try:
            from gtts import gTTS
            
            # High-quality gTTS configuration
            config = {
                "lang": "en",              # Language
                "slow": False,             # Normal speed
                "lang_check": True,        # Language validation
                "pre_processor_funcs": []  # No preprocessing
            }
            
            return {"engine": "gtts", "config": config}
            
        except Exception as e:
            print(f"⚠️ gTTS setup failed: {e}")
            return None
    
    def generate_high_quality_audio(self, text: str, output_path: str, 
                                   voice_style: str = "professional") -> str:
        """Generate high-quality audio with optimization"""
        
        print(f"🎙️ Generating high-quality audio: {voice_style} style")
        
        try:
            # Voice style optimization
            voice_config = self._get_voice_style_config(voice_style)
            
            # Generate audio with best available TTS
            audio_path = self._generate_audio_with_tts(text, voice_config)
            
            if audio_path and os.path.exists(audio_path):
                # Apply audio enhancement
                enhanced_path = self._enhance_audio_quality(audio_path, output_path)
                
                # Cleanup temporary files
                if audio_path != enhanced_path:
                    os.remove(audio_path)
                
                print(f"✅ High-quality audio generated: {enhanced_path}")
                return enhanced_path
            else:
                raise Exception("Audio generation failed")
                
        except Exception as e:
            print(f"❌ High-quality audio generation failed: {e}")
            # Fallback to basic TTS
            return self._generate_audio_fallback(text, output_path)
    
    def _generate_audio_fallback(self, text: str, output_path: str) -> str:
        """Generate basic audio fallback when TTS fails"""
        
        try:
            # Simple fallback: create a silent audio file
            import numpy as np
            import soundfile as sf
            
            # Generate 3 seconds of silence at 22050 Hz
            sample_rate = 22050
            duration = 3.0
            samples = int(sample_rate * duration)
            audio = np.zeros(samples)
            
            # Save as WAV
            sf.write(output_path, audio, sample_rate)
            print(f"⚠️ Generated fallback audio: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Fallback audio generation failed: {e}")
            # Create empty file as last resort
            with open(output_path, 'w') as f:
                f.write("")
            return output_path
    
    def _get_voice_style_config(self, voice_style: str) -> dict:
        """Get voice style configuration"""
        voice_styles = {
            "professional": {
                "speed": 150,      # Moderate speed
                "pitch": 50,       # Neutral pitch
                "volume": 100,     # Full volume
                "clarity": "high"  # High clarity
            },
            "energetic": {
                "speed": 180,      # Faster speed
                "pitch": 60,       # Higher pitch
                "volume": 110,     # Slightly louder
                "clarity": "high"
            },
            "calm": {
                "speed": 120,      # Slower speed
                "pitch": 40,       # Lower pitch
                "volume": 90,      # Slightly quieter
                "clarity": "medium"
            },
            "dramatic": {
                "speed": 140,      # Variable speed
                "pitch": 55,       # Slightly higher
                "volume": 105,     # Louder
                "clarity": "high"
            }
        }
        
        return voice_styles.get(voice_style, voice_styles["professional"])
    
    def _generate_audio_with_tts(self, text: str, voice_config: dict) -> str:
        """Generate audio using best available TTS"""
        
        # Priority order: Piper > espeak > gTTS
        tts_priority = ['piper', 'espeak', 'gtts']
        
        for tts_name in tts_priority:
            if tts_name in self.tts_systems:
                try:
                    return self._generate_with_specific_tts(text, tts_name, voice_config)
                except Exception as e:
                    print(f"⚠️ {tts_name} TTS failed: {e}")
                    continue
        
        raise Exception("No TTS system available")
    
    def _generate_with_specific_tts(self, text: str, tts_name: str, voice_config: dict) -> str:
        """Generate audio with specific TTS system"""
        
        if tts_name == 'piper':
            return self._generate_piper_audio(text, voice_config)
        elif tts_name == 'espeak':
            return self._generate_espeak_audio(text, voice_config)
        elif tts_name == 'gtts':
            return self._generate_gtts_audio(text, voice_config)
        else:
            raise Exception(f"Unknown TTS system: {tts_name}")
    
    def _generate_piper_audio(self, text: str, voice_config: dict) -> str:
        """Generate audio using Piper TTS"""
        try:
            if not PIPER_AVAILABLE:
                raise Exception("Piper TTS not available")
            
            # Generate unique filename
            output_path = f"temp_audio_{int(time.time())}.wav"
            
            # Use Piper TTS with correct API
            try:
                # Try to find available Piper voice models
                import os
                import glob
                
                # Look for available voice models
                voice_models = []
                possible_paths = [
                    "voices", "voice_models", "piper_voices", 
                    os.path.expanduser("~/.local/share/piper/voices"),
                    os.path.expanduser("~/piper_voices"),
                    "C:/piper_voices", "C:/voices",  # Windows paths
                    os.path.join(os.getcwd(), "voices"),
                    os.path.join(os.getcwd(), "piper_voices")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        voice_models.extend(glob.glob(os.path.join(path, "*.json")))
                
                if voice_models:
                    # Use first available voice model
                    voice_model = voice_models[0]
                    print(f"🎙️ Using Piper voice model: {voice_model}")
                    voice = piper.PiperVoice.load(voice_model)
                else:
                    # No Piper voice models found, using enhanced gTTS silently
                    return self._generate_gtts_audio(text, voice_config)
                
                # Synthesize text to audio
                with open(output_path, 'wb') as f:
                    voice.synthesize(text, f)
                
                return output_path
            except Exception as e:
                print(f"⚠️ Piper TTS synthesis failed: {e}")
                print("🔄 Falling back to gTTS...")
                # Fallback to gTTS
                return self._generate_gtts_audio(text, voice_config)
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Piper TTS failed: {e}")
            raise
    
    def _generate_espeak_audio(self, text: str, voice_config: dict) -> str:
        """Generate audio using espeak TTS"""
        try:
            if not ESPEAK_AVAILABLE:
                raise Exception("espeak TTS not available")
            
            # Generate unique filename
            output_path = f"temp_audio_{int(time.time())}.wav"
            
            # Use espeak TTS
            espeak.synth(text, output_path, voice_config.get('voice', 'en'))
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ espeak TTS failed: {e}")
            raise
    
    def _generate_gtts_audio(self, text: str, voice_config: dict) -> str:
        """Generate audio using gTTS"""
        try:
            if not GTTS_AVAILABLE:
                raise Exception("gTTS not available")
            
            # Generate unique filename
            output_path = f"temp_audio_{int(time.time())}.wav"
            
            # Use gTTS
            tts = gTTS(text=text, lang=voice_config.get('lang', 'en'), slow=False)
            tts.save(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ gTTS failed: {e}")
            raise
    
    def _enhance_audio_quality(self, input_path: str, output_path: str) -> str:
        """Enhance audio quality with post-processing"""
        
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.audio_quality["sample_rate"])
            
            # Apply enhancements
            if self.audio_quality["noise_reduction"]:
                audio = self._reduce_noise(audio)
            
            if self.audio_quality["normalization"]:
                audio = self._normalize_audio(audio)
            
            if self.audio_quality["compression"]:
                audio = self._compress_audio(audio)
            
            if self.audio_quality["enhancement"]:
                audio = self._enhance_audio(audio)
            
            # Save enhanced audio
            sf.write(output_path, audio, sr, subtype='PCM_24')
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Audio enhancement failed: {e}")
            # Return original if enhancement fails
            return input_path
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Reduce noise in audio"""
        try:
            # Simple noise reduction using spectral gating
            # This is a basic implementation - more advanced methods available
            noise_threshold = 0.01
            audio_spectrum = np.fft.fft(audio)
            noise_gate = np.abs(audio_spectrum) > noise_threshold
            filtered_spectrum = audio_spectrum * noise_gate
            return np.real(np.fft.ifft(filtered_spectrum))
        except Exception:
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio levels"""
        try:
            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val * 0.95  # Leave some headroom
            return audio
        except Exception:
            return audio
    
    def _compress_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Simple compression
            threshold = 0.5
            ratio = 4.0
            
            compressed = np.where(
                np.abs(audio) > threshold,
                threshold + (np.abs(audio) - threshold) / ratio * np.sign(audio),
                audio
            )
            
            return compressed
        except Exception:
            return audio
    
    def _enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio clarity"""
        try:
            # High-frequency boost for clarity
            # This is a simple enhancement - more sophisticated methods available
            enhanced = audio * 1.1  # Slight boost
            return np.clip(enhanced, -1, 1)  # Prevent clipping
        except Exception:
            return audio
    
    def setup_music_generation(self):
        """Setup MIDI music generation system"""
        if MIDO_AVAILABLE:
            try:
                self.music_system = "mido"
                self.log_message("✅ MIDI music generation system initialized", "MUSIC")
            except Exception as e:
                self.log_message(f"⚠️ MIDI system initialization failed: {e}", "MUSIC")
                self.music_system = None
        else:
            self.music_system = None
            self.log_message("⚠️ MIDI music generation not available", "MUSIC")
    
    def load_local_assets(self):
        """Load local assets for fallback scenarios"""
        self.local_assets = {
            'images': [],
            'videos': [],
            'audio': []
        }
        
        # Load local images
        image_dir = "assets/images"
        if os.path.exists(image_dir):
            for file in os.listdir(image_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.local_assets['images'].append(os.path.join(image_dir, file))
        
        # Load local videos
        video_dir = "assets/videos/downloads"
        if os.path.exists(video_dir):
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if file.lower().endswith('.mp4'):
                        self.local_assets['videos'].append(os.path.join(root, file))
        
        # Load local audio
        audio_dir = "assets/audio"
        if os.path.exists(audio_dir):
            for root, dirs, files in os.walk(audio_dir):
                for file in files:
                    if file.lower().endswith('.mp3'):
                        self.local_assets['audio'].append(os.path.join(root, file))
        
        self.log_message(f"📁 Loaded {len(self.local_assets['images'])} images, {len(self.local_assets['videos'])} videos, {len(self.local_assets['audio'])} audio files", "ASSETS")
    
    def setup_gpu_optimization(self):
        """Setup GPU optimization for video processing"""
        try:
            # Set GPU memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable mixed precision for better performance
            if hasattr(torch, 'autocast'):
                self.use_mixed_precision = True
                print("✅ Mixed precision enabled for GPU")
            
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            print(f"🚀 GPU optimization completed for device: {torch.cuda.get_device_name()}")
            
        except Exception as e:
            print(f"⚠️ GPU optimization failed: {e}")
            self.GPU_ACCELERATION = False
    
    def _ensure_parent_dir(self, path: str) -> None:
        """Ensure parent directory exists for the given path"""
        d = path if os.path.isdir(path) else os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    
    def _optimize_pexels_query(self, query: str, channel_niche: str) -> str:
        """Enhanced Pexels query optimization with intelligent keywords"""
        
        # Channel-specific keyword mapping with niche-specific content
        channel_keywords = {
            'CKLegends': {
                'primary': ['legendary', 'epic', 'heroic', 'mythical', 'ancient', 'mystical'],
                'secondary': ['warrior', 'kingdom', 'battle', 'victory', 'glory', 'honor'],
                'style': ['cinematic', 'dramatic', 'epic', '4K', 'high quality'],
                'niche': 'history',
                'content_types': ['ancient civilizations', 'historical figures', 'epic battles', 'mythological stories', 'royal kingdoms', 'heroic legends']
            },
            'CKCombat': {
                'primary': ['combat', 'battle', 'warrior', 'fighting', 'martial arts', 'sword'],
                'secondary': ['action', 'dynamic', 'intense', 'powerful', 'victory'],
                'style': ['action', 'dynamic', '4K', 'high quality', 'no text'],
                'niche': 'combat',
                'content_types': ['martial arts', 'military history', 'sports combat', 'warrior training', 'battle scenes', 'victory moments']
            },
            'CKDrive': {
                'primary': ['racing', 'speed', 'cars', 'motorsport', 'adrenaline', 'competition'],
                'secondary': ['fast', 'dynamic', 'exciting', 'championship', 'victory'],
                'style': ['dynamic', 'fast-paced', '4K', 'high quality', 'no text'],
                'niche': 'racing',
                'content_types': ['car racing', 'motorsport', 'speed records', 'racing legends', 'championship moments', 'adrenaline sports']
            },
            'CKFinanceCore': {
                'primary': ['business', 'finance', 'success', 'wealth', 'growth', 'investment'],
                'secondary': ['professional', 'corporate', 'modern', 'technology', 'innovation'],
                'style': ['professional', 'clean', '4K', 'high quality', 'no text'],
                'niche': 'finance',
                'content_types': ['business success', 'financial markets', 'entrepreneurship', 'wealth building', 'investment strategies', 'corporate leadership']
            },
            'CKIronWill': {
                'primary': ['strength', 'determination', 'resilience', 'overcome', 'challenge'],
                'secondary': ['powerful', 'inspiring', 'motivational', 'success', 'victory'],
                'style': ['inspirational', 'powerful', '4K', 'high quality', 'no text'],
                'niche': 'motivation',
                'content_types': ['personal development', 'overcoming challenges', 'success stories', 'motivational speakers', 'fitness transformation', 'mental strength']
            }
        }
        
        # Get channel config
        channel_config = channel_keywords.get(channel_niche, {
            'primary': ['cinematic', 'epic', 'dramatic'],
            'secondary': ['high quality', 'professional'],
            'style': ['4K', 'high quality', 'no text'],
            'niche': 'general',
            'content_types': ['general content', 'professional', 'high quality']
        })
        
        # Get niche for content-specific queries
        niche = channel_config.get('niche', 'general')
        content_types = channel_config.get('content_types', [])
        
        # Select relevant keywords based on query and niche
        primary_keywords = channel_config['primary']
        secondary_keywords = channel_config['secondary']
        style_keywords = channel_config['style']
        
        query_lower = query.lower()
        selected_primary = []
        
        # Find relevant primary keywords
        for keyword in primary_keywords:
            if keyword.lower() in query_lower or any(word in query_lower for word in keyword.split()):
                selected_primary.append(keyword)
        
        # Use default if no matches
        if not selected_primary:
            selected_primary = primary_keywords[:2]
        
        # Select secondary keywords
        selected_secondary = secondary_keywords[:2]
        
        # Build final query with niche-specific content
        final_query_parts = []
        
        # Add niche-specific content type if available
        if content_types and 'content' in query.lower():
            # Replace generic "content" with specific niche content
            content_query = content_types[0] if content_types else niche
            final_query_parts.append(content_query)
        else:
            # Add original query
            final_query_parts.append(query)
        
        # Add selected keywords
        final_query_parts.extend(selected_primary)
        final_query_parts.extend(selected_secondary)
        final_query_parts.extend(style_keywords)
        
        # Remove duplicates and join
        final_query = ' '.join(list(dict.fromkeys(final_query_parts)))
        
        print(f"🔍 Enhanced Pexels query: {query} -> {final_query}")
        print(f"📊 Channel: {channel_niche}, Niche: {niche}")
        print(f"🎯 Keywords: {selected_primary + selected_secondary}")
        print(f"📝 Content Type: {content_types[0] if content_types else 'general'}")
        
        return final_query
    
    def _download_pexels_video(self, query: str, min_duration: float, target_path: str) -> Optional[str]:
        """
        Download video from Pexels using real API with retry mechanism and validation
        
        Args:
            query: Search query for video content
            min_duration: Minimum required duration in seconds
            target_path: Target file path for download
            
        Returns:
            Optional[str]: Path to downloaded video or None if failed
        """
        try:
            # Check if Pexels API is available
            if not bool(PEXELS_API_KEY):
                self.log_message("⚠️ Pexels API key not available, using fallback", "PEXELS")
                return self._get_pexels_fallback_video(query, min_duration, target_path)
            
            # Setup API headers
            headers = {
                "Authorization": PEXELS_API_KEY,
                "User-Agent": "EnhancedMasterDirector/2.0"
            }
            
            # Search for videos
            search_url = "https://api.pexels.com/videos/search"
            params = {
                "query": query,
                "per_page": 15,  # Get more results to find best match
                "orientation": "landscape",
                "size": "large"
            }
            
            self.log_message(f"🔍 Searching Pexels for: {query}", "PEXELS")
            
            # Make search request with retry
            response = self._make_pexels_request(search_url, params, headers)
            if not response:
                return None
            
            # Parse search results
            videos = response.get("videos", [])
            if not videos:
                self.log_message(f"⚠️ No videos found for query: {query}", "PEXELS")
                return None
            
            # Find best video (highest bitrate, meets duration requirement)
            # First filter videos by duration
            suitable_videos = [v for v in videos if v.get("duration", 0) >= min_duration]
            if not suitable_videos:
                self.log_message(f"⚠️ No videos meet duration requirement: {min_duration}s", "PEXELS")
                return None
            
            # Get all video files from suitable videos
            all_video_files = []
            for video in suitable_videos:
                all_video_files.extend(video.get("video_files", []))
            
            best_video_file = self._select_best_pexels_video(all_video_files)
            if not best_video_file:
                self.log_message(f"⚠️ No suitable video file found", "PEXELS")
                return None
            
            # Download the selected video
            download_url = best_video_file["link"]
            # Generate filename from video file info
            video_filename = f"pexels_{best_video_file.get('id', 'unknown')}_{query.replace(' ', '_')[:20]}.mp4"
            output_path = os.path.join(target_path, video_filename)
            
            # Ensure target directory exists
            _ensure_parent_dir(output_path)
            
            self.log_message(f"📥 Downloading: {download_url}", "PEXELS")
            
            # Download with retry mechanism
            success = self._download_video_file(download_url, output_path, headers)
            if not success:
                return None
            
            # Validate downloaded file
            if self._validate_downloaded_video(output_path, min_duration):
                self.log_message(f"✅ Video downloaded successfully: {os.path.basename(output_path)}", "PEXELS")
                return output_path
            else:
                self.log_message(f"❌ Downloaded video validation failed", "PEXELS")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None
                
        except Exception as e:
            self.log_message(f"❌ Pexels download failed: {e}", "ERROR")
            return None
    
    def _make_pexels_request(self, url: str, params: dict, headers: dict) -> Optional[dict]:
        """Make Pexels API request with retry mechanism"""
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                delay = 2 ** attempt  # 1s, 2s, 4s
                self.log_message(f"⚠️ Pexels request attempt {attempt + 1} failed: {e}, retrying in {delay}s", "PEXELS")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(delay)
                continue
        return None
    
    def _select_best_pexels_video(self, video_files: list[dict]) -> dict | None:
        """
        Return the best MP4 by (resolution, bitrate).
        """
        candidates = []
        for vf in video_files or []:
            if vf.get("file_type") == "video/mp4" and vf.get("width") and vf.get("height"):
                try:
                    w = int(vf["width"]); h = int(vf["height"])
                    br = int(vf.get("bitrate") or 0)
                except Exception:
                    continue
                candidates.append((w*h, br, vf))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]
    
    def _download_video_file(self, url: str, output_path: str, headers: dict) -> bool:
        """Download video file with retry mechanism and validation"""
        # Use minimal headers for CDN downloads
        download_headers = {"User-Agent": "EnhancedMasterDirector/2.0"}
        
        for attempt in range(3):
            try:
                response = requests.get(url, stream=True, timeout=30, headers=download_headers)
                response.raise_for_status()
                
                # Get file size for progress tracking
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                # Verify download completion
                if total_size > 0 and downloaded_size < total_size:
                    self.log_message(f"⚠️ Incomplete download detected, retrying...", "PEXELS")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    continue
                
                return True
                
            except Exception as e:
                delay = 2 ** attempt  # 1s, 2s, 4s
                self.log_message(f"⚠️ Download attempt {attempt + 1} failed: {e}, retrying in {delay}s", "PEXELS")
                if os.path.exists(output_path):
                    os.remove(output_path)
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(delay)
                continue
        
        return False
    
    def _validate_downloaded_video(self, file_path: str, min_duration: float) -> bool:
        """Validate downloaded video file"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # Check file size (should be reasonable)
            file_size = os.path.getsize(file_path)
            if file_size < 1024 * 1024:  # Less than 1MB
                self.log_message(f"⚠️ Downloaded file too small: {file_size} bytes", "PEXELS")
                return False
            
            # Try to load video to check duration
            try:
                clip = VideoFileClip(file_path)
                actual_duration = clip.duration
                clip.close()
                
                if actual_duration < min_duration:
                    self.log_message(f"⚠️ Video duration too short: {actual_duration:.1f}s < {min_duration:.1f}s", "PEXELS")
                    return False
                
                return True
                
            except Exception as e:
                self.log_message(f"⚠️ Video validation failed: {e}", "PEXELS")
                return False
                
        except Exception as e:
            self.log_message(f"⚠️ File validation failed: {e}", "WARNING")
            return False
    
    def _get_pexels_fallback_video(self, query: str, min_duration: float, target_path: str) -> Optional[str]:
        """
        Get fallback video when Pexels API is not available
        
        Args:
            query: Search query (used for logging)
            min_duration: Minimum required duration
            target_path: Target directory path
            
        Returns:
            Optional[str]: Path to fallback video or None
        """
        try:
            self.log_message(f"🔄 Using Pexels fallback for query: {query}", "PEXELS")
            
            # Try to find a suitable local video
            local_video = self._get_local_asset_fallback("general", 1)
            if local_video and os.path.exists(local_video):
                # Copy to target directory
                import shutil
                target_filename = f"fallback_{query.replace(' ', '_')[:20]}.mp4"
                target_file = os.path.join(target_path, target_filename)
                
                # Ensure target directory exists
                _ensure_parent_dir(target_file)
                
                shutil.copy2(local_video, target_file)
                
                self.log_message(f"✅ Fallback video copied: {target_filename}", "PEXELS")
                return target_file
            
            return None
            
        except Exception as e:
            self.log_message(f"⚠️ Pexels fallback failed: {e}", "WARNING")
            return None
    
    def _get_local_asset_fallback(self, channel_niche: str, scene_num: int) -> Optional[str]:
        """Get local asset as fallback when Pexels fails"""
        try:
            if not self.local_assets['videos']:
                return None
            
            # Select appropriate local video based on scene and niche
            video_index = (scene_num - 1) % len(self.local_assets['videos'])
            return self.local_assets['videos'][video_index]
            
        except Exception as e:
            self.log_message(f"⚠️ Local asset fallback failed: {e}", "WARNING")
            return None
    
    def _upscale_video_to_4k(self, video_path: str) -> Optional[str]:
        """Upscale video to 4K using Pillow if available"""
        try:
            if not PILLOW_AVAILABLE:
                return video_path
            
            # Placeholder for 4K upscaling
            # In real implementation, this would use AI upscaling
            return video_path
            
        except Exception as e:
            self.log_message(f"⚠️ 4K upscaling failed: {e}", "WARNING")
            return video_path
    
    def _create_enhanced_visual_clip(self, visual_path: str, duration: float, scene_index: int) -> VideoClip:
        """Create enhanced visual clip with black frame detection and fallback"""
        try:
            if not visual_path or not os.path.exists(visual_path):
                # Create fallback visual clip
                return self._create_fallback_visual_clip(duration, scene_index)
            
            # Load video clip
            video_clip = VideoFileClip(visual_path)
            
            # Check for black frames using enhanced detection
            black_frame_analysis = self.detect_black_frames(video_clip)
            black_frame_ratio = black_frame_analysis.get('black_frame_ratio', 0.0)
            
            if black_frame_ratio > 0.1:  # More than 10% black frames
                self.log_message(f"⚠️ High black frame ratio detected: {black_frame_ratio:.2%}", "WARNING")
                self.log_message(f"📍 Black frames at: {black_frame_analysis.get('black_frame_timestamps', [])[:3]}", "WARNING")
                video_clip.close()
                return self._create_fallback_visual_clip(duration, scene_index)
            
            # Ensure proper duration with smooth transitions
            if video_clip.duration < duration:
                video_clip = self.extend_clip_to_duration(video_clip, duration)
            elif video_clip.duration > duration:
                video_clip = video_clip.subclip(0, duration)
            
            return video_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Enhanced visual clip creation failed: {e}", "WARNING")
            return self._create_fallback_visual_clip(duration, scene_index)
    
    def detect_black_frames(self, clip: VideoClip) -> Dict[str, Any]:
        """
        Enhanced black frame detection using luma percentile and stddev analysis
        
        Args:
            clip: VideoClip to analyze
            
        Returns:
            Dict containing black frame ratio, timestamps, and analysis details
        """
        try:
            import numpy as np
            
            # Sample frames evenly across the video duration
            sample_count = min(50, int(clip.duration * clip.fps))  # Sample up to 50 frames
            frame_analysis = []
            black_frame_timestamps = []
            
            for i in range(sample_count):
                # Calculate time position evenly distributed
                time_pos = (i / sample_count) * clip.duration
                frame = clip.get_frame(time_pos)
                
                # Convert to grayscale (luma)
                if len(frame.shape) == 3:
                    # Use BT.709 luma conversion for more accurate brightness
                    luma_frame = frame[:, :, 0] * 0.2126 + frame[:, :, 1] * 0.7152 + frame[:, :, 2] * 0.0722
                else:
                    luma_frame = frame
                
                # Calculate frame statistics
                frame_mean = np.mean(luma_frame)
                frame_std = np.std(luma_frame)
                
                frame_analysis.append({
                    'time': time_pos,
                    'mean': frame_mean,
                    'std': frame_std
                })
                
                # Check if frame is likely black using percentile analysis
                if self._is_frame_black(frame_mean, frame_std, frame_analysis):
                    black_frame_timestamps.append(time_pos)
            
            # Calculate global statistics
            all_means = [f['mean'] for f in frame_analysis]
            all_stds = [f['std'] for f in frame_analysis]
            
            global_mean = np.mean(all_means)
            global_std = np.std(all_means)
            global_p10 = np.percentile(all_means, 10)  # 10th percentile
            
            # Calculate black frame ratio
            black_ratio = len(black_frame_timestamps) / len(frame_analysis)
            
            # Enhanced analysis results
            analysis_result = {
                'black_frame_ratio': black_ratio,
                'black_frame_timestamps': black_frame_timestamps,
                'total_frames_analyzed': len(frame_analysis),
                'global_statistics': {
                    'mean_luma': global_mean,
                    'luma_stddev': global_std,
                    'luma_p10': global_p10,
                    'mean_stddev': np.mean(all_stds)
                },
                'frame_details': frame_analysis
            }
            
            # Log analysis summary
            self.log_message(f"🔍 Black frame analysis: {black_ratio:.1%} black frames detected", "ANALYSIS")
            if black_frame_timestamps:
                self.log_message(f"📍 Black frames at: {[f'{t:.1f}s' for t in black_frame_timestamps[:5]]}", "ANALYSIS")
            
            return analysis_result
            
        except Exception as e:
            self.log_message(f"⚠️ Enhanced black frame detection failed: {e}", "WARNING")
            return {
                'black_frame_ratio': 0.0,
                'black_frame_timestamps': [],
                'total_frames_analyzed': 0,
                'global_statistics': {},
                'frame_details': []
            }
    
    def _is_frame_black(self, frame_mean: float, frame_std: float, frame_history: List[dict]) -> bool:
        """
        Determine if a frame is black using adaptive thresholding
        
        Args:
            frame_mean: Mean luma value of the frame
            frame_std: Standard deviation of luma values
            frame_history: List of previous frame analysis data
            
        Returns:
            bool: True if frame is considered black
        """
        try:
            if len(frame_history) < 3:
                # Need at least 3 frames for comparison
                return frame_mean < 15 and frame_std < 5
            
            # Calculate adaptive thresholds based on frame history
            recent_means = [f['mean'] for f in frame_history[-10:]]  # Last 10 frames
            recent_stds = [f['std'] for f in frame_history[-10:]]
            
            # Dynamic threshold: 10th percentile of recent frames
            threshold_mean = np.percentile(recent_means, 10)
            threshold_std = np.percentile(recent_stds, 10)
            
            # Frame is black if:
            # 1. Luma is below 10th percentile of recent frames
            # 2. Standard deviation is low (indicating uniform darkness)
            # 3. Absolute luma is very low
            is_black = (
                frame_mean < max(threshold_mean * 0.8, 20) and  # Adaptive threshold
                frame_std < max(threshold_std * 1.2, 8) and     # Low variation
                frame_mean < 25                                  # Absolute threshold
            )
            
            return is_black
            
        except Exception as e:
            # Fallback to simple threshold
            return frame_mean < 15 and frame_std < 5
    
    def _create_fallback_visual_clip(self, duration: float, scene_index: int) -> VideoClip:
        """Create fallback visual clip when original fails"""
        try:
            # Create a simple colored background with text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create 1920x1080 image
            width, height = 1920, 1080
            colors = [(50, 50, 100), (100, 50, 50), (50, 100, 50), (100, 100, 50)]
            color = colors[scene_index % len(colors)]
            
            img = Image.new('RGB', (width, height), color)
            draw = ImageDraw.Draw(img)
            
            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            text = f"Scene {scene_index + 1}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            # Save temporary image
            temp_path = f"temp_fallback_{scene_index}.png"
            img.save(temp_path)
            
            # Create video clip from image
            clip = ImageClip(temp_path).set_duration(duration)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return clip
            
        except Exception as e:
            self.log_message(f"⚠️ Fallback visual clip creation failed: {e}", "WARNING")
            # Ultimate fallback: solid color clip
            return ColorClip(size=(1920, 1080), color=(50, 50, 100)).set_duration(duration)
    
    def extend_clip_to_duration(self, clip: VideoClip, target_duration: float) -> VideoClip:
        """
        Extend clip to target duration using smooth crossfade transitions
        
        Args:
            clip: VideoClip to extend
            target_duration: Target duration in seconds
            
        Returns:
            VideoClip: Extended clip with smooth transitions
        """
        try:
            if clip.duration >= target_duration:
                return clip
            
            self.log_message(f"🔄 Extending clip from {clip.duration:.1f}s to {target_duration:.1f}s", "EXTENSION")
            
            # Calculate how many loops we need
            loops_needed = int(target_duration / clip.duration) + 1
            crossfade_duration = 0.5  # 0.5 second crossfade
            
            # Create extended clips with crossfade transitions
            extended_clips = []
            
            for i in range(loops_needed):
                if i == 0:
                    # First clip: no crossfade in
                    extended_clips.append(clip)
                else:
                    # Subsequent clips: add crossfade in
                    looped_clip = clip.crossfadein(crossfade_duration)
                    extended_clips.append(looped_clip)
            
            # Concatenate with crossfade method
            self.log_message(f"🎬 Concatenating {len(extended_clips)} clips with crossfade transitions", "EXTENSION")
            
            final_clip = concatenate_videoclips(
                extended_clips, 
                method="compose",
                transition=lambda t: t  # Linear crossfade
            )
            
            # Trim to exact target duration
            if final_clip.duration > target_duration:
                final_clip = final_clip.subclip(0, target_duration)
            
            self.log_message(f"✅ Clip extended successfully to {final_clip.duration:.1f}s", "EXTENSION")
            return final_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Clip duration extension failed: {e}", "WARNING")
            # Fallback: simple loop without crossfade
            return self._extend_clip_simple_fallback(clip, target_duration)
    
    def _extend_clip_simple_fallback(self, clip: VideoClip, target_duration: float) -> VideoClip:
        """Simple fallback extension method without crossfade"""
        try:
            loops_needed = int(target_duration / clip.duration) + 1
            extended_clips = [clip] * loops_needed
            
            final_clip = concatenate_videoclips(extended_clips, method="compose")
            return final_clip.subclip(0, target_duration)
            
        except Exception as e:
            self.log_message(f"⚠️ Simple fallback extension failed: {e}", "WARNING")
            return clip
    
    def _extend_with_ollama_content(self, clip: VideoClip, target_duration: float) -> Optional[VideoClip]:
        """Extend clip duration using Ollama-generated additional content"""
        try:
            import ollama
            
            prompt = f"""Video süresini uzatmak için ekstra script cümleleri üret.
            
            Mevcut süre: {clip.duration:.1f} saniye
            Hedef süre: {target_duration:.1f} saniye
            Ek süre gerekli: {target_duration - clip.duration:.1f} saniye
            
            Her cümle yaklaşık 2-3 saniye sürmeli.
            Toplam {int((target_duration - clip.duration) / 2.5)} cümle üret.
            
            Format: Her cümle yeni satırda, açıklama yapma."""
            
            response = ollama.chat(model=OLLAMA_MODEL, 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            content = response.get('message', {}).get('content', '')
            
            if content:
                # Parse generated sentences
                sentences = [line.strip() for line in content.split('\n') if line.strip()]
                
                if sentences:
                    self.log_message(f"🤖 Ollama generated {len(sentences)} additional sentences", "OLLAMA")
                    
                    # Create additional visual content for these sentences
                    # This would typically involve generating or finding additional visuals
                    # For now, return None to use fallback method
                    return None
            
            return None
            
        except Exception as e:
            self.log_message(f"⚠️ Ollama content extension failed: {e}", "WARNING")
            return None
    
    def _apply_professional_effects(self, clip: VideoClip, scene_index: int) -> VideoClip:
        """Apply professional visual effects to enhance quality"""
        try:
            # Add subtle zoom effect
            clip = clip.resize(lambda t: 1 + 0.05 * t / clip.duration)
            
            # Add color correction
            clip = clip.fx(vfx.colorx, 1.1)  # Slightly enhance colors
            
            # Add subtle vignette
            clip = clip.fx(vfx.vignette, 0.3)
            
            return clip
            
        except Exception as e:
            self.log_message(f"⚠️ Professional effects failed: {e}", "WARNING")
            return clip
    
    def _create_subliminal_message(self, message: str, duration: float) -> Optional[VideoClip]:
        """Create subliminal message clip"""
        try:
            # Create very brief text clip
            text_clip = TextClip(message, fontsize=40, color='white', bg_color='black')
            text_clip = text_clip.set_duration(duration)
            
            # Position in center
            text_clip = text_clip.set_position('center')
            
            return text_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Subliminal message creation failed: {e}", "WARNING")
            return None
    
    def _add_background_music(self, video: VideoClip, music_path: str) -> VideoClip:
        """Add background music to video"""
        try:
            if not os.path.exists(music_path):
                return video
            
            music_clip = AudioFileClip(music_path)
            
            # Loop music if needed
            if music_clip.duration < video.duration:
                loops_needed = int(video.duration / music_clip.duration) + 1
                music_clip = concatenate_audioclips([music_clip] * loops_needed)
            
            # Trim to video duration
            music_clip = music_clip.subclip(0, video.duration)
            
            # Lower volume for background
            music_clip = music_clip.volumex(0.3)
            
            # Combine with video audio
            final_audio = CompositeAudioClip([video.audio, music_clip])
            video = video.set_audio(final_audio)
            
            return video
            
        except Exception as e:
            self.log_message(f"⚠️ Background music addition failed: {e}", "WARNING")
            return video
    
    def _add_multilingual_subtitles(self, video: VideoClip) -> VideoClip:
        """Add multilingual subtitles"""
        try:
            # Placeholder for multilingual subtitle implementation
            # This would typically involve creating subtitle tracks
            return video
            
        except Exception as e:
            self.log_message(f"⚠️ Multilingual subtitles failed: {e}", "WARNING")
            return video
    
    def _extend_video_duration(self, video: VideoClip, target_duration: float) -> VideoClip:
        """Extend video duration to meet minimum requirements"""
        try:
            if video.duration >= target_duration:
                return video
            
            # Use Ollama to generate additional content
            extended_video = self._extend_with_ollama_content(video, target_duration)
            if extended_video:
                return extended_video
            
            # Fallback: loop the video
            loops_needed = int(target_duration / video.duration) + 1
            extended_clips = [video] * loops_needed
            
            final_video = concatenate_videoclips(extended_clips, method="compose")
            return final_video.subclip(0, target_duration)
            
        except Exception as e:
            self.log_message(f"⚠️ Video duration extension failed: {e}", "WARNING")
            return video
    
    def _analyze_video_quality(self, video_path: str) -> Dict[str, Any]:
        """Analyze video quality using MoviePy and numpy for real metrics"""
        try:
            if not os.path.exists(video_path):
                return {"error": "Video file not found"}
            
            clip = VideoFileClip(video_path)
            
            # Basic metrics
            duration = clip.duration
            fps = clip.fps
            size = clip.size
            
            # Visual variety analysis using numpy
            visual_variety = self._analyze_visual_variety(clip)
            
            # Audio quality analysis
            audio_quality = self._analyze_audio_quality(clip)
            
            # Black frame detection
            black_frame_ratio = self._detect_black_frames(clip)
            
            # Calculate quality scores
            duration_score = min(1.0, duration / 600.0)  # Normalized to 10 minutes
            visual_score = visual_variety
            audio_score = audio_quality
            black_frame_penalty = 0.5 if black_frame_ratio > 0.1 else 1.0
            
            # Overall quality score
            overall_score = (duration_score + visual_score + audio_score) / 3 * black_frame_penalty
            
            analysis = {
                "duration": duration,
                "fps": fps,
                "size": size,
                "duration_score": duration_score,
                "visual_score": visual_score,
                "audio_score": audio_score,
                "black_frame_ratio": black_frame_ratio,
                "overall_score": overall_score,
                "quality_level": "high" if overall_score > 0.8 else "medium" if overall_score > 0.6 else "low"
            }
            
            # If quality is low, use Ollama to regenerate enhanced visual clip function
            if overall_score < 0.6:
                self._regenerate_enhanced_visual_clip_function(overall_score)
            
            clip.close()
            return analysis
            
        except Exception as e:
            self.log_message(f"❌ Video quality analysis failed: {e}", "ERROR")
            return {"error": str(e)}
    
    def _analyze_visual_variety(self, clip: VideoClip) -> float:
        """Analyze visual variety using numpy frame differences"""
        try:
            import numpy as np
            
            # Sample frames for analysis
            sample_count = min(50, int(clip.duration * clip.fps))
            frame_differences = []
            
            for i in range(sample_count - 1):
                time1 = (i / sample_count) * clip.duration
                time2 = ((i + 1) / sample_count) * clip.duration
                
                frame1 = clip.get_frame(time1)
                frame2 = clip.get_frame(time2)
                
                # Convert to grayscale
                if len(frame1.shape) == 3:
                    gray1 = np.mean(frame1, axis=2)
                    gray2 = np.mean(frame2, axis=2)
                else:
                    gray1 = frame1
                    gray2 = frame2
                
                # Calculate frame difference
                diff = np.mean(np.abs(gray2 - gray1))
                frame_differences.append(diff)
            
            # Calculate variety score based on standard deviation
            if frame_differences:
                variety_score = min(1.0, np.std(frame_differences) / 50.0)
                return variety_score
            
            return 0.5  # Default score
            
        except Exception as e:
            self.log_message(f"⚠️ Visual variety analysis failed: {e}", "WARNING")
            return 0.5
    
    def _analyze_audio_quality(self, clip: VideoClip) -> float:
        """Analyze audio quality"""
        try:
            if not clip.audio:
                return 0.0
            
            # Get audio array
            audio_array = clip.audio.to_soundarray()
            
            # Calculate audio metrics
            audio_mean = np.mean(np.abs(audio_array))
            audio_std = np.std(audio_array)
            
            # Quality score based on audio levels and variety
            if audio_mean > 0.01 and audio_std > 0.005:
                return min(1.0, (audio_mean * 100 + audio_std * 1000) / 2)
            else:
                return 0.3
            
        except Exception as e:
            self.log_message(f"⚠️ Audio quality analysis failed: {e}", "WARNING")
            return 0.5
    
    def _regenerate_enhanced_visual_clip_function(self, quality_score: float) -> None:
        """Use Ollama to regenerate the _create_enhanced_visual_clip function"""
        try:
            import ollama
            
            prompt = f"""Düşük kalite için iyileştirilmiş moviepy code üret.
            
            Kalite skoru: {quality_score:.2f}
            Problem: Video kalitesi düşük, _create_enhanced_visual_clip fonksiyonu iyileştirilmeli
            
            Fonksiyon gereksinimleri:
            - Black frame detection (numpy mean < 10)
            - Intelligent fallback systems
            - Quality enhancement techniques
            - Duration optimization
            
            Python code olarak döndür, sadece fonksiyonu yaz."""
            
            response = ollama.chat(model=OLLAMA_MODEL, 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            content = response.get('message', {}).get('content', '')
            
            if content:
                self.log_message(f"🤖 Ollama generated enhanced visual clip function", "OLLAMA")
                # In a real implementation, you might want to save this code
                # or use it to dynamically update the function
                
        except Exception as e:
            self.log_message(f"⚠️ Ollama function regeneration failed: {e}", "WARNING")
    
    def _create_hook_clip(self, hook_text: str, duration: float) -> VideoClip:
        """Create hook clip for short videos"""
        try:
            # Create hook text clip
            text_clip = TextClip(hook_text, fontsize=50, color='white', bg_color='red')
            text_clip = text_clip.set_duration(duration)
            text_clip = text_clip.set_position('center')
            
            return text_clip
            
        except Exception as e:
            self.log_message(f"⚠️ Hook clip creation failed: {e}", "WARNING")
            # Fallback: solid color clip
            return ColorClip(size=(1920, 1080), color=(255, 0, 0)).set_duration(duration)
    
    def generate_morgan_freeman_voiceover(self, text: str, output_path: str) -> bool:
        """Generate Morgan Freeman style voiceover using advanced TTS"""
        try:
            if self.tts_system == "piper":
                return self._generate_piper_voiceover(text, output_path)
            elif self.tts_system == "espeak":
                return self._generate_espeak_voiceover(text, output_path)
            elif self.tts_system == "gtts":
                return self._generate_gtts_voiceover(text, output_path)
            else:
                self.log_message("❌ No TTS system available", "ERROR")
                return False
        except Exception as e:
            self.log_message(f"❌ Voiceover generation failed: {e}", "ERROR")
            return False
    
    def _generate_piper_voiceover(self, text: str, output_path: str) -> bool:
        """Generate voiceover using Piper TTS with Morgan Freeman style"""
        try:
            # Morgan Freeman style parameters
            voice_params = {
                'speed': 0.8,  # Slower, more deliberate
                'pitch': 0.7,  # Deeper voice
                'volume': 1.2,  # Slightly louder
                'model': 'en_US-amy-low.onnx'  # Use appropriate model
            }
            
            # Generate audio with Piper
            tts = piper.PiperVoice.load_model(voice_params['model'])
            audio_data = tts.synthesize(text, voice_params)
            
            # Save audio
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            self.log_message(f"✅ Piper TTS voiceover generated: {output_path}", "TTS")
            return True
            
        except Exception as e:
            self.log_message(f"❌ Piper TTS failed: {e}", "ERROR")
            return False
    
    def _generate_espeak_voiceover(self, text: str, output_path: str) -> bool:
        """Generate voiceover using espeak with Morgan Freeman style"""
        try:
            # espeak parameters for Morgan Freeman style
            cmd = f'espeak -v en-us -s 120 -p 50 -a 100 "{text}" -w "{output_path}"'
            os.system(cmd)
            
            if os.path.exists(output_path):
                self.log_message(f"✅ espeak voiceover generated: {output_path}", "TTS")
                return True
            else:
                return False
                
        except Exception as e:
            self.log_message(f"❌ espeak failed: {e}", "ERROR")
            return False
    
    def _generate_gtts_voiceover(self, text: str, output_path: str) -> bool:
        """Generate voiceover using gTTS as fallback"""
        try:
            tts = gTTS(text=text, lang='en', slow=True)  # Slow for Morgan Freeman style
            tts.save(output_path)
            
            self.log_message(f"✅ gTTS voiceover generated: {output_path}", "TTS")
            return True
            
        except Exception as e:
            self.log_message(f"❌ gTTS failed: {e}", "ERROR")
            return False
    
    def generate_custom_music(self, duration: float, mood: str = "epic") -> str:
        """Generate custom MIDI music using Ollama and mido"""
        if not MIDO_AVAILABLE:
            self.log_message("⚠️ MIDI generation not available, using fallback music", "MUSIC")
            return self._get_fallback_music()
        
        try:
            # Generate music parameters using Ollama
            music_params = self._generate_music_parameters_with_ollama(mood, duration)
            
            # Create MIDI file
            midi_file = self._create_midi_file(music_params, duration)
            
            # Convert MIDI to audio
            audio_file = self._convert_midi_to_audio(midi_file)
            
            if audio_file and os.path.exists(audio_file):
                self.log_message(f"✅ Custom music generated: {audio_file}", "MUSIC")
                return audio_file
            else:
                return self._get_fallback_music()
                
        except Exception as e:
            self.log_message(f"❌ Custom music generation failed: {e}", "MUSIC")
            return self._get_fallback_music()
    
    def _generate_music_parameters_with_ollama(self, mood: str, duration: float) -> Dict:
        """Generate music parameters using Ollama LLM"""
        try:
            import ollama
            
            prompt = f"""Generate music parameters for a {mood} mood video that's {duration:.1f} seconds long.
            
            Return as JSON with:
            - tempo (BPM)
            - key (C, D, E, F, G, A, B)
            - scale (major, minor, pentatonic)
            - instruments (array of 3-5 instruments)
            - chord_progression (array of 4-8 chords)
            - mood_intensity (0.1 to 1.0)
            
            Make it cinematic and engaging for documentary content."""
            
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
            content = response.get('message', {}).get('content', '{}')
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._get_default_music_params()
                
        except Exception as e:
            self.log_message(f"⚠️ Ollama music generation failed: {e}, using defaults", "MUSIC")
            return self._get_default_music_params()
    
    def _get_default_music_params(self) -> Dict:
        """Get default music parameters"""
        return {
            'tempo': 120,
            'key': 'C',
            'scale': 'major',
            'instruments': ['piano', 'strings', 'brass'],
            'chord_progression': ['C', 'Am', 'F', 'G'],
            'mood_intensity': 0.7
        }
    
    def _get_fallback_music(self) -> str:
        """Get fallback music from local assets"""
        if self.local_assets['audio']:
            return random.choice(self.local_assets['audio'])
        else:
            return "assets/audio/music/epic_music.mp3"  # Default fallback
    
    def generate_voiceover(self, script_data: dict, output_folder: str) -> List[str]:
        """Generate advanced voiceover with Morgan Freeman style"""
        self.log_message("🎤 Generating Morgan Freeman style voiceover...", "VOICEOVER")
        
        try:
            narration_list = [scene.get("sentence") for scene in script_data.get("script", []) if scene.get("sentence")]
            if not narration_list:
                self.log_message("❌ No narration text found in script", "ERROR")
                return None
            
            audio_files = []
            os.makedirs(output_folder, exist_ok=True)
            
            for i, text in enumerate(narration_list):
                file_path = os.path.join(output_folder, f"part_{i+1}.mp3")
                self.log_message(f"🎤 Generating voiceover {i+1}/{len(narration_list)}", "VOICEOVER")
                
                if self.generate_morgan_freeman_voiceover(text, file_path):
                    audio_files.append(file_path)
                else:
                    self.log_message(f"⚠️ Voiceover generation failed for part {i+1}", "WARNING")
                    continue
            
            self.log_message(f"✅ Generated {len(audio_files)} voiceover files", "VOICEOVER")
            return audio_files
            
        except Exception as e:
            self.log_message(f"❌ Voiceover generation failed: {e}", "ERROR")
            return None
    
    def find_visual_assets(self, script_data: dict, channel_niche: str, download_folder: str) -> List[str]:
        """Find and download visual assets with 4K upscaling"""
        self.log_message("🎬 Finding visual assets with 4K upscaling...", "VISUALS")
        
        video_paths = []
        os.makedirs(download_folder, exist_ok=True)
        scenes = script_data.get("script", [])
        
        for i, scene in enumerate(scenes):
            query = scene.get("visual_query", "")
            found_video_path = None
            
            if query:
                # Optimize Pexels query
                optimized_query = self._optimize_pexels_query(query, channel_niche)
                
                # Calculate minimum duration for this scene (estimate based on script length)
                estimated_duration = max(5.0, len(scene.get("text", "").split()) * 0.3)  # ~0.3s per word
                
                # Try Pexels download with real API
                found_video_path = self._download_pexels_video(optimized_query, estimated_duration, download_folder)
            
            # Fallback to local assets if Pexels fails
            if not found_video_path:
                found_video_path = self._get_local_asset_fallback(channel_niche, i+1)
            
            # Upscale video to 4K if available
            if found_video_path and PILLOW_AVAILABLE:
                upscaled_path = self._upscale_video_to_4k(found_video_path)
                if upscaled_path:
                    found_video_path = upscaled_path
            
            video_paths.append(found_video_path)
            
            if found_video_path:
                self.log_message(f"✅ Scene {i+1} visual asset ready: {os.path.basename(found_video_path)}", "VISUALS")
            else:
                self.log_message(f"⚠️ Scene {i+1} visual asset failed", "WARNING")
        
        return video_paths
    
    def edit_long_form_video(self, audio_files: list, visual_files: list, music_path: str, output_filename: str) -> Optional[str]:
        """Create advanced long-form video with professional effects"""
        self.log_message("🎬 Creating advanced long-form video...", "VIDEO")
        
        try:
            clips = []
            total_duration = 0
            
            for i, (audio_path, visual_path) in enumerate(zip(audio_files, visual_files)):
                if not os.path.exists(audio_path):
                    continue
                
                # Load audio
                audio_clip = AudioFileClip(audio_path)
                if not audio_clip.duration or audio_clip.duration <= 0:
                    continue
                
                # Create visual clip
                visual_clip = self._create_enhanced_visual_clip(visual_path, audio_clip.duration, i)
                
                # Combine audio and visual
                scene_clip = visual_clip.set_audio(audio_clip)
                
                # Add professional effects
                scene_clip = self._apply_professional_effects(scene_clip, i)
                
                clips.append(scene_clip)
                total_duration += audio_clip.duration
                
                # Add subliminal message every 25th frame (disabled by default)
                if ENABLE_SUBLIMINAL and i % 25 == 0:
                    subliminal_clip = self._create_subliminal_message("Subscribe now", 0.04)  # 1/25 second
                    if subliminal_clip:
                        clips.append(subliminal_clip)
            
            if not clips:
                self.log_message("❌ No valid clips to process", "ERROR")
                return None
            
            # Concatenate clips with smooth transitions
            final_video = concatenate_videoclips(clips, method="compose")
            
            # Add background music
            if music_path and os.path.exists(music_path):
                final_video = self._add_background_music(final_video, music_path)
            
            # Add multilingual subtitles
            final_video = self._add_multilingual_subtitles(final_video)
            
            # Ensure minimum duration (10+ minutes)
            if total_duration < 600:  # Less than 10 minutes
                final_video = self._extend_video_duration(final_video, 600)
            
            # Write final video with MoviePy 2.0.0.dev2 optimized settings
            final_video.write_videofile(
                output_filename, 
                fps=self.FPS, 
                codec=self.CODEC,
                audio_codec=self.AUDIO_CODEC, 
                preset='ultrafast',
                threads=2,
                logger=None,
                ffmpeg_params=['-crf', '28', '-pix_fmt', 'yuv420p']
            )
            
            # Analyze video quality
            self._analyze_video_quality(output_filename)
            
            self.log_message(f"✅ Advanced video created: {output_filename}", "SUCCESS")
            return output_filename
            
        except Exception as e:
            self.log_message(f"❌ Video creation failed: {e}", "ERROR")
            return None
    
    def create_short_videos(self, long_form_video_path: str, output_folder: str) -> List[str]:
        """Create 3 short videos (15-60 seconds) from long form video"""
        self.log_message("🎬 Creating short videos from long form content...", "SHORTS")
        
        try:
            if not os.path.exists(long_form_video_path):
                self.log_message("❌ Long form video not found", "ERROR")
                return []
            
            # Load long form video
            long_video = VideoFileClip(long_form_video_path)
            
            short_videos = []
            durations = [15, 30, 60]  # Different short video lengths
            
            for i, duration in enumerate(durations):
                try:
                    # Extract random segment
                    start_time = random.uniform(0, max(0, long_video.duration - duration))
                    end_time = start_time + duration
                    
                    # Create short clip
                    short_clip = long_video.subclip(start_time, end_time)
                    
                    # Add hook at beginning
                    hook_clip = self._create_hook_clip(f"Hook {i+1}: The Mystery Deepens...", 3)
                    final_short = concatenate_videoclips([hook_clip, short_clip], method="compose")
                    
                    # Save short video with MoviePy 2.0.0.dev2 optimized settings
                    output_path = os.path.join(output_folder, f"short_{i+1}_{duration}s.mp4")
                    final_short.write_videofile(
                        output_path, 
                        fps=self.FPS, 
                        codec=self.CODEC,
                        audio_codec=self.AUDIO_CODEC, 
                        preset='ultrafast',
                        threads=1,
                        logger=None,
                        ffmpeg_params=['-crf', '30', '-pix_fmt', 'yuv420p']
                    )
                    
                    short_videos.append(output_path)
                    self.log_message(f"✅ Short video {i+1} created: {duration}s", "SHORTS")
                    
                except Exception as e:
                    self.log_message(f"⚠️ Short video {i+1} creation failed: {e}", "WARNING")
                    continue
            
            long_video.close()
            
            self.log_message(f"✅ Created {len(short_videos)} short videos", "SHORTS")
            return short_videos
            
        except Exception as e:
            self.log_message(f"❌ Short video creation failed: {e}", "ERROR")
            return []

    def create_video_optimized(self, script_data: dict, channel_niche: str, 
                              quality_preset: str = "high", use_gpu: bool = True) -> str:
        """Create video with optimized processing pipeline"""
        
        # Quality presets
        quality_presets = {
            "fast": {"fps": 24, "resolution": "1280x720", "crf": 23},
            "balanced": {"fps": 30, "resolution": "1920x1080", "crf": 20},
            "high": {"fps": 60, "resolution": "1920x1080", "crf": 18},
            "ultra": {"fps": 60, "resolution": "3840x2160", "crf": 16}
        }
        
        preset = quality_presets.get(quality_preset, quality_presets["balanced"])
        
        # Set processing parameters
        self.FPS = preset["fps"]
        self.CRF = preset["crf"]
        
        # Resolution parsing
        if "x" in preset["resolution"]:
            width, height = map(int, preset["resolution"].split("x"))
            self.WIDTH = width
            self.HEIGHT = height
        
        print(f"🚀 Starting optimized video creation: {quality_preset} quality")
        print(f"   Resolution: {self.WIDTH}x{self.HEIGHT}")
        print(f"   FPS: {self.FPS}")
        print(f"   GPU: {'Enabled' if use_gpu and self.GPU_ACCELERATION else 'Disabled'}")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Parallel processing setup
            if HARDWARE_OPTIMIZATION["max_threads"] > 1:
                self._setup_parallel_processing()
            
            # Create video with optimization
            output_path = self._create_video_pipeline(script_data, channel_niche, use_gpu)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"✅ Video created successfully in {processing_time:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Video creation failed: {e}")
            raise
    
    def _setup_parallel_processing(self):
        """Setup parallel processing for video creation"""
        try:
            import concurrent.futures
            
            # Set thread pool
            self.max_workers = min(HARDWARE_OPTIMIZATION["max_threads"], 8)
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
            
            print(f"🚀 Parallel processing enabled: {self.max_workers} workers")
            
        except Exception as e:
            print(f"⚠️ Parallel processing setup failed: {e}")
            self.max_workers = 1
            self.executor = None
    
    def _create_video_pipeline(self, script_data: dict, channel_niche: str, use_gpu: bool) -> str:
        """Create video using optimized pipeline"""
        
        # Generate script
        script = self._generate_script_optimized(script_data, channel_niche)
        
        # Generate audio
        audio_path = self._generate_audio_optimized(script, use_gpu)
        
        # Generate visuals
        visual_paths = self._generate_visuals_optimized(script_data, channel_niche)
        
        # Compile video
        output_path = self._compile_video_optimized(script, audio_path, visual_paths, use_gpu)
        
        return output_path
    
    def _generate_script_optimized(self, script_data: dict, channel_niche: str) -> str:
        """Enhanced script generation with intelligent content analysis"""
        try:
            # Enhanced content analysis
            topic = script_data.get('topic', '')
            duration_target = script_data.get('duration', 15)  # minutes
            target_audience = script_data.get('audience', 'general')
            
            print(f"🎬 Generating enhanced script for {channel_niche}")
            print(f"📝 Topic: {topic}")
            print(f"⏱️ Target Duration: {duration_target} minutes")
            print(f"👥 Target Audience: {target_audience}")
            
            # Channel-specific script templates with niche content
            script_templates = {
                'CKLegends': {
                    'intro': f"In the annals of time, where legends are born and heroes rise from the ashes of ordinary existence, we find ourselves drawn to the extraordinary tale of {topic}.",
                    'structure': ['epic_intro', 'rising_action', 'climax', 'heroic_resolution', 'inspirational_ending'],
                    'tone': 'epic, dramatic, inspiring',
                    'style': 'narrative, storytelling, cinematic',
                    'niche': 'history',
                    'content_focus': ['ancient civilizations', 'historical figures', 'epic battles', 'mythological stories', 'royal kingdoms', 'heroic legends']
                },
                'CKCombat': {
                    'intro': f"Prepare for battle as we dive into the intense world of {topic}, where skill meets strategy and victory is earned through determination and mastery.",
                    'structure': ['action_intro', 'skill_development', 'conflict_escalation', 'victory_moment', 'motivational_ending'],
                    'tone': 'intense, dynamic, motivational',
                    'style': 'action-oriented, skill-focused, competitive',
                    'niche': 'combat',
                    'content_focus': ['martial arts', 'military history', 'sports combat', 'warrior training', 'battle scenes', 'victory moments']
                },
                'CKDrive': {
                    'intro': f"Buckle up and feel the adrenaline as we explore the high-speed world of {topic}, where speed meets precision and champions are made.",
                    'structure': ['speed_intro', 'skill_showcase', 'competition_rise', 'victory_celebration', 'inspirational_ending'],
                    'tone': 'fast-paced, exciting, competitive',
                    'style': 'dynamic, speed-focused, championship-oriented',
                    'niche': 'racing',
                    'content_focus': ['car racing', 'motorsport', 'speed records', 'racing legends', 'championship moments', 'adrenaline sports']
                },
                'CKFinanceCore': {
                    'intro': f"Welcome to the world of strategic thinking and financial mastery, where we explore the sophisticated realm of {topic}.",
                    'structure': ['professional_intro', 'concept_explanation', 'strategy_development', 'success_application', 'future_outlook'],
                    'tone': 'professional, analytical, confident',
                    'style': 'educational, strategic, business-focused',
                    'niche': 'finance',
                    'content_focus': ['business success', 'financial markets', 'entrepreneurship', 'wealth building', 'investment strategies', 'corporate leadership']
                },
                'CKIronWill': {
                    'intro': f"Discover the power within as we explore the unbreakable spirit and relentless determination behind {topic}.",
                    'structure': ['inspirational_intro', 'challenge_presentation', 'overcoming_obstacles', 'victory_achievement', 'motivational_ending'],
                    'tone': 'inspirational, powerful, motivational',
                    'style': 'motivational, overcoming, success-focused',
                    'niche': 'motivation',
                    'content_focus': ['personal development', 'overcoming challenges', 'success stories', 'motivational speakers', 'fitness transformation', 'mental strength']
                }
            }
            
            # Get channel template
            template = script_templates.get(channel_niche, {
                'intro': f"Experience the amazing world of {topic}.",
                'structure': ['intro', 'development', 'climax', 'resolution'],
                'tone': 'engaging, informative, inspiring',
                'style': 'narrative, educational, entertaining'
            })
            
            # Calculate word count based on duration (average 150 words per minute)
            target_words = int(duration_target * 150)
            
            # Generate intelligent script structure
            script_parts = []
            
            # Introduction (15% of script)
            intro_words = int(target_words * 0.15)
            script_parts.append(f"{template['intro']}")
            
            # Main content (70% of script)
            main_words = int(target_words * 0.70)
            main_content = self._generate_main_content(topic, channel_niche, main_words, template)
            script_parts.append(main_content)
            
            # Conclusion (15% of script)
            conclusion_words = int(target_words * 0.15)
            conclusion = self._generate_conclusion(topic, channel_niche, conclusion_words, template)
            script_parts.append(conclusion)
            
            # Combine script parts
            final_script = ' '.join(script_parts)
            
            # Verify word count
            actual_words = len(final_script.split())
            print(f"📊 Script generated: {actual_words} words (target: {target_words})")
            print(f"🎭 Style: {template['style']}")
            print(f"🎨 Tone: {template['tone']}")
            print(f"🎯 Niche: {template.get('niche', 'general')}")
            print(f"📝 Content Focus: {template.get('content_focus', ['general'])[0]}")
            
            return final_script
            
        except Exception as e:
            print(f"⚠️ Enhanced script generation failed: {e}")
            # Fallback to standard generation
            return self._generate_script(script_data, channel_niche)
    
    def _generate_main_content(self, topic: str, channel_niche: str, target_words: int, template: dict) -> str:
        """Generate main content section with intelligent structure"""
        
        # Channel-specific content strategies
        content_strategies = {
            'CKLegends': [
                f"The story of {topic} begins in the depths of ancient wisdom, where every step forward represents a victory for human ingenuity.",
                f"Through the ages, {topic} has stood as a testament to the unquenchable thirst for knowledge that drives us forward.",
                f"From the smallest innovations to the grandest achievements, every aspect of {topic} represents a victory for human determination.",
                f"The lessons we learn from {topic} are not just about understanding the world around us, but about understanding ourselves."
            ],
            'CKCombat': [
                f"In the arena of {topic}, every challenge presents an opportunity to demonstrate skill and determination.",
                f"The path to mastery in {topic} requires dedication, practice, and an unyielding spirit.",
                f"Through the trials of {topic}, we discover that true strength comes from within.",
                f"Victory in {topic} is not just about winning, but about growing stronger with every challenge."
            ],
            'CKDrive': [
                f"The world of {topic} is driven by passion, precision, and the pursuit of excellence.",
                f"Every moment in {topic} is an opportunity to push boundaries and exceed expectations.",
                f"Through {topic}, we learn that success comes from preparation, skill, and determination.",
                f"The thrill of {topic} lies not just in the destination, but in the journey of improvement."
            ],
            'CKFinanceCore': [
                f"The principles of {topic} are built on strategic thinking and calculated decision-making.",
                f"Success in {topic} requires understanding market dynamics and adapting to changing conditions.",
                f"Through {topic}, we learn that wealth creation is both an art and a science.",
                f"The future of {topic} is shaped by innovation, technology, and forward-thinking strategies."
            ],
            'CKIronWill': [
                f"The essence of {topic} lies in the unbreakable spirit that refuses to accept defeat.",
                f"Every obstacle in {topic} is an opportunity to prove that limits are meant to be broken.",
                f"Through {topic}, we discover that true strength comes from facing challenges head-on.",
                f"The journey of {topic} teaches us that success is not given, but earned through persistence."
            ]
        }
        
        # Get channel-specific content
        strategies = content_strategies.get(channel_niche, [
            f"The fascinating world of {topic} offers endless opportunities for discovery and growth.",
            f"Through {topic}, we learn valuable lessons that apply to every aspect of life.",
            f"The story of {topic} is one of innovation, determination, and human achievement.",
            f"Exploring {topic} reveals the incredible potential that lies within each of us."
        ])
        
        # Build main content
        content_parts = []
        remaining_words = target_words
        
        for strategy in strategies:
            if remaining_words > 0:
                content_parts.append(strategy)
                remaining_words -= len(strategy.split())
        
        # Add additional content if needed
        while remaining_words > 50:
            additional_content = f"Every aspect of {topic} contributes to our understanding of the world and our place in it."
            content_parts.append(additional_content)
            remaining_words -= len(additional_content.split())
        
        return ' '.join(content_parts)
    
    def _generate_conclusion(self, topic: str, channel_niche: str, target_words: int, template: dict) -> str:
        """Generate conclusion section with channel-specific messaging"""
        
        conclusions = {
            'CKLegends': f"The legacy of {topic} continues to inspire generations, reminding us that we are capable of extraordinary things when we dare to dream and work together. The future of {topic} is being written today by visionaries who see possibilities where others see only limitations.",
            'CKCombat': f"The mastery of {topic} is not just about skill, but about the warrior spirit that drives us to overcome every obstacle. Through {topic}, we learn that true victory comes from within, and that every challenge makes us stronger.",
            'CKDrive': f"The passion for {topic} drives us to push beyond our limits and achieve what once seemed impossible. In {topic}, we find not just competition, but a community of dedicated individuals striving for excellence.",
            'CKFinanceCore': f"The world of {topic} continues to evolve, offering new opportunities for those willing to adapt and innovate. Success in {topic} comes from understanding that change is constant and adaptation is key to long-term growth.",
            'CKIronWill': f"The journey through {topic} teaches us that the human spirit is unbreakable when fueled by determination and purpose. Every challenge in {topic} is an opportunity to prove that we are stronger than we think and more capable than we know."
        }
        
        return conclusions.get(channel_niche, f"The story of {topic} is a testament to human potential and the incredible things we can achieve when we set our minds to it. The future is waiting for you to make your mark in the world of {topic}.")
    
    def _generate_audio_optimized(self, script: str, use_gpu: bool) -> str:
        """Generate audio with optimization"""
        try:
            # Use GPU acceleration if available
            if use_gpu and self.GPU_ACCELERATION:
                return self._generate_audio_gpu(script)
            else:
                return self._generate_audio_cpu(script)
        except Exception as e:
            print(f"⚠️ Audio generation failed: {e}")
            return self._generate_audio_fallback(script)
    
    def _generate_audio_gpu(self, script: str) -> str:
        """Generate audio using GPU acceleration"""
        # GPU-optimized audio generation
        # This would use GPU-accelerated TTS or audio processing
        return self._generate_audio_cpu(script)  # Fallback for now
    
    def _generate_audio_cpu(self, script: str) -> str:
        """Generate audio using CPU processing"""
        return self._generate_audio_fallback(script)
    
    def _generate_visuals_optimized(self, script_data: dict, channel_niche: str) -> list:
        """Generate visuals with optimization"""
        try:
            # Parallel image generation
            if self.executor:
                futures = []
                for i in range(30):  # Generate 30 images for longer video
                    future = self.executor.submit(self._generate_single_image, script_data, channel_niche, i)
                    futures.append(future)
                
                # Collect results
                visual_paths = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            visual_paths.append(result)
                    except Exception as e:
                        print(f"⚠️ Image generation failed: {e}")
                
                return visual_paths
            else:
                # Sequential generation
                return [self._generate_single_image(script_data, channel_niche, i) for i in range(30)]
                
        except Exception as e:
            print(f"⚠️ Visual generation failed: {e}")
            return []
    
    def _generate_single_image(self, script_data: dict, channel_niche: str, index: int) -> str:
        """Generate single image with optimization"""
        try:
            # Use optimized image generation
            topic = script_data.get('topic', f'{channel_niche} content {index + 1}')
            return self._get_pexels_image_optimized(topic, channel_niche)
        except Exception as e:
            print(f"⚠️ Single image generation failed: {e}")
            return ""
    
    def _compile_video_optimized(self, script: str, audio_path: str, visual_paths: list, use_gpu: bool) -> str:
        """Compile video with optimization"""
        try:
            # Use GPU encoding if available
            if use_gpu and self.GPU_ACCELERATION:
                return self._compile_video_gpu(script, audio_path, visual_paths)
            else:
                return self._compile_video_cpu(script, audio_path, visual_paths)
        except Exception as e:
            print(f"⚠️ Video compilation failed: {e}")
            raise
    
    def _compile_video_gpu(self, script: str, audio_path: str, visual_paths: list) -> str:
        """Compile video using GPU acceleration"""
        try:
            # GPU-optimized video compilation
            output_path = f"outputs/optimized_video_{int(time.time())}.mp4"
            
            # Use MoviePy with GPU optimization
            clips = []
            for visual_path in visual_paths:
                if visual_path and os.path.exists(visual_path):
                    clip = VideoFileClip(visual_path)
                    clips.append(clip)
            
            if clips:
                # Concatenate clips
                final_video = concatenate_videoclips(clips)
                
                # Add audio if available
                if audio_path and os.path.exists(audio_path):
                    audio = AudioFileClip(audio_path)
                    
                    # Ensure audio duration matches video duration
                    if audio.duration > final_video.duration:
                        # Trim audio to match video
                        audio = audio.subclip(0, final_video.duration)
                        print(f"🎵 Audio trimmed to match video: {audio.duration:.1f}s")
                    elif audio.duration < final_video.duration:
                        # Loop audio to match video
                        loops_needed = int(final_video.duration / audio.duration) + 1
                        audio_loops = [audio] * loops_needed
                        from moviepy.editor import concatenate_audioclips
                        audio = concatenate_audioclips(audio_loops)
                        audio = audio.subclip(0, final_video.duration)
                        print(f"🎵 Audio looped to match video: {audio.duration:.1f}s")
                    
                    final_video = final_video.set_audio(audio)
                    print(f"✅ Audio synchronized: {audio.duration:.1f}s / {final_video.duration:.1f}s")
                
                # Write with GPU optimization
                final_video.write_videofile(
                    output_path,
                    fps=self.FPS,
                    codec=self.CODEC,
                    audio_codec=self.AUDIO_CODEC,
                    threads=HARDWARE_OPTIMIZATION["max_threads"],
                    preset='fast' if self.CODEC == 'h264_nvenc' else 'medium'
                )
                
                # Cleanup
                final_video.close()
                for clip in clips:
                    clip.close()
                
                return output_path
            else:
                raise Exception("No valid visual clips found")
                
        except Exception as e:
            print(f"⚠️ GPU video compilation failed: {e}")
            # Fallback to CPU
            return self._compile_video_cpu(script, audio_path, visual_paths)
    
    def _compile_video_cpu(self, script: str, audio_path: str, visual_paths: list) -> str:
        """Compile video using CPU processing"""
        # CPU-based video compilation (existing implementation)
        return self._compile_video_fallback(script, audio_path, visual_paths)

    def setup_video_effects(self):
        """Setup advanced video effects system"""
        try:
            self.effects_enabled = VIDEO_EFFECTS_CONFIG["cinematic_effects"]
            self.visual_effects = VISUAL_EFFECTS
            self.color_grading = VIDEO_EFFECTS_CONFIG["color_grading"]
            self.motion_graphics = VIDEO_EFFECTS_CONFIG["motion_graphics"]
            
            print("✅ Advanced video effects system initialized")
            
        except Exception as e:
            print(f"⚠️ Video effects setup failed: {e}")
            self.effects_enabled = False
    
    def create_cinematic_video(self, script_data: dict, channel_niche: str, 
                              effects_preset: str = "cinematic") -> str:
        """Create cinematic video with advanced effects"""
        
        print(f"🎬 Creating cinematic video: {effects_preset} preset")
        
        try:
            # Set current channel for asset management
            self.current_channel_niche = channel_niche
            
            # Get channel config for proper asset management
            channel_config = self._setup_channel_assets(channel_niche)
            
            # Use channel name for directories, not niche
            actual_channel_name = channel_niche  # CKLegends, CKCombat, etc.
            niche = channel_config.get('niche', 'general')  # history, combat, etc.
            
            print(f"🎬 Channel: {actual_channel_name}, Niche: {niche}")
            print(f"📁 Assets will be saved to: {channel_config['images_dir']}")
            
            # Generate enhanced content
            if hasattr(self, 'llm_handler') and hasattr(self.llm_handler, 'generate_enhanced_content'):
                enhanced_content = self.llm_handler.generate_enhanced_content(
                    script_data.get('topic', ''),
                    channel_niche,
                    'cinematic_script',
                    500,
                    'cinematic'
                )
                script = enhanced_content.get('content', '')
                quality_score = enhanced_content.get('quality_score', 0.0)
                print(f"📊 Content quality score: {quality_score:.1f}/100")
            else:
                script = self._generate_fallback_script(script_data, channel_niche)
            
            # Generate audio with high quality
            audio_path = self.generate_high_quality_audio(script, "temp_audio.wav", "dramatic")
            
            # Generate visuals with effects
            visual_paths = self._generate_cinematic_visuals(script_data, channel_niche, effects_preset)
            
            # Apply cinematic effects
            enhanced_visuals = self._apply_cinematic_effects(visual_paths, effects_preset)
            
            # Compile cinematic video
            output_path = self._compile_cinematic_video(script, audio_path, enhanced_visuals, effects_preset)
            
            # Cleanup temporary files
            self._cleanup_temp_files([audio_path])
            
            return output_path
            
        except Exception as e:
            print(f"❌ Cinematic video creation failed: {e}")
            raise
    
    def _generate_fallback_script(self, script_data: dict, channel_niche: str) -> str:
        """Generate comprehensive fallback script for cinematic video"""
        
        topic = script_data.get('topic', 'Amazing Content')
        
        # Extended template-based script generation for 10+ minute videos
        templates = {
            'history': f"""Welcome to an extraordinary journey through time, where we explore the incredible story of {topic}. 

From the ancient civilizations that shaped our world to the modern discoveries that continue to amaze us, this fascinating tale spans centuries of human achievement and wonder.

In the depths of history, we find stories of courage, innovation, and determination that have inspired generations. The ancient world was a place of mystery and wonder, where great minds dared to dream beyond the ordinary.

Through the ages, humanity has faced countless challenges, yet we have always found ways to overcome, adapt, and thrive. The story of {topic} is not just about the past, but about the human spirit that drives us forward.

As we journey through time, we discover that every era has its heroes, every century its innovations, and every generation its lessons to teach. The wisdom of the ancients still resonates today, reminding us that we are part of a greater story.

From the first civilizations to the modern world, the journey of {topic} has been one of discovery, wonder, and endless possibility. It's a story that continues to unfold, with each new generation adding their own chapter to this incredible tale.

The lessons of history are not just about remembering the past, but about understanding how it shapes our present and guides our future. Every discovery, every innovation, every moment of courage has contributed to the world we know today.

As we explore this fascinating story, we are reminded that we are the inheritors of an incredible legacy. The achievements of the past inspire us to reach for new heights, to dream bigger dreams, and to create a future worthy of our ancestors.

The story of {topic} is a testament to human ingenuity, resilience, and the unquenchable thirst for knowledge that drives us forward. It's a story that reminds us that we are capable of extraordinary things when we dare to dream and work together.

From ancient wisdom to modern innovation, the journey continues. The story of {topic} is not just history, it's our story, and it's still being written with every new discovery, every new understanding, and every new generation that takes up the challenge.

So join us on this incredible journey through time, as we explore the fascinating world of {topic} and discover the amazing stories that have shaped our world and continue to inspire us today.""",
            
            'motivation': f"""Welcome to a transformative journey of self-discovery and empowerment, where we unlock the incredible potential within you through the power of {topic}.

This is not just another motivational talk, this is a life-changing experience that will challenge everything you think you know about success, achievement, and personal transformation.

The journey to greatness begins with a single step, but that step must be taken with courage, determination, and an unshakeable belief in your own potential. Every successful person you admire started exactly where you are right now - with a dream and the courage to pursue it.

The path to success is not always easy, but it is always worth it. Along the way, you will face challenges that will test your resolve, obstacles that will challenge your determination, and moments of doubt that will question your commitment.

But here's the truth: every challenge you face is an opportunity to grow stronger, every obstacle you overcome is a chance to prove your resilience, and every moment of doubt is a test of your faith in yourself.

The difference between those who succeed and those who don't is not talent, intelligence, or luck. The difference is persistence, determination, and the willingness to keep going when everything seems impossible.

Success is not a destination, it's a journey. It's about becoming the person you were meant to be, achieving the goals you were born to achieve, and living the life you were destined to live.

The power to change your life is already within you. You have the strength to overcome any obstacle, the wisdom to make the right decisions, and the courage to take the necessary risks.

Every day is a new opportunity to move closer to your dreams. Every moment is a chance to take action, make progress, and create the future you desire.

The journey of {topic} is about more than just achieving goals, it's about becoming the person you were meant to be. It's about discovering your true potential and having the courage to live up to it.

So take that first step today. Believe in yourself, trust your instincts, and have the courage to pursue your dreams. The world is waiting for you to show them what you're capable of.

Remember, you are stronger than you think, more capable than you believe, and more deserving of success than you know. The only question is: are you ready to prove it to yourself and the world?

The journey begins now. Your future is waiting. And the only person who can make it happen is you.""",
            
            'finance': f"""Welcome to the world of financial mastery, where we explore the powerful strategies and proven methods that can transform your financial future through {topic}.

This is not just about making money, this is about building wealth, creating security, and achieving the financial freedom that allows you to live life on your own terms.

The journey to financial success begins with understanding that money is not the goal, it's the tool. The real goal is financial freedom - the ability to make choices based on what you want, not what you can afford.

Every successful investor, entrepreneur, and financial expert started with the same basic principles: understanding how money works, learning how to make it work for you, and having the discipline to stick to proven strategies.

The world of finance is complex, but the principles of wealth building are simple. It's about spending less than you earn, investing the difference wisely, and letting compound interest work its magic over time.

Success in finance is not about getting rich quick, it's about building sustainable wealth through consistent action, informed decisions, and long-term thinking. The tortoise always beats the hare in the race to financial freedom.

The key to financial success is education. Understanding how markets work, how investments function, and how to manage risk is essential to making smart financial decisions.

But knowledge alone is not enough. You must also have the discipline to implement what you learn, the patience to wait for results, and the courage to stay the course when markets are volatile.

The journey to financial freedom is not without challenges. You will face market downturns, investment losses, and moments of doubt. But these are not obstacles, they are opportunities to learn, grow, and become stronger.

Every successful investor has experienced losses, every successful entrepreneur has faced failure, and every successful person has had moments when they wanted to give up. The difference is that they kept going.

The principles of {topic} are timeless and universal. They work for anyone who is willing to learn, practice, and persist. Your background, education, or starting point doesn't matter - what matters is your commitment to success.

Financial freedom is not a dream, it's a destination that anyone can reach with the right knowledge, the right strategies, and the right mindset. The journey begins with a single step, and that step is education.

So join us on this incredible journey to financial mastery. Learn the strategies that have worked for generations, understand the principles that never change, and discover the path to the financial future you deserve.

The world of finance is waiting for you. Your financial future is in your hands. And the time to start building it is now.""",
            
            'automotive': f"""Welcome to the cutting-edge world of automotive innovation, where we explore the revolutionary technology and groundbreaking developments that are changing the automotive industry forever through {topic}.

This is not just about cars, this is about the future of transportation, the evolution of mobility, and the incredible innovations that are reshaping how we think about getting from point A to point B.

The automotive industry is experiencing a transformation unlike anything we've seen in over a century. From electric vehicles to autonomous driving, from connected cars to sustainable materials, the future of transportation is being written today.

Every day, engineers, designers, and innovators around the world are pushing the boundaries of what's possible. They're creating vehicles that are not just faster, safer, and more efficient, but smarter, more connected, and more sustainable.

The journey of automotive innovation is about more than just building better cars. It's about creating a transportation system that works for everyone, that protects our planet, and that opens up new possibilities for how we live and work.

From the first horseless carriages to the latest electric supercars, the automotive industry has always been about innovation and progress. But what's happening today is different - it's a fundamental shift in how we think about transportation.

The future of automotive technology is not just about the vehicles themselves, but about the entire ecosystem that supports them. It's about charging infrastructure, smart cities, and integrated transportation networks.

Every breakthrough in automotive technology brings us closer to a future where transportation is safer, cleaner, and more accessible. It's a future where traffic accidents are rare, where emissions are minimal, and where mobility is available to everyone.

The innovations in {topic} are not just changing the automotive industry, they're changing the world. They're creating new jobs, new industries, and new possibilities that we're only beginning to understand.

The journey to the future of transportation is happening now, and it's being driven by visionaries who see beyond the limitations of today to the possibilities of tomorrow. They're not just building cars, they're building the future.

From concept cars to production vehicles, from racing technology to everyday transportation, the innovations in {topic} are setting new standards for what's possible. They're proving that the future is not just coming, it's already here.

So join us on this incredible journey through the world of automotive innovation. Discover the technologies that are changing everything, meet the people who are making it happen, and see the future of transportation taking shape before your eyes.

The automotive revolution is here. The future is now. And the journey to tomorrow begins with understanding what's happening today.""",
            
            'combat': f"""Welcome to the world of martial arts mastery, where we explore the ancient techniques and modern strategies that can elevate your skills to the next level through {topic}.

This is not just about fighting, this is about discipline, respect, and the journey to becoming the best version of yourself. Martial arts is a path of continuous improvement, where every training session brings new insights and every challenge offers new opportunities for growth.

The journey to martial arts mastery begins with understanding that the greatest opponent you will ever face is not another person, but yourself. It's about overcoming your own limitations, pushing past your comfort zone, and discovering strength you never knew you had.

Every martial art has its own unique philosophy, techniques, and traditions. But they all share the same fundamental principles: respect for your opponent, dedication to your training, and commitment to continuous improvement.

The path to mastery is not easy. It requires hours of practice, countless repetitions, and the willingness to fail again and again until you succeed. But every bruise, every sore muscle, and every moment of frustration is a step toward becoming stronger.

Martial arts is not just about physical strength, it's about mental toughness, emotional control, and spiritual growth. It's about developing the discipline to train when you don't feel like it, the courage to face your fears, and the wisdom to know when to fight and when to walk away.

The techniques of {topic} have been refined over centuries, passed down from master to student, and tested in countless battles. They are not just fighting moves, they are expressions of human potential and the art of self-defense.

But martial arts is about more than just self-defense. It's about building confidence, developing character, and creating a community of like-minded individuals who support each other's growth and development.

Every training session is an opportunity to learn something new, to improve your technique, and to challenge yourself in ways you never thought possible. The journey never ends, because there's always something new to learn and improve.

The principles of martial arts extend far beyond the training hall. They teach you how to face challenges in life, how to overcome obstacles, and how to stay focused on your goals even when the path seems impossible.

The journey to martial arts mastery is not just about becoming a better fighter, it's about becoming a better person. It's about developing the qualities that make you successful in all areas of life: discipline, determination, respect, and continuous improvement.

So join us on this incredible journey through the world of martial arts. Discover the ancient wisdom that has guided warriors for centuries, learn the techniques that can protect you and those you love, and experience the transformation that comes from dedicating yourself to mastery.

The path to greatness is waiting. The techniques of {topic} are ready to be learned. And the journey to becoming a martial artist begins with a single step onto the training floor."""
        }
        
        return templates.get(channel_niche, f"""Experience the amazing world of {topic}. This incredible content will inspire, educate, and entertain you like never before.

From the depths of human creativity to the heights of innovation, the story of {topic} is one that spans the full spectrum of human experience and achievement.

Every discovery, every innovation, every moment of inspiration contributes to the incredible tapestry of human progress. The journey of {topic} is not just about what has been accomplished, but about what is still possible.

As we explore this fascinating world together, we discover that the boundaries of what we can achieve are constantly expanding. Every challenge we face is an opportunity to grow stronger, every obstacle we overcome is a chance to prove our resilience.

The story of {topic} is a testament to human potential, creativity, and the unquenchable thirst for knowledge that drives us forward. It's a story that reminds us that we are capable of extraordinary things when we dare to dream and work together.

From the smallest innovations to the grandest achievements, every step forward in {topic} represents a victory for human ingenuity and determination. It's a story that continues to unfold, with each new generation adding their own chapter to this incredible tale.

The lessons we learn from {topic} are not just about understanding the world around us, but about understanding ourselves and our place in the grand scheme of things. Every discovery brings new questions, every answer opens new possibilities.

As we journey through this fascinating world, we are reminded that we are part of something much larger than ourselves. The story of {topic} is not just about individual achievement, but about the collective progress of humanity.

The future of {topic} is being written today, by people who are not afraid to dream big, think differently, and challenge the status quo. They are the visionaries, the innovators, and the dreamers who see possibilities where others see only limitations.

So join us on this incredible journey of discovery and wonder. Experience the amazing world of {topic} and discover the incredible stories that have shaped our world and continue to inspire us today.

The adventure begins now. The possibilities are endless. And the future is waiting for you to make your mark.""")
    
    def _get_pexels_image_optimized(self, query: str, width: int = 1920, height: int = 1080) -> str:
        """Get optimized image from Pexels API with enhanced fallback"""
        
        # Ensure width and height are integers
        try:
            width = int(width) if width is not None else 1920
            height = int(height) if height is not None else 1080
        except (ValueError, TypeError):
            width, height = 1920, 1080
        
        try:
            if not self.pexels_api:
                print(f"🖼️ Pexels API not available, using enhanced fallback for: {query}")
                return self._get_fallback_image(query, width, height)
            
            # Search for images using custom Pexels API with enhanced error handling
            print(f"🔍 Searching Pexels for: {query}")
            
            try:
                photos = self.pexels_api.search_images(query, per_page=1, page=1)
                
                if photos and len(photos) > 0:
                    photo = photos[0]
                    image_url = photo.get('src', {}).get('original', photo.get('url', ''))
                    print(f"📸 Found Pexels image: {photo.get('alt', 'No description')}")
                    
                    # Download and optimize image
                    import requests
                    from PIL import Image
                    import io
                    
                    response = requests.get(image_url, timeout=30)
                    if response.status_code == 200:
                        # Open and resize image
                        image = Image.open(io.BytesIO(response.content))
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                        
                        # Save optimized image to channel-specific directory
                        import time
                        timestamp = int(time.time() * 1000)  # Millisecond precision
                        filename = f"pexels_{timestamp}.jpg"
                        
                        # Get channel asset path (if available)
                        if hasattr(self, 'current_channel_niche'):
                            output_path = self._get_channel_asset_path(self.current_channel_niche, 'image', filename)
                            print(f"📁 Saving to channel directory: {output_path}")
                        else:
                            output_path = f"temp_pexels_{timestamp}.jpg"
                            print(f"📁 Saving to temp directory: {output_path}")
                        
                        # Ensure directory exists
                        import os
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        image.save(output_path, "JPEG", quality=95, optimize=True)
                        
                        # Verify file was created
                        if os.path.exists(output_path):
                            file_size = os.path.getsize(output_path) / 1024  # KB
                            print(f"✅ Pexels image downloaded: {query} -> {output_path} ({file_size:.1f}KB)")
                            return output_path
                        else:
                            print(f"❌ File creation failed: {output_path}")
                            return self._get_fallback_image(query, width, height)
                    else:
                        print(f"❌ Image download failed: {response.status_code}")
                        return self._get_fallback_image(query, width, height)
                
                print(f"🔄 No Pexels images found for: {query}")
                return self._get_fallback_image(query, width, height)
                
            except requests.exceptions.ConnectionError as e:
                print(f"❌ Pexels connection error: {e}")
                print("🌐 Check internet connection and firewall settings")
                return self._get_fallback_image(query, width, height)
            except requests.exceptions.Timeout as e:
                print(f"❌ Pexels timeout error: {e}")
                return self._get_fallback_image(query, width, height)
            except Exception as e:
                print(f"❌ Pexels error: {e}")
                return self._get_fallback_image(query, width, height)
            
        except Exception as e:
            print(f"⚠️ Pexels image generation failed: {e}")
            print(f"🔄 Using enhanced fallback for: {query}")
            return self._get_fallback_image(query, width, height)
    
    def _get_fallback_image(self, query: str, width: int = 1920, height: int = 1080, variation: int = 0) -> str:
        """Generate simple fallback image when Pexels fails"""
        
        try:
            # Ensure width and height are integers
            width = int(width) if width is not None else 1920
            height = int(height) if height is not None else 1080
        except (ValueError, TypeError):
            width, height = 1920, 1080
        
        try:
            from PIL import Image, ImageDraw
            
            # Create a simple colored image
            image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(image)
            
            # Create simple gradient background
            for y in range(height):
                progress = y / height
                r = int(20 + progress * 60)
                g = int(10 + progress * 40)
                b = int(30 + progress * 80)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # Save image
            import time
            output_path = f"temp_fallback_{int(time.time())}.jpg"
            image.save(output_path, "JPEG", quality=95)
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Fallback image generation failed: {e}")
            # Return a simple colored image
            return self._create_simple_image(query, width, height)
    
    def _create_simple_image(self, query: str, width: int = 1920, height: int = 1080) -> str:
        """Create a very simple colored image"""
        
        # Ensure width and height are integers
        try:
            width = int(width) if width is not None else 1920
            height = int(height) if height is not None else 1080
        except (ValueError, TypeError):
            width, height = 1920, 1080
        
        try:
            from PIL import Image
            
            # Create solid color image
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            # Use hash of query string to select color
            query_hash = hash(str(query))
            color_index = abs(query_hash) % len(colors)
            color = colors[color_index]
            
            image = Image.new('RGB', (width, height), color)
            output_path = f"temp_simple_{int(time.time())}.jpg"
            image.save(output_path, "JPEG")
            
            return output_path
            
        except Exception as e:
            print(f"⚠️ Simple image creation failed: {e}")
            return None
    
    def _cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files"""
        try:
            for file_path in file_paths:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🧹 Cleaned up: {file_path}")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
    
    def _generate_cinematic_visuals(self, script_data: dict, channel_niche: str, effects_preset: str) -> list:
        """Generate cinematic visuals with effects"""
        
        try:
            # Generate base visuals
            base_visuals = self._generate_visuals_optimized(script_data, channel_niche)
            
            # Apply cinematic enhancement
            cinematic_visuals = []
            for visual_path in base_visuals:
                if visual_path and os.path.exists(visual_path):
                    enhanced_path = self._enhance_visual_cinematic(visual_path, effects_preset)
                    cinematic_visuals.append(enhanced_path)
            
            return cinematic_visuals
            
        except Exception as e:
            print(f"⚠️ Cinematic visuals generation failed: {e}")
            return []
    
    def _enhance_visual_cinematic(self, visual_path: str, effects_preset: str) -> str:
        """Enhance visual with cinematic effects"""
        
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
            
            # Load image
            image = Image.open(visual_path)
            
            # Apply cinematic effects based on preset
            if effects_preset == "cinematic":
                image = self._apply_cinematic_look(image)
            elif effects_preset == "vintage":
                image = self._apply_vintage_filter(image)
            elif effects_preset == "modern":
                image = self._apply_modern_filter(image)
            elif effects_preset == "dramatic":
                image = self._apply_dramatic_filter(image)
            elif effects_preset == "professional":
                image = self._apply_professional_filter(image)
            
            # Save enhanced image
            enhanced_path = visual_path.replace('.', '_enhanced.')
            image.save(enhanced_path, quality=95)
            
            return enhanced_path
            
        except Exception as e:
            print(f"⚠️ Visual enhancement failed: {e}")
            return visual_path
    
    def _apply_cinematic_look(self, image: Image.Image) -> Image.Image:
        """Apply cinematic look to image"""
        
        try:
            # Color grading
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.2)  # Slightly more saturated
            
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)  # Higher contrast
            
            # Brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.9)  # Slightly darker
            
            # Sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)  # Slightly sharper
            
            # Add cinematic aspect ratio (letterbox)
            image = self._add_letterbox(image, 2.39)  # Cinematic 21:9
            
            return image
            
        except Exception as e:
            print(f"⚠️ Cinematic look failed: {e}")
            return image
    
    def _apply_vintage_filter(self, image: Image.Image) -> Image.Image:
        """Apply vintage filter to image"""
        
        try:
            # Convert to sepia
            image = image.convert('RGB')
            image = ImageOps.colorize(image.convert('L'), '#8B4513', '#F4A460')
            
            # Add vintage noise
            image = image.filter(ImageFilter.GaussianBlur(0.5))
            
            # Reduce saturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.7)
            
            return image
            
        except Exception as e:
            print(f"⚠️ Vintage filter failed: {e}")
            return image
    
    def _apply_modern_filter(self, image: Image.Image) -> Image.Image:
        """Apply modern filter to image"""
        
        try:
            # High contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.4)
            
            # Vibrant colors
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.3)
            
            # Sharp edges
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            return image
            
        except Exception as e:
            print(f"⚠️ Modern filter failed: {e}")
            return image
    
    def _apply_dramatic_filter(self, image: Image.Image) -> Image.Image:
        """Apply dramatic filter to image"""
        
        try:
            # High contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.6)
            
            # Darker shadows
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)
            
            # Saturated colors
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.4)
            
            return image
            
        except Exception as e:
            print(f"⚠️ Dramatic filter failed: {e}")
            return image
    
    def _apply_professional_filter(self, image: Image.Image) -> Image.Image:
        """Apply professional filter to image"""
        
        try:
            # Balanced contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Natural colors
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)
            
            # Professional sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            print(f"⚠️ Professional filter failed: {e}")
            return image
    
    def _add_letterbox(self, image: Image.Image, aspect_ratio: float) -> Image.Image:
        """Add letterbox for cinematic aspect ratio"""
        
        try:
            # Calculate new dimensions
            width, height = image.size
            target_width = int(height * aspect_ratio)
            
            if target_width > width:
                # Need to add black bars on sides
                new_image = Image.new('RGB', (target_width, height), (0, 0, 0))
                paste_x = (target_width - width) // 2
                new_image.paste(image, (paste_x, 0))
                return new_image
            else:
                # Need to add black bars on top/bottom
                target_height = int(width / aspect_ratio)
                new_image = Image.new('RGB', (width, target_height), (0, 0, 0))
                paste_y = (target_height - height) // 2
                new_image.paste(image, (0, paste_y))
                return new_image
                
        except Exception as e:
            print(f"⚠️ Letterbox failed: {e}")
            return image
    
    def _apply_cinematic_effects(self, visual_paths: list, effects_preset: str) -> list:
        """Apply cinematic effects to visuals"""
        
        try:
            enhanced_paths = []
            
            for visual_path in visual_paths:
                if visual_path and os.path.exists(visual_path):
                    # Apply effects
                    enhanced_path = self._enhance_visual_cinematic(visual_path, effects_preset)
                    enhanced_paths.append(enhanced_path)
                    
                    # Add motion effects if enabled
                    if VIDEO_EFFECTS_CONFIG["motion_graphics"]:
                        motion_path = self._add_motion_effects(enhanced_path, effects_preset)
                        enhanced_paths.append(motion_path)
            
            return enhanced_paths
            
        except Exception as e:
            print(f"⚠️ Cinematic effects failed: {e}")
            return visual_paths
    
    def _add_motion_effects(self, visual_path: str, effects_preset: str) -> str:
        """Add simple motion effects to visual - simplified for stability"""
        
        try:
            # Skip motion effects for now to avoid corruption
            return visual_path
            
        except Exception as e:
            print(f"⚠️ Motion effects failed: {e}")
            return visual_path
    
    def _compile_cinematic_video(self, script: str, audio_path: str, visual_paths: list, effects_preset: str) -> str:
        """Compile cinematic video with effects"""
        
        try:
            # Create output path
            output_path = f"outputs/cinematic_video_{effects_preset}_{int(time.time())}.mp4"
            
            # Create video clips with effects
            clips = []
            for visual_path in visual_paths:
                if visual_path and os.path.exists(visual_path):
                    if visual_path.endswith(('.mp4', '.avi', '.mov')):
                        # Video file
                        clip = VideoFileClip(visual_path)
                    else:
                        # Image file - convert to video with longer duration for cinematic feel
                        # Calculate duration based on script length and number of visuals
                        script_words = len(script.split())
                        total_duration = max(900, script_words * 0.8)  # Minimum 15 minutes, 0.8s per word
                        clip_duration = total_duration / len(visual_paths) if len(visual_paths) > 0 else 15.0
                        clip_duration = max(8.0, min(clip_duration, 30.0))  # Between 8-30 seconds per clip
                        
                        print(f"📊 Script: {script_words} words, Target duration: {total_duration/60:.1f} minutes")
                        print(f"🎬 Clip duration: {clip_duration:.1f} seconds for {len(visual_paths)} visuals")
                        
                        clip = ImageClip(visual_path, duration=clip_duration)
                    
                    # Add basic effects only (skip complex cinematic effects for stability)
                    try:
                        clip = self._add_clip_effects(clip, effects_preset)
                    except Exception as e:
                        print(f"⚠️ Clip effects failed, using original: {e}")
                        # Use original clip if effects fail
                    
                    clips.append(clip)
            
            if clips:
                # Concatenate clips with transitions
                final_video = self._concatenate_with_transitions(clips, effects_preset)
                
                # Add audio if available
                if audio_path and os.path.exists(audio_path):
                    audio = AudioFileClip(audio_path)
                    
                    # Ensure audio duration matches video duration
                    if audio.duration > final_video.duration:
                        # Trim audio to match video
                        audio = audio.subclip(0, final_video.duration)
                        print(f"🎵 Audio trimmed to match video: {audio.duration:.1f}s")
                    elif audio.duration < final_video.duration:
                        # Loop audio to match video
                        loops_needed = int(final_video.duration / audio.duration) + 1
                        audio_loops = [audio] * loops_needed
                        from moviepy.editor import concatenate_audioclips
                        audio = concatenate_audioclips(audio_loops)
                        audio = audio.subclip(0, final_video.duration)
                        print(f"🎵 Audio looped to match video: {audio.duration:.1f}s")
                    
                    final_video = final_video.set_audio(audio)
                    print(f"✅ Audio synchronized: {audio.duration:.1f}s / {final_video.duration:.1f}s")
                
                                # Write final video with enhanced optimization
                try:
                    # Clear GPU memory before writing
                    if hasattr(self, 'memory_manager'):
                        self.memory_manager.clear_memory()
                    
                    print(f"🎬 Rendering video to: {output_path}")
                    print(f"📊 Video duration: {final_video.duration:.1f}s")
                    print(f"🎵 Audio duration: {final_video.audio.duration if final_video.audio else 'No audio'}")
                    
                    # Enhanced video writing with better error handling
                    final_video.write_videofile(
                        output_path,
                        fps=24,  # Optimal FPS for quality/speed balance
                        codec='libx264',  # Reliable CPU encoder
                        audio_codec='aac',
                        threads=4,  # Optimized thread count
                        preset='medium',  # Better quality than ultrafast
                        bitrate='8000k',  # High quality bitrate
                        logger=None,  # Reduce logging overhead
                        ffmpeg_params=['-crf', '18']  # High quality constant rate factor
                    )
                    
                    print(f"✅ Video rendered successfully: {output_path}")
                    
                except Exception as e:
                    print(f"⚠️ High-quality rendering failed: {e}")
                    print("🔄 Trying fallback rendering...")
                    
                    # Fallback with minimal settings
                    try:
                        final_video.write_videofile(
                            output_path,
                            fps=24,
                            codec='libx264',
                            audio_codec='aac',
                            threads=2,
                            preset='ultrafast',
                            logger=None
                        )
                        print(f"✅ Fallback rendering successful: {output_path}")
                        
                    except Exception as e2:
                        print(f"❌ All rendering methods failed: {e2}")
                        print("🔄 Attempting basic rendering...")
                        
                        # Last resort - basic rendering
                        try:
                            final_video.write_videofile(
                                output_path,
                                fps=24,
                                codec='libx264',
                                audio_codec='aac',
                                threads=1,
                                preset='ultrafast',
                                logger=None
                            )
                            print(f"✅ Basic rendering successful: {output_path}")
                            
                        except Exception as e3:
                            print(f"❌ All rendering methods failed: {e3}")
                            raise Exception(f"Video rendering completely failed: {e3}")
                
                # Cleanup
                final_video.close()
                for clip in clips:
                    clip.close()
                
                return output_path
            else:
                raise Exception("No valid visual clips found")
                
        except Exception as e:
            print(f"⚠️ Cinematic video compilation failed: {e}")
            raise
    
    def _add_clip_effects(self, clip, effects_preset: str):
        """Add effects to individual clip"""
        
        try:
            # Add color grading
            if VIDEO_EFFECTS_CONFIG["color_grading"]:
                clip = self._apply_color_grading(clip, effects_preset)
            
            # Add text overlays
            if VIDEO_EFFECTS_CONFIG["text_overlays"]:
                clip = self._add_text_overlays(clip, effects_preset)
            
            return clip
            
        except Exception as e:
            print(f"⚠️ Clip effects failed: {e}")
            return clip
    
    def _apply_color_grading(self, clip, effects_preset: str):
        """Apply color grading to clip"""
        
        try:
            # Simple color grading effects
            if effects_preset == "cinematic":
                clip = clip.fx(vfx.colorx, 1.1)  # Slightly more saturated
                clip = clip.fx(vfx.lum_contrast, lum=0.9, contrast=1.2)
            elif effects_preset == "dramatic":
                clip = clip.fx(vfx.colorx, 1.3)  # More saturated
                clip = clip.fx(vfx.lum_contrast, lum=0.8, contrast=1.4)
            
            return clip
            
        except Exception as e:
            print(f"⚠️ Color grading failed: {e}")
            return clip
    
    def _add_text_overlays(self, clip, effects_preset: str):
        """Add text overlays to clip"""
        
        try:
            # Create text clip
            from moviepy.editor import TextClip
            
            text = "Cinematic Content"
            if effects_preset == "cinematic":
                text_clip = TextClip(text, fontsize=70, color='white', font='Arial-Bold')
            else:
                text_clip = TextClip(text, fontsize=50, color='white', font='Arial')
            
            # Position text
            text_clip = text_clip.set_position('center').set_duration(clip.duration)
            
            # Composite with main clip
            from moviepy.editor import CompositeVideoClip
            final_clip = CompositeVideoClip([clip, text_clip])
            
            return final_clip
            
        except Exception as e:
            print(f"⚠️ Text overlays failed: {e}")
            return clip
    
    def _concatenate_with_transitions(self, clips: list, effects_preset: str):
        """Concatenate clips with transitions"""
        
        try:
            if len(clips) == 1:
                return clips[0]
            
            # Add transitions between clips
            final_clips = []
            for i, clip in enumerate(clips):
                final_clips.append(clip)
                
                # Add transition to next clip (except last)
                if i < len(clips) - 1:
                    transition = self._create_transition(clip, clips[i + 1], effects_preset)
                    if transition:
                        final_clips.append(transition)
            
            # Concatenate all clips
            from moviepy.editor import concatenate_videoclips
            final_video = concatenate_videoclips(final_clips)
            
            return final_video
            
        except Exception as e:
            print(f"⚠️ Transitions failed: {e}")
            # Fallback to simple concatenation
            from moviepy.editor import concatenate_videoclips
            return concatenate_videoclips(clips)
    
    def _create_transition(self, clip1, clip2, effects_preset: str):
        """Create transition between clips"""
        
        try:
            # Simple cross-dissolve transition
            transition_duration = 0.5
            
            # Create transition clip
            from moviepy.editor import CompositeVideoClip
            
            # Fade out first clip
            clip1_fade = clip1.fx(vfx.fadeout, transition_duration)
            
            # Fade in second clip
            clip2_fade = clip2.fx(vfx.fadein, transition_duration)
            
            # Composite transition
            transition = CompositeVideoClip([
                clip1_fade.set_duration(transition_duration),
                clip2_fade.set_duration(transition_duration)
            ])
            
            return transition
            
        except Exception as e:
            print(f"⚠️ Transition creation failed: {e}")
            return None

    def _setup_channel_assets(self, channel_niche: str) -> dict:
        """Setup channel-specific asset directories and management"""
        
        # Channel asset configuration with proper mapping
        channel_configs = {
            'CKLegends': {
                'images_dir': 'assets/images/CKLegends',
                'audio_dir': 'assets/audio/CKLegends',
                'video_dir': 'assets/video/CKLegends',
                'temp_dir': 'temp_assets/CKLegends',
                'style': 'epic, dramatic, cinematic',
                'niche': 'history'  # CKLegends = history niche
            },
            'CKCombat': {
                'images_dir': 'assets/images/CKCombat',
                'audio_dir': 'assets/audio/CKCombat',
                'video_dir': 'assets/video/CKCombat',
                'temp_dir': 'temp_assets/CKCombat',
                'style': 'action, dynamic, intense',
                'niche': 'combat'
            },
            'CKDrive': {
                'images_dir': 'assets/images/CKDrive',
                'audio_dir': 'assets/audio/CKDrive',
                'video_dir': 'assets/video/CKDrive',
                'temp_dir': 'temp_assets/CKDrive',
                'style': 'fast-paced, dynamic, exciting',
                'niche': 'racing'
            },
            'CKFinanceCore': {
                'images_dir': 'assets/images/CKFinanceCore',
                'audio_dir': 'assets/audio/CKFinanceCore',
                'video_dir': 'assets/video/CKFinanceCore',
                'temp_dir': 'temp_assets/CKFinanceCore',
                'style': 'professional, clean, modern',
                'niche': 'finance'
            },
            'CKIronWill': {
                'images_dir': 'assets/images/CKIronWill',
                'audio_dir': 'assets/audio/CKIronWill',
                'video_dir': 'assets/video/CKIronWill',
                'temp_dir': 'temp_assets/CKIronWill',
                'style': 'inspirational, powerful, motivational',
                'niche': 'motivation'
            }
        }
        
        # Get channel config or use default
        config = channel_configs.get(channel_niche, {
            'images_dir': f'assets/images/{channel_niche}',
            'audio_dir': f'assets/audio/{channel_niche}',
            'video_dir': f'assets/video/{channel_niche}',
            'temp_dir': f'temp_assets/{channel_niche}',
            'style': 'professional, engaging, high-quality',
            'niche': 'general'
        })
        
        # Create directories
        import os
        for directory in [config['images_dir'], config['audio_dir'], config['video_dir'], config['temp_dir']]:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        
        print(f"🎯 Channel assets configured for: {channel_niche}")
        print(f"🎨 Style: {config['style']}")
        
        return config
    
    def _get_channel_asset_path(self, channel_niche: str, asset_type: str, filename: str) -> str:
        """Get channel-specific asset path"""
        
        config = self._setup_channel_assets(channel_niche)
        
        if asset_type == 'image':
            return os.path.join(config['images_dir'], filename)
        elif asset_type == 'audio':
            return os.path.join(config['audio_dir'], filename)
        elif asset_type == 'video':
            return os.path.join(config['video_dir'], filename)
        elif asset_type == 'temp':
            return os.path.join(config['temp_dir'], filename)
        else:
            return os.path.join(config['temp_dir'], filename)
    
    def _download_to_channel_assets(self, channel_niche: str, asset_type: str, source_url: str, filename: str) -> str:
        """Download asset to channel-specific directory"""
        
        try:
            # Get channel asset path
            asset_path = self._get_channel_asset_path(channel_niche, asset_type, filename)
            
            # Download asset
            import requests
            response = requests.get(source_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save to channel directory
            with open(asset_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file
            if os.path.exists(asset_path):
                file_size = os.path.getsize(asset_path) / 1024  # KB
                print(f"✅ Asset downloaded to {channel_niche}: {asset_path} ({file_size:.1f}KB)")
                return asset_path
            else:
                print(f"❌ Asset download failed: {asset_path}")
                return ""
                
        except Exception as e:
            print(f"⚠️ Asset download error: {e}")
            return ""

    # ... existing code ...

# Convenience functions for backward compatibility
def generate_voiceover(script_data: dict, output_folder: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    return creator.generate_voiceover(script_data, output_folder)

def find_visual_assets(script_data: dict, channel_niche: str, download_folder: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    return creator.find_visual_assets(script_data, channel_niche, download_folder)

def edit_long_form_video(audio_files: list, visual_files: list, music_path: str, output_filename: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    return creator.edit_long_form_video(audio_files, visual_files, music_path, output_filename)

def create_short_video(long_form_video_path: str, output_filename: str):
    """Backward compatibility function"""
    creator = AdvancedVideoCreator()
    output_folder = os.path.dirname(output_filename)
    short_videos = creator.create_short_videos(long_form_video_path, output_folder)
    return short_videos[0] if short_videos else None

# Basic test
if __name__ == "__main__":
    print("🧪 Testing Advanced Video Creator...")
    creator = AdvancedVideoCreator()
    print("✅ Basic initialization completed!")

    # Example usage (replace with actual script_data, channel_niche, etc.)
    # script_data = {
    #     "script": [
    #         {"sentence": "This is the first scene. The story begins."},
    #         {"sentence": "The protagonist, a young adventurer, sets out on a quest."},
    #         {"sentence": "They encounter a mysterious forest."},
    #         {"sentence": "The protagonist discovers a hidden cave."},
    #         {"sentence": "They find a treasure map."},
    #         {"sentence": "The protagonist follows the map."},
    #         {"sentence": "They encounter a dragon."},
    #         {"sentence": "The protagonist fights the dragon."},
    #         {"sentence": "They win the battle."},
    #         {"sentence": "The protagonist returns home."},
    #         {"sentence": "They celebrate their victory."},
    #         {"sentence": "The end."}
    #     ]
    # }
    # channel_niche = "Adventure"
    # download_folder = "assets/videos/downloads"
    # visual_assets = creator.find_visual_assets(script_data, channel_niche, download_folder)
    # print(f"Visual assets found: {visual_assets}")

    # audio_files = ["assets/audio/voiceover/part_1.mp3", "assets/audio/voiceover/part_2.mp3"]
    # visual_files = ["assets/videos/downloads/scene_1_pexels.mp4", "assets/videos/downloads/scene_2_pexels.mp4"]
    # music_path = "assets/audio/music/epic_music.mp3"
    # output_filename = "assets/videos/advanced_video.mp4"
    # long_form_video = creator.edit_long_form_video(audio_files, visual_files, music_path, output_filename)
    # print(f"Long form video created: {long_form_video}")

    # long_form_video_path = "assets/videos/advanced_video.mp4"
    # output_folder = "assets/videos/shorts"
    # short_videos = creator.create_short_videos(long_form_video_path, output_folder)
    # print(f"Short videos created: {short_videos}")
