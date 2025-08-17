# -*- coding: utf-8 -*-
"""
🎭 AI ADVANCED VOICE ACTING ENGINE
Professional AI-Powered Voice Acting & Character Voice Generation
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import requests
from dataclasses import dataclass, field
from enum import Enum

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("⚠️ Audio processing libraries not available - some features will be limited")

# AI/ML imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - AI features will be limited")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - NLP features will be limited")

# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class Emotion(Enum):
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CALM = "calm"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    INSPIRATIONAL = "inspirational"
    MYSTERIOUS = "mysterious"
    DRAMATIC = "dramatic"
    EPIC = "epic"
    INTIMATE = "intimate"

class VoiceStyle(Enum):
    AUTHORITATIVE = "authoritative"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    MYSTERIOUS = "mysterious"
    ENERGETIC = "energetic"
    CALM = "calm"
    DRAMATIC = "dramatic"
    NARRATIVE = "narrative"
    CHARACTER = "character"

class CharacterPersonality(Enum):
    WISE = "wise"
    HEROIC = "heroic"
    MYSTERIOUS = "mysterious"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    INSPIRATIONAL = "inspirational"
    DRAMATIC = "dramatic"
    NEUTRAL = "neutral"

class AudioQuality(Enum):
    LOW = "low"           # 16kHz, 64kbps
    MEDIUM = "medium"     # 22kHz, 128kbps
    HIGH = "high"         # 44kHz, 256kbps
    PROFESSIONAL = "professional"  # 48kHz, 320kbps

@dataclass
class VoiceProfile:
    """Voice profile configuration"""
    emotion: Emotion
    style: VoiceStyle
    personality: CharacterPersonality
    pitch_adjustment: float = 0.0  # -12 to +12 semitones
    speed_adjustment: float = 1.0  # 0.5 to 2.0
    energy_level: float = 0.5      # 0.0 to 1.0
    clarity_level: float = 0.8     # 0.0 to 1.0

@dataclass
class AudioOutput:
    """Audio output configuration"""
    quality: AudioQuality
    format: str = "mp3"
    sample_rate: int = 44100
    bitrate: str = "256k"
    channels: int = 1

@dataclass
class VoiceActingResult:
    """Result of voice acting generation"""
    audio_file_path: str
    voice_profile: VoiceProfile
    audio_output: AudioOutput
    processing_time: float
    quality_score: float
    metadata: Dict[str, Any]

# =============================================================================
# AI ADVANCED VOICE ACTING ENGINE CLASS
# =============================================================================

class AdvancedVoiceActingEngine:
    """
    🎭 AI-Powered Advanced Voice Acting Engine
    
    Features:
    - Emotional voice synthesis
    - Character personality adaptation
    - Voice style customization
    - Professional audio quality
    - Multi-format output support
    - Voice cloning and adaptation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Advanced Voice Acting Engine"""
        self.config = config or {}
        self.setup_logging()
        
        # Initialize AI models and TTS engines
        self.initialize_voice_engines()
        
        # Voice profiles and templates
        self.voice_templates = self._load_voice_templates()
        self.emotion_mappings = self._load_emotion_mappings()
        
        # Audio processing capabilities
        self.audio_processor = self._initialize_audio_processor()
        
        logging.info("🎭 Advanced Voice Acting Engine initialized successfully!")
    
    def setup_logging(self):
        """Setup logging for the voice acting engine"""
        self.logger = logging.getLogger(__name__)
    
    def initialize_voice_engines(self):
        """Initialize various TTS and voice processing engines"""
        self.tts_engines = {}
        self.voice_models = {}
        
        # Initialize TTS engines
        self._initialize_tts_engines()
        
        # Initialize voice models if available
        if TORCH_AVAILABLE:
            self._initialize_voice_models()
        
        # Initialize NLP pipeline if available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_nlp_pipeline()
    
    def _initialize_tts_engines(self):
        """Initialize TTS engines"""
        try:
            # Try to initialize different TTS engines
            engines_to_try = [
                ("gtts", self._init_gtts),
                ("espeak", self._init_espeak),
                ("piper", self._init_piper),
                ("elevenlabs", self._init_elevenlabs)
            ]
            
            for engine_name, init_func in engines_to_try:
                try:
                    engine = init_func()
                    if engine:
                        self.tts_engines[engine_name] = engine
                        self.logger.info(f"✅ {engine_name.upper()} TTS engine initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ {engine_name.upper()} TTS engine failed: {e}")
            
            if not self.tts_engines:
                self.logger.warning("⚠️ No TTS engines available - voice generation will be limited")
                
        except Exception as e:
            self.logger.error(f"❌ TTS engine initialization failed: {e}")
    
    def _init_gtts(self):
        """Initialize Google Text-to-Speech"""
        try:
            from gtts import gTTS
            return {"type": "gtts", "available": True}
        except ImportError:
            return None
    
    def _init_espeak(self):
        """Initialize eSpeak TTS"""
        try:
            import subprocess
            # Check if espeak is available
            result = subprocess.run(["espeak", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return {"type": "espeak", "available": True}
            return None
        except Exception:
            return None
    
    def _init_piper(self):
        """Initialize Piper TTS"""
        try:
            import subprocess
            # Check if piper is available
            result = subprocess.run(["piper", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return {"type": "piper", "available": True}
            return None
        except Exception:
            return None
    
    def _init_elevenlabs(self):
        """Initialize ElevenLabs TTS"""
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                return {"type": "elevenlabs", "available": True, "api_key": api_key}
            return None
        except Exception:
            return None
    
    def _initialize_voice_models(self):
        """Initialize AI voice models"""
        try:
            # Initialize voice cloning model (placeholder)
            self.voice_models["cloning"] = self._create_voice_cloning_model()
            
            # Initialize emotion detection model
            self.voice_models["emotion_detection"] = self._create_emotion_detection_model()
            
            self.logger.info("✅ Voice AI models initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Voice AI models initialization failed: {e}")
    
    def _create_voice_cloning_model(self):
        """Create voice cloning model (placeholder)"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Simple placeholder model for voice cloning
            model = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            return model
        except Exception as e:
            self.logger.error(f"❌ Voice cloning model creation failed: {e}")
            return None
    
    def _create_emotion_detection_model(self):
        """Create emotion detection model (placeholder)"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Simple placeholder model for emotion detection
            model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, len(Emotion))
            )
            return model
        except Exception as e:
            self.logger.error(f"❌ Emotion detection model creation failed: {e}")
            return None
    
    def _initialize_nlp_pipeline(self):
        """Initialize NLP pipeline for text analysis"""
        try:
            # Initialize sentiment analysis
            self.nlp_pipeline = pipeline("sentiment-analysis", 
                                       model="distilbert-base-uncased-finetuned-sst-2-english")
            self.logger.info("✅ NLP pipeline initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ NLP pipeline initialization failed: {e}")
            self.nlp_pipeline = None
    
    def _initialize_audio_processor(self):
        """Initialize audio processing capabilities"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return None
        
        try:
            # Initialize audio processing
            processor = {
                "available": True,
                "effects": ["pitch_shift", "speed_change", "noise_reduction", "equalization"],
                "formats": ["mp3", "wav", "ogg", "flac"]
            }
            return processor
            
        except Exception as e:
            self.logger.error(f"❌ Audio processor initialization failed: {e}")
            return None
    
    def _load_voice_templates(self) -> Dict[str, Dict]:
        """Load predefined voice templates"""
        return {
            "wise_elder": {
                "personality": CharacterPersonality.WISE,
                "style": VoiceStyle.AUTHORITATIVE,
                "emotion": Emotion.CALM,
                "pitch_adjustment": -2.0,
                "speed_adjustment": 0.8,
                "energy_level": 0.4,
                "clarity_level": 0.9
            },
            "heroic_warrior": {
                "personality": CharacterPersonality.HEROIC,
                "style": VoiceStyle.DRAMATIC,
                "emotion": Emotion.EXCITED,
                "pitch_adjustment": 1.0,
                "speed_adjustment": 1.2,
                "energy_level": 0.9,
                "clarity_level": 0.8
            },
            "mysterious_sage": {
                "personality": CharacterPersonality.MYSTERIOUS,
                "style": VoiceStyle.MYSTERIOUS,
                "emotion": Emotion.MYSTERIOUS,
                "pitch_adjustment": -1.0,
                "speed_adjustment": 0.9,
                "energy_level": 0.6,
                "clarity_level": 0.7
            },
            "friendly_guide": {
                "personality": CharacterPersonality.FRIENDLY,
                "style": VoiceStyle.FRIENDLY,
                "emotion": Emotion.HAPPY,
                "pitch_adjustment": 0.0,
                "speed_adjustment": 1.0,
                "energy_level": 0.7,
                "clarity_level": 0.9
            },
            "professional_narrator": {
                "personality": CharacterPersonality.NEUTRAL,
                "style": VoiceStyle.PROFESSIONAL,
                "emotion": Emotion.NEUTRAL,
                "pitch_adjustment": 0.0,
                "speed_adjustment": 1.0,
                "energy_level": 0.6,
                "clarity_level": 0.95
            }
        }
    
    def _load_emotion_mappings(self) -> Dict[str, Dict]:
        """Load emotion to voice parameter mappings"""
        return {
            Emotion.HAPPY: {
                "pitch_adjustment": 1.0,
                "speed_adjustment": 1.1,
                "energy_level": 0.8,
                "clarity_level": 0.9
            },
            Emotion.SAD: {
                "pitch_adjustment": -1.0,
                "speed_adjustment": 0.8,
                "energy_level": 0.3,
                "clarity_level": 0.7
            },
            Emotion.EXCITED: {
                "pitch_adjustment": 2.0,
                "speed_adjustment": 1.3,
                "energy_level": 0.95,
                "clarity_level": 0.8
            },
            Emotion.CALM: {
                "pitch_adjustment": 0.0,
                "speed_adjustment": 0.9,
                "energy_level": 0.5,
                "clarity_level": 0.9
            },
            Emotion.ANGRY: {
                "pitch_adjustment": 1.5,
                "speed_adjustment": 1.2,
                "energy_level": 0.9,
                "clarity_level": 0.8
            },
            Emotion.INSPIRATIONAL: {
                "pitch_adjustment": 0.5,
                "speed_adjustment": 1.1,
                "energy_level": 0.8,
                "clarity_level": 0.95
            },
            Emotion.DRAMATIC: {
                "pitch_adjustment": 1.0,
                "speed_adjustment": 1.0,
                "energy_level": 0.85,
                "clarity_level": 0.9
            }
        }
    
    def create_character_voice(self, 
                              text: str, 
                              character_personality: CharacterPersonality = CharacterPersonality.WISE,
                              emotion: Emotion = Emotion.INSPIRATIONAL,
                              voice_style: VoiceStyle = VoiceStyle.AUTHORITATIVE,
                              output_quality: AudioQuality = AudioQuality.HIGH,
                              output_format: str = "mp3") -> VoiceActingResult:
        """
        Create character voice with specified personality and emotion
        
        Args:
            text: Text to convert to speech
            character_personality: Character personality type
            emotion: Emotional tone
            voice_style: Voice style
            output_quality: Audio quality
            output_format: Output format
            
        Returns:
            VoiceActingResult: Complete voice acting result
        """
        try:
            start_time = time.time()
            self.logger.info(f"🎭 Creating character voice: {character_personality.value} with {emotion.value} emotion")
            
            # Create voice profile
            voice_profile = self._create_voice_profile(
                character_personality, emotion, voice_style
            )
            
            # Create audio output configuration
            audio_output = self._create_audio_output(output_quality, output_format)
            
            # Generate speech
            audio_file_path = self._generate_speech(text, voice_profile, audio_output)
            
            if not audio_file_path:
                raise Exception("Speech generation failed")
            
            # Process audio with character-specific enhancements
            enhanced_audio_path = self._apply_character_enhancements(
                audio_file_path, voice_profile, audio_output
            )
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(enhanced_audio_path, voice_profile)
            
            # Create result
            result = VoiceActingResult(
                audio_file_path=enhanced_audio_path,
                voice_profile=voice_profile,
                audio_output=audio_output,
                processing_time=time.time() - start_time,
                quality_score=quality_score,
                metadata={
                    "text_length": len(text),
                    "character_personality": character_personality.value,
                    "emotion": emotion.value,
                    "voice_style": voice_style.value,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"✅ Character voice created successfully: {enhanced_audio_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Character voice creation failed: {e}")
            return self._create_fallback_result(text, character_personality, emotion)
    
    def _create_voice_profile(self, personality: CharacterPersonality, 
                             emotion: Emotion, style: VoiceStyle) -> VoiceProfile:
        """Create voice profile based on personality, emotion, and style"""
        try:
            # Get base template
            template_key = f"{personality.value}_{style.value}"
            base_template = self.voice_templates.get(template_key, self.voice_templates["professional_narrator"])
            
            # Get emotion adjustments
            emotion_adjustments = self.emotion_mappings.get(emotion, {})
            
            # Create profile with adjustments
            profile = VoiceProfile(
                emotion=emotion,
                style=style,
                personality=personality,
                pitch_adjustment=base_template.get("pitch_adjustment", 0.0) + emotion_adjustments.get("pitch_adjustment", 0.0),
                speed_adjustment=base_template.get("speed_adjustment", 1.0) * emotion_adjustments.get("speed_adjustment", 1.0),
                energy_level=base_template.get("energy_level", 0.5) * emotion_adjustments.get("energy_level", 1.0),
                clarity_level=base_template.get("clarity_level", 0.8) * emotion_adjustments.get("clarity_level", 1.0)
            )
            
            # Ensure values are within valid ranges
            profile.pitch_adjustment = max(-12.0, min(12.0, profile.pitch_adjustment))
            profile.speed_adjustment = max(0.5, min(2.0, profile.speed_adjustment))
            profile.energy_level = max(0.0, min(1.0, profile.energy_level))
            profile.clarity_level = max(0.0, min(1.0, profile.clarity_level))
            
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Voice profile creation failed: {e}")
            # Return default profile
            return VoiceProfile(
                emotion=emotion,
                style=style,
                personality=personality
            )
    
    def _create_audio_output(self, quality: AudioQuality, format_type: str) -> AudioOutput:
        """Create audio output configuration"""
        quality_settings = {
            AudioQuality.LOW: {"sample_rate": 16000, "bitrate": "64k"},
            AudioQuality.MEDIUM: {"sample_rate": 22050, "bitrate": "128k"},
            AudioQuality.HIGH: {"sample_rate": 44100, "bitrate": "256k"},
            AudioQuality.PROFESSIONAL: {"sample_rate": 48000, "bitrate": "320k"}
        }
        
        settings = quality_settings.get(quality, quality_settings[AudioQuality.HIGH])
        
        return AudioOutput(
            quality=quality,
            format=format_type,
            sample_rate=settings["sample_rate"],
            bitrate=settings["bitrate"]
        )
    
    def _generate_speech(self, text: str, voice_profile: VoiceProfile, 
                         audio_output: AudioOutput) -> Optional[str]:
        """Generate speech using available TTS engines"""
        try:
            # Try different TTS engines in order of preference
            engines_to_try = ["elevenlabs", "piper", "gtts", "espeak"]
            
            for engine_name in engines_to_try:
                if engine_name in self.tts_engines:
                    try:
                        audio_path = self._generate_with_engine(
                            engine_name, text, voice_profile, audio_output
                        )
                        if audio_path:
                            return audio_path
                    except Exception as e:
                        self.logger.warning(f"⚠️ {engine_name} generation failed: {e}")
                        continue
            
            # If all engines fail, try fallback
            return self._generate_fallback_speech(text, voice_profile, audio_output)
            
        except Exception as e:
            self.logger.error(f"❌ Speech generation failed: {e}")
            return None
    
    def _generate_with_engine(self, engine_name: str, text: str, 
                             voice_profile: VoiceProfile, audio_output: AudioOutput) -> Optional[str]:
        """Generate speech with specific TTS engine"""
        try:
            if engine_name == "gtts":
                return self._generate_with_gtts(text, audio_output)
            elif engine_name == "espeak":
                return self._generate_with_espeak(text, voice_profile, audio_output)
            elif engine_name == "piper":
                return self._generate_with_piper(text, voice_profile, audio_output)
            elif engine_name == "elevenlabs":
                return self._generate_with_elevenlabs(text, voice_profile, audio_output)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ {engine_name} generation failed: {e}")
            return None
    
    def _generate_with_gtts(self, text: str, audio_output: AudioOutput) -> Optional[str]:
        """Generate speech with Google TTS"""
        try:
            from gtts import gTTS
            
            # Create output directory
            output_dir = Path("assets/audio/generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"gtts_{timestamp}.{audio_output.format}"
            output_path = output_dir / filename
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ gTTS generation failed: {e}")
            return None
    
    def _generate_with_espeak(self, text: str, voice_profile: VoiceProfile, 
                              audio_output: AudioOutput) -> Optional[str]:
        """Generate speech with eSpeak"""
        try:
            import subprocess
            
            # Create output directory
            output_dir = Path("assets/audio/generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"espeak_{timestamp}.{audio_output.format}"
            output_path = output_dir / filename
            
            # Build espeak command
            cmd = [
                "espeak",
                "-w", str(output_path),
                "-s", str(int(150 * voice_profile.speed_adjustment)),
                "-p", str(int(50 + voice_profile.pitch_adjustment * 10)),
                "-a", str(int(100 * voice_profile.energy_level)),
                text
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_path.exists():
                return str(output_path)
            else:
                self.logger.error(f"eSpeak command failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ eSpeak generation failed: {e}")
            return None
    
    def _generate_with_piper(self, text: str, voice_profile: VoiceProfile, 
                            audio_output: AudioOutput) -> Optional[str]:
        """Generate speech with Piper TTS"""
        try:
            import subprocess
            
            # Create output directory
            output_dir = Path("assets/audio/generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"piper_{timestamp}.{audio_output.format}"
            output_path = output_dir / filename
            
            # Build piper command (simplified)
            cmd = [
                "piper",
                "--output", str(output_path),
                "--model", "en_US-amy-low.onnx",  # Default model
                text
            ]
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_path.exists():
                return str(output_path)
            else:
                self.logger.error(f"Piper command failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Piper generation failed: {e}")
            return None
    
    def _generate_with_elevenlabs(self, text: str, voice_profile: VoiceProfile, 
                                 audio_output: AudioOutput) -> Optional[str]:
        """Generate speech with ElevenLabs"""
        try:
            import requests
            
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                return None
            
            # Create output directory
            output_dir = Path("assets/audio/generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f"elevenlabs_{timestamp}.{audio_output.format}"
            output_path = output_dir / filename
            
            # ElevenLabs API endpoint
            url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": voice_profile.clarity_level,
                    "similarity_boost": voice_profile.energy_level
                }
            }
            
            # Make API request
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save audio file
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                return str(output_path)
            else:
                self.logger.error(f"ElevenLabs API failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ ElevenLabs generation failed: {e}")
            return None
    
    def _generate_fallback_speech(self, text: str, voice_profile: VoiceProfile, 
                                 audio_output: AudioOutput) -> Optional[str]:
        """Generate fallback speech when all engines fail"""
        try:
            self.logger.warning("⚠️ Using fallback speech generation")
            
            # Try to use any available TTS engine with basic settings
            for engine_name, engine in self.tts_engines.items():
                try:
                    if engine_name == "gtts":
                        return self._generate_with_gtts(text, audio_output)
                    elif engine_name == "espeak":
                        return self._generate_with_espeak(text, voice_profile, audio_output)
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Fallback speech generation failed: {e}")
            return None
    
    def _apply_character_enhancements(self, audio_path: str, voice_profile: VoiceProfile, 
                                    audio_output: AudioOutput) -> str:
        """Apply character-specific audio enhancements"""
        try:
            if not self.audio_processor or not AUDIO_PROCESSING_AVAILABLE:
                return audio_path
            
            # Create enhanced output path
            output_dir = Path("assets/audio/enhanced")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            enhanced_filename = f"enhanced_{timestamp}.{audio_output.format}"
            enhanced_path = output_dir / enhanced_filename
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=audio_output.sample_rate)
            
            # Apply pitch shift if needed
            if abs(voice_profile.pitch_adjustment) > 0.1:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=voice_profile.pitch_adjustment)
            
            # Apply speed change if needed
            if abs(voice_profile.speed_adjustment - 1.0) > 0.1:
                y = librosa.effects.time_stretch(y, rate=voice_profile.speed_adjustment)
            
            # Save enhanced audio
            sf.write(str(enhanced_path), y, sr)
            
            return str(enhanced_path)
            
        except Exception as e:
            self.logger.error(f"❌ Character enhancements failed: {e}")
            return audio_path  # Return original if enhancement fails
    
    def _calculate_quality_score(self, audio_path: str, voice_profile: VoiceProfile) -> float:
        """Calculate quality score for generated audio"""
        try:
            base_score = 0.7  # Base quality score
            
            # Adjust based on voice profile
            if voice_profile.clarity_level > 0.8:
                base_score += 0.1
            if voice_profile.energy_level > 0.7:
                base_score += 0.1
            if abs(voice_profile.pitch_adjustment) < 2.0:
                base_score += 0.05
            if abs(voice_profile.speed_adjustment - 1.0) < 0.2:
                base_score += 0.05
            
            # Check file size and existence
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                if file_size > 1000:  # At least 1KB
                    base_score += 0.1
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            self.logger.error(f"❌ Quality score calculation failed: {e}")
            return 0.5  # Default moderate quality
    
    def _create_fallback_result(self, text: str, personality: CharacterPersonality, 
                               emotion: Emotion) -> VoiceActingResult:
        """Create fallback result when voice generation fails"""
        try:
            # Create minimal voice profile
            voice_profile = VoiceProfile(
                emotion=emotion,
                style=VoiceStyle.NEUTRAL,
                personality=personality
            )
            
            # Create basic audio output
            audio_output = AudioOutput(
                quality=AudioQuality.MEDIUM,
                format="mp3"
            )
            
            # Create placeholder result
            result = VoiceActingResult(
                audio_file_path="",
                voice_profile=voice_profile,
                audio_output=audio_output,
                processing_time=0.0,
                quality_score=0.0,
                metadata={
                    "text_length": len(text),
                    "character_personality": personality.value,
                    "emotion": emotion.value,
                    "error": "Voice generation failed, using fallback",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Fallback result creation failed: {e}")
            # Return minimal result
            return VoiceActingResult(
                audio_file_path="",
                voice_profile=VoiceProfile(Emotion.NEUTRAL, VoiceStyle.NEUTRAL, CharacterPersonality.NEUTRAL),
                audio_output=AudioOutput(AudioQuality.LOW),
                processing_time=0.0,
                quality_score=0.0,
                metadata={"error": f"Complete failure: {e}"}
            )

    def generate_cinematic_audio_plan(self, script_data: Dict[str, Any], 
                                     target_duration_minutes: float = 15.0,
                                     niche: str = "general") -> Dict[str, Any]:
        """
        Generate comprehensive audio plan for cinematic content
        
        Args:
            script_data: Script with scene breakdown
            target_duration_minutes: Target duration
            niche: Content niche
            
        Returns:
            Complete audio production plan
        """
        try:
            # Analyze script for voice acting requirements
            voice_requirements = self._analyze_voice_requirements(script_data, niche)
            
            # Generate music score plan
            music_plan = self._generate_music_plan(script_data, niche)
            
            # Create sound effects plan
            sound_effects_plan = self._generate_sound_effects_plan(script_data, niche)
            
            # Audio mixing and mastering plan
            mixing_plan = self._create_mixing_plan(voice_requirements, music_plan, sound_effects_plan)
            
            return {
                "voice_requirements": voice_requirements,
                "music_plan": music_plan,
                "sound_effects_plan": sound_effects_plan,
                "mixing_plan": mixing_plan,
                "quality_standards": self.quality_standards,
                "production_timeline": self._create_audio_timeline(target_duration_minutes),
                "technical_requirements": self._get_technical_requirements(niche)
            }
            
        except Exception as e:
            self.logger.error(f"Audio plan generation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_voice_requirements(self, script_data: Dict[str, Any], 
                                  niche: str) -> Dict[str, Any]:
        """Analyze script for voice acting requirements"""
        voice_requirements = {
            "narrator_profile": self._get_narrator_profile_for_niche(niche),
            "emotional_arcs": [],
            "pacing_requirements": {},
            "character_voices": [],
            "technical_requirements": {}
        }
        
        if "scenes" in script_data:
            for scene in script_data["scenes"]:
                emotional_beat = scene.get("emotional_beat", "neutral")
                duration = scene.get("duration_seconds", 0)
                
                # Map emotional beat to voice requirements
                voice_profile = self._map_emotion_to_voice(emotional_beat, niche)
                
                voice_requirements["emotional_arcs"].append({
                    "scene": scene.get("scene_number", 0),
                    "emotional_beat": emotional_beat,
                    "voice_profile": voice_profile,
                    "duration_seconds": duration,
                    "intensity_level": self._get_intensity_level(emotional_beat),
                    "pacing": self._get_pacing_for_beat(emotional_beat)
                })
        
        return voice_requirements
    
    def _get_narrator_profile_for_niche(self, niche: str) -> Dict[str, Any]:
        """Get appropriate narrator profile for the niche"""
        narrator_profiles = {
            "history": {
                "style": VoiceStyle.AUTHORITATIVE,
                "personality": CharacterPersonality.WISE,
                "emotion": Emotion.NEUTRAL,
                "pitch": "medium-low",
                "pace": "measured",
                "examples": ["David Attenborough", "Morgan Freeman", "Werner Herzog"]
            },
            "motivation": {
                "style": VoiceStyle.INSPIRATIONAL,
                "personality": CharacterPersonality.HEROIC,
                "emotion": Emotion.EXCITED,
                "pitch": "medium-high",
                "pace": "energetic",
                "examples": ["Tony Robbins", "Les Brown", "Eric Thomas"]
            },
            "finance": {
                "style": VoiceStyle.PROFESSIONAL,
                "personality": CharacterPersonality.AUTHORITATIVE,
                "emotion": Emotion.CALM,
                "pitch": "medium",
                "pace": "steady",
                "examples": ["Warren Buffett", "Ray Dalio", "Warren Buffett"]
            },
            "automotive": {
                "style": VoiceStyle.ENERGETIC,
                "personality": CharacterPersonality.PLAYFUL,
                "emotion": Emotion.EXCITED,
                "pitch": "medium-high",
                "pace": "dynamic",
                "examples": ["Jeremy Clarkson", "Richard Hammond", "James May"]
            },
            "combat": {
                "style": VoiceStyle.DRAMATIC,
                "personality": CharacterPersonality.HEROIC,
                "emotion": Emotion.INTENSE,
                "pitch": "medium-low",
                "pace": "powerful",
                "examples": ["Bruce Lee", "Mike Tyson", "Conor McGregor"]
            }
        }
        
        return narrator_profiles.get(niche, narrator_profiles["history"])
    
    def _map_emotion_to_voice(self, emotional_beat: str, niche: str) -> Dict[str, Any]:
        """Map emotional beat to specific voice characteristics"""
        emotion_mapping = {
            "opening": {
                "emotion": Emotion.CALM,
                "intensity": 0.6,
                "pace": "measured",
                "volume": 0.8
            },
            "hook": {
                "emotion": Emotion.EXCITED,
                "intensity": 0.9,
                "pace": "energetic",
                "volume": 0.9
            },
            "conflict": {
                "emotion": Emotion.DRAMATIC,
                "intensity": 0.8,
                "pace": "building",
                "volume": 0.85
            },
            "climax": {
                "emotion": Emotion.EPIC,
                "intensity": 1.0,
                "pace": "powerful",
                "volume": 1.0
            },
            "resolution": {
                "emotion": Emotion.INSPIRATIONAL,
                "intensity": 0.85,
                "pace": "steady",
                "volume": 0.9
            }
        }
        
        base_profile = emotion_mapping.get(emotional_beat, emotion_mapping["opening"])
        
        # Adjust for niche-specific characteristics
        if niche == "motivation":
            base_profile["intensity"] = min(1.0, base_profile["intensity"] * 1.2)
        elif niche == "combat":
            base_profile["intensity"] = min(1.0, base_profile["intensity"] * 1.3)
        
        return base_profile
    
    def _get_intensity_level(self, emotional_beat: str) -> float:
        """Get intensity level for emotional beat"""
        intensity_levels = {
            "opening": 0.6,
            "hook": 0.9,
            "conflict": 0.8,
            "climax": 1.0,
            "resolution": 0.85
        }
        return intensity_levels.get(emotional_beat, 0.7)
    
    def _get_pacing_for_beat(self, emotional_beat: str) -> str:
        """Get pacing recommendation for emotional beat"""
        pacing_map = {
            "opening": "measured",
            "hook": "energetic",
            "conflict": "building",
            "climax": "powerful",
            "resolution": "steady"
        }
        return pacing_map.get(emotional_beat, "moderate")
    
    def _generate_music_plan(self, script_data: Dict[str, Any], 
                            niche: str) -> Dict[str, Any]:
        """Generate comprehensive music score plan"""
        music_plan = {
            "overall_theme": self._get_music_theme_for_niche(niche),
            "scene_music": [],
            "emotional_arcs": [],
            "transitions": [],
            "technical_requirements": {}
        }
        
        if "scenes" in script_data:
            for scene in script_data["scenes"]:
                emotional_beat = scene.get("emotional_beat", "neutral")
                duration = scene.get("duration_seconds", 0)
                
                # Get music for this scene
                scene_music = self._get_music_for_scene(emotional_beat, duration, niche)
                
                music_plan["scene_music"].append({
                    "scene": scene.get("scene_number", 0),
                    "emotional_beat": emotional_beat,
                    "music_style": scene_music["style"],
                    "duration_seconds": duration,
                    "intensity": scene_music["intensity"],
                    "instruments": scene_music["instruments"],
                    "tempo": scene_music["tempo"]
                })
        
        return music_plan
    
    def _get_music_theme_for_niche(self, niche: str) -> Dict[str, Any]:
        """Get music theme for specific niche"""
        music_themes = {
            "history": {
                "style": "epic_orchestral",
                "mood": "majestic",
                "instruments": ["orchestra", "choir", "brass", "strings"],
                "examples": ["Gladiator", "Braveheart", "Kingdom of Heaven"]
            },
            "motivation": {
                "style": "uplifting_cinematic",
                "mood": "inspirational",
                "instruments": ["orchestra", "piano", "strings", "drums"],
                "examples": ["Rocky", "Chariots of Fire", "The Pursuit of Happyness"]
            },
            "finance": {
                "style": "modern_corporate",
                "mood": "sophisticated",
                "instruments": ["electronic", "strings", "piano", "bass"],
                "examples": ["The Social Network", "Wall Street", "The Big Short"]
            },
            "automotive": {
                "style": "high_energy_electronic",
                "mood": "thrilling",
                "instruments": ["electronic", "drums", "bass", "synths"],
                "examples": ["Fast & Furious", "Drive", "Baby Driver"]
            },
            "combat": {
                "style": "intense_action",
                "mood": "powerful",
                "instruments": ["drums", "brass", "strings", "percussion"],
                "examples": ["300", "John Wick", "Mad Max: Fury Road"]
            }
        }
        
        return music_themes.get(niche, music_themes["history"])

    def _get_music_for_scene(self, emotional_beat: str, duration: float, niche: str) -> Dict[str, Any]:
        """Get music specifications for a specific scene"""
        try:
            # Base music profile for emotional beat
            base_music = {
                "opening": {
                    "style": "ambient",
                    "intensity": 0.4,
                    "tempo": "slow",
                    "instruments": ["strings", "piano"]
                },
                "hook": {
                    "style": "energetic",
                    "intensity": 0.8,
                    "tempo": "fast",
                    "instruments": ["drums", "brass", "strings"]
                },
                "conflict": {
                    "style": "tension",
                    "intensity": 0.7,
                    "tempo": "building",
                    "instruments": ["strings", "percussion"]
                },
                "climax": {
                    "style": "epic",
                    "intensity": 1.0,
                    "tempo": "maximum",
                    "instruments": ["full_orchestra", "choir"]
                },
                "resolution": {
                    "style": "peaceful",
                    "intensity": 0.5,
                    "tempo": "slowing",
                    "instruments": ["piano", "soft_strings"]
                }
            }
            
            scene_music = base_music.get(emotional_beat, base_music["opening"])
            
            # Adjust for niche-specific characteristics
            niche_adjustments = {
                "history": {"intensity_multiplier": 1.2, "style_modifier": "orchestral"},
                "motivation": {"intensity_multiplier": 1.1, "style_modifier": "inspirational"},
                "finance": {"intensity_multiplier": 0.9, "style_modifier": "sophisticated"},
                "automotive": {"intensity_multiplier": 1.3, "style_modifier": "electronic"},
                "combat": {"intensity_multiplier": 1.4, "style_modifier": "intense"}
            }
            
            adjustment = niche_adjustments.get(niche, {"intensity_multiplier": 1.0, "style_modifier": "standard"})
            
            # Apply adjustments
            scene_music["intensity"] = min(1.0, scene_music["intensity"] * adjustment["intensity_multiplier"])
            scene_music["style"] = f"{adjustment['style_modifier']}_{scene_music['style']}"
            
            # Add duration-based adjustments
            if duration > 60:  # Long scenes
                scene_music["variation"] = "extended"
                scene_music["transitions"] = ["build_up", "main_theme", "wind_down"]
            else:  # Short scenes
                scene_music["variation"] = "focused"
                scene_music["transitions"] = ["direct"]
            
            return scene_music
            
        except Exception as e:
            logging.error(f"Music scene generation failed: {e}")
            return {
                "style": "standard",
                "intensity": 0.7,
                "tempo": "moderate",
                "instruments": ["strings"]
            }
    
    def _generate_sound_effects_plan(self, script_data: Dict[str, Any], 
                                   niche: str) -> Dict[str, Any]:
        """Generate sound effects plan"""
        sfx_plan = {
            "atmospheric": [],
            "action": [],
            "emotional": [],
            "transition": [],
            "technical_requirements": {}
        }
        
        if "scenes" in script_data:
            for scene in script_data["scenes"]:
                emotional_beat = scene.get("emotional_beat", "neutral")
                
                # Get SFX for emotional beat
                scene_sfx = self._get_sfx_for_beat(emotional_beat, niche)
                sfx_plan["atmospheric"].extend(scene_sfx.get("atmospheric", []))
                sfx_plan["action"].extend(scene_sfx.get("action", []))
                sfx_plan["emotional"].extend(scene_sfx.get("emotional", []))
                sfx_plan["transition"].extend(scene_sfx.get("transition", []))
        
        return sfx_plan

    def _create_mixing_plan(self, voice_requirements: Dict[str, Any], 
                           music_plan: Dict[str, Any], 
                           sfx_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive audio mixing plan"""
        try:
            mixing_plan = {
                "voice_track": {
                    "volume": 0.0,  # dB
                    "compression": {
                        "threshold": -20,
                        "ratio": 4,
                        "attack": 5,
                        "release": 50
                    },
                    "eq": {
                        "low_cut": 80,
                        "high_cut": 8000,
                        "presence_boost": 3,
                        "clarity_boost": 2
                    },
                    "reverb": {
                        "wet_level": 0.1,
                        "room_size": 0.3,
                        "damping": 0.5
                    }
                },
                "music_track": {
                    "volume": -12.0,  # dB
                    "compression": {
                        "threshold": -25,
                        "ratio": 3,
                        "attack": 10,
                        "release": 100
                    },
                    "eq": {
                        "low_cut": 60,
                        "high_cut": 16000,
                        "mid_boost": 1
                    },
                    "sidechain": {
                        "enabled": True,
                        "source": "voice",
                        "threshold": -30,
                        "ratio": 2
                    }
                },
                "sfx_track": {
                    "volume": -15.0,  # dB
                    "compression": {
                        "threshold": -30,
                        "ratio": 5,
                        "attack": 1,
                        "release": 20
                    },
                    "eq": {
                        "low_cut": 40,
                        "high_cut": 18000
                    }
                },
                "master_bus": {
                    "limiter": {
                        "threshold": -1.0,
                        "ceiling": -0.1
                    },
                    "loudness": {
                        "target": -14.0,  # LUFS
                        "range": 7.0
                    }
                }
            }
            
            # Adjust for niche-specific requirements
            if "niche" in voice_requirements:
                niche = voice_requirements["niche"]
                if niche == "combat":
                    mixing_plan["voice_track"]["volume"] = 2.0  # Louder for action
                    mixing_plan["music_track"]["volume"] = -8.0  # More prominent music
                elif niche == "motivation":
                    mixing_plan["voice_track"]["eq"]["presence_boost"] = 4  # Clearer voice
                    mixing_plan["music_track"]["volume"] = -10.0  # Balanced music
            
            return mixing_plan
            
        except Exception as e:
            logging.error(f"Mixing plan creation failed: {e}")
            return {}

    def _get_sfx_for_beat(self, emotional_beat: str, niche: str) -> Dict[str, List[str]]:
        """Get sound effects for emotional beat"""
        sfx_mapping = {
            "opening": {
                "atmospheric": ["ambient_wind", "distant_crowd", "nature_sounds"],
                "action": [],
                "emotional": ["soft_transition"],
                "transition": ["fade_in"]
            },
            "hook": {
                "atmospheric": ["energy_build", "tension_rise"],
                "action": ["impact", "whoosh"],
                "emotional": ["excitement", "anticipation"],
                "transition": ["quick_cut"]
            },
            "conflict": {
                "atmospheric": ["tension", "darkness", "pressure"],
                "action": ["struggle", "impact", "movement"],
                "emotional": ["fear", "determination"],
                "transition": ["dramatic_shift"]
            },
            "climax": {
                "atmospheric": ["intensity", "overwhelming"],
                "action": ["explosion", "clash", "victory"],
                "emotional": ["triumph", "overwhelm"],
                "transition": ["epic_shift"]
            },
            "resolution": {
                "atmospheric": ["calm", "peace", "relief"],
                "action": [],
                "emotional": ["satisfaction", "hope"],
                "transition": ["gentle_fade"]
            }
        }
        
        base_sfx = sfx_mapping.get(emotional_beat, sfx_mapping["opening"])
        
        # Add niche-specific SFX
        niche_sfx = {
            "history": ["ancient_echoes", "medieval_ambience", "historical_atmosphere"],
            "motivation": ["inspirational_rise", "achievement_sounds", "success_fanfare"],
            "finance": ["modern_tech", "corporate_ambience", "digital_sounds"],
            "automotive": ["engine_sounds", "speed_effects", "mechanical_ambience"],
            "combat": ["fight_sounds", "training_ambience", "athletic_effects"]
        }
        
        niche_effects = niche_sfx.get(niche, [])
        base_sfx["atmospheric"].extend(niche_effects)
        
        return base_sfx

    def _create_audio_timeline(self, duration_minutes: float) -> Dict[str, Any]:
        """Create audio production timeline"""
        try:
            total_seconds = duration_minutes * 60
            
            timeline = {
                "pre_production": {
                    "script_analysis": "1 hour",
                    "voice_casting": "2 hours",
                    "music_selection": "3 hours",
                    "sfx_planning": "2 hours"
                },
                "production": {
                    "voice_recording": f"{duration_minutes * 2} hours",
                    "music_composition": f"{duration_minutes * 3} hours",
                    "sfx_creation": f"{duration_minutes * 1.5} hours"
                },
                "post_production": {
                    "audio_editing": f"{duration_minutes * 2} hours",
                    "mixing": f"{duration_minutes * 2} hours",
                    "mastering": f"{duration_minutes * 1} hours"
                },
                "total_time": f"{duration_minutes * 11.5} hours",
                "parallel_tasks": [
                    "Voice recording + Music composition",
                    "SFX creation + Audio editing",
                    "Mixing + Quality review"
                ]
            }
            
            return timeline
            
        except Exception as e:
            logging.error(f"Audio timeline creation failed: {e}")
            return {}

    def _get_technical_requirements(self, niche: str) -> Dict[str, Any]:
        """Get technical requirements for niche"""
        try:
            base_requirements = {
                "sample_rate": 48000,
                "bit_depth": 24,
                "channels": 2,
                "format": "WAV",
                "compression": "Lossless"
            }
            
            niche_requirements = {
                "history": {
                    "sample_rate": 96000,  # Higher quality for historical content
                    "bit_depth": 24,
                    "channels": 2,
                    "format": "WAV",
                    "compression": "Lossless",
                    "special_notes": "High fidelity for archival quality"
                },
                "motivation": {
                    "sample_rate": 48000,
                    "bit_depth": 24,
                    "channels": 2,
                    "format": "WAV",
                    "compression": "Lossless",
                    "special_notes": "Clear voice reproduction"
                },
                "finance": {
                    "sample_rate": 48000,
                    "bit_depth": 24,
                    "channels": 2,
                    "format": "WAV",
                    "compression": "Lossless",
                    "special_notes": "Professional broadcast quality"
                },
                "automotive": {
                    "sample_rate": 48000,
                    "bit_depth": 24,
                    "channels": 2,
                    "format": "WAV",
                    "compression": "Lossless",
                    "special_notes": "Dynamic range for engine sounds"
                },
                "combat": {
                    "sample_rate": 48000,
                    "bit_depth": 24,
                    "channels": 2,
                    "format": "WAV",
                    "compression": "Lossless",
                    "special_notes": "High impact sound reproduction"
                }
            }
            
            return niche_requirements.get(niche, base_requirements)
            
        except Exception as e:
            logging.error(f"Technical requirements gathering failed: {e}")
            return {
                "sample_rate": 48000,
                "bit_depth": 24,
                "channels": 2,
                "format": "WAV",
                "compression": "Lossless"
            }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_voice_acting_engine(config: Optional[Dict] = None) -> AdvancedVoiceActingEngine:
    """Create and initialize Advanced Voice Acting Engine"""
    return AdvancedVoiceActingEngine(config)

def generate_character_voice(text: str, personality: CharacterPersonality, 
                           emotion: Emotion) -> VoiceActingResult:
    """Generate character voice with default settings"""
    engine = create_voice_acting_engine()
    return engine.create_character_voice(text, personality, emotion)

# =============================================================================
# TEST & DEMO
# =============================================================================

if __name__ == "__main__":
    print("🎭 Testing Advanced Voice Acting Engine...")
    
    # Create engine
    engine = create_voice_acting_engine()
    
    # Test character voice generation
    test_text = "Welcome to an incredible journey through the world of artificial intelligence."
    
    result = engine.create_character_voice(
        text=test_text,
        character_personality=CharacterPersonality.WISE,
        emotion=Emotion.INSPIRATIONAL,
        voice_style=VoiceStyle.AUTHORITATIVE
    )
    
    print(f"Voice Acting Result: {json.dumps(result.__dict__, indent=2, default=str)}")
    print("✅ Advanced Voice Acting Engine test completed!")

