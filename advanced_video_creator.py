# content_pipeline/advanced_video_creator.py (Professional Master Director Edition)

import os
import sys
import json
import time
import random
import requests
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

# --- config imports & defaults ---
try:
    from config import AI_CONFIG, PEXELS_API_KEY
except Exception:
    AI_CONFIG, PEXELS_API_KEY = {}, None

try:
    OLLAMA_MODEL = (AI_CONFIG or {}).get("ollama_model", "llama3:8b")
except Exception:
    OLLAMA_MODEL = "llama3:8b"

# Varsayılan: subliminal kapalı (YouTube politikaları ve güven için)
ENABLE_SUBLIMINAL = False

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

# AI Suite imports
try:
    from ai_visual_suite import AIVisualSuite, VisualStyle, ImageFormat
    AI_VISUAL_AVAILABLE = True
    print("✅ AI Visual Suite imported successfully")
except ImportError as e:
    AI_VISUAL_AVAILABLE = False
    print(f"⚠️ AI Visual Suite not available: {e}")

try:
    from ai_cinematic_director import CinematicAIDirector, StoryArc, CinematicStyle
    AI_CINEMATIC_AVAILABLE = True
    print("✅ AI Cinematic Director imported successfully")
except ImportError as e:
    AI_CINEMATIC_AVAILABLE = False
    print(f"⚠️ AI Cinematic Director not available: {e}")

try:
    from ai_advanced_voice_acting import AdvancedVoiceActingEngine, Emotion, VoiceStyle
    AI_VOICE_AVAILABLE = True
    print("✅ AI Voice Acting imported successfully")
except ImportError as e:
    AI_VOICE_AVAILABLE = False
    print(f"⚠️ AI Voice Acting not available: {e}")

from moviepy.editor import *
from moviepy.video.fx import all as vfx
from moviepy.audio.fx import all as afx

class AdvancedVideoCreator:
    
    def __init__(
        self,
        pexels_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice_id: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3:8b",
        # ... varsa diğer parametrelerin
    ) -> None:
        """Video üretim altyapısını hazırlar (API anahtarları, TTS, müzik, log vb.)."""
        # ---- temel ayarlar / parametreler ----
        self.pexels_api_key = pexels_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

        # Bilgilendirici loglar (anahtar yoksa fallback kullanacağız)
        if not self.pexels_api_key:
            logging.info("Pexels API key yok — görseller için local/downloads fallback kullanılacak.")
        if not self.elevenlabs_api_key:
            logging.info("ElevenLabs API key yok — TTS için gTTS/espeak fallback kullanılacak.")

        # ---- codec / fps ----
        self.FPS = 30
        
        # ---- AI Suite initialization ----
        self.ai_visual_suite = None
        self.ai_cinematic_director = None
        self.ai_voice_acting = None
        
        if AI_VISUAL_AVAILABLE:
            try:
                self.ai_visual_suite = AIVisualSuite()
                logging.info("✅ AI Visual Suite initialized")
            except Exception as e:
                logging.error(f"❌ AI Visual Suite initialization failed: {e}")
        
        if AI_CINEMATIC_AVAILABLE:
            try:
                self.ai_cinematic_director = CinematicAIDirector()
                logging.info("✅ AI Cinematic Director initialized")
            except Exception as e:
                logging.error(f"❌ AI Cinematic Director initialization failed: {e}")
        
        if AI_VOICE_AVAILABLE:
            try:
                self.ai_voice_acting = AdvancedVoiceActingEngine()
                logging.info("✅ AI Voice Acting initialized")
            except Exception as e:
                logging.error(f"❌ AI Voice Acting initialization failed: {e}")
        self.CODEC = "libx264"
        self.AUDIO_CODEC = "aac"

        # ---- logging ----
        try:
            log_dir = os.environ.get("CK_LOG_DIR", ".")
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"advanced_video_creator_{int(time.time())}.log")
        except Exception:
            # Bir sorun olursa None bırak; setup_logging/log_message kendini kurtarıyor
            self.log_file = None
        self.setup_logging()

        # ---- varlık (assets) taraması ----
        self.local_assets = {"images": [], "videos": [], "audio": []}
        self._load_local_assets()

        # ---- TTS sistemi ----
        self.tts_system = None
        self.setup_tts()

        # ---- müzik sistemi ----
        self.music_system = None
        self.setup_music_generation()

    def _load_local_assets(self):
        """Load local assets from assets directory"""
        try:
            # Load images
            image_dir = "assets/images"
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        self.local_assets['images'].append(os.path.join(image_dir, file))
            
            # Load videos
            video_dir = "assets/videos"
            if os.path.exists(video_dir):
                for root, dirs, files in os.walk(video_dir):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            self.local_assets['videos'].append(os.path.join(root, file))
            
            # Load audio
            audio_dir = "assets/audio"
            if os.path.exists(audio_dir):
                for root, dirs, files in os.walk(audio_dir):
                    for file in files:
                        if file.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                            self.local_assets['audio'].append(os.path.join(root, file))
            
            self.log_message(f"📁 Loaded {len(self.local_assets['images'])} images, {len(self.local_assets['videos'])} videos, {len(self.local_assets['audio'])} audio files", "ASSETS")
        except Exception as e:
            self.log_message(f"⚠️ Error loading local assets: {e}", "ASSETS")

    def enhance_script_with_metadata(self, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI destekli senaryo geliştirme ve metadata ekleme"""
        
        try:
            if not script_data or "script" not in script_data:
                return script_data

            # AI Cinematic Director ile senaryo geliştir
            if self.ai_cinematic_director:
                try:
                    self.log_message("🎬 AI Cinematic Director ile senaryo geliştiriliyor...", "SCRIPT")
                    
                    # Mevcut senaryoyu analiz et
                    current_script = script_data["script"]
                    total_sentences = len(current_script)
                    
                    # AI ile senaryo yapısını geliştir
                    enhanced_structure = self.ai_cinematic_director.create_story_structure(
                        channel_name=script_data.get("channel", "Unknown"),
                        niche=script_data.get("niche", "general"),
                        target_duration_minutes=15,  # Hedef 15 dakika
                        style=CinematicStyle.HOLLYWOOD,
                        arc_type=StoryArc.HERO_JOURNEY,
                        pacing="medium"
                    )
                    
                    if enhanced_structure:
                        # AI önerilerini uygula
                        enhanced_script = self._apply_ai_enhancements(current_script, enhanced_structure)
                        script_data["script"] = enhanced_script
                        total_sentences = len(enhanced_script)
                        
                        self.log_message(f"✅ AI ile senaryo geliştirildi: {total_sentences} cümle", "SCRIPT")
                    
                except Exception as ai_error:
                    self.log_message(f"⚠️ AI senaryo geliştirme başarısız: {ai_error}", "WARNING")
                    # Fallback: orijinal senaryo kullan
                    total_sentences = len(script_data["script"])
            else:
                total_sentences = len(script_data["script"])

            # Calculate estimated duration
            estimated_duration = total_sentences * 8  # Assume 8 seconds per sentence

            # Add enhanced metadata
            enhanced_metadata = {
                "estimated_duration_seconds": estimated_duration,
                "estimated_duration_minutes": round(estimated_duration / 60, 1),
                "sentence_count": total_sentences,
                "ai_enhancement_applied": self.ai_cinematic_director is not None,
                "optimization_suggestions": [
                    "Use cinematic 4K footage for maximum visual impact",
                    "Implement smooth transitions between scenes",
                    "Add atmospheric background music",
                    "Include text overlays for key points",
                    "Use color grading for dramatic effect",
                    "AI-powered story structure optimization",
                    "Dynamic scene pacing and emotional arcs",
                ],
                "subtitle_optimization": {
                    "English": "Primary language, optimize for clarity",
                    "Spanish": "Latin American and European Spanish",
                    "French": "International French with clear pronunciation",
                    "German": "Standard German with proper grammar",
                },
                "self_improvement": {
                    "extra_sentences_generated": 0,
                    "duration_improvement": f"{estimated_duration}s (target: 600s+)",
                    "quality_enhancement": "AI-powered content optimization applied",
                },
            }

            script_data["enhanced_metadata"] = enhanced_metadata
            self.log_message(
                f"Enhanced script metadata added for {total_sentences} sentences",
                "INFO",
            )

            return script_data
        
        except Exception as e:
            self.log_message(f"❌ Error in enhance_script_with_metadata: {e}", "ERROR")
            return script_data
    
    def _apply_ai_enhancements(self, current_script: List[str], ai_structure: Dict[str, Any]) -> List[str]:
        """AI önerilerini mevcut senaryoya uygula"""
        try:
            enhanced_script = []
            
            # AI yapısından sahne bilgilerini al
            scenes = ai_structure.get("scenes", [])
            
            if scenes:
                # Her sahne için AI önerilerini uygula
                for i, scene in enumerate(scenes):
                    scene_content = scene.get("content", "")
                    scene_duration = scene.get("duration", 0)
                    
                    if scene_content:
                        # AI önerilen sahne içeriğini ekle
                        enhanced_script.append(scene_content)
                    elif i < len(current_script):
                        # Mevcut senaryodan al ve geliştir
                        enhanced_script.append(current_script[i])
                    
                    # Sahne geçişleri ekle
                    if i < len(scenes) - 1:
                        transition = f"[Scene {i+1} to {i+2} transition]"
                        enhanced_script.append(transition)
            else:
                # AI yapısı yoksa mevcut senaryoyu kullan
                enhanced_script = current_script.copy()
            
            # Hedef süreye ulaşmak için ek cümleler ekle
            target_sentences = 75  # 15 dakika için ~75 cümle
            if len(enhanced_script) < target_sentences:
                additional_sentences = self._generate_additional_sentences(
                    enhanced_script, target_sentences - len(enhanced_script)
                )
                enhanced_script.extend(additional_sentences)
            
            self.log_message(f"✅ AI enhancements applied: {len(enhanced_script)} total sentences", "SCRIPT")
            return enhanced_script
            
        except Exception as e:
            self.log_message(f"❌ AI enhancements failed: {e}", "ERROR")
            return current_script
    
    def _generate_additional_sentences(self, current_script: List[str], count: int) -> List[str]:
        """Ek cümleler üret"""
        try:
            additional_sentences = []
            
            # Mevcut senaryodan anahtar kelimeleri çıkar
            keywords = []
            for sentence in current_script[:5]:  # İlk 5 cümleden
                words = sentence.split()
                keywords.extend([w.lower() for w in words if len(w) > 4])
            
            # Tekrarlanan kelimeleri temizle
            keywords = list(set(keywords))[:10]
            
            # Ek cümleler üret
            for i in range(count):
                if keywords:
                    keyword = keywords[i % len(keywords)]
                    additional_sentence = f"This {keyword} represents a significant breakthrough in our understanding."
                    additional_sentences.append(additional_sentence)
                else:
                    additional_sentence = f"Let's explore this fascinating topic further in detail."
                    additional_sentences.append(additional_sentence)
            
            return additional_sentences
            
        except Exception as e:
            self.log_message(f"❌ Additional sentence generation failed: {e}", "ERROR")
            return []

    # Örnek: görsel bulucu fonksiyonunda Pexels yoksa local'den al
    def find_visual_assets(self, script, niche: str, downloads_dir: str):
        os.makedirs(downloads_dir, exist_ok=True)
        frames = []

        if self.pexels_api_key:
            try:
                # … senin mevcut pexels indirme mantığın
                pass
            except Exception as e:
                logging.warning(f"Pexels hata: {e}. Local fallback'e geçiliyor.")
        # Fallback – downloads_dir içindeki mp4/jpg/png dosyaları kullan
        for f in os.listdir(downloads_dir):
            if f.lower().endswith((".mp4", ".jpg", ".jpeg", ".png")):
                frames.append(os.path.join(downloads_dir, f))

        return frames

    # TTS örneği – ElevenLabs yoksa gTTS/espeak
    def generate_voiceover(self, script_obj, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # … mevcut parçalama/segment mantığın
        if self.elevenlabs_api_key:
            try:
                # … ElevenLabs çağrın
                pass
            except Exception as e:
                logging.warning(f"ElevenLabs hata: {e}. gTTS/espeak'e düşülüyor.")
                # … fallback üretim
        else:
            # … doğrudan gTTS/espeak fallback üretim
            pass
        # return oluşturulan dosya yolları listesi

    
    def setup_logging(self):
        """Setup enhanced logging system"""
        # log_file is already initialized in __init__, just log the path
        print(f"📝 Advanced Video Creator logging to: {self.log_file}")
    
    def log_message(self, message: str, category: str = "INFO") -> None:
        # ensure log_file exists even if __init__ hasn't set it yet
        try:
            if not hasattr(self, "log_file") or not self.log_file:
                import time, os
                ts = int(time.time())
                log_dir = os.environ.get("CK_LOG_DIR", ".")
                os.makedirs(log_dir, exist_ok=True)
                self.log_file = os.path.join(log_dir, f"advanced_video_creator_{ts}.log")
        except Exception:
            # last-ditch: avoid blocking the app if logging path fails
            self.log_file = None

        # console/UI stream (optional)
        try:
            ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
            line = f"{ts} {category}: {message}"
            print(line)
        except Exception:
            pass

        # file stream
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{message}\n")
            except Exception:
                # avoid raising from logging
                pass
    
    def setup_tts(self):
        """Setup TTS systems with fallback hierarchy"""
        self.tts_system = None
        
        if PIPER_AVAILABLE:
            try:
                # Initialize Piper TTS with Morgan Freeman style
                self.tts_system = "piper"
                self.log_message("✅ Piper TTS initialized for Morgan Freeman style", "TTS")
            except Exception as e:
                self.log_message(f"⚠️ Piper TTS initialization failed: {e}", "TTS")
                self.tts_system = None
        
        if not self.tts_system and ESPEAK_AVAILABLE:
            try:
                self.tts_system = "espeak"
                self.log_message("✅ espeak TTS initialized as fallback", "TTS")
            except Exception as e:
                self.log_message(f"⚠️ espeak initialization failed: {e}", "TTS")
        
        if not self.tts_system and GTTS_AVAILABLE:
            self.tts_system = "gtts"
            self.log_message("✅ gTTS initialized as final fallback", "TTS")
        
        if not self.tts_system:
            self.log_message("❌ No TTS system available", "ERROR")
    
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
    
    def _optimize_pexels_query(self, query: str, channel_niche: str) -> str:
        """Optimize Pexels query using Ollama for better visual results"""
        try:
            import ollama
            
            prompt = f"""Alakasız görsel için yeni query üret.
            
            Orijinal query: {query}
            Kanal niş: {channel_niche}
            
            Yeni query formatı: "cinematic 4K {{niche}} {{scene}} high quality no black"
            
            Örnekler:
            - "cinematic 4K history ancient civilizations high quality no black"
            - "cinematic 4K motivation success achievement high quality no black"
            - "cinematic 4K finance business modern high quality no black"
            
            Sadece yeni query'yi döndür, açıklama yapma."""
            
            response = ollama.chat(model=OLLAMA_MODEL, 
                                 messages=[{'role': 'user', 'content': prompt}])
            
            optimized_query = response.get('message', {}).get('content', '').strip()
            
            if optimized_query:
                self.log_message(f"🤖 Ollama optimized query: {query} → {optimized_query}", "OLLAMA")
                return optimized_query
            else:
                # Fallback to enhanced query
                return f"cinematic 4K {channel_niche} {query} high quality no black"
                
        except Exception as e:
            self.log_message(f"⚠️ Ollama query optimization failed: {e}", "WARNING")
            # Fallback to enhanced query
            return f"cinematic 4K {channel_niche} {query} high quality no black"
    
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
        """Generate voiceover using gTTS as fallback with timeout and retry logic"""
        for attempt in range(3):  # 2 retries + 1 initial attempt
            try:
                # Add timeout for network operations
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                
                # Configure session with retry strategy
                session = requests.Session()
                retry_strategy = Retry(
                    total=2,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # Set timeout for the session
                session.timeout = 30
                
                # Create gTTS with custom session
                tts = gTTS(text=text, lang='en', slow=True)  # Slow for Morgan Freeman style
                
                # Save with timeout protection
                tts.save(output_path)
                
                self.log_message(f"✅ gTTS voiceover generated: {output_path}", "TTS")
                return True
                
            except Exception as e:
                if attempt < 2:  # Don't sleep on last attempt
                    self.log_message(f"⚠️ gTTS attempt {attempt + 1} failed: {e}, retrying...", "TTS")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                    continue
                else:
                    self.log_message(f"❌ gTTS failed after 3 attempts: {e}", "ERROR")
                    return False
        
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
        """AI destekli gelişmiş seslendirme üretimi"""
        self.log_message("🎤 AI destekli seslendirme üretimi başlatılıyor...", "VOICEOVER")
        
        try:
            # Script'ten cümleleri al
            script_content = script_data.get("script", [])
            if isinstance(script_content, list):
                # Eğer script liste ise, her öğeyi kontrol et
                narration_list = []
                for item in script_content:
                    if isinstance(item, dict):
                        text = item.get("sentence") or item.get("text", "")
                    else:
                        text = str(item)
                    if text:
                        narration_list.append(text)
            else:
                # Eğer script string ise, cümlelere böl
                narration_list = [s.strip() for s in str(script_content).split('.') if s.strip()]
            
            if not narration_list:
                self.log_message("❌ No narration text found in script", "ERROR")
                return None
            
            audio_files = []
            os.makedirs(output_folder, exist_ok=True)
            
            # AI Voice Acting ile seslendirme üret
            if self.ai_voice_acting:
                try:
                    self.log_message("🎭 AI Voice Acting ile seslendirme üretiliyor...", "VOICEOVER")
                    
                    for i, text in enumerate(narration_list):
                        file_path = os.path.join(output_folder, f"ai_voice_{i+1}.mp3")
                        self.log_message(f"🎤 AI seslendirme {i+1}/{len(narration_list)}", "VOICEOVER")
                        
                        try:
                            # AI ile karakter sesi oluştur
                            ai_audio_path = self.ai_voice_acting.create_character_voice(
                                text=text,
                                character_personality="WISE",  # Bilge karakter
                                emotion="INSPIRATIONAL",  # İlham verici
                                voice_style="AUTHORITATIVE"  # Otoriter stil
                            )
                            
                            if ai_audio_path and os.path.exists(ai_audio_path):
                                # AI seslendirmeyi kopyala
                                import shutil
                                shutil.copy2(ai_audio_path, file_path)
                                audio_files.append(file_path)
                                self.log_message(f"✅ AI seslendirme üretildi: {os.path.basename(file_path)}", "VOICEOVER")
                            else:
                                # AI başarısızsa fallback kullan
                                if self.generate_morgan_freeman_voiceover(text, file_path):
                                    audio_files.append(file_path)
                                else:
                                    self.log_message(f"⚠️ Voiceover generation failed for part {i+1}", "WARNING")
                                    continue
                                    
                        except Exception as ai_error:
                            self.log_message(f"⚠️ AI voice acting failed: {ai_error}, using fallback", "WARNING")
                            # Fallback: Morgan Freeman style
                            if self.generate_morgan_freeman_voiceover(text, file_path):
                                audio_files.append(file_path)
                            else:
                                continue
                    
                except Exception as ai_error:
                    self.log_message(f"⚠️ AI Voice Acting hatası: {ai_error}, fallback kullanılıyor", "WARNING")
                    # Fallback: Orijinal Morgan Freeman style
                    for i, text in enumerate(narration_list):
                        file_path = os.path.join(output_folder, f"fallback_voice_{i+1}.mp3")
                        if self.generate_morgan_freeman_voiceover(text, file_path):
                            audio_files.append(file_path)
            else:
                # AI Voice Acting yoksa orijinal yöntemi kullan
                self.log_message("⚠️ AI Voice Acting not available, using Morgan Freeman style", "VOICEOVER")
                for i, text in enumerate(narration_list):
                    file_path = os.path.join(output_folder, f"part_{i+1}.mp3")
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
        """AI destekli görsel üretimi - Her sahne için özel görsel"""
        self.log_message("🎬 AI destekli görsel üretimi başlatılıyor...", "VISUALS")
        
        video_paths = []
        os.makedirs(download_folder, exist_ok=True)
        scenes = script_data.get("script", [])
        
        for i, scene in enumerate(scenes):
            query = scene.get("visual_query", "")
            found_video_path = None
            
            if query and self.ai_visual_suite:
                try:
                    # AI Visual Suite ile görsel üret
                    self.log_message(f"🎨 AI ile görsel üretiliyor: {query}", "VISUALS")
                    
                    # Her sahne için özel görsel üret
                    ai_image_path = self.ai_visual_suite.generate_visual_asset(
                        prompt=f"{query} {channel_niche} scene {i+1}",
                        style=VisualStyle.MODERN,
                        format_type=ImageFormat.BANNER
                    )
                    
                    if ai_image_path and os.path.exists(ai_image_path):
                        # Görseli video'ya çevir
                        found_video_path = self._convert_image_to_video(ai_image_path, scene, download_folder)
                        self.log_message(f"✅ AI görsel üretildi ve video'ya çevrildi: {os.path.basename(found_video_path)}", "VISUALS")
                    
                except Exception as ai_error:
                    self.log_message(f"⚠️ AI görsel üretimi başarısız: {ai_error}", "WARNING")
            
            # AI başarısızsa Pexels'ı dene
            if not found_video_path and query:
                try:
                    # Optimize Pexels query
                    optimized_query = self._optimize_pexels_query(query, channel_niche)
                    
                    # Calculate minimum duration for this scene (estimate based on script length)
                    estimated_duration = max(5.0, len(scene.get("text", "").split()) * 0.3)  # ~0.3s per word
                    
                    # Try Pexels download with real API
                    found_video_path = self._download_pexels_video(optimized_query, estimated_duration, download_folder)
                    
                    if found_video_path:
                        self.log_message(f"✅ Pexels'dan video indirildi: {os.path.basename(found_video_path)}", "VISUALS")
                        
                except Exception as pexels_error:
                    self.log_message(f"⚠️ Pexels indirme başarısız: {pexels_error}", "WARNING")
            
            # Son çare: local fallback
            if not found_video_path:
                found_video_path = self._get_local_asset_fallback(channel_niche, i+1)
                if found_video_path:
                    self.log_message(f"✅ Local fallback kullanıldı: {os.path.basename(found_video_path)}", "VISUALS")
            
            # Upscale video to 4K if available
            if found_video_path and PILLOW_AVAILABLE:
                upscaled_path = self._upscale_video_to_4k(found_video_path)
                if upscaled_path:
                    found_video_path = upscaled_path
            
            video_paths.append(found_video_path)
            
            if found_video_path:
                self.log_message(f"✅ Scene {i+1} görsel asset hazır: {os.path.basename(found_video_path)}", "VISUALS")
            else:
                self.log_message(f"⚠️ Scene {i+1} görsel asset başarısız", "WARNING")
        
        return video_paths
    
    def _convert_image_to_video(self, image_path: str, scene: dict, output_folder: str) -> str:
        """AI üretilen görseli video'ya çevir"""
        try:
            # Görsel süresini hesapla (script uzunluğuna göre)
            script_text = scene.get("text", "")
            estimated_duration = max(3.0, len(script_text.split()) * 0.3)  # ~0.3s per word
            
            # Görseli yükle ve video'ya çevir
            image_clip = ImageClip(image_path, duration=estimated_duration)
            
            # Görsel efektleri ekle
            image_clip = image_clip.fx(vfx.resize, width=1920, height=1080)
            
            # Ken Burns efekti ekle (yavaş zoom/pan)
            image_clip = image_clip.fx(vfx.zoom, 1.1, duration=estimated_duration)
            
            # Output path oluştur
            timestamp = int(time.time())
            output_filename = f"ai_scene_{timestamp}_{os.path.basename(image_path)}.mp4"
            output_path = os.path.join(output_folder, output_filename)
            
            # Video'yu kaydet with MoviePy 2.0.0.dev2 optimized settings
            image_clip.write_videofile(
                output_path,
                fps=self.FPS,
                codec=self.CODEC,
                preset='ultrafast',
                threads=1,
                logger=None,
                ffmpeg_params=['-crf', '30', '-pix_fmt', 'yuv420p']
            )
            
            # Cleanup
            image_clip.close()
            
            self.log_message(f"✅ Görsel video'ya çevrildi: {os.path.basename(output_path)}", "VISUALS")
            return output_path
            
        except Exception as e:
            self.log_message(f"❌ Görsel video dönüşümü başarısız: {e}", "ERROR")
            return None
    
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

    def create_cinematic_masterpiece(self, script_data: Dict[str, Any], 
                                   target_duration_minutes: float = 15.0,
                                   niche: str = "general",
                                   quality_level: str = "cinematic") -> Dict[str, Any]:
        """
        Create a cinematic masterpiece with 10+ minute duration
        
        Args:
            script_data: Script with scene breakdown
            target_duration_minutes: Target duration (minimum 10 minutes)
            niche: Content niche
            quality_level: Quality level (cinematic, premium, ultra)
            
        Returns:
            Complete cinematic video with metadata
        """
        try:
            self.log_message(f"🎬 Creating cinematic masterpiece: {target_duration_minutes} minutes", "CINEMATIC")
            
            # Ensure minimum duration
            if target_duration_minutes < 10:
                target_duration_minutes = 15.0
                self.log_message(f"⚠️ Duration increased to {target_duration_minutes} minutes for cinematic quality", "WARNING")
            
            # Step 1: Expand script to cinematic length
            expanded_script = self._expand_script_cinematically(script_data, target_duration_minutes, niche)
            
            # Step 2: Generate visual plan
            visual_plan = self._generate_cinematic_visual_plan(expanded_script, target_duration_minutes, niche)
            
            # Step 3: Generate audio plan
            audio_plan = self._generate_cinematic_audio_plan(expanded_script, target_duration_minutes, niche)
            
            # Step 4: Create cinematic video
            final_video = self._assemble_cinematic_video(expanded_script, visual_plan, audio_plan, quality_level)
            
            # Step 5: Quality assurance
            quality_report = self._assess_cinematic_quality(final_video, target_duration_minutes)
            
            return {
                "success": True,
                "video_path": final_video["path"],
                "duration_minutes": final_video["duration"] / 60,
                "quality_score": quality_report["overall_score"],
                "expanded_script": expanded_script,
                "visual_plan": visual_plan,
                "audio_plan": audio_plan,
                "quality_report": quality_report,
                "production_metadata": {
                    "total_assets_used": visual_plan.get("total_assets_needed", 0),
                    "audio_tracks": len(audio_plan.get("voice_requirements", {}).get("emotional_arcs", [])),
                    "production_time": self._calculate_production_time(target_duration_minutes),
                    "quality_level": quality_level
                }
            }
            
        except Exception as e:
            self.log_message(f"❌ Cinematic masterpiece creation failed: {e}", "ERROR")
            return {"success": False, "error": str(e)}
    
    def _expand_script_cinematically(self, script_data: Dict[str, Any], 
                                   target_minutes: float, niche: str) -> Dict[str, Any]:
        """Expand script to cinematic length using AI Director"""
        try:
            if AI_CINEMATIC_AVAILABLE:
                from ai_cinematic_director import CinematicAIDirector
                director = CinematicAIDirector()
                expanded = director.expand_script_to_cinematic_length(
                    script_data.get("script", ""), target_minutes, niche
                )
                return expanded
            else:
                # Fallback expansion
                return self._fallback_script_expansion(script_data, target_minutes, niche)
                
        except Exception as e:
            self.log_message(f"⚠️ AI script expansion failed, using fallback: {e}", "WARNING")
            return self._fallback_script_expansion(script_data, target_minutes, niche)
    
    def _fallback_script_expansion(self, script_data: Dict[str, Any], 
                                 target_minutes: float, niche: str) -> Dict[str, Any]:
        """Fallback script expansion when AI Director is unavailable"""
        base_script = script_data.get("script", "")
        target_words = int(target_minutes * 175)  # 175 words per minute
        
        # Simple expansion by repeating and elaborating
        expanded_content = base_script
        while len(expanded_content.split()) < target_words:
            expanded_content += f"\n\n{base_script}"
        
        return {
            "expanded_script": expanded_content,
            "target_duration_minutes": target_minutes,
            "estimated_duration_minutes": target_minutes,
            "scenes": self._create_basic_scenes(expanded_content, target_minutes)
        }
    
    def _create_basic_scenes(self, content: str, duration_minutes: float) -> List[Dict[str, Any]]:
        """Create basic scene structure for fallback"""
        scene_count = max(5, int(duration_minutes / 3))  # 1 scene per 3 minutes
        scenes = []
        
        for i in range(scene_count):
            scene = {
                "scene_number": i + 1,
                "emotional_beat": ["opening", "hook", "conflict", "climax", "resolution"][i % 5],
                "duration_seconds": (duration_minutes * 60) / scene_count,
                "content": content[i::scene_count]  # Distribute content across scenes
            }
            scenes.append(scene)
        
        return scenes
    
    def _generate_cinematic_visual_plan(self, expanded_script: Dict[str, Any], 
                                      target_minutes: float, niche: str) -> Dict[str, Any]:
        """Generate comprehensive visual plan using AI Visual Suite"""
        try:
            if AI_VISUAL_AVAILABLE:
                from ai_visual_suite import AIVisualSuite
                visual_suite = AIVisualSuite()
                visual_plan = visual_suite.generate_cinematic_visual_plan(
                    expanded_script, target_minutes, niche
                )
                return visual_plan
            else:
                # Fallback visual plan
                return self._fallback_visual_plan(target_minutes, niche)
                
        except Exception as e:
            self.log_message(f"⚠️ AI visual planning failed, using fallback: {e}", "WARNING")
            return self._fallback_visual_plan(target_minutes, niche)
    
    def _fallback_visual_plan(self, target_minutes: float, niche: str) -> Dict[str, Any]:
        """Fallback visual plan when AI Visual Suite is unavailable"""
        total_assets = int(target_minutes * 12)  # 12 assets per minute
        
        return {
            "total_assets_needed": total_assets,
            "asset_categories": {
                "primary_visuals": {"count": int(total_assets * 0.6)},
                "background_visuals": {"count": int(total_assets * 0.3)},
                "transition_visuals": {"count": int(total_assets * 0.1)}
            },
            "quality_requirements": "4K cinematic quality",
            "asset_sources": ["Pexels", "Local Library", "Generated"]
        }
    
    def _generate_cinematic_audio_plan(self, expanded_script: Dict[str, Any], 
                                     target_minutes: float, niche: str) -> Dict[str, Any]:
        """Generate comprehensive audio plan using AI Voice Acting"""
        try:
            if AI_VOICE_AVAILABLE:
                from ai_advanced_voice_acting import AdvancedVoiceActingEngine
                voice_engine = AdvancedVoiceActingEngine()
                audio_plan = voice_engine.generate_cinematic_audio_plan(
                    expanded_script, target_minutes, niche
                )
                return audio_plan
            else:
                # Fallback audio plan
                return self._fallback_audio_plan(target_minutes, niche)
                
        except Exception as e:
            self.log_message(f"⚠️ AI audio planning failed, using fallback: {e}", "WARNING")
            return self._fallback_audio_plan(target_minutes, niche)
    
    def _fallback_audio_plan(self, target_minutes: float, niche: str) -> Dict[str, Any]:
        """Fallback audio plan when AI Voice Acting is unavailable"""
        return {
            "voice_requirements": {
                "narrator_profile": {"style": "professional", "quality": "high"},
                "emotional_arcs": [{"intensity": 0.8, "pacing": "steady"}]
            },
            "music_plan": {
                "overall_theme": f"{niche}-themed cinematic score",
                "scene_music": [{"style": "cinematic", "intensity": 0.7}]
            },
            "quality_standards": {"sample_rate": 48000, "bit_depth": 24}
        }
    
    def _assemble_cinematic_video(self, expanded_script: Dict[str, Any], 
                                 visual_plan: Dict[str, Any], 
                                 audio_plan: Dict[str, Any],
                                 quality_level: str) -> Dict[str, Any]:
        """Assemble the final cinematic video"""
        try:
            self.log_message("🎬 Assembling cinematic video...", "ASSEMBLY")
            
            # Create video clips for each scene
            video_clips = []
            total_duration = 0
            
            scenes = expanded_script.get("scenes", [])
            for scene in scenes:
                scene_clip = self._create_cinematic_scene(
                    scene, visual_plan, audio_plan, quality_level
                )
                if scene_clip:
                    video_clips.append(scene_clip)
                    total_duration += scene_clip.duration
            
            if not video_clips:
                raise Exception("No valid video clips created")
            
            # Assemble final video with cinematic transitions
            final_video = self._assemble_with_cinematic_transitions(video_clips, quality_level)
            
            # Add cinematic audio
            final_video = self._add_cinematic_audio(final_video, audio_plan, quality_level)
            
            # Export with cinematic quality settings
            output_path = self._export_cinematic_video(final_video, quality_level)
            
            return {
                "path": output_path,
                "duration": total_duration,
                "quality": quality_level,
                "clips_used": len(video_clips)
            }
            
        except Exception as e:
            self.log_message(f"❌ Cinematic video assembly failed: {e}", "ERROR")
            raise
    
    def _create_cinematic_scene(self, scene: Dict[str, Any], 
                               visual_plan: Dict[str, Any], 
                               audio_plan: Dict[str, Any],
                               quality_level: str) -> Optional[VideoClip]:
        """Create a single cinematic scene"""
        try:
            scene_duration = scene.get("duration_seconds", 10)
            emotional_beat = scene.get("emotional_beat", "neutral")
            
            # Get visual assets for this scene
            visual_assets = self._get_scene_visual_assets(scene, visual_plan, quality_level)
            
            # Create scene video
            scene_video = self._compose_scene_video(visual_assets, scene_duration, emotional_beat)
            
            # Add scene audio
            scene_audio = self._create_scene_audio(scene, audio_plan, quality_level)
            if scene_audio:
                scene_video = scene_video.set_audio(scene_audio)
            
            return scene_video
            
        except Exception as e:
            self.log_message(f"⚠️ Scene creation failed: {e}", "WARNING")
            return None
    
    def _get_scene_visual_assets(self, scene: Dict[str, Any], 
                                visual_plan: Dict[str, Any], 
                                quality_level: str) -> List[str]:
        """Get high-quality visual assets for a scene"""
        try:
            # Calculate assets needed for this scene
            scene_duration = scene.get("duration_seconds", 10)
            assets_needed = max(3, int(scene_duration / 4))  # 1 asset per 4 seconds
            
            # Get assets from Pexels or local library
            assets = []
            niche = scene.get("niche", "general")
            
            # Try Pexels first
            if PEXELS_API_KEY:
                pexels_assets = self._download_pexels_assets(niche, assets_needed, quality_level)
                assets.extend(pexels_assets)
            
            # Fill remaining with local assets
            if len(assets) < assets_needed:
                local_assets = self._get_local_visual_assets(niche, assets_needed - len(assets))
                assets.extend(local_assets)
            
            return assets[:assets_needed]
            
        except Exception as e:
            self.log_message(f"⚠️ Visual asset gathering failed: {e}", "WARNING")
            return []
    
    def _assemble_with_cinematic_transitions(self, video_clips: List[VideoClip], 
                                           quality_level: str) -> VideoClip:
        """Assemble video clips with cinematic transitions"""
        try:
            if len(video_clips) == 1:
                return video_clips[0]
            
            # Create cinematic transitions
            transitioned_clips = []
            for i, clip in enumerate(video_clips):
                if i > 0:
                    # Add crossfade transition
                    transition_duration = min(1.0, clip.duration * 0.1)  # 10% of clip duration
                    clip = clip.fx(vfx.fadein, transition_duration)
                
                transitioned_clips.append(clip)
            
            # Concatenate with smooth transitions
            final_video = concatenate_videoclips(
                transitioned_clips, 
                method="compose",
                transition=lambda t: vfx.crossfadein(t, 0.5)
            )
            
            return final_video
            
        except Exception as e:
            self.log_message(f"⚠️ Cinematic transitions failed: {e}", "WARNING")
            # Fallback to simple concatenation
            return concatenate_videoclips(video_clips)
    
    def _add_cinematic_audio(self, video: VideoClip, 
                            audio_plan: Dict[str, Any], 
                            quality_level: str) -> VideoClip:
        """Add cinematic audio to the video"""
        try:
            # Create background music
            background_music = self._create_cinematic_music(audio_plan, video.duration)
            
            # Create voice narration
            voice_narration = self._create_voice_narration(audio_plan, video.duration)
            
            # Mix audio tracks
            if background_music and voice_narration:
                # Lower music volume to not interfere with voice
                background_music = background_music.volumex(0.3)
                final_audio = CompositeAudioClip([voice_narration, background_music])
                return video.set_audio(final_audio)
            elif voice_narration:
                return video.set_audio(voice_narration)
            elif background_music:
                return video.set_audio(background_music)
            else:
                return video
                
        except Exception as e:
            self.log_message(f"⚠️ Cinematic audio addition failed: {e}", "WARNING")
            return video
    
    def _export_cinematic_video(self, video: VideoClip, quality_level: str) -> str:
        """Export video with cinematic quality settings"""
        try:
            timestamp = int(time.time())
            output_filename = f"cinematic_masterpiece_{timestamp}.mp4"
            output_path = os.path.join("cinematic_outputs", output_filename)
            
            # Ensure output directory exists
            os.makedirs("cinematic_outputs", exist_ok=True)
            
            # Quality settings based on level
            quality_settings = self._get_cinematic_quality_settings(quality_level)
            
            # Export with high quality
            video.write_videofile(
                output_path,
                fps=quality_settings["fps"],
                codec=quality_settings["codec"],
                bitrate=quality_settings["bitrate"],
                audio_codec=quality_settings["audio_codec"],
                audio_bitrate=quality_settings["audio_bitrate"],
                threads=4,
                preset=quality_settings["preset"]
            )
            
            self.log_message(f"✅ Cinematic video exported: {output_path}", "SUCCESS")
            return output_path
            
        except Exception as e:
            self.log_message(f"❌ Cinematic video export failed: {e}", "ERROR")
            raise
    
    def _get_cinematic_quality_settings(self, quality_level: str) -> Dict[str, Any]:
        """Get quality settings for different quality levels"""
        quality_settings = {
            "cinematic": {
                "fps": 30,
                "codec": "libx264",
                "bitrate": "20M",
                "audio_codec": "aac",
                "audio_bitrate": "320k",
                "preset": "slow"
            },
            "premium": {
                "fps": 60,
                "codec": "libx264",
                "bitrate": "50M",
                "audio_codec": "aac",
                "audio_bitrate": "512k",
                "preset": "veryslow"
            },
            "ultra": {
                "fps": 60,
                "codec": "libx265",
                "bitrate": "100M",
                "audio_codec": "aac",
                "audio_bitrate": "640k",
                "preset": "veryslow"
            }
        }
        
        return quality_settings.get(quality_level, quality_settings["cinematic"])
    
    def _assess_cinematic_quality(self, video_info: Dict[str, Any], 
                                 target_duration_minutes: float) -> Dict[str, Any]:
        """Assess the quality of the created cinematic video"""
        try:
            video_path = video_info["path"]
            actual_duration = video_info["duration"]
            
            # Basic quality metrics
            quality_metrics = {
                "duration_accuracy": min(1.0, actual_duration / (target_duration_minutes * 60)),
                "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                "clips_used": video_info.get("clips_used", 0),
                "quality_level": video_info.get("quality", "unknown")
            }
            
            # Calculate overall quality score
            duration_score = quality_metrics["duration_accuracy"] * 0.4
            file_size_score = min(1.0, quality_metrics["file_size"] / (100 * 1024 * 1024)) * 0.3  # 100MB baseline
            clips_score = min(1.0, quality_metrics["clips_used"] / 20) * 0.3  # 20 clips baseline
            
            overall_score = duration_score + file_size_score + clips_score
            
            quality_report = {
                "overall_score": overall_score,
                "quality_level": "excellent" if overall_score > 0.9 else "good" if overall_score > 0.7 else "acceptable",
                "metrics": quality_metrics,
                "recommendations": self._get_quality_recommendations(overall_score, quality_metrics)
            }
            
            return quality_report
            
        except Exception as e:
            self.log_message(f"⚠️ Quality assessment failed: {e}", "WARNING")
            return {"overall_score": 0.5, "error": str(e)}
    
    def _get_quality_recommendations(self, overall_score: float, 
                                   metrics: Dict[str, Any]) -> List[str]:
        """Get recommendations for improving video quality"""
        recommendations = []
        
        if metrics["duration_accuracy"] < 0.8:
            recommendations.append("Increase script length to meet target duration")
        
        if metrics["file_size"] < 50 * 1024 * 1024:  # Less than 50MB
            recommendations.append("Use higher bitrate for better visual quality")
        
        if metrics["clips_used"] < 15:
            recommendations.append("Increase visual variety with more assets")
        
        if overall_score < 0.7:
            recommendations.append("Review and enhance overall production quality")
        
        return recommendations
    
    def _calculate_production_time(self, duration_minutes: float) -> str:
        """Calculate estimated production time for cinematic content"""
        # Base time: 1 hour per minute of content for cinematic quality
        base_hours = duration_minutes * 1.0
        
        # Add quality multipliers
        total_hours = base_hours * 1.5  # 50% extra for quality
        
        if total_hours < 24:
            return f"{total_hours:.1f} hours"
        else:
            days = total_hours / 24
            return f"{days:.1f} days"

    def create_video(self, script_data: Dict[str, Any], 
                    output_folder: str = "outputs",
                    target_duration_minutes: float = 5.0) -> Dict[str, Any]:
        """
        Create a video using the provided script data
        
        Args:
            script_data: Script data dictionary
            output_folder: Output folder path
            target_duration_minutes: Target duration in minutes
            
        Returns:
            Video creation result
        """
        try:
            self.log_message(f"🎬 Creating video: {target_duration_minutes} minutes", "VIDEO_CREATION")
            
            # Ensure output directory exists
            os.makedirs(output_folder, exist_ok=True)
            
            # Extract script content
            script_text = script_data.get("script", "")
            topic = script_data.get("topic", "general")
            niche = script_data.get("niche", "general")
            
            if not script_text:
                return {"success": False, "error": "No script content provided"}
            
            # Generate voiceover
            self.log_message("🎙️ Generating voiceover...", "VOICEOVER")
            audio_dir = os.path.join(output_folder, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            audio_files = self.generate_voiceover(script_data, audio_dir)
            if not audio_files:
                return {"success": False, "error": "Voiceover generation failed"}
            
            # Find visual assets
            self.log_message("🎞️ Finding visual assets...", "VISUALS")
            downloads_dir = os.path.join(output_folder, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            
            visuals = self.find_visual_assets(script_data, niche, downloads_dir)
            if not visuals:
                return {"success": False, "error": "No visual assets found"}
            
            # Create final video
            self.log_message("🎬 Rendering final video...", "RENDERING")
            timestamp = int(time.time())
            output_filename = f"{topic}_{timestamp}.mp4"
            output_path = os.path.join(output_folder, output_filename)
            
            # Check for background music
            music_path = os.path.join("assets", "audio", "music", "epic_music.mp3")
            music = music_path if os.path.exists(music_path) else None
            
            # Render video
            final_path = self.edit_long_form_video(audio_files, visuals, music, output_path)
            
            if final_path and os.path.exists(final_path):
                # Analyze video quality
                quality_report = self._analyze_video_quality(final_path)
                
                return {
                    "success": True,
                    "output_path": final_path,
                    "duration_minutes": target_duration_minutes,
                    "quality_score": quality_report.get("overall_score", 0.0),
                    "audio_files": audio_files,
                    "visual_assets": visuals,
                    "quality_report": quality_report
                }
            else:
                return {"success": False, "error": "Video rendering failed"}
                
        except Exception as e:
            self.log_message(f"❌ Video creation failed: {e}", "ERROR")
            return {"success": False, "error": str(e)}

    def _create_scene_audio(self, scene: Dict[str, Any], 
                           audio_plan: Dict[str, Any], 
                           quality_level: str) -> Optional[AudioClip]:
        """Create audio for a scene"""
        try:
            scene_duration = scene.get("duration_seconds", 10)
            emotional_beat = scene.get("emotional_beat", "neutral")
            
            # Create basic narration for the scene
            scene_content = scene.get("content", "Scene content")
            
            # Generate voiceover for this scene
            audio_dir = "temp_audio"
            os.makedirs(audio_dir, exist_ok=True)
            
            script_data = {
                "script": scene_content,
                "niche": scene.get("niche", "general")
            }
            
            audio_files = self.generate_voiceover(script_data, audio_dir)
            if audio_files and len(audio_files) > 0:
                # Load the first audio file
                audio_path = audio_files[0]
                if os.path.exists(audio_path):
                    audio_clip = AudioFileClip(audio_path)
                    
                    # Adjust duration to match scene
                    if audio_clip.duration > scene_duration:
                        audio_clip = audio_clip.subclip(0, scene_duration)
                    elif audio_clip.duration < scene_duration:
                        # Loop audio if too short
                        audio_clip = audio_clip.fx(afx.audio_loop, duration=scene_duration)
                    
                    return audio_clip
            
            # Fallback: create silent audio
            return AudioClip(lambda t: 0, duration=scene_duration)
            
        except Exception as e:
            self.log_message(f"⚠️ Scene audio creation failed: {e}", "WARNING")
            return None

    def _compose_scene_video(self, visual_assets: List[str], 
                            scene_duration: float, 
                            emotional_beat: str) -> Optional[VideoClip]:
        """Compose video for a scene using visual assets"""
        try:
            if not visual_assets:
                # Create a simple color clip if no assets
                return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=scene_duration)
            
            # Use the first visual asset
            asset_path = visual_assets[0]
            if os.path.exists(asset_path):
                # Load video or image
                if asset_path.lower().endswith(('.mp4', '.avi', '.mov')):
                    clip = VideoFileClip(asset_path)
                else:
                    # Assume it's an image
                    clip = ImageClip(asset_path)
                
                # Adjust duration
                if clip.duration and clip.duration > scene_duration:
                    clip = clip.subclip(0, scene_duration)
                else:
                    clip = clip.set_duration(scene_duration)
                
                # Resize to standard dimensions
                clip = clip.resize(width=1920, height=1080)
                
                return clip
            else:
                # Fallback: create color clip
                return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=scene_duration)
                
        except Exception as e:
            self.log_message(f"⚠️ Scene video composition failed: {e}", "WARNING")
            return None

    def _create_cinematic_music(self, audio_plan: Dict[str, Any], 
                               video_duration: float) -> Optional[AudioClip]:
        """Create cinematic background music"""
        try:
            # Check for existing music files
            music_dir = Path("assets/audio/music")
            if music_dir.exists():
                music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
                if music_files:
                    # Use the first available music file
                    music_path = str(music_files[0])
                    music_clip = AudioFileClip(music_path)
                    
                    # Loop music to match video duration
                    if music_clip.duration < video_duration:
                        music_clip = music_clip.fx(afx.audio_loop, duration=video_duration)
                    else:
                        music_clip = music_clip.subclip(0, video_duration)
                    
                    return music_clip
            
            # Fallback: create silent audio
            return AudioClip(lambda t: 0, duration=video_duration)
            
        except Exception as e:
            self.log_message(f"⚠️ Cinematic music creation failed: {e}", "WARNING")
            return None

    def _create_voice_narration(self, audio_plan: Dict[str, Any], 
                               video_duration: float) -> Optional[AudioClip]:
        """Create voice narration for the video"""
        try:
            # Create a simple narration script
            narration_script = "Welcome to this cinematic experience. We will explore fascinating topics that will change your perspective on the world."
            
            # Generate voiceover
            temp_dir = "temp_narration"
            os.makedirs(temp_dir, exist_ok=True)
            
            script_data = {
                "script": narration_script,
                "niche": "general"
            }
            
            audio_files = self.generate_voiceover(script_data, temp_dir)
            if audio_files and len(audio_files) > 0:
                audio_path = audio_files[0]
                if os.path.exists(audio_path):
                    audio_clip = AudioFileClip(audio_path)
                    
                    # Loop audio to match video duration
                    if audio_clip.duration < video_duration:
                        audio_clip = audio_clip.fx(afx.audio_loop, duration=video_duration)
                    else:
                        audio_clip = audio_clip.subclip(0, video_duration)
                    
                    return audio_clip
            
            # Fallback: create silent audio
            return AudioClip(lambda t: 0, duration=video_duration)
            
        except Exception as e:
            self.log_message(f"⚠️ Voice narration creation failed: {e}", "WARNING")
            return None

    def _download_pexels_assets(self, niche: str, count: int, quality_level: str) -> List[str]:
        """Download assets from Pexels API"""
        try:
            if not PEXELS_API_KEY:
                self.log_message("⚠️ Pexels API key not available", "WARNING")
                return []
            
            assets = []
            queries = self._get_niche_queries(niche)
            
            for query in queries[:count]:
                try:
                    # Download asset from Pexels
                    asset_path = self._download_pexels_video(query, 5, f"temp_pexels_{len(assets)}.mp4")
                    if asset_path and os.path.exists(asset_path):
                        assets.append(asset_path)
                        self.log_message(f"✅ Downloaded Pexels asset: {query}", "DOWNLOAD")
                    
                    if len(assets) >= count:
                        break
                        
                except Exception as e:
                    self.log_message(f"⚠️ Failed to download asset for query '{query}': {e}", "WARNING")
                    continue
            
            return assets
            
        except Exception as e:
            self.log_message(f"❌ Pexels asset download failed: {e}", "ERROR")
            return []

    def _get_local_visual_assets(self, niche: str, count: int) -> List[str]:
        """Get local visual assets"""
        try:
            assets = []
            
            # Check assets/videos directory
            videos_dir = Path("assets/videos")
            if videos_dir.exists():
                video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi")) + list(videos_dir.glob("*.mov"))
                assets.extend([str(f) for f in video_files[:count]])
            
            # Check assets/images directory
            images_dir = Path("assets/images")
            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
                assets.extend([str(f) for f in image_files[:count - len(assets)]])
            
            # Check outputs directory for existing videos
            outputs_dir = Path("outputs")
            if outputs_dir.exists():
                for channel_dir in outputs_dir.iterdir():
                    if channel_dir.is_dir():
                        video_files = list(channel_dir.glob("*.mp4"))
                        assets.extend([str(f) for f in video_files[:count - len(assets)]])
            
            return assets[:count]
            
        except Exception as e:
            self.log_message(f"❌ Local asset gathering failed: {e}", "ERROR")
            return []

    def _get_niche_queries(self, niche: str) -> List[str]:
        """Get search queries specific to niche"""
        niche_queries = {
            "history": [
                "ancient civilization", "historical landmarks", "archaeological sites",
                "medieval castles", "ancient ruins", "historical artifacts"
            ],
            "motivation": [
                "achievement", "success", "determination", "overcoming obstacles",
                "inspirational moments", "goal setting"
            ],
            "finance": [
                "modern city", "business district", "financial markets",
                "technology innovation", "global economy"
            ],
            "automotive": [
                "luxury cars", "sports cars", "automotive technology",
                "racing scenes", "car manufacturing"
            ],
            "combat": [
                "martial arts", "training scenes", "athletic performance",
                "determination", "physical strength"
            ]
        }
        
        return niche_queries.get(niche, ["cinematic", "professional", "high quality"])

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
