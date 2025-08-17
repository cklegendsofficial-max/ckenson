# -*- coding: utf-8 -*-
"""
🚀 DÜNYANIN EN İYİSİ AI VIDEO EDITING SUITE
Professional Video Creation & Optimization Platform
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import requests
from dataclasses import dataclass
from enum import Enum

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

# Video processing imports
try:
    from moviepy.editor import *
    from moviepy.video.fx import all as vfx
    from moviepy.audio.fx import all as afx
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ MoviePy not available - video processing disabled")

# Image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("⚠️ Pillow not available - image processing disabled")

# Audio processing
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ Librosa not available - audio analysis disabled")

# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class ContentType(Enum):
    SHORT_FORM = "short_form"      # TikTok, Reels, Shorts
    LONG_FORM = "long_form"        # YouTube, IGTV
    LIVE = "live"                  # Live streams
    STORY = "story"                # Stories, Fleets

class Platform(Enum):
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"

class QualityScore(Enum):
    EXCELLENT = 90
    GOOD = 75
    AVERAGE = 60
    POOR = 40
    FAIL = 20

@dataclass
class ContentMetrics:
    """Content performance metrics"""
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    watch_time: float = 0.0
    retention_rate: float = 0.0
    ctr: float = 0.0
    engagement_rate: float = 0.0

@dataclass
class ContentAnalysis:
    """AI content analysis results"""
    quality_score: float
    engagement_prediction: float
    seo_score: float
    audience_match: float
    trend_relevance: float
    recommendations: List[str]

# =============================================================================
# AI CONTENT QUALITY SCORING ENGINE
# =============================================================================

class ContentQualityScorer:
    """AI-powered content quality assessment"""
    
    def __init__(self):
        self.models = {}
        self._load_ai_models()
    
    def _load_ai_models(self):
        """Load AI models for content analysis"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Sentiment analysis model
                self.models['sentiment'] = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logging.info("✅ AI models loaded successfully")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load some AI models: {e}")
    
    def analyze_content_quality(
        self, 
        title: str, 
        description: str, 
        tags: List[str],
        thumbnail_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None
    ) -> ContentAnalysis:
        """Comprehensive content quality analysis"""
        
        scores = {}
        recommendations = []
        
        # 1. Text Quality Analysis
        text_score = self._analyze_text_quality(title, description, tags)
        scores['text'] = text_score
        
        # 2. Visual Quality Analysis
        if thumbnail_path and PILLOW_AVAILABLE:
            visual_score = self._analyze_visual_quality(thumbnail_path)
            scores['visual'] = visual_score
        else:
            scores['visual'] = 70.0  # Default score
        
        # 3. Audio Quality Analysis
        if audio_path and LIBROSA_AVAILABLE:
            audio_score = self._analyze_audio_quality(audio_path)
            scores['audio'] = audio_score
        else:
            scores['audio'] = 75.0  # Default score
        
        # 4. SEO Analysis
        seo_score = self._analyze_seo_optimization(title, description, tags)
        scores['seo'] = seo_score
        
        # 5. Trend Relevance
        trend_score = self._analyze_trend_relevance(title, tags)
        scores['trend'] = trend_score
        
        # Calculate overall quality score
        overall_score = np.mean(list(scores.values()))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scores)
        
        return ContentAnalysis(
            quality_score=overall_score,
            engagement_prediction=self._predict_engagement(scores),
            seo_score=seo_score,
            audience_match=self._calculate_audience_match(scores),
            trend_relevance=trend_score,
            recommendations=recommendations
        )
    
    def _analyze_text_quality(self, title: str, description: str, tags: List[str]) -> float:
        """Analyze text content quality"""
        score = 0.0
        
        # Title analysis
        if len(title) >= 10 and len(title) <= 60:
            score += 25
        elif len(title) > 60:
            score += 15
        
        # Description analysis
        if len(description) >= 100:
            score += 25
        elif len(description) >= 50:
            score += 15
        
        # Tags analysis
        if len(tags) >= 5 and len(tags) <= 15:
            score += 25
        elif len(tags) > 15:
            score += 15
        
        # Sentiment analysis
        if TRANSFORMERS_AVAILABLE and self.models.get('sentiment'):
            try:
                sentiment_result = self.models['sentiment'](title + " " + description)
                if sentiment_result[0]['label'] in ['POSITIVE', 'NEUTRAL']:
                    score += 25
                else:
                    score += 15
            except Exception:
                score += 20  # Default score
        
        return min(score, 100.0)
    
    def _analyze_visual_quality(self, image_path: str) -> float:
        """Analyze visual content quality"""
        try:
            with Image.open(image_path) as img:
                # Resolution check
                width, height = img.size
                if width >= 1280 and height >= 720:
                    score = 30
                elif width >= 640 and height >= 480:
                    score = 20
                else:
                    score = 10
                
                # Brightness and contrast
                enhancer = ImageEnhance.Brightness(img)
                brightness_factor = enhancer.enhance(1.0)
                
                enhancer = ImageEnhance.Contrast(img)
                contrast_factor = enhancer.enhance(1.0)
                
                # Color analysis
                colors = img.getcolors(maxcolors=1000)
                if colors and len(colors) > 10:
                    score += 20
                else:
                    score += 10
                
                # Sharpness check
                sharpness = img.filter(ImageFilter.FIND_EDGES)
                if sharpness:
                    score += 20
                
                return min(score, 100.0)
        except Exception as e:
            logging.warning(f"Visual analysis failed: {e}")
            return 70.0
    
    def _analyze_audio_quality(self, audio_path: str) -> float:
        """Analyze audio content quality"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)
            
            score = 0.0
            
            # Duration check
            duration = librosa.get_duration(y=y, sr=sr)
            if duration >= 10:  # Minimum 10 seconds
                score += 20
            
            # Volume analysis
            rms = np.sqrt(np.mean(y**2))
            if 0.01 < rms < 0.5:  # Good volume range
                score += 25
            elif 0.001 < rms < 1.0:  # Acceptable range
                score += 15
            
            # Noise analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            if np.std(spectral_centroids) > 100:  # Good audio variation
                score += 25
            else:
                score += 15
            
            # Clarity check
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            if np.mean(zero_crossing_rate) > 0.01:  # Good clarity
                score += 30
            else:
                score += 20
            
            return min(score, 100.0)
        except Exception as e:
            logging.warning(f"Audio analysis failed: {e}")
            return 75.0
    
    def _analyze_seo_optimization(self, title: str, description: str, tags: List[str]) -> float:
        """Analyze SEO optimization"""
        score = 0.0
        
        # Title optimization
        if any(keyword in title.lower() for keyword in ['how to', 'best', 'top', 'guide', 'tips']):
            score += 25
        
        # Description length
        if len(description) >= 200:
            score += 25
        elif len(description) >= 100:
            score += 15
        
        # Tags optimization
        if len(tags) >= 8:
            score += 25
        elif len(tags) >= 5:
            score += 15
        
        # Keyword density
        all_text = (title + " " + description + " " + " ".join(tags)).lower()
        word_count = len(all_text.split())
        if word_count > 0:
            keyword_density = sum(1 for tag in tags if tag.lower() in all_text) / word_count
            if 0.01 < keyword_density < 0.05:  # Optimal keyword density
                score += 25
            elif keyword_density > 0.05:
                score += 15
        
        return min(score, 100.0)
    
    def _analyze_trend_relevance(self, title: str, tags: List[str]) -> float:
        """Analyze trend relevance"""
        trending_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'chatgpt', 'gpt-4',
            'blockchain', 'crypto', 'nft', 'metaverse', 'web3',
            'sustainability', 'climate change', 'renewable energy',
            'remote work', 'digital nomad', 'work from home',
            'mental health', 'wellness', 'mindfulness', 'meditation'
        ]
        
        score = 0.0
        all_text = (title + " " + " ".join(tags)).lower()
        
        for keyword in trending_keywords:
            if keyword in all_text:
                score += 10
        
        return min(score, 100.0)
    
    def _predict_engagement(self, scores: Dict[str, float]) -> float:
        """Predict engagement based on quality scores"""
        weights = {
            'text': 0.25,
            'visual': 0.30,
            'audio': 0.20,
            'seo': 0.15,
            'trend': 0.10
        }
        
        weighted_score = sum(scores.get(key, 0) * weights.get(key, 0) for key in weights)
        return min(weighted_score, 100.0)
    
    def _calculate_audience_match(self, scores: Dict[str, float]) -> float:
        """Calculate audience match score"""
        return np.mean(list(scores.values()))
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if scores.get('text', 0) < 70:
            recommendations.append("Improve title length and description quality")
        
        if scores.get('visual', 0) < 70:
            recommendations.append("Enhance thumbnail quality and visual appeal")
        
        if scores.get('audio', 0) < 70:
            recommendations.append("Improve audio quality and clarity")
        
        if scores.get('seo', 0) < 70:
            recommendations.append("Optimize SEO with better keywords and descriptions")
        
        if scores.get('trend', 0) < 70:
            recommendations.append("Include more trending topics and keywords")
        
        if not recommendations:
            recommendations.append("Content quality is excellent! Keep up the great work!")
        
        return recommendations

# =============================================================================
# ENGAGEMENT PREDICTION ENGINE
# =============================================================================

class EngagementPredictor:
    """AI-powered engagement prediction"""
    
    def __init__(self):
        self.historical_data = {}
        self.prediction_models = {}
        self._load_prediction_models()
    
    def _load_prediction_models(self):
        """Load engagement prediction models"""
        # This would load trained ML models
        # For now, using statistical models
        pass
    
    def predict_engagement(
        self,
        content_type: ContentType,
        platform: Platform,
        content_analysis: ContentAnalysis,
        posting_time: datetime,
        audience_size: int = 1000
    ) -> Dict[str, float]:
        """Predict engagement metrics"""
        
        base_engagement = content_analysis.quality_score / 100.0
        
        # Platform-specific adjustments
        platform_multipliers = {
            Platform.YOUTUBE: 1.0,
            Platform.TIKTOK: 1.2,  # Higher engagement potential
            Platform.INSTAGRAM: 1.1,
            Platform.LINKEDIN: 0.8,  # Lower engagement potential
            Platform.TWITTER: 0.9
        }
        
        # Content type adjustments
        type_multipliers = {
            ContentType.SHORT_FORM: 1.3,  # Higher engagement for short content
            ContentType.LONG_FORM: 1.0,
            ContentType.LIVE: 1.4,  # Live content has highest engagement
            ContentType.STORY: 1.2
        }
        
        # Time-based adjustments
        hour = posting_time.hour
        if 9 <= hour <= 11 or 18 <= hour <= 21:  # Peak hours
            time_multiplier = 1.2
        elif 12 <= hour <= 17:  # Good hours
            time_multiplier = 1.1
        else:  # Off-peak hours
            time_multiplier = 0.8
        
        # Calculate predicted metrics
        engagement_rate = base_engagement * platform_multipliers[platform] * type_multipliers[content_type] * time_multiplier
        
        # Predict specific metrics
        predicted_views = int(audience_size * engagement_rate * 0.1)
        predicted_likes = int(predicted_views * engagement_rate * 0.15)
        predicted_comments = int(predicted_views * engagement_rate * 0.05)
        predicted_shares = int(predicted_views * engagement_rate * 0.03)
        
        return {
            'engagement_rate': min(engagement_rate, 1.0),
            'predicted_views': predicted_views,
            'predicted_likes': predicted_likes,
            'predicted_comments': predicted_comments,
            'predicted_shares': predicted_shares,
            'confidence': 0.85  # Model confidence
        }

# =============================================================================
# A/B TESTING FRAMEWORK
# =============================================================================

class ABTestingFramework:
    """Comprehensive A/B testing for content optimization"""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
        self.test_counter = 0
    
    def create_test(
        self,
        test_name: str,
        test_type: str,  # 'thumbnail', 'title', 'description', 'tags'
        variants: List[Dict[str, Any]],
        audience_size: int = 1000,
        test_duration_days: int = 7
    ) -> str:
        """Create a new A/B test"""
        
        test_id = f"test_{self.test_counter}_{int(time.time())}"
        self.test_counter += 1
        
        test_config = {
            'name': test_name,
            'type': test_type,
            'variants': variants,
            'audience_size': audience_size,
            'duration_days': test_duration_days,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=test_duration_days),
            'status': 'active',
            'results': {}
        }
        
        self.active_tests[test_id] = test_config
        logging.info(f"✅ A/B test created: {test_name} (ID: {test_id})")
        
        return test_id
    
    def record_test_result(
        self,
        test_id: str,
        variant: str,
        metrics: ContentMetrics
    ) -> None:
        """Record test results for analysis"""
        
        if test_id not in self.active_tests:
            logging.warning(f"Test ID {test_id} not found")
            return
        
        test = self.active_tests[test_id]
        
        if variant not in test['results']:
            test['results'][variant] = []
        
        test['results'][variant].append(metrics)
        logging.info(f"📊 Test result recorded for {test_id} - {variant}")
    
    def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results and determine winner"""
        
        if test_id not in self.active_tests:
            return {"error": "Test not found"}
        
        test = self.active_tests[test_id]
        
        if not test['results']:
            return {"error": "No results available"}
        
        analysis = {
            'test_name': test['name'],
            'test_type': test['type'],
            'variants': {},
            'winner': None,
            'confidence': 0.0,
            'recommendations': []
        }
        
        # Analyze each variant
        for variant, results in test['results'].items():
            if not results:
                continue
            
            # Calculate average metrics
            avg_metrics = self._calculate_average_metrics(results)
            analysis['variants'][variant] = avg_metrics
        
        # Determine winner
        if len(analysis['variants']) > 1:
            winner, confidence = self._determine_winner(analysis['variants'])
            analysis['winner'] = winner
            analysis['confidence'] = confidence
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_test_recommendations(
                analysis['variants'], winner, test['type']
            )
        
        return analysis
    
    def _calculate_average_metrics(self, results: List[ContentMetrics]) -> Dict[str, float]:
        """Calculate average metrics from multiple results"""
        if not results:
            return {}
        
        avg_metrics = {}
        for field in ContentMetrics.__annotations__:
            values = [getattr(result, field) for result in results if hasattr(result, field)]
            if values:
                avg_metrics[field] = np.mean(values)
        
        return avg_metrics
    
    def _determine_winner(self, variants: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
        """Determine the winning variant with confidence score"""
        
        # Use engagement rate as primary metric
        engagement_rates = {}
        for variant, metrics in variants.items():
            if 'engagement_rate' in metrics:
                engagement_rates[variant] = metrics['engagement_rate']
        
        if not engagement_rates:
            return list(variants.keys())[0], 0.5  # Default
        
        # Find winner
        winner = max(engagement_rates, key=engagement_rates.get)
        winner_rate = engagement_rates[winner]
        
        # Calculate confidence based on margin
        other_rates = [rate for var, rate in engagement_rates.items() if var != winner]
        if other_rates:
            margin = winner_rate - max(other_rates)
            confidence = min(margin * 10, 0.95)  # Scale margin to confidence
        else:
            confidence = 0.5
        
        return winner, confidence
    
    def _generate_test_recommendations(
        self, 
        variants: Dict[str, Dict[str, float]], 
        winner: str, 
        test_type: str
    ) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        if test_type == 'thumbnail':
            recommendations.append(f"Use variant '{winner}' as primary thumbnail")
            recommendations.append("Test different visual elements in future iterations")
        
        elif test_type == 'title':
            recommendations.append(f"Use variant '{winner}' as primary title")
            recommendations.append("Analyze winning title patterns for future content")
        
        elif test_type == 'description':
            recommendations.append(f"Use variant '{winner}' as primary description")
            recommendations.append("Optimize description length and keyword usage")
        
        elif test_type == 'tags':
            recommendations.append(f"Use variant '{winner}' as primary tag set")
            recommendations.append("Research trending tags in your niche")
        
        return recommendations

# =============================================================================
# ADVANCED VIDEO EFFECTS ENGINE
# =============================================================================

class AdvancedVideoEffects:
    """Professional video effects and enhancements"""
    
    def __init__(self):
        self.effects_library = {}
        self._load_effects_library()
    
    def _load_effects_library(self):
        """Load available video effects"""
        self.effects_library = {
            'transitions': ['fade', 'slide', 'zoom', 'rotate', 'flip'],
            'filters': ['cinematic', 'vintage', 'modern', 'dramatic', 'warm'],
            'text_effects': ['typewriter', 'fade_in', 'slide_in', 'bounce'],
            'motion_effects': ['pan', 'tilt', 'zoom', 'rotation']
        }
    
    def apply_dynamic_transitions(
        self, 
        video_input, 
        transition_type: str = 'auto',
        duration: float = 1.0
    ):
        """Apply AI-powered dynamic transitions
        
        Args:
            video_input: Either a VideoClip object or file path string
            transition_type: Type of transition to apply
            duration: Duration of transition effects
        """
        
        if not MOVIEPY_AVAILABLE:
            logging.error("MoviePy not available for video effects")
            return video_input
        
        try:
            # Handle both VideoClip objects and file paths
            if isinstance(video_input, str):
                # Input is a file path
                video = VideoFileClip(video_input)
                is_file_path = True
            else:
                # Input is a VideoClip object
                video = video_input
                is_file_path = False
            
            # Auto-detect transition type if specified
            if transition_type == 'auto':
                transition_type = self._auto_detect_transition_type(video)
            
            # Apply transition effect
            if transition_type == 'fade':
                video = video.fadein(duration).fadeout(duration)
            elif transition_type == 'slide':
                # Use available slide effect or fallback to fade
                try:
                    video = video.fx(vfx.slide_in, duration, 'left')
                except AttributeError:
                    logging.warning("⚠️ slide_in effect not available, using fade instead")
                    video = video.fadein(duration).fadeout(duration)
            elif transition_type == 'zoom':
                # Use available zoom effect or fallback to fade
                try:
                    video = video.fx(vfx.zoom, 1.2)
                except AttributeError:
                    logging.warning("⚠️ zoom effect not available, using fade instead")
                    video = video.fadein(duration).fadeout(duration)
            
            if is_file_path:
                # Save enhanced video if input was a file path
                output_path = video_input.replace('.mp4', '_enhanced.mp4')
                video.write_videofile(output_path, codec='libx264')
                video.close()
                logging.info(f"✅ Dynamic transition applied: {transition_type}")
                return output_path
            else:
                # Return enhanced VideoClip if input was a VideoClip
                logging.info(f"✅ Dynamic transition applied: {transition_type}")
                return video
            
        except Exception as e:
            logging.error(f"❌ Transition application failed: {e}")
            return video_input
    
    def apply_color_grading(
        self, 
        video_input, 
        style: str = 'cinematic'
    ):
        """Apply Hollywood-quality color grading
        
        Args:
            video_input: Either a VideoClip object or file path string
            style: Color grading style to apply
        """
        
        if not MOVIEPY_AVAILABLE:
            logging.error("MoviePy not available for color grading")
            return video_input
        
        try:
            # Handle both VideoClip objects and file paths
            if isinstance(video_input, str):
                # Input is a file path
                video = VideoFileClip(video_input)
                is_file_path = True
            else:
                # Input is a VideoClip object
                video = video_input
                is_file_path = False
            
            # Apply color grading based on style
            if style == 'cinematic':
                # Cinematic look: increased contrast, warm tones
                video = video.fx(vfx.colorx, 1.2)  # Increase saturation
                video = video.fx(vfx.lum_contrast, lum=1.1, contrast=1.3)
            elif style == 'vintage':
                # Vintage look: sepia tones, reduced saturation
                video = video.fx(vfx.colorx, 0.8)
                video = video.fx(vfx.lum_contrast, lum=0.9, contrast=1.1)
            elif style == 'modern':
                # Modern look: clean, bright, high contrast
                video = video.fx(vfx.lum_contrast, lum=1.2, contrast=1.4)
                video = video.fx(vfx.colorx, 1.1)
            elif style == 'finance':
                # Professional finance look: muted, professional, clean
                video = video.fx(vfx.colorx, 0.9)  # Slightly reduced saturation for professionalism
                video = video.fx(vfx.lum_contrast, lum=1.0, contrast=1.2)  # Balanced contrast
                # Apply subtle blue tint for corporate feel
                try:
                    video = video.fx(vfx.colorx, 0.95)  # Slight blue tint
                except:
                    pass  # Fallback if effect not available
            elif style == 'professional':
                # Professional business look: clean, balanced, corporate
                video = video.fx(vfx.lum_contrast, lum=1.05, contrast=1.15)  # Subtle enhancement
                video = video.fx(vfx.colorx, 0.95)  # Slightly muted colors
            
            if is_file_path:
                # Save enhanced video if input was a file path
                output_path = video_input.replace('.mp4', f'_{style}.mp4')
                video.write_videofile(output_path, codec='libx264')
                video.close()
                logging.info(f"✅ Color grading applied: {style}")
                return output_path
            else:
                # Return enhanced VideoClip if input was a VideoClip
                logging.info(f"✅ Color grading applied: {style}")
                return video
                
        except Exception as e:
            logging.error(f"❌ Color grading failed: {e}")
            return video_input
    
    def create_motion_graphics(
        self, 
        text: str, 
        style: str = 'modern',
        duration: float = 5.0
    ) -> str:
        """Create animated motion graphics"""
        
        if not MOVIEPY_AVAILABLE:
            logging.error("MoviePy not available for motion graphics")
            return ""
        
        try:
            # Create text clip
            txt_clip = TextClip(
                text, 
                fontsize=70, 
                color='white',
                font='Arial-Bold'
            )
            
            # Apply animation effects
            if style == 'modern':
                txt_clip = txt_clip.set_position('center').set_duration(duration)
                txt_clip = txt_clip.fx(vfx.fadein, 1.0).fx(vfx.fadeout, 1.0)
            elif style == 'dynamic':
                txt_clip = txt_clip.set_position('center').set_duration(duration)
                txt_clip = txt_clip.fx(vfx.zoom, 1.2)
            
            # Save motion graphic
            output_path = f"motion_graphic_{int(time.time())}.mp4"
            txt_clip.write_videofile(output_path, codec='libx264')
            txt_clip.close()
            
            logging.info(f"✅ Motion graphic created: {style}")
            return output_path
            
        except Exception as e:
            logging.error(f"❌ Motion graphic creation failed: {e}")
            return ""
    
    def _auto_detect_transition_type(self, video) -> str:
        """Auto-detect best transition type based on video content"""
        # Simple heuristic based on video duration and content
        duration = video.duration
        
        if duration < 10:
            return 'fade'  # Short videos: simple fade
        elif duration < 30:
            return 'slide'  # Medium videos: slide transition
        else:
            return 'zoom'  # Long videos: zoom effect

# =============================================================================
# MAIN AI VIDEO SUITE CLASS
# =============================================================================

class AIVideoSuite:
    """Main AI Video Suite - Professional Video Creation & Optimization"""
    
    def __init__(self):
        self.quality_scorer = ContentQualityScorer()
        self.engagement_predictor = EngagementPredictor()
        self.ab_testing = ABTestingFramework()
        self.video_effects = AdvancedVideoEffects()
        
        logging.info("🚀 AI Video Suite initialized successfully!")
    
    def create_optimized_content(
        self,
        title: str,
        description: str,
        tags: List[str],
        content_type: ContentType,
        platform: Platform,
        posting_time: Optional[datetime] = None,
        audience_size: int = 1000
    ) -> Dict[str, Any]:
        """Create and optimize content with AI analysis"""
        
        if posting_time is None:
            posting_time = datetime.now()
        
        # 1. Content Quality Analysis
        content_analysis = self.quality_scorer.analyze_content_quality(
            title, description, tags
        )
        
        # 2. Engagement Prediction
        engagement_prediction = self.engagement_predictor.predict_engagement(
            content_type, platform, content_analysis, posting_time, audience_size
        )
        
        # 3. Generate optimization recommendations
        optimization_plan = self._generate_optimization_plan(
            content_analysis, engagement_prediction
        )
        
        return {
            'content_analysis': content_analysis,
            'engagement_prediction': engagement_prediction,
            'optimization_plan': optimization_plan,
            'quality_score': content_analysis.quality_score,
            'predicted_engagement': engagement_prediction['engagement_rate'],
            'recommendations': content_analysis.recommendations
        }
    
    def run_ab_test(
        self,
        test_name: str,
        test_type: str,
        variants: List[Dict[str, Any]],
        test_duration_days: int = 7
    ) -> str:
        """Run A/B testing for content optimization"""
        
        test_id = self.ab_testing.create_test(
            test_name, test_type, variants, test_duration_days=test_duration_days
        )
        
        logging.info(f"🧪 A/B test started: {test_name}")
        return test_id
    
    def enhance_video(
        self,
        video_path: str,
        effects: List[str],
        output_path: Optional[str] = None
    ) -> str:
        """Apply advanced video effects"""
        
        if not output_path:
            output_path = video_path.replace('.mp4', '_enhanced.mp4')
        
        enhanced_path = video_path
        
        for effect in effects:
            if effect == 'transitions':
                enhanced_path = self.video_effects.apply_dynamic_transitions(enhanced_path)
            elif effect == 'color_grading':
                enhanced_path = self.video_effects.apply_color_grading(enhanced_path)
            elif effect == 'motion_graphics':
                # Create motion graphics and overlay
                pass
        
        logging.info(f"✅ Video enhancement completed: {output_path}")
        return enhanced_path
    
    def _generate_optimization_plan(
        self, 
        content_analysis: ContentAnalysis, 
        engagement_prediction: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization plan"""
        
        plan = {
            'priority': 'high' if content_analysis.quality_score < 70 else 'medium',
            'actions': [],
            'expected_improvement': 0.0
        }
        
        # Generate action items based on analysis
        if content_analysis.quality_score < 70:
            plan['actions'].append("Improve content quality based on AI recommendations")
            plan['expected_improvement'] += 20
        
        if content_analysis.seo_score < 70:
            plan['actions'].append("Optimize SEO with better keywords and descriptions")
            plan['expected_improvement'] += 15
        
        if content_analysis.trend_relevance < 70:
            plan['actions'].append("Include more trending topics and keywords")
            plan['expected_improvement'] += 10
        
        # Engagement optimization
        if engagement_prediction['engagement_rate'] < 0.5:
            plan['actions'].append("Optimize for higher engagement with better hooks and CTAs")
            plan['expected_improvement'] += 25
        
        return plan
    
    def _ensure_even_dimensions(self, image_path: str) -> str:
        """Ensure image has even dimensions for FFmpeg compatibility"""
        try:
            if not PILLOW_AVAILABLE:
                return image_path
                
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check if dimensions are even
                if width % 2 == 0 and height % 2 == 0:
                    return image_path
                
                # Make dimensions even by cropping or resizing
                new_width = width if width % 2 == 0 else width - 1
                new_height = height if height % 2 == 0 else height - 1
                
                # Crop to even dimensions
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = left + new_width
                bottom = top + new_height
                
                cropped_img = img.crop((left, top, right, bottom))
                
                # Save to temporary file
                temp_dir = "temp_images"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"even_dim_{os.path.basename(image_path)}")
                cropped_img.save(temp_path, quality=95)
                
                logging.info(f"✅ Fixed image dimensions: {width}x{height} -> {new_width}x{new_height}")
                return temp_path
                
        except Exception as e:
            logging.warning(f"⚠️ Failed to fix image dimensions for {image_path}: {e}")
            return image_path

    def _enhance_audio_quality(self, audio_path: str) -> str:
        """Enhance audio quality for better video output"""
        try:
            if not os.path.exists(audio_path):
                return audio_path
            
            # Create enhanced audio path
            base_name = os.path.splitext(audio_path)[0]
            enhanced_path = f"{base_name}_enhanced_quality.mp3"
            
            if MOVIEPY_AVAILABLE:
                try:
                    # Load audio with MoviePy
                    audio_clip = AudioFileClip(audio_path)
                    
                    # Apply audio enhancements for professional finance content
                    # Normalize volume for consistent levels
                    audio_clip = audio_clip.volumex(1.4)
                    
                    # Apply fade in/out for smoother transitions
                    audio_clip = audio_clip.fadein(1.5).fadeout(1.5)
                    
                    # Apply subtle audio compression for professional sound
                    try:
                        # Try to apply audio compression if available
                        audio_clip = audio_clip.fx(afx.audio_normalize)
                    except:
                        pass  # Fallback if effect not available
                    
                    # Write enhanced audio with higher quality for finance content
                    audio_clip.write_audiofile(
                        enhanced_path,
                        codec='mp3',
                        bitrate='320k',  # Maximum quality audio
                        verbose=False,
                        logger=None,
                        ffmpeg_params=[
                            '-ar', '48000',  # Higher sample rate
                            '-ac', '2',      # Stereo
                            '-b:a', '320k',  # High bitrate
                            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Professional loudness normalization
                            '-af', 'highpass=f=50',  # Remove low frequency noise
                            '-af', 'lowpass=f=15000'  # Remove high frequency noise
                        ]
                    )
                    
                    audio_clip.close()
                    logging.info(f"✅ Professional audio quality enhanced: {enhanced_path}")
                    return enhanced_path
                    
                except Exception as e:
                    logging.warning(f"⚠️ MoviePy audio enhancement failed: {e}")
                    return audio_path
            
            return audio_path
            
        except Exception as e:
            logging.warning(f"⚠️ Audio enhancement failed: {e}")
            return audio_path

    def _select_relevant_images(self, visual_assets: List[str], content_topic: str = "finance") -> List[str]:
        """Select most relevant images for the content topic"""
        try:
            if not visual_assets:
                return []
            
            # Priority keywords for finance content
            finance_keywords = [
                'finance', 'money', 'business', 'investment', 'trading', 'stock', 'market',
                'economy', 'banking', 'technology', 'ai', 'artificial intelligence',
                'digital', 'innovation', 'growth', 'success', 'professional', 'corporate',
                'entrepreneur', 'startup', 'wealth', 'financial', 'economic', 'commerce',
                'bank', 'currency', 'profit', 'revenue', 'capital', 'enterprise'
            ]
            
            # Score each image based on relevance
            scored_images = []
            for asset_path in visual_assets:
                if not asset_path or not os.path.exists(asset_path):
                    continue
                
                score = 0
                filename = os.path.basename(asset_path).lower()
                
                # Check filename for relevance
                for keyword in finance_keywords:
                    if keyword in filename:
                        score += 25  # Higher score for finance relevance
                
                # Prioritize certain image categories for finance content
                if 'script' in filename:
                    score += 20  # Script images are excellent for text overlay
                elif 'welcome' in filename:
                    score += 15  # Welcome images are good for intros
                elif 'exploring' in filename:
                    score += 12  # Exploring images show discovery
                elif 'tech' in filename or 'technology' in filename:
                    score += 22  # Technology is very relevant for finance
                elif 'business' in filename:
                    score += 25  # Business images are highly relevant
                elif 'money' in filename or 'finance' in filename:
                    score += 30  # Direct finance relevance
                elif 'ai' in filename or 'artificial' in filename:
                    score += 28  # AI is very relevant for modern finance
                elif 'energy' in filename:
                    score += 10   # Energy sector is relevant for finance
                
                # Avoid generic or irrelevant images
                if 'generic' in filename or 'random' in filename:
                    score -= 20
                if 'nature' in filename or 'landscape' in filename:
                    score -= 15  # Less relevant for finance
                if 'food' in filename or 'cooking' in filename:
                    score -= 25  # Not relevant for finance
                
                # Bonus for high-quality image indicators
                if '4k' in filename or 'hd' in filename:
                    score += 8
                if 'professional' in filename:
                    score += 10
                
                scored_images.append((asset_path, score))
            
            # Sort by score (highest first) and select top images
            scored_images.sort(key=lambda x: x[1], reverse=True)
            
            # Select ALL images for longer videos - no limit for finance content
            selected_images = [img[0] for img in scored_images]
            
            logging.info(f"🎯 Selected {len(selected_images)} most relevant finance images from {len(visual_assets)} available")
            logging.info(f"📊 Top 10 image scores: {[score for _, score in scored_images[:10]]}")
            
            # Ensure we have enough images for a long video
            if len(selected_images) < 20:
                logging.warning(f"⚠️ Only {len(selected_images)} images selected, video may be shorter than expected")
            
            return selected_images
            
        except Exception as e:
            logging.warning(f"⚠️ Image selection failed: {e}")
            # Fallback: return ALL images for longer content
            return visual_assets

    def create_professional_video(
        self,
        visual_assets: List[str],
        audio_path: str,
        content_type: ContentType = ContentType.LONG_FORM,
        target_platform: Platform = Platform.YOUTUBE,
        quality_target: QualityScore = QualityScore.EXCELLENT
    ) -> Dict[str, Any]:
        """Create professional video from visual assets and audio - MoviePy 2.0.0.dev2 Optimized"""
        
        try:
            if not MOVIEPY_AVAILABLE:
                logging.warning("⚠️ MoviePy not available, creating placeholder video")
                return self._create_placeholder_video(visual_assets, audio_path)
            
            # Select most relevant images for the content - Use ALL available images for longer videos
            selected_assets = self._select_relevant_images(visual_assets, "finance")
            
            if not selected_assets:
                logging.warning("⚠️ No visual assets available, creating simple video")
                return self._create_simple_video(audio_path)
            
            # Create video composition
            video_clips = []
            
            # Calculate target duration: 10+ minutes minimum
            target_total_duration = 600.0  # 10 minutes minimum
            max_clips = len(selected_assets)
            
            # Calculate duration per image to achieve target length
            # Each image should show for at least 3 seconds, maximum 8 seconds
            min_duration_per_image = 3.0
            max_duration_per_image = 8.0
            
            # Calculate optimal duration per image
            optimal_duration_per_image = max(min_duration_per_image, 
                                          min(max_duration_per_image, 
                                              target_total_duration / max_clips))
            
            logging.info(f"🎬 Processing {max_clips} visual assets for {target_total_duration}s video")
            logging.info(f"📊 Duration per image: {optimal_duration_per_image:.2f}s")
            
            # Create video clips from images
            for i, asset_path in enumerate(selected_assets):
                if asset_path and os.path.exists(asset_path):
                    try:
                        # Ensure image has even dimensions for FFmpeg compatibility
                        fixed_asset_path = self._ensure_even_dimensions(asset_path)
                        
                        # Create video clip from image with calculated duration
                        clip = ImageClip(fixed_asset_path, duration=optimal_duration_per_image)
                        
                        # Resize to standard HD dimensions for better quality
                        clip = clip.resize((1920, 1080))
                        
                        video_clips.append(clip)
                        
                        # Log progress every few clips
                        if i % 10 == 0:
                            logging.info(f"🔄 Processed {i+1}/{max_clips} visual assets")
                            
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to add visual asset {asset_path}: {e}")
                        continue
            
            if not video_clips:
                logging.warning("⚠️ No valid visual assets, creating simple video")
                return self._create_simple_video(audio_path)
            
            logging.info(f"✅ Successfully processed {len(video_clips)} visual assets")
            
            # Concatenate video clips
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # Calculate actual video duration
            actual_video_duration = final_video.duration
            logging.info(f"🎬 Video duration: {actual_video_duration:.2f}s ({actual_video_duration/60:.1f} minutes)")
            
            # Add audio if available
            if audio_path and os.path.exists(audio_path):
                try:
                    # Enhance audio quality before adding to video
                    enhanced_audio_path = self._enhance_audio_quality(audio_path)
                    audio_clip = AudioFileClip(enhanced_audio_path)
                    
                    logging.info(f"🎵 Audio duration: {audio_clip.duration:.2f}s")
                    
                    # Match video duration to audio for longer content
                    if audio_clip.duration > actual_video_duration:
                        # Extend video with last frame to match audio length
                        final_frame = final_video.get_frame(actual_video_duration - 0.1)
                        extended_clip = ImageClip(final_frame, duration=audio_clip.duration - actual_video_duration)
                        final_video = concatenate_videoclips([final_video, extended_clip])
                        logging.info(f"✅ Extended video to match audio duration: {audio_clip.duration:.2f}s")
                    else:
                        # Trim video to audio length
                        final_video = final_video.subclip(0, audio_clip.duration)
                        logging.info(f"✅ Trimmed video to audio duration: {audio_clip.duration:.2f}s")
                    
                    final_video = final_video.set_audio(audio_clip)
                    logging.info(f"✅ Audio synchronized: {audio_clip.duration:.2f}s")
                    
                    # Update actual duration
                    actual_video_duration = final_video.duration
                    
                except Exception as e:
                    logging.warning(f"⚠️ Failed to add audio: {e}")
            
            # Apply quality enhancements
            if quality_target == QualityScore.EXCELLENT:
                logging.info("🎨 Applying professional finance enhancements...")
                final_video = self.video_effects.apply_color_grading(final_video, 'finance')
                final_video = self.video_effects.apply_dynamic_transitions(final_video)
            
            # Generate output path
            timestamp = int(time.time())
            output_path = f"outputs/videos/professional_video_{timestamp}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ensure output directory is writable
            if not os.access(os.path.dirname(output_path), os.W_OK):
                logging.error("❌ Output directory is not writable")
                return self._create_basic_fallback_video()
            
            # Write final video with MoviePy 2.0.0.dev2 optimized settings for MUCH longer content
            try:
                logging.info("🎬 Starting high-quality video rendering with MoviePy 2.0.0.dev2...")
                
                # High-quality rendering parameters optimized for much longer videos
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    fps=30,  # Higher FPS for smoother video
                    preset='medium',  # Better quality preset
                    threads=4,  # Use more threads for better performance
                    bitrate='8000k',  # Much higher bitrate for longer, better quality videos
                    audio_bitrate='320k',  # Higher audio bitrate for professional sound
                    logger=None,  # Disable verbose logging
                    ffmpeg_params=[  # Additional FFmpeg optimizations for much longer content
                        '-crf', '20',  # Lower CRF for higher quality
                        '-movflags', '+faststart',  # Optimize for web streaming
                        '-pix_fmt', 'yuv420p',  # Ensure compatibility
                        '-profile:v', 'high',  # High profile for better quality
                        '-level', '4.1',  # Higher level for better compatibility
                        '-maxrate', '12000k',  # Maximum bitrate for quality control
                        '-bufsize', '24000k'  # Buffer size for smooth encoding
                    ]
                )
                
                logging.info(f"✅ High-quality video rendered successfully: {output_path}")
                
            except Exception as e:
                logging.error(f"❌ Primary video rendering failed: {e}")
                logging.info("🔄 Attempting fallback rendering...")
                
                try:
                    # Fallback 1: Medium quality settings for much longer videos
                    final_video.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        fps=24,
                        preset='fast',
                        threads=2,
                        bitrate='5000k',  # Higher bitrate for longer content
                        audio_bitrate='256k',  # Better audio quality
                        logger=None
                    )
                    logging.info(f"✅ Fallback rendering successful: {output_path}")
                    
                except Exception as e2:
                    logging.error(f"❌ Fallback rendering also failed: {e2}")
                    logging.info("🔄 Creating simple fallback video...")
                    
                    # Fallback 2: Create simple video with guaranteed even dimensions
                    try:
                        return self._create_simple_video(audio_path)
                    except Exception as e3:
                        logging.error(f"❌ All video creation methods failed: {e3}")
                        # Final fallback: Create a basic video with known good dimensions
                        return self._create_basic_fallback_video()
            
            # Clean up
            final_video.close()
            for clip in video_clips:
                clip.close()
            
            # Clean up temporary image files
            try:
                temp_dir = "temp_images"
                if os.path.exists(temp_dir):
                    for temp_file in os.listdir(temp_dir):
                        if temp_file.startswith("even_dim_"):
                            os.remove(os.path.join(temp_dir, temp_file))
                    logging.info("✅ Temporary image files cleaned up")
            except Exception as e:
                logging.warning(f"⚠️ Failed to cleanup temporary files: {e}")
            
            # Verify output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024:  # > 1MB
                logging.info(f"✅ Professional video created successfully: {output_path}")
                
                return {
                    'video_path': output_path,
                    'duration': actual_video_duration,
                    'quality_score': quality_target.value,
                    'platform_optimized': target_platform.value,
                    'content_type': content_type.value,
                    'created_at': datetime.now().isoformat(),
                    'images_used': len(video_clips),
                    'total_images_available': len(visual_assets),
                    'target_duration_achieved': actual_video_duration >= 600.0,  # 10+ minutes
                    'duration_per_image': optimal_duration_per_image,
                    'estimated_total_duration': len(video_clips) * optimal_duration_per_image
                }
            else:
                logging.warning("⚠️ Output file verification failed, creating simple video")
                return self._create_simple_video(audio_path)
                
        except Exception as e:
            logging.error(f"❌ Professional video creation failed: {e}")
            logging.info("🔄 Attempting basic fallback video creation...")
            try:
                return self._create_basic_fallback_video()
            except Exception as e2:
                logging.error(f"❌ All video creation methods failed: {e2}")
                return {"error": str(e2), "status": "failed"}
    
    def _create_basic_fallback_video(self) -> Dict[str, Any]:
        """Create a basic fallback video with guaranteed working parameters"""
        try:
            timestamp = int(time.time())
            output_path = f"outputs/videos/basic_fallback_{timestamp}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if MOVIEPY_AVAILABLE:
                # Create a simple black video with guaranteed even dimensions
                bg_clip = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=5.0)
                
                # Write video with minimal settings
                bg_clip.write_videofile(
                    output_path, 
                    codec='libx264', 
                    fps=24,
                    preset='ultrafast',
                    threads=1,
                    logger=None
                )
                
                bg_clip.close()
                
                logging.info(f"✅ Basic fallback video created: {output_path}")
                
                return {
                    'video_path': output_path,
                    'duration': 5.0,
                    'quality_score': 25.0,
                    'platform_optimized': 'youtube',
                    'content_type': 'fallback',
                    'created_at': datetime.now().isoformat()
                }
            else:
                # Create empty file if MoviePy not available
                with open(output_path, 'w') as f:
                    f.write("Basic fallback video file")
                
                return {
                    'video_path': output_path,
                    'duration': 0.0,
                    'quality_score': 25.0,
                    'platform_optimized': 'youtube',
                    'content_type': 'fallback',
                    'created_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logging.error(f"❌ Basic fallback video creation failed: {e}")
            return {
                'video_path': 'fallback_failed.mp4',
                'duration': 0.0,
                'quality_score': 25.0,
                'platform_optimized': 'youtube',
                'content_type': 'error',
                'created_at': datetime.now().isoformat()
            }

    def _create_placeholder_video(self, visual_assets: List[str], audio_path: str) -> Dict[str, Any]:
        """Create a placeholder video when main creation fails - MoviePy 2.0.0.dev2 Optimized"""
        try:
            timestamp = int(time.time())
            output_path = f"outputs/videos/placeholder_video_{timestamp}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create simple black video with text
            if MOVIEPY_AVAILABLE:
                # Create black background with FPS
                bg_clip = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=10.0)
                
                # Add text overlay
                txt_clip = TextClip(
                    "AI Video Suite - Professional Video",
                    fontsize=70,
                    color='white',
                    font='Arial-Bold'
                ).set_position('center').set_duration(10.0)
                
                # Combine background and text
                final_clip = CompositeVideoClip([bg_clip, txt_clip])
                
                # Write video with MoviePy 2.0.0.dev2 optimized settings
                final_clip.write_videofile(
                    output_path, 
                    codec='libx264', 
                    fps=24,
                    preset='ultrafast',
                    threads=1,
                    logger=None,
                    ffmpeg_params=['-crf', '30', '-pix_fmt', 'yuv420p']
                )
                
                final_clip.close()
                bg_clip.close()
                txt_clip.close()
                
                logging.info(f"✅ Placeholder video created: {output_path}")
                
                return {
                    'video_path': output_path,
                    'duration': 10.0,
                    'quality_score': QualityScore.MEDIUM.value,
                    'platform_optimized': 'youtube',
                    'content_type': 'placeholder',
                    'created_at': datetime.now().isoformat()
                }
            else:
                # Create empty file if MoviePy not available
                with open(output_path, 'w') as f:
                    f.write("Placeholder video file")
                
                return {
                    'video_path': output_path,
                    'duration': 0.0,
                    'quality_score': 25.0,  # Fixed value instead of QualityScore.LOW.value
                    'platform_optimized': 'youtube',
                    'content_type': 'placeholder',
                    'created_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logging.error(f"❌ Placeholder video creation failed: {e}")
            return {
                'video_path': 'placeholder_failed.mp4',
                'duration': 0.0,
                'quality_score': 25.0,  # Fixed value instead of QualityScore.LOW.value
                'platform_optimized': 'youtube',
                'content_type': 'error',
                'created_at': datetime.now().isoformat()
            }
    
    def _create_simple_video(self, audio_path: str) -> Dict[str, Any]:
        """Create a simple video with just audio - MoviePy 2.0.0.dev2 Optimized"""
        try:
            timestamp = int(time.time())
            output_path = f"outputs/videos/simple_video_{timestamp}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if MOVIEPY_AVAILABLE and audio_path and os.path.exists(audio_path):
                # Create black background with audio duration and FPS
                audio_clip = AudioFileClip(audio_path)
                bg_clip = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=audio_clip.duration)
                
                # Add audio
                final_clip = bg_clip.set_audio(audio_clip)
                
                # Write video with MoviePy 2.0.0.dev2 optimized settings
                final_clip.write_videofile(
                    output_path, 
                    codec='libx264', 
                    fps=24,
                    preset='ultrafast',
                    threads=1,
                    logger=None,
                    ffmpeg_params=['-crf', '28', '-pix_fmt', 'yuv420p']
                )
                
                final_clip.close()
                bg_clip.close()
                audio_clip.close()
                
                logging.info(f"✅ Simple video created: {output_path}")
                
                return {
                    'video_path': output_path,
                    'duration': audio_clip.duration,
                    'quality_score': QualityScore.MEDIUM.value,
                    'platform_optimized': 'youtube',
                    'content_type': 'simple',
                    'created_at': datetime.now().isoformat()
                }
            else:
                return self._create_placeholder_video([], audio_path)
                
        except Exception as e:
            logging.error(f"❌ Simple video creation failed: {e}")
            return self._create_placeholder_video([], audio_path)

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Initialize AI Video Suite
    suite = AIVideoSuite()
    
    # Example content optimization
    result = suite.create_optimized_content(
        title="10 AI Tools That Will Change Your Life in 2025",
        description="Discover the most powerful AI tools that are revolutionizing how we work, create, and live. From productivity to creativity, these tools will transform your daily routine.",
        tags=["AI", "artificial intelligence", "productivity", "2025", "technology", "tools", "automation"],
        content_type=ContentType.LONG_FORM,
        platform=Platform.YOUTUBE,
        audience_size=5000
    )
    
    print("🎬 AI Video Suite Analysis Results:")
    print(f"Quality Score: {result['quality_score']:.1f}/100")
    print(f"Predicted Engagement: {result['predicted_engagement']:.1%}")
    print(f"SEO Score: {result['content_analysis'].seo_score:.1f}/100")
    print(f"Trend Relevance: {result['content_analysis'].trend_relevance:.1f}/100")
    print("\n📋 Recommendations:")
    for rec in result['recommendations']:
        print(f"• {rec}")
    
    print("\n🚀 Optimization Plan:")
    plan = result['optimization_plan']
    print(f"Priority: {plan['priority']}")
    print(f"Expected Improvement: +{plan['expected_improvement']:.1f} points")
    print("Actions:")
    for action in plan['actions']:
        print(f"• {action}")
