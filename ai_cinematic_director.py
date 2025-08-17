# -*- coding: utf-8 -*-
"""
🎬 CINEMATIC AI DIRECTOR
AI-Powered Cinematic Direction & Storytelling Platform
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

class StoryArc(Enum):
    HERO_JOURNEY = "hero_journey"
    THREE_ACT = "three_act"
    FIVE_ACT = "five_act"
    CIRCULAR = "circular"
    LINEAR = "linear"
    NON_LINEAR = "non_linear"

class CinematicStyle(Enum):
    HOLLYWOOD = "hollywood"
    INDIE = "indie"
    DOCUMENTARY = "documentary"
    EXPERIMENTAL = "experimental"
    CLASSIC = "classic"
    MODERN = "modern"
    EPIC = "epic"
    INTIMATE = "intimate"

class PacingType(Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    VARIABLE = "variable"
    BUILDING = "building"
    EXPLOSIVE = "explosive"

class EmotionalBeat(Enum):
    OPENING = "opening"
    HOOK = "hook"
    CONFLICT = "conflict"
    CLIMAX = "climax"
    RESOLUTION = "resolution"
    TWIST = "twist"
    REVELATION = "revelation"

@dataclass
class StoryStructure:
    """Story structure and pacing"""
    arc_type: StoryArc
    pacing: PacingType
    duration_minutes: float
    emotional_beats: List[EmotionalBeat]
    key_moments: List[str]
    target_engagement: float

@dataclass
class CinematicGuidelines:
    """Cinematic direction guidelines"""
    style: CinematicStyle
    target_emotion: str
    visual_theme: str
    audio_mood: str
    pacing_target: PacingType
    engagement_curve: List[float]

@dataclass
class SceneBreakdown:
    """Scene-by-scene breakdown"""
    scene_number: int
    duration_seconds: float
    purpose: str
    emotional_tone: str
    visual_elements: List[str]
    audio_requirements: List[str]
    engagement_target: float

@dataclass
class CinematicAnalysis:
    """Complete cinematic analysis"""
    story_structure: StoryStructure
    guidelines: CinematicGuidelines
    scene_breakdown: List[SceneBreakdown]
    pacing_analysis: Dict[str, float]
    engagement_prediction: float
    optimization_suggestions: List[str]

# =============================================================================
# CINEMATIC AI DIRECTOR CLASS
# =============================================================================

class CinematicAIDirector:
    """
    🎬 AI-Powered Cinematic Director
    
    Features:
    - Story structure analysis and optimization
    - Emotional pacing and beat mapping
    - Scene-by-scene breakdown
    - Engagement prediction and optimization
    - Cinematic style adaptation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Cinematic AI Director with memory management"""
        self.config = config or {}
        self.setup_logging()
        
        # Memory management
        self._clear_gpu_memory()
        
        # Initialize AI models if available
        self.initialize_ai_models()
        
        # Story templates and patterns
        self.story_templates = self._load_story_templates()
        self.emotional_patterns = self._load_emotional_patterns()
        
        self.story_structures = {
            "epic": StoryStructure(
                arc_type=StoryArc.HERO_JOURNEY,
                pacing=PacingType.BUILDING,
                duration_minutes=15.0,
                emotional_beats=[
                    EmotionalBeat.OPENING,
                    EmotionalBeat.HOOK,
                    EmotionalBeat.CONFLICT,
                    EmotionalBeat.CLIMAX,
                    EmotionalBeat.RESOLUTION
                ],
                key_moments=[
                    "Opening hook (0-2 min)",
                    "Problem establishment (2-5 min)",
                    "Rising action (5-10 min)",
                    "Climax (10-13 min)",
                    "Resolution (13-15 min)"
                ],
                target_engagement=0.95
            ),
            "documentary": StoryStructure(
                arc_type=StoryArc.THREE_ACT,
                pacing=PacingType.MEDIUM,
                duration_minutes=12.0,
                emotional_beats=[
                    EmotionalBeat.OPENING,
                    EmotionalBeat.REVELATION,
                    EmotionalBeat.CONFLICT,
                    EmotionalBeat.TWIST,
                    EmotionalBeat.RESOLUTION
                ],
                key_moments=[
                    "Introduction (0-2 min)",
                    "Background context (2-5 min)",
                    "Main narrative (5-9 min)",
                    "Key insights (9-11 min)",
                    "Conclusion (11-12 min)"
                ],
                target_engagement=0.90
            )
        }
        
        logging.info("🎬 Cinematic AI Director initialized successfully!")
    
    def setup_logging(self):
        """Setup logging for the cinematic director"""
        self.logger = logging.getLogger(__name__)
    
    def initialize_ai_models(self):
        """Initialize AI models for enhanced analysis with memory management"""
        self.nlp_pipeline = None
        self.sentiment_analyzer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Check GPU memory availability and use CPU if needed
                if TORCH_AVAILABLE:
                    try:
                        # Try GPU first with memory check
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                            if gpu_memory < 4.0:  # Less than 4GB GPU
                                print(f"⚠️ GPU memory limited ({gpu_memory:.1f}GB), using CPU for sentiment analysis")
                                device = -1  # CPU
                            else:
                                device = 0  # GPU
                        else:
                            device = -1  # CPU
                    except Exception:
                        device = -1  # CPU fallback
                else:
                    device = -1  # CPU
                
                # Initialize sentiment analysis pipeline with device specification
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                model="distilbert-base-uncased-finetuned-sst-2-english",
                                                device=device)
                self.logger.info(f"✅ Sentiment analyzer initialized on {'GPU' if device >= 0 else 'CPU'}")
            except Exception as e:
                self.logger.warning(f"⚠️ Sentiment analyzer initialization failed: {e}")
                # Try with smaller model as fallback
                try:
                    self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                                    device=-1)  # Force CPU
                    self.logger.info("✅ Sentiment analyzer initialized with fallback model on CPU")
                except Exception as e2:
                    self.logger.warning(f"⚠️ Fallback sentiment analyzer also failed: {e2}")
        
        if TORCH_AVAILABLE:
            try:
                # Initialize basic neural network for engagement prediction
                self.engagement_predictor = self._create_engagement_predictor()
                self.logger.info("✅ Engagement predictor initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Engagement predictor initialization failed: {e}")
    
    def _create_engagement_predictor(self):
        """Create a simple neural network for engagement prediction"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            model = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            return model
        except Exception as e:
            self.logger.error(f"❌ Engagement predictor creation failed: {e}")
            return None
    
    def _load_story_templates(self) -> Dict[str, Dict]:
        """Load predefined story templates"""
        return {
            "hero_journey": {
                "structure": ["Call to Adventure", "Crossing Threshold", "Tests & Allies", "Approach", "Ordeal", "Reward", "Return"],
                "emotional_arc": [0.3, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5],
                "duration_weights": [0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05]
            },
            "three_act": {
                "structure": ["Setup", "Confrontation", "Resolution"],
                "emotional_arc": [0.4, 0.7, 0.9],
                "duration_weights": [0.25, 0.5, 0.25]
            },
            "five_act": {
                "structure": ["Exposition", "Rising Action", "Climax", "Falling Action", "Denouement"],
                "emotional_arc": [0.3, 0.5, 0.9, 0.7, 0.4],
                "duration_weights": [0.15, 0.25, 0.2, 0.25, 0.15]
            }
        }
    
    def _load_emotional_patterns(self) -> Dict[str, List[float]]:
        """Load emotional pacing patterns"""
        return {
            "fast": [0.3, 0.6, 0.9, 0.7, 0.5],
            "medium": [0.4, 0.5, 0.7, 0.8, 0.6],
            "slow": [0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5],
            "building": [0.2, 0.3, 0.5, 0.7, 0.9],
            "explosive": [0.8, 0.9, 0.7, 0.5, 0.3]
        }
    
    def create_story_structure(self, 
                              channel_name: str, 
                              niche: str, 
                              target_duration_minutes: float,
                              style: CinematicStyle = CinematicStyle.HOLLYWOOD,
                              arc_type: StoryArc = StoryArc.HERO_JOURNEY,
                              pacing: PacingType = PacingType.MEDIUM) -> Dict[str, Any]:
        """
        Create cinematic story structure for content creation
        
        Args:
            channel_name: Name of the channel
            niche: Content niche/topic
            target_duration_minutes: Target duration in minutes
            style: Cinematic style to apply
            arc_type: Story arc type
            pacing: Pacing type
            
        Returns:
            Dict: Complete story structure
        """
        try:
            self.logger.info(f"🎬 Creating story structure for {channel_name} ({niche})")
            
            # Get story template
            template = self.story_templates.get(arc_type.value, self.story_templates["hero_journey"])
            
            # Calculate scene durations
            scene_durations = self._calculate_scene_durations(
                template["duration_weights"], 
                target_duration_minutes
            )
            
            # Create emotional beats
            emotional_beats = self._create_emotional_beats(
                template["emotional_arc"], 
                pacing, 
                len(template["structure"])
            )
            
            # Generate scene breakdown
            scenes = self._create_scene_breakdown(
                template["structure"],
                scene_durations,
                emotional_beats,
                niche,
                style
            )
            
            # Create story structure
            story_structure = {
                "channel_name": channel_name,
                "niche": niche,
                "target_duration_minutes": target_duration_minutes,
                "cinematic_style": style.value,
                "story_arc": arc_type.value,
                "pacing": pacing.value,
                "scenes": scenes,
                "emotional_arc": emotional_beats,
                "total_scenes": len(scenes),
                "engagement_prediction": self._predict_engagement(scenes, style, pacing),
                "created_at": datetime.now().isoformat(),
                "optimization_suggestions": self._generate_optimization_suggestions(scenes, style)
            }
            
            self.logger.info(f"✅ Story structure created with {len(scenes)} scenes")
            return story_structure
            
        except Exception as e:
            self.logger.error(f"❌ Story structure creation failed: {e}")
            return self._create_fallback_story_structure(channel_name, niche, target_duration_minutes)
    
    def _calculate_scene_durations(self, duration_weights: List[float], 
                                  total_duration: float) -> List[float]:
        """Calculate scene durations based on weights"""
        try:
            # Convert to seconds
            total_seconds = total_duration * 60
            
            # Calculate scene durations
            scene_durations = []
            for weight in duration_weights:
                scene_duration = (weight * total_seconds) / sum(duration_weights)
                scene_durations.append(scene_duration)
            
            return scene_durations
            
        except Exception as e:
            self.logger.error(f"❌ Scene duration calculation failed: {e}")
            # Fallback: equal distribution
            num_scenes = len(duration_weights)
            scene_duration = (total_duration * 60) / num_scenes
            return [scene_duration] * num_scenes
    
    def _create_emotional_beats(self, base_arc: List[float], 
                               pacing: PacingType, 
                               num_scenes: int) -> List[float]:
        """Create emotional beats based on pacing and base arc"""
        try:
            if pacing == PacingType.FAST:
                # Compress emotional arc
                emotional_beats = np.linspace(0.3, 0.9, num_scenes).tolist()
            elif pacing == PacingType.SLOW:
                # Expand emotional arc
                emotional_beats = np.linspace(0.2, 0.8, num_scenes).tolist()
            elif pacing == PacingType.BUILDING:
                # Building tension
                emotional_beats = np.linspace(0.2, 0.9, num_scenes).tolist()
            elif pacing == PacingType.EXPLOSIVE:
                # Start high, then decline
                emotional_beats = [0.9 - (i * 0.6 / (num_scenes - 1)) for i in range(num_scenes)]
            else:
                # Medium pacing - use base arc
                emotional_beats = base_arc[:num_scenes]
                if len(emotional_beats) < num_scenes:
                    # Extend with interpolation
                    extended = np.linspace(emotional_beats[-1], 0.5, num_scenes - len(emotional_beats))
                    emotional_beats.extend(extended.tolist())
            
            # Ensure values are between 0 and 1
            emotional_beats = [max(0.1, min(0.95, beat)) for beat in emotional_beats]
            
            return emotional_beats
            
        except Exception as e:
            self.logger.error(f"❌ Emotional beats creation failed: {e}")
            # Fallback: linear progression
            return np.linspace(0.3, 0.8, num_scenes).tolist()
    
    def _create_scene_breakdown(self, scene_names: List[str], 
                               durations: List[float], 
                               emotional_beats: List[float],
                               niche: str, 
                               style: CinematicStyle) -> List[Dict[str, Any]]:
        """Create detailed scene breakdown"""
        try:
            scenes = []
            
            for i, (name, duration, emotion) in enumerate(zip(scene_names, durations, emotional_beats)):
                scene = {
                    "scene_number": i + 1,
                    "name": name,
                    "duration_seconds": duration,
                    "duration_minutes": duration / 60,
                    "emotional_tone": self._get_emotional_tone(emotion),
                    "purpose": self._get_scene_purpose(name, niche),
                    "visual_elements": self._get_visual_elements(name, niche, style),
                    "audio_requirements": self._get_audio_requirements(emotion, style),
                    "engagement_target": emotion,
                    "cinematic_notes": self._get_cinematic_notes(name, style)
                }
                scenes.append(scene)
            
            return scenes
            
        except Exception as e:
            self.logger.error(f"❌ Scene breakdown creation failed: {e}")
            return []
    
    def _get_emotional_tone(self, emotion_value: float) -> str:
        """Convert emotion value to descriptive tone"""
        if emotion_value < 0.3:
            return "calm"
        elif emotion_value < 0.5:
            return "neutral"
        elif emotion_value < 0.7:
            return "engaging"
        elif emotion_value < 0.85:
            return "exciting"
        else:
            return "thrilling"
    
    def _get_scene_purpose(self, scene_name: str, niche: str) -> str:
        """Get scene purpose based on name and niche"""
        purpose_map = {
            "Call to Adventure": "Introduce the main topic and hook the audience",
            "Setup": "Establish context and background",
            "Exposition": "Provide essential information and context",
            "Crossing Threshold": "Transition to main content",
            "Rising Action": "Build tension and interest",
            "Tests & Allies": "Present challenges and solutions",
            "Confrontation": "Address the main conflict or topic",
            "Approach": "Prepare for the climax",
            "Climax": "Deliver the most important information",
            "Ordeal": "Present the core challenge or revelation",
            "Falling Action": "Begin resolution and conclusion",
            "Reward": "Provide value and insights",
            "Resolution": "Wrap up and provide closure",
            "Return": "Bring the audience back to reality",
            "Denouement": "Final thoughts and call to action"
        }
        
        return purpose_map.get(scene_name, f"Develop the {niche} narrative")
    
    def _get_visual_elements(self, scene_name: str, niche: str, style: CinematicStyle) -> List[str]:
        """Get visual elements for the scene"""
        base_elements = {
            "Call to Adventure": ["hook_visual", "topic_introduction", "audience_connection"],
            "Setup": ["background_visuals", "context_images", "setting_establishment"],
            "Climax": ["key_visuals", "important_graphics", "highlighted_content"],
            "Resolution": ["summary_visuals", "conclusion_graphics", "call_to_action"]
        }
        
        # Add style-specific elements
        style_elements = {
            CinematicStyle.HOLLYWOOD: ["dynamic_transitions", "professional_graphics", "cinematic_angles"],
            CinematicStyle.DOCUMENTARY: ["authentic_imagery", "real_world_photos", "informative_graphics"],
            CinematicStyle.MODERN: ["clean_design", "minimal_graphics", "smooth_animations"],
            CinematicStyle.EPIC: ["grand_visuals", "dramatic_imagery", "powerful_graphics"]
        }
        
        elements = base_elements.get(scene_name, [f"{niche}_content", "relevant_visuals"])
        elements.extend(style_elements.get(style, ["standard_visuals"]))
        
        return elements
    
    def _get_audio_requirements(self, emotion: float, style: CinematicStyle) -> List[str]:
        """Get audio requirements based on emotion and style"""
        requirements = []
        
        # Emotion-based requirements
        if emotion < 0.3:
            requirements.extend(["calm_background", "soft_music", "gentle_narration"])
        elif emotion < 0.5:
            requirements.extend(["neutral_background", "balanced_music", "clear_narration"])
        elif emotion < 0.7:
            requirements.extend(["engaging_background", "building_music", "dynamic_narration"])
        elif emotion < 0.85:
            requirements.extend(["exciting_background", "intense_music", "powerful_narration"])
        else:
            requirements.extend(["thrilling_background", "epic_music", "dramatic_narration"])
        
        # Style-based requirements
        if style == CinematicStyle.HOLLYWOOD:
            requirements.extend(["professional_audio", "cinematic_sound_effects"])
        elif style == CinematicStyle.DOCUMENTARY:
            requirements.extend(["authentic_audio", "natural_sounds"])
        elif style == CinematicStyle.EPIC:
            requirements.extend(["epic_music", "dramatic_sound_effects"])
        
        return requirements
    
    def _get_cinematic_notes(self, scene_name: str, style: CinematicStyle) -> str:
        """Get cinematic direction notes for the scene"""
        notes_map = {
            "Call to Adventure": "Start with a strong visual hook that immediately captures attention",
            "Setup": "Establish visual rhythm and pacing for the content",
            "Climax": "Use the most impactful visuals and audio for maximum engagement",
            "Resolution": "Create a satisfying conclusion that reinforces the main message"
        }
        
        style_notes = {
            CinematicStyle.HOLLYWOOD: "Maintain professional, cinematic quality throughout",
            CinematicStyle.DOCUMENTARY: "Focus on authenticity and information clarity",
            CinematicStyle.MODERN: "Keep visuals clean and contemporary",
            CinematicStyle.EPIC: "Create grand, memorable moments"
        }
        
        base_note = notes_map.get(scene_name, "Maintain consistent visual and audio quality")
        style_note = style_notes.get(style, "Follow established cinematic guidelines")
        
        return f"{base_note}. {style_note}."
    
    def _predict_engagement(self, scenes: List[Dict], style: CinematicStyle, 
                           pacing: PacingType) -> float:
        """Predict overall engagement score"""
        try:
            if not scenes:
                return 0.5
            
            # Calculate average emotional engagement
            avg_emotion = sum(scene.get("engagement_target", 0.5) for scene in scenes) / len(scenes)
            
            # Style multiplier
            style_multipliers = {
                CinematicStyle.HOLLYWOOD: 1.1,
                CinematicStyle.EPIC: 1.15,
                CinematicStyle.MODERN: 1.05,
                CinematicStyle.DOCUMENTARY: 0.95,
                CinematicStyle.INDIE: 0.9
            }
            style_multiplier = style_multipliers.get(style, 1.0)
            
            # Pacing multiplier
            pacing_multipliers = {
                PacingType.FAST: 1.1,
                PacingType.MEDIUM: 1.0,
                PacingType.SLOW: 0.9,
                PacingType.BUILDING: 1.05,
                PacingType.EXPLOSIVE: 1.1
            }
            pacing_multiplier = pacing_multipliers.get(pacing, 1.0)
            
            # Calculate final engagement score
            engagement_score = avg_emotion * style_multiplier * pacing_multiplier
            
            # Ensure score is between 0 and 1
            return max(0.1, min(0.95, engagement_score))
            
        except Exception as e:
            self.logger.error(f"❌ Engagement prediction failed: {e}")
            return 0.6  # Default moderate engagement
    
    def _generate_optimization_suggestions(self, scenes: List[Dict], 
                                         style: CinematicStyle) -> List[str]:
        """Generate optimization suggestions for the story structure"""
        suggestions = []
        
        try:
            # Analyze scene distribution
            if len(scenes) < 3:
                suggestions.append("Consider adding more scenes for better pacing")
            elif len(scenes) > 8:
                suggestions.append("Consider consolidating scenes to maintain focus")
            
            # Analyze emotional arc
            emotions = [scene.get("engagement_target", 0.5) for scene in scenes]
            if emotions:
                emotion_range = max(emotions) - min(emotions)
                if emotion_range < 0.3:
                    suggestions.append("Increase emotional variation for more engaging content")
                elif emotion_range > 0.8:
                    suggestions.append("Consider smoothing emotional transitions for better flow")
            
            # Style-specific suggestions
            if style == CinematicStyle.HOLLYWOOD:
                suggestions.append("Maintain consistent professional quality across all scenes")
            elif style == CinematicStyle.DOCUMENTARY:
                suggestions.append("Ensure each scene provides valuable information")
            elif style == CinematicStyle.EPIC:
                suggestions.append("Create memorable visual moments in key scenes")
            
            # General suggestions
            suggestions.append("Ensure smooth transitions between scenes")
            suggestions.append("Maintain consistent visual and audio quality")
            suggestions.append("Optimize scene durations for audience retention")
            
        except Exception as e:
            self.logger.error(f"❌ Optimization suggestions generation failed: {e}")
            suggestions = ["Focus on maintaining consistent quality", "Ensure smooth pacing"]
        
        return suggestions
    
    def _create_fallback_story_structure(self, channel_name: str, niche: str, 
                                       target_duration: float) -> Dict[str, Any]:
        """Create fallback story structure when main creation fails"""
        try:
            self.logger.warning("⚠️ Using fallback story structure")
            
            # Simple 3-act structure
            scenes = [
                {
                    "scene_number": 1,
                    "name": "Introduction",
                    "duration_seconds": target_duration * 20,  # 20% of total
                    "duration_minutes": target_duration * 0.2,
                    "emotional_tone": "engaging",
                    "purpose": f"Introduce {niche} topic and hook the audience",
                    "visual_elements": [f"{niche}_introduction", "hook_visual"],
                    "audio_requirements": ["clear_narration", "engaging_background"],
                    "engagement_target": 0.6,
                    "cinematic_notes": "Start strong to capture attention"
                },
                {
                    "scene_number": 2,
                    "name": "Main Content",
                    "duration_seconds": target_duration * 60,  # 60% of total
                    "duration_minutes": target_duration * 0.6,
                    "emotional_tone": "exciting",
                    "purpose": f"Present main {niche} content and insights",
                    "visual_elements": [f"{niche}_content", "key_visuals"],
                    "audio_requirements": ["dynamic_narration", "building_music"],
                    "engagement_target": 0.8,
                    "cinematic_notes": "Maintain high engagement throughout"
                },
                {
                    "scene_number": 3,
                    "name": "Conclusion",
                    "duration_seconds": target_duration * 20,  # 20% of total
                    "duration_minutes": target_duration * 0.2,
                    "emotional_tone": "satisfying",
                    "purpose": f"Summarize {niche} insights and provide call to action",
                    "visual_elements": [f"{niche}_summary", "conclusion_graphics"],
                    "audio_requirements": ["clear_narration", "satisfying_music"],
                    "engagement_target": 0.7,
                    "cinematic_notes": "End with a strong, memorable conclusion"
                }
            ]
            
            return {
                "channel_name": channel_name,
                "niche": niche,
                "target_duration_minutes": target_duration,
                "cinematic_style": "fallback",
                "story_arc": "three_act",
                "pacing": "medium",
                "scenes": scenes,
                "emotional_arc": [0.6, 0.8, 0.7],
                "total_scenes": 3,
                "engagement_prediction": 0.7,
                "created_at": datetime.now().isoformat(),
                "optimization_suggestions": ["Use fallback structure as starting point", "Customize based on specific needs"],
                "fallback_used": True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback story structure creation failed: {e}")
            return {
                "channel_name": channel_name,
                "niche": niche,
                "target_duration_minutes": target_duration,
                "error": f"Story structure creation failed: {e}",
                "created_at": datetime.now().isoformat()
            }

    def expand_script_to_cinematic_length(self, base_script: str, target_minutes: float = 15.0, 
                                        niche: str = "general") -> Dict[str, Any]:
        """
        Expand a basic script to cinematic length with proper story structure
        
        Args:
            base_script: Original script text
            target_minutes: Target duration in minutes
            niche: Content niche (history, motivation, finance, etc.)
            
        Returns:
            Expanded script with cinematic structure
        """
        try:
            # Calculate target word count (150-200 words per minute for cinematic content)
            target_words = int(target_minutes * 175)
            
            # Analyze current script
            current_words = len(base_script.split())
            expansion_needed = max(0, target_words - current_words)
            
            # Get story structure for the niche
            story_structure = self._get_story_structure_for_niche(niche, target_minutes)
            
            # Generate expanded content
            expanded_content = self._generate_cinematic_expansion(
                base_script, story_structure, expansion_needed, niche
            )
            
            # Structure the expanded content
            structured_script = self._structure_cinematic_content(expanded_content, story_structure)
            
            return {
                "original_script": base_script,
                "expanded_script": structured_script,
                "target_duration_minutes": target_minutes,
                "estimated_duration_minutes": self._calculate_estimated_duration(structured_script),
                "story_structure": story_structure,
                "scene_breakdown": self._create_scene_breakdown(structured_script, story_structure),
                "visual_requirements": self._generate_visual_requirements(structured_script, niche),
                "audio_requirements": self._generate_audio_requirements(structured_script, niche),
                "engagement_curve": self._generate_engagement_curve(story_structure)
            }
            
        except Exception as e:
            logging.error(f"Script expansion failed: {e}")
            return {"error": str(e)}
    
    def _get_story_structure_for_niche(self, niche: str, target_minutes: float) -> StoryStructure:
        """Get appropriate story structure for niche and duration"""
        if niche in ["history", "documentary"]:
            return self.story_structures["documentary"]
        elif niche in ["motivation", "inspiration"]:
            return self.story_structures["epic"]
        else:
            # Default to epic structure for most niches
            return self.story_structures["epic"]

    def _get_beat_purpose(self, beat: EmotionalBeat) -> str:
        """Get the purpose of an emotional beat"""
        beat_purposes = {
            EmotionalBeat.OPENING: "Establish context and hook the audience",
            EmotionalBeat.HOOK: "Grab attention and create immediate interest",
            EmotionalBeat.CONFLICT: "Introduce tension and build engagement",
            EmotionalBeat.CLIMAX: "Peak emotional moment and key revelation",
            EmotionalBeat.RESOLUTION: "Provide satisfying conclusion and call to action",
            EmotionalBeat.TWIST: "Unexpected development that changes perspective",
            EmotionalBeat.REVELATION: "Important information that transforms understanding"
        }
        
        return beat_purposes.get(beat, "Advance the narrative")

    def _extract_scene_content(self, expanded_content: str, scene_index: int) -> str:
        """Extract content for a specific scene"""
        try:
            # Split content into sections
            sections = expanded_content.split('\n\n')
            
            if scene_index < len(sections):
                return sections[scene_index].strip()
            else:
                # Fallback: return a portion of the content
                words = expanded_content.split()
                start_idx = (scene_index * len(words)) // 5  # 5 scenes
                end_idx = ((scene_index + 1) * len(words)) // 5
                return ' '.join(words[start_idx:end_idx])
                
        except Exception as e:
            logging.error(f"Scene content extraction failed: {e}")
            return "Scene content"

    def _get_visual_requirements_for_beat(self, beat: EmotionalBeat) -> Dict[str, Any]:
        """Get visual requirements for an emotional beat"""
        visual_requirements = {
            EmotionalBeat.OPENING: {
                "style": "establishing_shot",
                "mood": "calm",
                "color_palette": "neutral",
                "movement": "slow"
            },
            EmotionalBeat.HOOK: {
                "style": "dynamic",
                "mood": "exciting",
                "color_palette": "vibrant",
                "movement": "fast"
            },
            EmotionalBeat.CONFLICT: {
                "style": "tension",
                "mood": "intense",
                "color_palette": "dramatic",
                "movement": "building"
            },
            EmotionalBeat.CLIMAX: {
                "style": "epic",
                "mood": "overwhelming",
                "color_palette": "intense",
                "movement": "explosive"
            },
            EmotionalBeat.RESOLUTION: {
                "style": "peaceful",
                "mood": "satisfying",
                "color_palette": "warm",
                "movement": "gentle"
            }
        }
        
        return visual_requirements.get(beat, {
            "style": "standard",
            "mood": "neutral",
            "color_palette": "balanced",
            "movement": "moderate"
        })

    def _get_audio_requirements_for_beat(self, beat: EmotionalBeat) -> Dict[str, Any]:
        """Get audio requirements for an emotional beat"""
        audio_requirements = {
            EmotionalBeat.OPENING: {
                "music_style": "ambient",
                "volume": 0.6,
                "tempo": "slow",
                "instruments": ["strings", "piano"]
            },
            EmotionalBeat.HOOK: {
                "music_style": "energetic",
                "volume": 0.8,
                "tempo": "fast",
                "instruments": ["drums", "brass", "strings"]
            },
            EmotionalBeat.CONFLICT: {
                "music_style": "tension",
                "volume": 0.7,
                "tempo": "building",
                "instruments": ["strings", "percussion"]
            },
            EmotionalBeat.CLIMAX: {
                "music_style": "epic",
                "volume": 1.0,
                "tempo": "maximum",
                "instruments": ["full_orchestra", "choir"]
            },
            EmotionalBeat.RESOLUTION: {
                "music_style": "peaceful",
                "volume": 0.5,
                "tempo": "slowing",
                "instruments": ["piano", "soft_strings"]
            }
        }
        
        return audio_requirements.get(beat, {
            "music_style": "neutral",
            "volume": 0.7,
            "tempo": "moderate",
            "instruments": ["strings"]
        })

    def _split_audio_segments(self, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split scene into audio segments"""
        try:
            duration = scene.get("duration_seconds", 10)
            content = scene.get("content", "")
            
            # Split content into sentences
            sentences = content.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return [{"start": 0, "end": duration, "content": content}]
            
            # Calculate segment duration
            segment_duration = duration / len(sentences)
            
            segments = []
            current_time = 0
            
            for sentence in sentences:
                end_time = min(current_time + segment_duration, duration)
                segments.append({
                    "start": current_time,
                    "end": end_time,
                    "content": sentence,
                    "duration": end_time - current_time
                })
                current_time = end_time
                
                if current_time >= duration:
                    break
            
            return segments
            
        except Exception as e:
            logging.error(f"Audio segment splitting failed: {e}")
            return [{"start": 0, "end": duration, "content": content}]

    def _get_transition_recommendations(self, scene: Dict[str, Any]) -> List[str]:
        """Get transition recommendations for a scene"""
        emotional_beat = scene.get("emotional_beat", "neutral")
        
        transition_map = {
            "opening": ["fade_in", "zoom_in", "slide_in"],
            "hook": ["quick_cut", "zoom_blast", "slide_transition"],
            "conflict": ["crossfade", "dissolve", "push_transition"],
            "climax": ["explosive_cut", "zoom_explosion", "dramatic_transition"],
            "resolution": ["gentle_fade", "soft_dissolve", "peaceful_transition"]
        }
        
        return transition_map.get(emotional_beat, ["crossfade", "dissolve"])

    def _get_special_effects_recommendations(self, scene: Dict[str, Any]) -> List[str]:
        """Get special effects recommendations for a scene"""
        emotional_beat = scene.get("emotional_beat", "neutral")
        
        effects_map = {
            "opening": ["color_grading", "subtle_glow", "atmospheric_fog"],
            "hook": ["dynamic_movement", "color_pop", "energy_trails"],
            "conflict": ["tension_effects", "dramatic_lighting", "motion_blur"],
            "climax": ["explosive_effects", "intense_lighting", "particle_systems"],
            "resolution": ["soft_glow", "warm_colors", "gentle_movement"]
        }
        
        return effects_map.get(emotional_beat, ["color_grading", "subtle_enhancement"])

    def _get_visual_style_for_beat(self, emotional_beat: str, niche: str) -> str:
        """Get visual style for an emotional beat"""
        style_map = {
            "opening": "establishing",
            "hook": "dynamic",
            "conflict": "tension",
            "climax": "epic",
            "resolution": "peaceful"
        }
        
        base_style = style_map.get(emotional_beat, "standard")
        
        # Add niche-specific modifiers
        niche_modifiers = {
            "history": "vintage",
            "motivation": "inspirational",
            "finance": "modern",
            "automotive": "dynamic",
            "combat": "intense"
        }
        
        niche_modifier = niche_modifiers.get(niche, "")
        if niche_modifier:
            return f"{niche_modifier}_{base_style}"
        
        return base_style

    def _get_mood_for_beat(self, emotional_beat: str) -> str:
        """Get mood for an emotional beat"""
        mood_map = {
            "opening": "calm",
            "hook": "exciting",
            "conflict": "tense",
            "climax": "overwhelming",
            "resolution": "satisfying"
        }
        
        return mood_map.get(emotional_beat, "neutral")

    def _calculate_estimated_duration(self, structured_script: Dict[str, Any]) -> float:
        """Calculate estimated duration for the structured script"""
        try:
            # Get total duration from structured script
            total_duration_seconds = structured_script.get("total_duration_seconds", 0)
            
            if total_duration_seconds > 0:
                return total_duration_seconds / 60.0  # Convert to minutes
            
            # Fallback: estimate based on content length
            full_script = structured_script.get("full_script", "")
            word_count = len(full_script.split())
            
            # Assume 150-200 words per minute for cinematic content
            estimated_minutes = word_count / 175.0
            
            return max(1.0, estimated_minutes)  # Minimum 1 minute
            
        except Exception as e:
            logging.error(f"Duration calculation failed: {e}")
            return 15.0  # Default 15 minutes

    def _generate_cinematic_expansion(self, base_script: str, story_structure: StoryStructure, 
                                    expansion_needed: int, niche: str) -> str:
        """Generate cinematic expansion of the base script"""
        try:
            # Use Ollama to expand the script
            prompt = f"""
            Expand this script to create a {story_structure.duration_minutes}-minute cinematic masterpiece.
            
            Original script: {base_script}
            
            Requirements:
            - Target length: {expansion_needed} additional words
            - Niche: {niche}
            - Story arc: {story_structure.arc_type.value}
            - Emotional beats: {[beat.value for beat in story_structure.emotional_beats]}
            - Pacing: {story_structure.pacing.value}
            
            Create engaging, cinematic content that:
            1. Hooks viewers in the first 30 seconds
            2. Builds emotional connection throughout
            3. Includes specific visual descriptions
            4. Has natural speech patterns
            5. Ends with powerful impact
            
            Return only the expanded script text, no explanations.
            """
            
            # This would integrate with your Ollama system
            # For now, return a structured expansion template
            return self._create_expansion_template(base_script, story_structure, niche)
            
        except Exception as e:
            logging.error(f"Cinematic expansion generation failed: {e}")
            return base_script
    
    def _create_expansion_template(self, base_script: str, story_structure: StoryStructure, 
                                 niche: str) -> str:
        """Create expansion template based on story structure"""
        template_sections = []
        
        # Opening hook (0-2 minutes)
        template_sections.append(f"""
        [OPENING HOOK - 0:00-2:00]
        {base_script}
        
        [EXPANSION: Add compelling opening that immediately grabs attention]
        """)
        
        # Problem establishment (2-5 minutes)
        template_sections.append(f"""
        [PROBLEM ESTABLISHMENT - 2:00-5:00]
        [EXPANSION: Deep dive into the core issue, add context and background]
        """)
        
        # Rising action (5-10 minutes)
        template_sections.append(f"""
        [RISING ACTION - 5:00-10:00]
        [EXPANSION: Build tension, add examples, create emotional investment]
        """)
        
        # Climax (10-13 minutes)
        template_sections.append(f"""
        [CLIMAX - 10:00-13:00]
        [EXPANSION: Peak emotional moment, key revelation, turning point]
        """)
        
        # Resolution (13-15 minutes)
        template_sections.append(f"""
        [RESOLUTION - 13:00-15:00]
        [EXPANSION: Satisfying conclusion, call to action, lasting impact]
        """)
        
        return "\n\n".join(template_sections)
    
    def _structure_cinematic_content(self, expanded_content: str, 
                                   story_structure: StoryStructure) -> Dict[str, Any]:
        """Structure the expanded content into cinematic format"""
        scenes = []
        current_time = 0.0
        
        # Split content into scenes based on emotional beats
        for i, beat in enumerate(story_structure.emotional_beats):
            scene_duration = story_structure.duration_minutes * 60 / len(story_structure.emotional_beats)
            
            scene = {
                "scene_number": i + 1,
                "emotional_beat": beat.value,
                "start_time": current_time,
                "end_time": current_time + scene_duration,
                "duration_seconds": scene_duration,
                "purpose": self._get_beat_purpose(beat),
                "content": self._extract_scene_content(expanded_content, i),
                "visual_requirements": self._get_visual_requirements_for_beat(beat),
                "audio_requirements": self._get_audio_requirements_for_beat(beat)
            }
            
            scenes.append(scene)
            current_time += scene_duration
        
        return {
            "full_script": expanded_content,
            "scenes": scenes,
            "total_duration_seconds": story_structure.duration_minutes * 60,
            "story_arc": story_structure.arc_type.value
        }
    
    def _create_scene_breakdown(self, structured_script: Dict[str, Any], 
                               story_structure: StoryStructure) -> List[Dict[str, Any]]:
        """Create detailed scene breakdown for video production"""
        scene_breakdown = []
        
        for scene in structured_script.get("scenes", []):
            breakdown = {
                "scene_number": scene["scene_number"],
                "duration_seconds": scene["duration_seconds"],
                "emotional_beat": scene["emotional_beat"],
                "visual_assets_needed": self._calculate_visual_assets_needed(scene),
                "audio_segments": self._split_audio_segments(scene),
                "transitions": self._get_transition_recommendations(scene),
                "special_effects": self._get_special_effects_recommendations(scene)
            }
            scene_breakdown.append(breakdown)
        
        return scene_breakdown
    
    def _calculate_visual_assets_needed(self, scene: Dict[str, Any]) -> int:
        """Calculate how many visual assets needed for a scene"""
        # Rule: 1 visual asset per 3-5 seconds for cinematic quality
        base_assets = max(3, int(scene["duration_seconds"] / 4))
        
        # Add more assets for high-impact scenes
        if scene["emotional_beat"] in ["climax", "hook"]:
            base_assets = int(base_assets * 1.5)
        
        return base_assets
    
    def _generate_visual_requirements(self, structured_script: Dict[str, Any], 
                                    niche: str) -> Dict[str, Any]:
        """Generate comprehensive visual requirements"""
        total_assets_needed = 0
        visual_styles = []
        
        for scene in structured_script.get("scenes", []):
            assets_count = self._calculate_visual_assets_needed(scene)
            total_assets_needed += assets_count
            
            # Determine visual style for each scene
            style = self._get_visual_style_for_beat(scene["emotional_beat"], niche)
            visual_styles.append({
                "scene": scene["scene_number"],
                "style": style,
                "assets_count": assets_count,
                "mood": self._get_mood_for_beat(scene["emotional_beat"])
            })
        
        return {
            "total_visual_assets_needed": total_assets_needed,
            "scene_visual_styles": visual_styles,
            "quality_requirements": "4K cinematic quality",
            "diversity_requirements": "No repeated visuals within 30 seconds",
            "transition_requirements": "Smooth crossfades and cinematic transitions"
        }
    
    def _generate_audio_requirements(self, structured_script: Dict[str, Any], 
                                   niche: str) -> Dict[str, Any]:
        """Generate comprehensive audio requirements"""
        return {
            "voice_acting_quality": "Professional studio quality",
            "background_music": f"{niche}-themed cinematic score",
            "sound_effects": "Atmospheric and immersive",
            "audio_mixing": "Professional cinema standards",
            "duration": f"{structured_script.get('total_duration_seconds', 0) / 60:.1f} minutes"
        }
    
    def _generate_engagement_curve(self, story_structure: StoryStructure) -> List[float]:
        """Generate engagement curve for the story"""
        # Create engagement curve that peaks at climax
        total_beats = len(story_structure.emotional_beats)
        curve = []
        
        for i, beat in enumerate(story_structure.emotional_beats):
            if beat == EmotionalBeat.HOOK:
                engagement = 0.9  # High engagement at hook
            elif beat == EmotionalBeat.CLIMAX:
                engagement = 1.0  # Peak engagement at climax
            elif beat == EmotionalBeat.OPENING:
                engagement = 0.7  # Moderate opening
            elif beat == EmotionalBeat.RESOLUTION:
                engagement = 0.8  # Strong ending
            else:
                engagement = 0.75 + (i / total_beats) * 0.2  # Gradual build
            
            curve.append(engagement)
        
        return curve

    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent conflicts"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("✅ GPU memory cleared")
        except Exception as e:
            self.logger.debug(f"GPU memory clearing not available: {e}")
    
    def get_capabilities(self) -> List[str]:
        """Get available capabilities"""
        capabilities = ['story_creation', 'pacing_control', 'emotional_arcs']
        
        if self.sentiment_analyzer:
            capabilities.append('sentiment_analysis')
        
        if self.engagement_predictor:
            capabilities.append('engagement_prediction')
        
        return capabilities

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_cinematic_director(config: Optional[Dict] = None) -> CinematicAIDirector:
    """Create and initialize Cinematic AI Director"""
    return CinematicAIDirector(config)

def analyze_story_structure(story_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze existing story structure"""
    try:
        director = CinematicAIDirector()
        return director._analyze_existing_story(story_data)
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

# =============================================================================
# TEST & DEMO
# =============================================================================

if __name__ == "__main__":
    print("🎬 Testing Cinematic AI Director...")
    
    # Create director
    director = CinematicAIDirector()
    
    # Test story structure creation
    story = director.create_story_structure(
        channel_name="CKLegends",
        niche="history",
        target_duration_minutes=15,
        style=CinematicStyle.HOLLYWOOD,
        arc_type=StoryArc.HERO_JOURNEY,
        pacing=PacingType.MEDIUM
    )
    
    print(f"Story Structure: {json.dumps(story, indent=2, default=str)}")
    print("✅ Cinematic AI Director test completed!")
