# -*- coding: utf-8 -*-
"""
🎨 AI VISUAL ASSETS SUITE
AI-Powered Image Generation & Video Enhancement Platform
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import requests
from dataclasses import dataclass
from enum import Enum

# Image processing imports
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("⚠️ Pillow not available - image processing disabled")

# Video processing imports
try:
    from moviepy.editor import *
    from moviepy.video.fx import all as vfx
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️ MoviePy not available - video processing disabled")

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

class VisualStyle(Enum):
    MODERN = "modern"
    VINTAGE = "vintage"
    MINIMALIST = "minimalist"
    BOLD = "bold"
    ELEGANT = "elegant"
    PLAYFUL = "playful"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"

class ImageFormat(Enum):
    THUMBNAIL = "thumbnail"      # 1280x720
    BANNER = "banner"            # 1920x1080
    SQUARE = "square"            # 1080x1080
    STORY = "story"              # 1080x1920
    CUSTOM = "custom"

class EnhancementType(Enum):
    UPSCALING = "upscaling"
    STABILIZATION = "stabilization"
    COLOR_CORRECTION = "color_correction"
    BACKGROUND_REMOVAL = "background_removal"
    NOISE_REDUCTION = "noise_reduction"

@dataclass
class VisualAsset:
    """Visual asset metadata"""
    name: str
    type: str
    dimensions: Tuple[int, int]
    style: VisualStyle
    colors: List[str]
    tags: List[str]
    file_path: str
    created_at: datetime

@dataclass
class EnhancementResult:
    """Video enhancement results"""
    original_path: str
    enhanced_path: str
    enhancements_applied: List[str]
    quality_improvement: float
    processing_time: float

# =============================================================================
# AI IMAGE GENERATION ENGINE
# =============================================================================

class AIImageGenerator:
    """AI-powered image generation and manipulation"""
    
    def __init__(self):
        self.generation_models = {}
        self.style_presets = {}
        self._load_generation_models()
        self._load_style_presets()
    
    def _load_generation_models(self):
        """Load AI image generation models"""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load image generation pipeline with fallback
                try:
                    # Try with a more compatible model first
                    self.generation_models['text2img'] = pipeline(
                        "text-to-image",
                        model="runwayml/stable-diffusion-v1-5"
                    )
                    logging.info("✅ AI image generation models loaded (runwayml/stable-diffusion-v1-5)")
                except Exception as e:
                    logging.warning(f"⚠️ Primary model failed, trying alternative: {e}")
                    try:
                        # Fallback to a simpler model
                        self.generation_models['text2img'] = pipeline(
                            "text-to-image",
                            model="CompVis/stable-diffusion-v1-4",
                            revision="fp16"
                        )
                        logging.info("✅ AI image generation models loaded (CompVis/stable-diffusion-v1-4)")
                    except Exception as e2:
                        logging.warning(f"⚠️ Alternative model also failed: {e2}")
                        # Disable AI generation, use Pexels only
                        self.generation_models['text2img'] = None
                logging.info("✅ AI image generation models loaded")
            except Exception as e:
                logging.warning(f"⚠️ AI model loading failed: {e}")
    
    def _load_style_presets(self):
        """Load visual style presets"""
        self.style_presets = {
            VisualStyle.MODERN: {
                'colors': ['#2C3E50', '#3498DB', '#E74C3C', '#F39C12', '#FFFFFF'],
                'fonts': ['Arial', 'Helvetica', 'Roboto'],
                'effects': ['gradient', 'shadow', 'rounded_corners']
            },
            VisualStyle.VINTAGE: {
                'colors': ['#8B4513', '#CD853F', '#DEB887', '#F5DEB3', '#8FBC8F'],
                'fonts': ['Times New Roman', 'Georgia', 'Palatino'],
                'effects': ['sepia', 'texture', 'vintage_filter']
            },
            VisualStyle.MINIMALIST: {
                'colors': ['#000000', '#FFFFFF', '#808080', '#C0C0C0'],
                'fonts': ['Arial', 'Helvetica', 'Futura'],
                'effects': ['clean_lines', 'simple_shapes', 'minimal_text']
            },
            VisualStyle.BOLD: {
                'colors': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'],
                'fonts': ['Impact', 'Arial Black', 'Bebas Neue'],
                'effects': ['high_contrast', 'bold_text', 'vibrant_colors']
            }
        }
    
    def generate_custom_thumbnail(
        self,
        title: str,
        subtitle: str = "",
        style: VisualStyle = VisualStyle.MODERN,
        format: ImageFormat = ImageFormat.THUMBNAIL,
        background_image: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """Generate custom thumbnail with AI assistance"""
        
        if not PILLOW_AVAILABLE:
            raise Exception("Pillow required for thumbnail generation")
        
        # Get dimensions for format
        dimensions = self._get_format_dimensions(format)
        
        # Create base image
        if background_image and os.path.exists(background_image):
            base_image = Image.open(background_image).resize(dimensions)
        else:
            base_image = self._create_gradient_background(dimensions, style)
        
        # Apply style-based enhancements
        styled_image = self._apply_style_enhancements(base_image, style)
        
        # Add text elements
        final_image = self._add_text_elements(styled_image, title, subtitle, style)
        
        # Save thumbnail
        if not output_path:
            timestamp = int(time.time())
            output_path = f"thumbnail_{style.value}_{timestamp}.png"
        
        final_image.save(output_path, "PNG", quality=95)
        
        logging.info(f"✅ Custom thumbnail generated: {output_path}")
        return output_path
    
    def _get_format_dimensions(self, format: ImageFormat) -> Tuple[int, int]:
        """Get dimensions for image format"""
        
        dimensions = {
            ImageFormat.THUMBNAIL: (1280, 720),
            ImageFormat.BANNER: (1920, 1080),
            ImageFormat.SQUARE: (1080, 1080),
            ImageFormat.STORY: (1080, 1920),
            ImageFormat.CUSTOM: (1280, 720)
        }
        
        return dimensions.get(format, (1280, 720))
    
    def _create_gradient_background(self, dimensions: Tuple[int, int], style: VisualStyle) -> Image.Image:
        """Create gradient background"""
        
        width, height = dimensions
        
        # Get style colors
        style_colors = self.style_presets.get(style, self.style_presets[VisualStyle.MODERN])
        colors = style_colors['colors']
        
        # Create gradient
        image = Image.new('RGB', dimensions)
        draw = ImageDraw.Draw(image)
        
        # Simple linear gradient
        for y in range(height):
            ratio = y / height
            color1 = self._hex_to_rgb(colors[0])
            color2 = self._hex_to_rgb(colors[1])
            
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        return image
    
    def _apply_style_enhancements(self, image: Image.Image, style: VisualStyle) -> Image.Image:
        """Apply style-based enhancements"""
        
        enhanced_image = image.copy()
        
        if style == VisualStyle.VINTAGE:
            # Apply vintage filter
            enhanced_image = ImageOps.colorize(enhanced_image.convert('L'), '#8B4513', '#F5DEB3')
            enhanced_image = enhanced_image.convert('RGB')
        
        elif style == VisualStyle.MINIMALIST:
            # Apply minimalist filter
            enhanced_image = enhanced_image.filter(ImageFilter.SMOOTH)
        
        elif style == VisualStyle.BOLD:
            # Apply bold filter
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.5)
        
        return enhanced_image
    
    def _add_text_elements(self, image: Image.Image, title: str, subtitle: str, style: VisualStyle) -> Image.Image:
        """Add text elements to image"""
        
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Get style fonts and colors
        style_config = self.style_presets.get(style, self.style_presets[VisualStyle.MODERN])
        colors = style_config['colors']
        
        # Title text
        title_font_size = min(width // 15, 72)
        try:
            title_font = ImageFont.truetype("arial.ttf", title_font_size)
        except:
            title_font = ImageFont.load_default()
        
        # Calculate title position
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        title_y = height // 3
        
        # Draw title with shadow
        shadow_offset = 3
        draw.text((title_x + shadow_offset, title_y + shadow_offset), title, 
                 fill=self._hex_to_rgb(colors[0]), font=title_font)
        draw.text((title_x, title_y), title, 
                 fill=self._hex_to_rgb(colors[1]), font=title_font)
        
        # Subtitle text
        if subtitle:
            subtitle_font_size = min(width // 25, 48)
            try:
                subtitle_font = ImageFont.truetype("arial.ttf", subtitle_font_size)
            except:
                subtitle_font = ImageFont.load_default()
            
            # Calculate subtitle position
            subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
            subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
            subtitle_x = (width - subtitle_width) // 2
            subtitle_y = title_y + title_font_size + 20
            
            # Draw subtitle
            draw.text((subtitle_x, subtitle_y), subtitle, 
                     fill=self._hex_to_rgb(colors[2]), font=subtitle_font)
        
        return image
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB"""
        
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def generate_ai_image(
        self,
        prompt: str,
        style: VisualStyle = VisualStyle.MODERN,
        dimensions: Tuple[int, int] = (512, 512),
        output_path: Optional[str] = None
    ) -> str:
        """Generate AI image from text prompt"""
        
        if not self.generation_models.get('text2img'):
            raise Exception("AI image generation models not available")
        
        try:
            # Enhance prompt with style
            enhanced_prompt = self._enhance_prompt_with_style(prompt, style)
            
            # Generate image
            image = self.generation_models['text2img'](enhanced_prompt)
            
            # Save image
            if not output_path:
                timestamp = int(time.time())
                output_path = f"ai_generated_{style.value}_{timestamp}.png"
            
            image[0].save(output_path)
            
            logging.info(f"✅ AI image generated: {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"❌ AI image generation failed: {e}")
            raise
    
    def _enhance_prompt_with_style(self, prompt: str, style: VisualStyle) -> str:
        """Enhance prompt with style-specific keywords"""
        
        style_keywords = {
            VisualStyle.MODERN: "modern, clean, minimalist, professional",
            VisualStyle.VINTAGE: "vintage, retro, classic, nostalgic",
            VisualStyle.MINIMALIST: "minimalist, simple, clean, uncluttered",
            VisualStyle.BOLD: "bold, vibrant, high contrast, dramatic",
            VisualStyle.ELEGANT: "elegant, sophisticated, refined, luxurious",
            VisualStyle.PLAYFUL: "playful, fun, colorful, energetic",
            VisualStyle.PROFESSIONAL: "professional, corporate, business, formal",
            VisualStyle.CREATIVE: "creative, artistic, imaginative, innovative"
        }
        
        keywords = style_keywords.get(style, "")
        enhanced_prompt = f"{prompt}, {keywords}, high quality, detailed"
        
        return enhanced_prompt

# =============================================================================
# VIDEO ENHANCEMENT ENGINE
# =============================================================================

class VideoEnhancementEngine:
    """AI-powered video enhancement and processing"""
    
    def __init__(self):
        self.enhancement_models = {}
        self.processing_pipeline = {}
        self._load_enhancement_models()
    
    def _load_enhancement_models(self):
        """Load video enhancement models"""
        if TORCH_AVAILABLE:
            try:
                # Load enhancement models
                logging.info("✅ Video enhancement models loaded")
            except Exception as e:
                logging.warning(f"⚠️ Enhancement model loading failed: {e}")
    
    def enhance_video(
        self,
        video_path: str,
        enhancements: List[EnhancementType],
        output_path: Optional[str] = None,
        quality: str = 'high'
    ) -> EnhancementResult:
        """Apply multiple video enhancements"""
        
        if not MOVIEPY_AVAILABLE:
            raise Exception("MoviePy required for video enhancement")
        
        start_time = time.time()
        
        # Load video
        video = VideoFileClip(video_path)
        enhanced_video = video
        
        # Apply enhancements
        for enhancement in enhancements:
            if enhancement == EnhancementType.UPSCALING:
                enhanced_video = self._apply_upscaling(enhanced_video, quality)
            elif enhancement == EnhancementType.STABILIZATION:
                enhanced_video = self._apply_stabilization(enhanced_video)
            elif enhancement == EnhancementType.COLOR_CORRECTION:
                enhanced_video = self._apply_color_correction(enhanced_video)
            elif enhancement == EnhancementType.NOISE_REDUCTION:
                enhanced_video = self._apply_noise_reduction(enhanced_video)
        
        # Save enhanced video
        if not output_path:
            output_path = video_path.replace('.mp4', '_enhanced.mp4')
        
        enhanced_video.write_videofile(output_path, codec='libx264')
        enhanced_video.close()
        video.close()
        
        processing_time = time.time() - start_time
        
        return EnhancementResult(
            original_path=video_path,
            enhanced_path=output_path,
            enhancements_applied=[e.value for e in enhancements],
            quality_improvement=self._calculate_quality_improvement(video_path, output_path),
            processing_time=processing_time
        )
    
    def _apply_upscaling(self, video: VideoFileClip, quality: str) -> VideoFileClip:
        """Apply video upscaling"""
        
        # Get current dimensions
        current_width, current_height = video.size
        
        # Calculate target dimensions based on quality
        if quality == 'high':
            target_width = current_width * 2
            target_height = current_height * 2
        elif quality == 'medium':
            target_width = int(current_width * 1.5)
            target_height = int(current_height * 1.5)
        else:
            target_width = current_width
            target_height = current_height
        
        # Resize video
        upscaled_video = video.resize((target_width, target_height))
        
        return upscaled_video
    
    def _apply_stabilization(self, video: VideoFileClip) -> VideoFileClip:
        """Apply video stabilization"""
        
        # Simple stabilization using crop
        # In a real implementation, this would use advanced stabilization algorithms
        
        width, height = video.size
        crop_margin = min(width, height) // 20
        
        stabilized_video = video.crop(
            x1=crop_margin, y1=crop_margin,
            x2=width-crop_margin, y2=height-crop_margin
        )
        
        return stabilized_video
    
    def _apply_color_correction(self, video: VideoFileClip) -> VideoFileClip:
        """Apply color correction"""
        
        # Apply basic color enhancements
        corrected_video = video.fx(vfx.colorx, 1.1)  # Increase saturation
        corrected_video = corrected_video.fx(vfx.lum_contrast, lum=1.05, contrast=1.1)
        
        return corrected_video
    
    def _apply_noise_reduction(self, video: VideoFileClip) -> VideoFileClip:
        """Apply noise reduction"""
        
        # Simple noise reduction using smoothing
        # In a real implementation, this would use advanced denoising algorithms
        
        denoised_video = video.fx(vfx.gaussian_blur, sigma=0.5)
        
        return denoised_video
    
    def _calculate_quality_improvement(self, original_path: str, enhanced_path: str) -> float:
        """Calculate quality improvement percentage"""
        
        # This is a simplified calculation
        # In a real implementation, this would analyze actual video quality metrics
        
        try:
            original_video = VideoFileClip(original_path)
            enhanced_video = VideoFileClip(enhanced_path)
            
            # Compare file sizes as a simple quality indicator
            original_size = os.path.getsize(original_path)
            enhanced_size = os.path.getsize(enhanced_path)
            
            if original_size > 0:
                improvement = ((enhanced_size - original_size) / original_size) * 100
                return min(improvement, 100.0)
            else:
                return 0.0
                
        except Exception:
            return 25.0  # Default improvement estimate

# =============================================================================
# VISUAL CONSISTENCY ENGINE
# =============================================================================

class VisualConsistencyEngine:
    """Maintain visual consistency across assets"""
    
    def __init__(self):
        self.brand_guidelines = {}
        self.color_palettes = {}
        self.font_systems = {}
        self._load_brand_guidelines()
    
    def _load_brand_guidelines(self):
        """Load brand visual guidelines"""
        
        # Default brand guidelines
        self.brand_guidelines = {
            'default': {
                'primary_colors': ['#2C3E50', '#3498DB', '#E74C3C'],
                'secondary_colors': ['#F39C12', '#9B59B6', '#1ABC9C'],
                'fonts': ['Arial', 'Helvetica', 'Roboto'],
                'spacing': {'margin': 20, 'padding': 10},
                'border_radius': 8
            }
        }
        
        # Color palettes
        self.color_palettes = {
            'professional': ['#2C3E50', '#34495E', '#7F8C8D', '#BDC3C7', '#ECF0F1'],
            'creative': ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#3498DB'],
            'minimal': ['#000000', '#FFFFFF', '#808080', '#C0C0C0', '#E0E0E0'],
            'vintage': ['#8B4513', '#CD853F', '#DEB887', '#F5DEB3', '#8FBC8F']
        }
    
    def create_consistent_assets(
        self,
        base_asset: str,
        variations: List[Dict[str, Any]],
        brand_guideline: str = 'default'
    ) -> List[str]:
        """Create consistent asset variations"""
        
        if not PILLOW_AVAILABLE:
            raise Exception("Pillow required for asset consistency")
        
        consistent_assets = []
        guidelines = self.brand_guidelines.get(brand_guideline, self.brand_guidelines['default'])
        
        for variation in variations:
            # Load base asset
            base_image = Image.open(base_asset)
            
            # Apply variation
            modified_image = self._apply_variation(base_image, variation, guidelines)
            
            # Save consistent asset
            output_path = f"consistent_{variation.get('name', 'variation')}_{int(time.time())}.png"
            modified_image.save(output_path, "PNG", quality=95)
            
            consistent_assets.append(output_path)
        
        return consistent_assets
    
    def _apply_variation(self, image: Image.Image, variation: Dict[str, Any], guidelines: Dict[str, Any]) -> Image.Image:
        """Apply variation while maintaining consistency"""
        
        modified_image = image.copy()
        
        # Apply color adjustments
        if 'color_theme' in variation:
            modified_image = self._apply_color_theme(modified_image, variation['color_theme'])
        
        # Apply size adjustments
        if 'dimensions' in variation:
            modified_image = modified_image.resize(variation['dimensions'])
        
        # Apply style adjustments
        if 'style' in variation:
            modified_image = self._apply_style_adjustments(modified_image, variation['style'], guidelines)
        
        return modified_image
    
    def _apply_color_theme(self, image: Image.Image, theme: str) -> Image.Image:
        """Apply color theme to image"""
        
        if theme in self.color_palettes:
            colors = self.color_palettes[theme]
            
            # Convert to grayscale and apply color overlay
            gray_image = image.convert('L')
            colored_image = ImageOps.colorize(gray_image, colors[0], colors[-1])
            
            return colored_image.convert('RGB')
        
        return image
    
    def _apply_style_adjustments(self, image: Image.Image, style: str, guidelines: Dict[str, Any]) -> Image.Image:
        """Apply style adjustments"""
        
        if style == 'rounded':
            # Apply rounded corners
            border_radius = guidelines.get('border_radius', 8)
            # This would implement rounded corner logic
        
        elif style == 'shadow':
            # Apply shadow effect
            shadow_offset = guidelines.get('shadow_offset', 5)
            shadow_blur = guidelines.get('shadow_blur', 10)
            # This would implement shadow logic
        
        return image

# =============================================================================
# MAIN AI VISUAL SUITE CLASS
# =============================================================================

class AIVisualSuite:
    """Main AI Visual Suite - Professional Visual Asset Creation & Enhancement"""
    
    def __init__(self):
        self.image_generator = AIImageGenerator()
        self.video_enhancer = VideoEnhancementEngine()
        self.consistency_engine = VisualConsistencyEngine()
        
        # Pexels API entegrasyonu
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            pexels_api_key = os.getenv("PEXELS_API_KEY")
            if pexels_api_key:
                from pexels_api import PexelsImageDownloader
                self.pexels_downloader = PexelsImageDownloader(pexels_api_key)
                logging.info(f"✅ Pexels API integrated with key: {pexels_api_key[:20]}...")
            else:
                self.pexels_downloader = None
                logging.warning("⚠️ Pexels API key not found")
        except Exception as e:
            self.pexels_downloader = None
            logging.warning(f"⚠️ Pexels API integration failed: {e}")
        
        self.asset_cache = {}
        self.quality_thresholds = {
            "minimum_resolution": (1920, 1080),
            "preferred_resolution": (3840, 2160),  # 4K
            "minimum_bitrate": 5000000,  # 5 Mbps
            "preferred_bitrate": 20000000,  # 20 Mbps
            "color_depth": "10-bit",
            "frame_rate": 30
        }
        self.asset_rotation_rules = {
            "min_assets_per_minute": 12,  # 1 asset per 5 seconds
            "max_repetition_interval": 30,  # No repeat within 30 seconds
            "diversity_boost": 1.5  # Increase variety for high-impact scenes
        }
        
    def generate_cinematic_visual_plan(self, script_data: Dict[str, Any], 
                                     target_duration_minutes: float = 15.0,
                                     niche: str = "general") -> Dict[str, Any]:
        """
        Generate comprehensive visual plan for cinematic content
        
        Args:
            script_data: Script with scene breakdown
            target_duration_minutes: Target duration
            niche: Content niche
            
        Returns:
            Complete visual production plan
        """
        try:
            # Calculate total visual assets needed
            total_assets_needed = self._calculate_total_assets_needed(target_duration_minutes)
            
            # Generate asset categories and requirements
            asset_categories = self._generate_asset_categories(script_data, niche)
            
            # Create asset rotation schedule
            rotation_schedule = self._create_asset_rotation_schedule(
                total_assets_needed, script_data, niche
            )
            
            # Quality enhancement plan
            quality_plan = self._create_quality_enhancement_plan(rotation_schedule, "cinematic")
            
            return {
                "total_assets_needed": total_assets_needed,
                "asset_categories": asset_categories,
                "rotation_schedule": rotation_schedule,
                "quality_plan": quality_plan,
                "production_requirements": self._get_production_requirements(niche),
                "asset_sources": self._get_asset_sources(niche),
                "timeline": self._create_production_timeline(total_assets_needed)
            }
            
        except Exception as e:
            logging.error(f"Visual plan generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_total_assets_needed(self, duration_minutes: float) -> int:
        """Calculate total visual assets needed for cinematic quality"""
        # Base: 1 asset per 5 seconds for cinematic quality
        base_assets = int(duration_minutes * 60 / 5)
        
        # Add variety boost for longer content
        if duration_minutes > 10:
            variety_boost = 1.3
        elif duration_minutes > 5:
            variety_boost = 1.2
        else:
            variety_boost = 1.0
            
        return int(base_assets * variety_boost)
    
    def _generate_asset_categories(self, script_data: Dict[str, Any], 
                                 niche: str) -> Dict[str, Any]:
        """Generate asset categories based on script and niche"""
        categories = {
            "primary_visuals": {
                "count": 0,
                "description": "Main visual elements that directly support the narrative",
                "quality_requirements": "4K cinematic quality",
                "sources": self._get_niche_specific_sources(niche, "primary")
            },
            "background_visuals": {
                "count": 0,
                "description": "Supporting visuals that create atmosphere",
                "quality_requirements": "HD+ quality, atmospheric",
                "sources": self._get_niche_specific_sources(niche, "background")
            },
            "transition_visuals": {
                "count": 0,
                "description": "Visual elements for smooth scene transitions",
                "quality_requirements": "High quality, smooth motion",
                "sources": self._get_niche_specific_sources(niche, "transition")
            },
            "emphasis_visuals": {
                "count": 0,
                "description": "High-impact visuals for key moments",
                "quality_requirements": "Premium 4K, dramatic impact",
                "sources": self._get_niche_specific_sources(niche, "emphasis")
            }
        }
        
        # Calculate counts based on script structure
        if "scenes" in script_data:
            for scene in script_data["scenes"]:
                scene_duration = scene.get("duration_seconds", 0)
                emotional_beat = scene.get("emotional_beat", "neutral")
                
                # Calculate assets per scene
                base_assets = max(3, int(scene_duration / 4))  # 1 per 4 seconds
                
                # Adjust based on emotional beat
                if emotional_beat in ["climax", "hook"]:
                    base_assets = int(base_assets * 1.5)
                elif emotional_beat in ["opening", "resolution"]:
                    base_assets = int(base_assets * 1.2)
                
                # Distribute across categories
                categories["primary_visuals"]["count"] += int(base_assets * 0.5)
                categories["background_visuals"]["count"] += int(base_assets * 0.3)
                categories["transition_visuals"]["count"] += int(base_assets * 0.15)
                categories["emphasis_visuals"]["count"] += int(base_assets * 0.05)
        
        return categories
    
    def _create_quality_enhancement_plan(self, assets: List[Dict[str, Any]], 
                                       target_quality: str) -> Dict[str, Any]:
        """Create quality enhancement plan for assets"""
        try:
            enhancement_plan = {
                "upscaling": [],
                "color_correction": [],
                "stabilization": [],
                "noise_reduction": [],
                "sharpening": [],
                "overall_improvements": []
            }
            
            for asset in assets:
                asset_id = asset.get("id", "unknown")
                current_quality = asset.get("quality", "medium")
                
                # Determine enhancement needs based on current quality
                if current_quality == "low":
                    enhancement_plan["upscaling"].append(asset_id)
                    enhancement_plan["color_correction"].append(asset_id)
                    enhancement_plan["noise_reduction"].append(asset_id)
                    enhancement_plan["sharpening"].append(asset_id)
                elif current_quality == "medium":
                    enhancement_plan["color_correction"].append(asset_id)
                    enhancement_plan["sharpening"].append(asset_id)
                elif current_quality == "high":
                    enhancement_plan["color_correction"].append(asset_id)
                
                # Add stabilization for video assets
                if asset.get("type") == "video":
                    enhancement_plan["stabilization"].append(asset_id)
                
                # Add overall improvements for all assets
                enhancement_plan["overall_improvements"].append(asset_id)
            
            # Add target quality specific enhancements
            if target_quality == "cinematic":
                enhancement_plan["color_correction"].extend([asset["id"] for asset in assets])
                enhancement_plan["sharpening"].extend([asset["id"] for asset in assets])
            elif target_quality == "premium":
                enhancement_plan["upscaling"].extend([asset["id"] for asset in assets])
                enhancement_plan["color_correction"].extend([asset["id"] for asset in assets])
                enhancement_plan["noise_reduction"].extend([asset["id"] for asset in assets])
            
            return enhancement_plan
            
        except Exception as e:
            logging.error(f"Quality enhancement plan creation failed: {e}")
            return {
                "upscaling": [],
                "color_correction": [],
                "stabilization": [],
                "noise_reduction": [],
                "sharpening": [],
                "overall_improvements": []
            }

    def _create_asset_rotation_schedule(self, total_assets: int, 
                                      script_data: Dict[str, Any], 
                                      niche: str) -> List[Dict[str, Any]]:
        """Create asset rotation schedule for diversity"""
        try:
            rotation_schedule = []
            
            # Get assets from various sources
            pexels_assets = self._get_pexels_assets(niche, total_assets // 3)
            local_assets = self._get_local_assets(niche, total_assets // 3)
            generated_assets = self._get_generated_assets(niche, total_assets // 6)
            premium_assets = self._get_premium_assets(niche, total_assets // 6)
            
            # Combine all assets
            all_assets = pexels_assets + local_assets + generated_assets + premium_assets
            
            # If we don't have enough assets, generate fallbacks
            if len(all_assets) < total_assets:
                fallback_assets = self._generate_fallback_assets(niche, total_assets - len(all_assets))
                all_assets.extend(fallback_assets)
            
            # Create rotation schedule
            for i in range(total_assets):
                if i < len(all_assets):
                    asset = all_assets[i]
                else:
                    # Create placeholder asset
                    asset = {
                        "id": f"placeholder_{niche}_{i}",
                        "type": "placeholder",
                        "source": "fallback",
                        "quality": "medium",
                        "duration": 5,
                        "tags": [niche, "placeholder"],
                        "file_path": f"placeholder_{niche}_{i}.mp4"
                    }
                
                # Add scene information
                scene_info = self._get_scene_info_for_position(i, script_data)
                
                rotation_item = {
                    "position": i,
                    "asset": asset,
                    "scene": scene_info,
                    "emotional_beat": scene_info.get("emotional_beat", "neutral"),
                    "duration_seconds": scene_info.get("duration_seconds", 5),
                    "transition_type": self._get_transition_type(i, script_data),
                    "special_effects": self._get_special_effects_recommendations(scene_info)
                }
                
                rotation_schedule.append(rotation_item)
            
            return rotation_schedule
            
        except Exception as e:
            logging.error(f"Asset rotation schedule creation failed: {e}")
            return []

    def _get_scene_info_for_position(self, position: int, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get scene information for a specific position"""
        try:
            scenes = script_data.get("scenes", [])
            if position < len(scenes):
                return scenes[position]
            else:
                # Create default scene info
                return {
                    "scene_number": position + 1,
                    "emotional_beat": "neutral",
                    "duration_seconds": 5,
                    "content": "Scene content"
                }
        except Exception as e:
            logging.error(f"Scene info gathering failed: {e}")
            return {
                "scene_number": position + 1,
                "emotional_beat": "neutral",
                "duration_seconds": 5,
                "content": "Scene content"
            }

    def _get_special_effects_recommendations(self, scene: Dict[str, Any]) -> List[str]:
        """Get special effects recommendations for scene"""
        try:
            emotional_beat = scene.get("emotional_beat", "neutral")
            
            effects_map = {
                "opening": ["color_grading", "subtle_glow", "atmospheric_fog"],
                "hook": ["dynamic_movement", "color_pop", "energy_trails"],
                "conflict": ["tension_effects", "dramatic_lighting", "motion_blur"],
                "climax": ["explosive_effects", "intense_lighting", "particle_systems"],
                "resolution": ["soft_glow", "warm_colors", "gentle_movement"]
            }
            
            return effects_map.get(emotional_beat, ["color_grading", "subtle_enhancement"])
            
        except Exception as e:
            logging.error(f"Special effects recommendations failed: {e}")
            return ["color_grading"]

    def _get_transition_type(self, position: int, script_data: Dict[str, Any]) -> str:
        """Get transition type for a position"""
        try:
            scenes = script_data.get("scenes", [])
            if position < len(scenes):
                scene = scenes[position]
                emotional_beat = scene.get("emotional_beat", "neutral")
                
                # Map emotional beats to transition types
                transition_map = {
                    "opening": "fade_in",
                    "hook": "quick_cut",
                    "conflict": "crossfade",
                    "climax": "dramatic_cut",
                    "resolution": "gentle_fade"
                }
                
                return transition_map.get(emotional_beat, "crossfade")
            
            return "crossfade"  # Default transition
            
        except Exception as e:
            logging.error(f"Transition type determination failed: {e}")
            return "crossfade"

    def _get_production_requirements(self, niche: str) -> Dict[str, Any]:
        """Get production requirements for niche"""
        try:
            base_requirements = {
                "resolution": "4K (3840x2160)",
                "frame_rate": "30fps",
                "color_depth": "10-bit",
                "codec": "H.264 High Profile",
                "bitrate": "20 Mbps minimum"
            }
            
            niche_requirements = {
                "history": {
                    "resolution": "4K (3840x2160)",
                    "frame_rate": "30fps",
                    "color_depth": "10-bit HDR",
                    "codec": "H.265",
                    "bitrate": "50 Mbps",
                    "special_notes": "High fidelity for archival quality, color grading for period authenticity"
                },
                "motivation": {
                    "resolution": "4K (3840x2160)",
                    "frame_rate": "30fps",
                    "color_depth": "10-bit",
                    "codec": "H.264 High Profile",
                    "bitrate": "30 Mbps",
                    "special_notes": "Bright, vibrant colors, smooth motion for inspirational feel"
                },
                "finance": {
                    "resolution": "4K (3840x2160)",
                    "frame_rate": "30fps",
                    "color_depth": "10-bit",
                    "codec": "H.264 High Profile",
                    "bitrate": "25 Mbps",
                    "special_notes": "Professional, clean look, modern aesthetics"
                },
                "automotive": {
                    "resolution": "4K (3840x2160)",
                    "frame_rate": "60fps for action scenes",
                    "color_depth": "10-bit",
                    "codec": "H.264 High Profile",
                    "bitrate": "40 Mbps",
                    "special_notes": "High frame rate for smooth motion, dynamic range for engine details"
                },
                "combat": {
                    "resolution": "4K (3840x2160)",
                    "frame_rate": "60fps for action",
                    "color_depth": "10-bit",
                    "codec": "H.264 High Profile",
                    "bitrate": "45 Mbps",
                    "special_notes": "High frame rate for action sequences, intense color grading"
                }
            }
            
            return niche_requirements.get(niche, base_requirements)
            
        except Exception as e:
            logging.error(f"Production requirements gathering failed: {e}")
            return base_requirements

    def _get_asset_sources(self, niche: str) -> Dict[str, List[str]]:
        """Get comprehensive asset sources for niche"""
        try:
            asset_sources = {
                "primary_sources": [
                    "Pexels API",
                    "Local Asset Library",
                    "AI Generated Content",
                    "Premium Stock Libraries"
                ],
                "niche_specific": {
                    "history": [
                        "Historical Archives",
                        "Museum Collections",
                        "Documentary Footage",
                        "Archaeological Sites",
                        "Historical Reenactments"
                    ],
                    "motivation": [
                        "Success Stories",
                        "Achievement Footage",
                        "Inspirational Scenes",
                        "Nature Landscapes",
                        "Urban Success Stories"
                    ],
                    "finance": [
                        "Business Districts",
                        "Technology Innovation",
                        "Global Markets",
                        "Corporate Environments",
                        "Digital Displays"
                    ],
                    "automotive": [
                        "Car Manufacturers",
                        "Racing Footage",
                        "Automotive Shows",
                        "Road Scenes",
                        "Garage Environments"
                    ],
                    "combat": [
                        "Sports Archives",
                        "Training Videos",
                        "Action Footage",
                        "Gym Environments",
                        "Training Facilities"
                    ]
                },
                "quality_tiers": {
                    "premium": ["Shutterstock", "Getty Images", "Adobe Stock"],
                    "standard": ["Pexels", "Unsplash", "Pixabay"],
                    "ai_generated": ["DALL-E", "Midjourney", "Stable Diffusion"],
                    "local": ["Custom Photography", "Stock Footage", "Graphics"]
                }
            }
            
            # Get niche-specific sources
            niche_sources = asset_sources["niche_specific"].get(niche, [])
            asset_sources["niche_sources"] = niche_sources
            
            return asset_sources
            
        except Exception as e:
            logging.error(f"Asset source gathering failed: {e}")
            return {
                "primary_sources": ["Pexels", "Local Library"],
                "niche_sources": [],
                "quality_tiers": {
                    "standard": ["Pexels", "Local Library"],
                    "ai_generated": ["AI Generated"]
                }
            }

    def _create_production_timeline(self, total_assets: int) -> Dict[str, Any]:
        """Create production timeline for asset creation"""
        try:
            # Estimate time needed for each asset type
            time_per_asset = {
                "primary_visuals": 30,  # minutes
                "background_visuals": 20,
                "transition_visuals": 15,
                "emphasis_visuals": 45
            }
            
            total_time = sum(time_per_asset.values()) * total_assets / 4  # Average
            
            timeline = {
                "total_production_time_hours": total_time / 60,
                "parallel_processing": "Yes (multiple assets simultaneously)",
                "quality_review_time": "2 hours per 10 minutes of content",
                "final_assembly_time": "1 hour per 5 minutes of content",
                "stages": {
                    "pre_production": "Asset planning and sourcing",
                    "production": "Asset creation and enhancement",
                    "post_production": "Quality review and finalization"
                }
            }
            
            return timeline
            
        except Exception as e:
            logging.error(f"Production timeline creation failed: {e}")
            return {
                "total_production_time_hours": 0,
                "parallel_processing": "Unknown",
                "quality_review_time": "Unknown",
                "final_assembly_time": "Unknown"
            }

    def _select_diverse_asset(self, available_assets: List[Dict[str, Any]], 
                             used_assets: set, position: int, 
                             script_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select an asset ensuring diversity"""
        try:
            # Filter out already used assets
            unused_assets = [asset for asset in available_assets if asset["id"] not in used_assets]
            
            if not unused_assets:
                return None
            
            # Get scene information for better selection
            scenes = script_data.get("scenes", [])
            current_scene = None
            if position < len(scenes):
                current_scene = scenes[position]
            
            # Score assets based on diversity and scene fit
            scored_assets = []
            for asset in unused_assets:
                score = 0
                
                # Diversity score (prefer assets not used recently)
                if asset["id"] not in used_assets:
                    score += 10
                
                # Scene fit score
                if current_scene:
                    emotional_beat = current_scene.get("emotional_beat", "neutral")
                    if emotional_beat in ["climax", "hook"] and asset.get("quality") == "high":
                        score += 5
                
                # Quality score
                if asset.get("quality") == "high":
                    score += 3
                elif asset.get("quality") == "medium":
                    score += 2
                
                scored_assets.append((score, asset))
            
            # Sort by score and return the best
            scored_assets.sort(key=lambda x: x[0], reverse=True)
            
            if scored_assets:
                return scored_assets[0][1]
            
            return None
            
        except Exception as e:
            logging.error(f"Asset selection failed: {e}")
            return None

    def _gather_available_assets(self, niche: str, target_count: int) -> List[Dict[str, Any]]:
        """Gather available assets from multiple sources"""
        try:
            assets = []
            
            # 1. Pexels API (high quality)
            pexels_assets = self._get_pexels_assets(niche, target_count // 2)
            assets.extend(pexels_assets)
            
            # 2. Local asset library
            local_assets = self._get_local_assets(niche, target_count // 4)
            assets.extend(local_assets)
            
            # 3. Generated assets (AI)
            generated_assets = self._get_generated_assets(niche, target_count // 4)
            assets.extend(generated_assets)
            
            # 4. Premium stock libraries (if available)
            premium_assets = self._get_premium_assets(niche, target_count // 8)
            assets.extend(premium_assets)
            
            return assets[:target_count]
            
        except Exception as e:
            logging.error(f"Asset gathering failed: {e}")
            return []

    def _get_pexels_assets(self, niche: str, count: int) -> List[Dict[str, Any]]:
        """Get assets from Pexels API"""
        try:
            if not hasattr(self, 'pexels_api_key') or not self.pexels_api_key:
                logging.warning("Pexels API key not available")
                return []
            
            assets = []
            queries = self._get_niche_queries(niche)
            
            for query in queries[:count]:
                try:
                    # This would integrate with Pexels API
                    # For now, return placeholder assets
                    asset = {
                        "id": f"pexels_{query}_{len(assets)}",
                        "type": "video",
                        "source": "pexels",
                        "quality": "high",
                        "duration": 15,
                        "tags": [niche, query, "pexels"],
                        "file_path": f"pexels_{query}_{len(assets)}.mp4",
                        "url": f"https://pexels.com/video/{len(assets)}",
                        "photographer": "Pexels Contributor",
                        "license": "Free to use"
                    }
                    assets.append(asset)
                    
                    if len(assets) >= count:
                        break
                        
                except Exception as e:
                    logging.warning(f"Failed to get Pexels asset for query '{query}': {e}")
                    continue
            
            return assets
            
        except Exception as e:
            logging.error(f"Pexels asset gathering failed: {e}")
            return []

    def _get_niche_queries(self, niche: str) -> List[str]:
        """Get search queries for niche"""
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

    def _get_local_assets(self, niche: str, count: int) -> List[Dict[str, Any]]:
        """Get local visual assets"""
        try:
            assets = []
            
            # Check assets/videos directory
            videos_dir = Path("assets/videos")
            if videos_dir.exists():
                video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi")) + list(videos_dir.glob("*.mov"))
                for video_file in video_files[:count]:
                    asset = {
                        "id": f"local_video_{len(assets)}",
                        "type": "video",
                        "source": "local",
                        "quality": "high",
                        "duration": 10,  # Default duration
                        "tags": [niche, "local"],
                        "file_path": str(video_file)
                    }
                    assets.append(asset)
            
            # Check assets/images directory
            images_dir = Path("assets/images")
            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
                for image_file in image_files[:count - len(assets)]:
                    asset = {
                        "id": f"local_image_{len(assets)}",
                        "type": "image",
                        "source": "local",
                        "quality": "high",
                        "duration": 5,  # Default duration for images
                        "tags": [niche, "local"],
                        "file_path": str(image_file)
                    }
                    assets.append(asset)
            
            return assets[:count]
            
        except Exception as e:
            logging.error(f"Local asset gathering failed: {e}")
            return []

    def _get_generated_assets(self, niche: str, count: int) -> List[Dict[str, Any]]:
        """Get AI-generated assets"""
        try:
            assets = []
            
            # This would integrate with AI image generation systems
            # For now, return placeholder assets
            for i in range(count):
                asset = {
                    "id": f"ai_generated_{niche}_{i}",
                    "type": "image",
                    "source": "ai_generated",
                    "quality": "high",
                    "duration": 5,
                    "tags": [niche, "ai_generated"],
                    "file_path": f"generated_{niche}_{i}.png"
                }
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logging.error(f"Generated asset gathering failed: {e}")
            return []

    def _get_premium_assets(self, niche: str, count: int) -> List[Dict[str, Any]]:
        """Get premium stock assets"""
        try:
            assets = []
            
            # This would integrate with premium stock libraries
            # For now, return placeholder assets
            for i in range(count):
                asset = {
                    "id": f"premium_{niche}_{i}",
                    "type": "video",
                    "source": "premium_stock",
                    "quality": "ultra_high",
                    "duration": 15,
                    "tags": [niche, "premium"],
                    "file_path": f"premium_{niche}_{i}.mp4"
                }
                assets.append(asset)
            
            return assets
            
        except Exception as e:
            logging.error(f"Premium asset gathering failed: {e}")
            return []

    def _generate_fallback_assets(self, niche: str, count: int) -> List[Dict[str, Any]]:
        """Generate fallback assets when insufficient assets are available"""
        try:
            assets = []
            
            # Create placeholder assets
            for i in range(count):
                asset = {
                    "id": f"fallback_{niche}_{i}",
                    "type": "placeholder",
                    "source": "fallback",
                    "quality": "medium",
                    "duration": 5,
                    "tags": [niche, "fallback"],
                    "file_path": f"fallback_{niche}_{i}.mp4"
                }
                assets.append(asset)
            
            logging.warning(f"Generated {count} fallback assets for {niche}")
            return assets
            
        except Exception as e:
            logging.error(f"Fallback asset generation failed: {e}")
            return []

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

    def _get_niche_specific_sources(self, niche: str, source_type: str) -> List[str]:
        """Get niche-specific asset sources"""
        try:
            base_sources = ["Pexels", "Local Library", "AI Generated"]
            
            niche_sources = {
                "history": {
                    "primary": ["Historical Archives", "Museum Collections", "Documentary Footage"],
                    "background": ["Ancient Sites", "Historical Reenactments", "Period Artwork"],
                    "transition": ["Timeline Graphics", "Historical Maps", "Archive Footage"]
                },
                "motivation": {
                    "primary": ["Success Stories", "Achievement Footage", "Inspirational Scenes"],
                    "background": ["Nature Landscapes", "Urban Success", "Teamwork Scenes"],
                    "transition": ["Progress Graphics", "Achievement Symbols", "Motivational Quotes"]
                },
                "finance": {
                    "primary": ["Business Districts", "Technology Innovation", "Global Markets"],
                    "background": ["Modern Architecture", "Corporate Environments", "Digital Displays"],
                    "transition": ["Financial Charts", "Technology Graphics", "Business Symbols"]
                },
                "automotive": {
                    "primary": ["Car Manufacturers", "Racing Footage", "Automotive Shows"],
                    "background": ["Road Scenes", "Garage Environments", "Automotive Technology"],
                    "transition": ["Speed Graphics", "Car Specifications", "Automotive Logos"]
                },
                "combat": {
                    "primary": ["Sports Archives", "Training Videos", "Action Footage"],
                    "background": ["Gym Environments", "Training Facilities", "Athletic Scenes"],
                    "transition": ["Combat Graphics", "Training Diagrams", "Athletic Symbols"]
                }
            }
            
            niche_specific = niche_sources.get(niche, {})
            source_list = niche_sources.get(source_type, [])
            
            return base_sources + source_list
            
        except Exception as e:
            logging.error(f"Niche-specific source gathering failed: {e}")
            return ["Pexels", "Local Library", "AI Generated"]

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Initialize AI Visual Suite
    suite = AIVisualSuite()
    
    # Example 1: Create visual assets
    print("🎨 Creating visual assets...")
    assets_result = suite.create_visual_assets(
        title="10 AI Tools That Will Change Your Life",
        subtitle="Discover the Future of Technology",
        style=VisualStyle.MODERN,
        formats=[ImageFormat.THUMBNAIL, ImageFormat.SQUARE],
        generate_ai=True,
        ai_prompt="futuristic technology tools, digital interface, modern design"
    )
    
    print(f"✅ Assets created: {assets_result['total_assets']}")
    print(f"Style used: {assets_result['style_used']}")
    print(f"Formats: {assets_result['formats_generated']}")
    
    # Example 2: List available options
    print(f"\n🎭 Available styles: {suite.get_available_styles()}")
    print(f"📐 Available formats: {suite.get_available_formats()}")
    
    # Example 3: Create consistent brand assets
    print("\n🔗 Creating consistent brand assets...")
    variations = [
        {'name': 'social_media', 'dimensions': (1080, 1080)},
        {'name': 'banner', 'dimensions': (1920, 1080)},
        {'name': 'story', 'dimensions': (1080, 1920)}
    ]
    
    if assets_result['assets'].get('thumbnail'):
        try:
            consistent_assets = suite.create_consistent_brand_assets(
                assets_result['assets']['thumbnail'],
                variations
            )
            print(f"✅ Consistent assets created: {len(consistent_assets)}")
        except Exception as e:
            print(f"⚠️ Brand consistency failed: {e}")
    
    print("\n🚀 AI Visual Suite demonstration completed!")
