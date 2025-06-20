#!/usr/bin/env python3
"""
Targeted Transcript Collector for 6 Specific Lex Fridman Episodes
Focused on quality over quantity for better training data
"""

import os
import json
import time
import random
from datetime import datetime
import yt_dlp
import pandas as pd
import logging
from pathlib import Path
import re

class TargetedCollector:
    def __init__(self):
        # Specific 6 videos requested by user
        self.target_videos = [
            {
                'url': 'https://www.youtube.com/watch?v=HUkBz-cdB-k',
                'id': 'HUkBz-cdB-k',
                'title': 'Terence Tao - Hardest Problems in Mathematics, Physics & AI',
                'episode': '#472'
            },
            {
                'url': 'https://www.youtube.com/watch?v=_1f-o0nqpEI',
                'id': '_1f-o0nqpEI', 
                'title': 'DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate',
                'episode': '#459'
            },
            {
                'url': 'https://www.youtube.com/watch?v=ugvHCXCOmm4',
                'id': 'ugvHCXCOmm4',
                'title': 'Javier Milei - President of Argentina',
                'episode': '#453'
            },
            {
                'url': 'https://www.youtube.com/watch?v=F3Jd9GI6XqE',
                'id': 'F3Jd9GI6XqE',
                'title': 'Michael Malice - Anarchism, Democracy, Libertarianism',
                'episode': '#400'
            },
            {
                'url': 'https://www.youtube.com/watch?v=1X_KdkoGxSs',
                'id': '1X_KdkoGxSs',
                'title': 'Andrew Huberman - Focus, Stress, Relationships, and Friendship',
                'episode': '#392'
            },
            {
                'url': 'https://www.youtube.com/watch?v=Kbk9BiPhm7o',
                'id': 'Kbk9BiPhm7o',
                'title': 'Elon Musk - Neuralink and the Future of Humanity',
                'episode': '#438'
            }
        ]
        
        self.output_dir = Path("data")
        self.transcripts_dir = self.output_dir / "transcripts"
        self.metadata_dir = self.output_dir / "metadata"
        self.txt_dir = self.output_dir / "txt_files"
        
        # Create directories
        for dir_path in [self.transcripts_dir, self.metadata_dir, self.txt_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('targeted_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_videos': len(self.target_videos),
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_words': 0,
            'start_time': datetime.now()
        }

def main():
    print("🚀 TARGETED LEX FRIDMAN PODCAST - 6 HIGH-QUALITY EPISODES")
    print("=" * 65)
    print("📺 Target: 6 carefully selected diverse episodes")
    print("🎯 Quality over quantity approach for optimal training")
    
    collector = TargetedCollector()
    
    # Show target episodes
    print("🎬 Target Episodes:")
    for i, video in enumerate(collector.target_videos, 1):
        print(f"   {i}. {video['episode']}: {video['title']}")
    print()

if __name__ == "__main__":
    main()
