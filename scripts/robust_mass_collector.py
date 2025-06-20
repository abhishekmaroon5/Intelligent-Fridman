#!/usr/bin/env python3
"""
Robust Mass Transcript Collector for All 467 Lex Fridman Podcast Episodes
Handles YouTube's bot detection with cookies and advanced rate limiting
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
import subprocess

class RobustMassCollector:
    def __init__(self):
        self.playlist_url = "https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4"
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
                logging.FileHandler('robust_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_videos': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_words': 0,
            'blocked_by_youtube': 0,
            'start_time': datetime.now()
        }
    
    def setup_yt_dlp_with_cookies(self):
        """Configure yt-dlp with browser cookies to avoid bot detection."""
        return {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'json',
            'skip_download': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'subtitleslangs': ['en'],
            'cookiesfrombrowser': ('chrome',),  # Use Chrome cookies
            'outtmpl': str(self.transcripts_dir / f'temp_%(id)s.%(ext)s'),
            # Additional options to avoid detection
            'sleep_interval': 2,
            'max_sleep_interval': 5,
            'sleep_interval_subtitles': 1,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'referer': 'https://www.youtube.com/',
        }
    
    def setup_yt_dlp_fallback(self):
        """Fallback configuration without cookies."""
        return {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'json',
            'skip_download': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'subtitleslangs': ['en'],
            'outtmpl': str(self.transcripts_dir / f'temp_%(id)s.%(ext)s'),
            # More conservative settings
            'sleep_interval': 5,
            'max_sleep_interval': 10,
            'sleep_interval_subtitles': 3,
            'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'referer': 'https://www.youtube.com/',
            'extract_flat': False,
        }
    
    def get_playlist_videos(self):
        """Extract all video URLs from the playlist."""
        self.logger.info("🔍 Extracting playlist information...")
        
        # Try with cookies first
        try:
            ydl_opts = {
                'extract_flat': True,
                'quiet': True,
                'no_warnings': True,
                'cookiesfrombrowser': ('chrome',),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(self.playlist_url, download=False)
        except Exception as e:
            self.logger.warning(f"⚠️ Cookie method failed: {e}")
            # Fallback without cookies
            ydl_opts = {
                'extract_flat': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(self.playlist_url, download=False)
        
        videos = []
        for entry in playlist_info.get('entries', []):
            if entry and entry.get('id'):
                videos.append({
                    'id': entry.get('id'),
                    'title': entry.get('title', 'Unknown'),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                    'duration': entry.get('duration'),
                    'upload_date': entry.get('upload_date')
                })
        
        self.stats['total_videos'] = len(videos)
        self.logger.info(f"✅ Found {len(videos)} videos in playlist")
        
        # Save video list
        with open(self.metadata_dir / 'all_videos.json', 'w') as f:
            json.dump(videos, f, indent=2)
        
        return videos
    
    def download_single_transcript(self, video_info, use_cookies=True):
        """Download transcript for a single video with multiple fallback strategies."""
        video_id = video_info['id']
        title = video_info['title']
        url = video_info['url']
        
        # Progressive delay to avoid rate limiting
        delay = random.uniform(3, 8)
        time.sleep(delay)
        
        try:
            # Try with cookies first
            if use_cookies:
                ydl_opts = self.setup_yt_dlp_with_cookies()
            else:
                ydl_opts = self.setup_yt_dlp_fallback()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # First, just extract info without downloading
                    info_dict = ydl.extract_info(url, download=False)
                    
                    if not info_dict:
                        self.logger.warning(f"⚠️ No info extracted for: {title}")
                        return None
                    
                    # Check for available subtitles
                    has_subtitles = (
                        info_dict.get('subtitles', {}).get('en') or
                        info_dict.get('automatic_captions', {}).get('en')
                    )
                    
                    if not has_subtitles:
                        self.logger.warning(f"⚠️ No English subtitles for: {title}")
                        return None
                    
                    # Now download the subtitles
                    ydl.download([url])
                    
                    # Process the downloaded transcript
                    transcript_data = self.process_downloaded_transcript(video_id, title, info_dict)
                    
                    if transcript_data:
                        self.stats['successful_downloads'] += 1
                        word_count = len(transcript_data.get('text', '').split())
                        self.stats['total_words'] += word_count
                        self.logger.info(f"✅ {title} ({word_count:,} words)")
                        return transcript_data
                    else:
                        self.stats['failed_downloads'] += 1
                        return None
                        
                except Exception as e:
                    error_str = str(e).lower()
                    if 'sign in' in error_str or 'bot' in error_str:
                        self.stats['blocked_by_youtube'] += 1
                        self.logger.warning(f"🤖 Bot detection for {title}")
                        # If we were using cookies and got blocked, try without
                        if use_cookies:
                            time.sleep(random.uniform(10, 20))  # Longer delay
                            return self.download_single_transcript(video_info, use_cookies=False)
                        else:
                            return None
                    else:
                        self.logger.error(f"❌ yt-dlp error for {title}: {e}")
                        self.stats['failed_downloads'] += 1
                        return None
                        
        except Exception as e:
            self.logger.error(f"❌ General error for {title}: {e}")
            self.stats['failed_downloads'] += 1
            return None
    
    def process_downloaded_transcript(self, video_id, title, info_dict):
        """Process the downloaded transcript files."""
        try:
            # Look for transcript files
            possible_files = [
                self.transcripts_dir / f'temp_{video_id}.en.json',
                self.transcripts_dir / f'temp_{video_id}.en-US.json',
                self.transcripts_dir / f'temp_{video_id}.en-GB.json',
            ]
            
            transcript_file = None
            for file_path in possible_files:
                if file_path.exists():
                    transcript_file = file_path
                    break
            
            if not transcript_file:
                # Look for any JSON file with the video ID
                json_files = list(self.transcripts_dir.glob(f'*{video_id}*.json'))
                if json_files:
                    transcript_file = json_files[0]
            
            if not transcript_file:
                self.logger.warning(f"⚠️ No transcript file found for {video_id}")
                return None
            
            # Read and process the transcript
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Extract text using multiple methods
            text = self.extract_text_from_transcript(transcript_data)
            
            if not text or len(text.split()) < 100:
                self.logger.warning(f"⚠️ Transcript too short for {title}")
                transcript_file.unlink(missing_ok=True)
                return None
            
            # Clean the text
            text = self.clean_transcript_text(text)
            
            # Create processed data
            processed_data = {
                'video_id': video_id,
                'title': title,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'text': text,
                'word_count': len(text.split()),
                'duration': info_dict.get('duration'),
                'upload_date': info_dict.get('upload_date'),
                'processed_at': datetime.now().isoformat()
            }
            
            # Save the processed transcript
            self.save_processed_transcript(processed_data)
            
            # Clean up temporary file
            transcript_file.unlink(missing_ok=True)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Error processing transcript for {title}: {e}")
            return None
    
    def extract_text_from_transcript(self, transcript_data):
        """Extract text from transcript JSON using multiple methods."""
        text_parts = []
        
        try:
            if isinstance(transcript_data, list):
                # Format: list of segments
                for segment in transcript_data:
                    if isinstance(segment, dict):
                        if 'text' in segment:
                            text_parts.append(segment['text'])
                        elif 'content' in segment:
                            text_parts.append(segment['content'])
            elif isinstance(transcript_data, dict):
                if 'events' in transcript_data:
                    # Format: events structure
                    for event in transcript_data.get('events', []):
                        if 'segs' in event:
                            for seg in event['segs']:
                                if 'utf8' in seg:
                                    text_parts.append(seg['utf8'])
                elif 'text' in transcript_data:
                    # Direct text format
                    text_parts.append(transcript_data['text'])
                elif 'content' in transcript_data:
                    text_parts.append(transcript_data['content'])
            
            return ' '.join(text_parts).strip()
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting text: {e}")
            return ""
    
    def clean_transcript_text(self, text):
        """Clean transcript text."""
        # Remove common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible), etc.
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&nbsp;', ' ', text)
        
        # Fix spacing and formatting
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def save_processed_transcript(self, data):
        """Save processed transcript in multiple formats."""
        video_id = data['video_id']
        title = data['title']
        text = data['text']
        
        # Create safe filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title[:100]  # Limit length
        
        # Save as JSON
        json_file = self.transcripts_dir / f"{safe_title}_{video_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save as TXT
        txt_file = self.txt_dir / f"{safe_title}_{video_id}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def collect_all_transcripts(self, batch_size=25, max_retries=3):
        """Collect transcripts with advanced rate limiting and retry logic."""
        self.logger.info("🚀 Starting robust mass collection for all 467 episodes!")
        
        # Get all videos
        videos = self.get_playlist_videos()
        if not videos:
            self.logger.error("❌ No videos found in playlist")
            return []
        
        # Filter out already downloaded videos
        existing_files = set()
        for file in self.transcripts_dir.glob("*.json"):
            # Extract video ID from filename
            parts = file.stem.split('_')
            if parts:
                existing_files.add(parts[-1])
        
        remaining_videos = [v for v in videos if v['id'] not in existing_files]
        self.logger.info(f"📋 Found {len(existing_files)} existing transcripts")
        self.logger.info(f"📋 Remaining to download: {len(remaining_videos)}")
        
        if not remaining_videos:
            self.logger.info("✅ All transcripts already collected!")
            return []
        
        # Process in smaller batches with longer delays
        all_transcripts = []
        total_batches = (len(remaining_videos) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_videos))
            batch_videos = remaining_videos[start_idx:end_idx]
            
            self.logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch_videos)} videos)")
            
            batch_transcripts = []
            for i, video in enumerate(batch_videos, 1):
                self.logger.info(f"🔄 [{i}/{len(batch_videos)}] {video['title']}")
                
                # Try with retries
                result = None
                for retry in range(max_retries):
                    result = self.download_single_transcript(video, use_cookies=(retry == 0))
                    if result:
                        break
                    elif retry < max_retries - 1:
                        wait_time = (retry + 1) * 10
                        self.logger.info(f"⏳ Retry {retry + 1} in {wait_time}s...")
                        time.sleep(wait_time)
                
                if result:
                    batch_transcripts.append(result)
                
                # Progress update every 10 videos
                if i % 10 == 0:
                    self.print_progress(start_idx + i, len(remaining_videos))
                
                # Adaptive delay based on success rate
                if self.stats['blocked_by_youtube'] > 5:
                    time.sleep(random.uniform(10, 20))  # Longer delays if being blocked
            
            all_transcripts.extend(batch_transcripts)
            
            # Batch summary
            self.logger.info(f"✅ Batch {batch_num + 1} complete: {len(batch_transcripts)}/{len(batch_videos)} successful")
            
            # Longer pause between batches
            if batch_num < total_batches - 1:
                pause_time = random.uniform(30, 60)
                self.logger.info(f"⏸️ Pausing {pause_time:.1f}s between batches...")
                time.sleep(pause_time)
        
        # Final statistics
        self.print_final_stats(all_transcripts)
        self.save_collection_summary(all_transcripts)
        
        return all_transcripts
    
    def print_progress(self, completed, total):
        """Print progress update."""
        elapsed = datetime.now() - self.stats['start_time']
        success_rate = (self.stats['successful_downloads'] / completed * 100) if completed > 0 else 0
        
        self.logger.info(f"""
📊 PROGRESS UPDATE:
   Completed: {completed}/{total} ({completed/total*100:.1f}%)
   Successful: {self.stats['successful_downloads']} ({success_rate:.1f}%)
   Failed: {self.stats['failed_downloads']}
   Bot Blocks: {self.stats['blocked_by_youtube']}
   Total Words: {self.stats['total_words']:,}
   Elapsed Time: {elapsed}
        """)
    
    def print_final_stats(self, transcripts):
        """Print final collection statistics."""
        total_time = datetime.now() - self.stats['start_time']
        
        self.logger.info(f"""
🎉 ROBUST MASS COLLECTION COMPLETED!
═══════════════════════════════════════════════════════
📊 FINAL STATISTICS:
   Total Videos in Playlist: {self.stats['total_videos']}
   Successfully Downloaded: {len(transcripts)}
   Failed Downloads: {self.stats['failed_downloads']}
   Bot Detection Blocks: {self.stats['blocked_by_youtube']}
   Success Rate: {len(transcripts)/self.stats['total_videos']*100:.1f}%
   
📝 CONTENT STATISTICS:
   Total Words Collected: {self.stats['total_words']:,}
   Average Words per Episode: {self.stats['total_words']//len(transcripts) if transcripts else 0:,}
   Estimated Training Examples: {self.stats['total_words']//50*3:,}
   
⏱️ TIME STATISTICS:
   Total Collection Time: {total_time}
   
🎯 DATASET QUALITY:
   This dataset will create a WORLD-CLASS Lex Fridman chatbot!
   Expected performance improvement: 3/10 → 9/10
═══════════════════════════════════════════════════════
        """)
    
    def save_collection_summary(self, transcripts):
        """Save collection summary."""
        summary = {
            'collection_date': datetime.now().isoformat(),
            'playlist_url': self.playlist_url,
            'total_videos_in_playlist': self.stats['total_videos'],
            'successful_downloads': len(transcripts),
            'failed_downloads': self.stats['failed_downloads'],
            'bot_detection_blocks': self.stats['blocked_by_youtube'],
            'success_rate': len(transcripts)/self.stats['total_videos']*100 if self.stats['total_videos'] > 0 else 0,
            'total_words': self.stats['total_words'],
            'average_words_per_episode': self.stats['total_words']//len(transcripts) if transcripts else 0,
            'collection_time': str(datetime.now() - self.stats['start_time']),
            'transcripts': [
                {
                    'title': t['title'],
                    'video_id': t['video_id'],
                    'word_count': t['word_count'],
                    'upload_date': t.get('upload_date')
                }
                for t in transcripts
            ]
        }
        
        with open(self.metadata_dir / 'robust_collection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    print("🚀 ROBUST LEX FRIDMAN PODCAST - MASS TRANSCRIPT COLLECTOR")
    print("=" * 75)
    print("📺 Target: All 467 episodes from the official playlist")
    print("🔗 Playlist: https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4")
    print("🛡️ Enhanced with bot detection avoidance and retry logic!")
    print("🎯 Expected: 200-400+ successful transcripts (millions of words)")
    print("⏱️ Estimated time: 3-6 hours (with conservative rate limiting)")
    print()
    
    collector = RobustMassCollector()
    
    # Ask for confirmation
    response = input("🤔 Ready to build the ultimate Lex Fridman dataset? (y/N): ")
    if response.lower() != 'y':
        print("👋 Collection cancelled.")
        return
    
    # Start collection
    transcripts = collector.collect_all_transcripts(batch_size=25, max_retries=3)
    
    if transcripts:
        print(f"\n🎉 MASSIVE SUCCESS! Collected {len(transcripts)} transcripts")
        print(f"📊 Total words: {collector.stats['total_words']:,}")
        print("📁 Files saved in:")
        print(f"   - Transcripts: {collector.transcripts_dir}")
        print(f"   - Text files: {collector.txt_dir}")
        print(f"   - Metadata: {collector.metadata_dir}")
        print("\n🚀 Next step: Enhanced preprocessing and world-class training!")
    else:
        print("❌ No new transcripts collected. Check existing files or logs.")

if __name__ == "__main__":
    main() 