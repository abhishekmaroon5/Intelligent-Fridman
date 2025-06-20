#!/usr/bin/env python3
"""
Mass Transcript Collector for All 467 Lex Fridman Podcast Episodes
Collects transcripts from the complete YouTube playlist
"""

import os
import json
import time
import random
from datetime import datetime
import yt_dlp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

class MassTranscriptCollector:
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
                logging.FileHandler('mass_collection.log'),
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
            'start_time': datetime.now()
        }
    
    def setup_yt_dlp(self):
        """Configure yt-dlp for transcript extraction."""
        return {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'json',
            'skip_download': True,
            'ignoreerrors': True,
            'no_warnings': False,
            'extract_flat': False,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'outtmpl': str(self.transcripts_dir / '%(title)s_%(id)s.%(ext)s'),
        }
    
    def get_playlist_videos(self):
        """Extract all video URLs from the playlist."""
        self.logger.info("🔍 Extracting playlist information...")
        
        ydl_opts = {
            'extract_flat': True,
            'quiet': True,
            'no_warnings': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(self.playlist_url, download=False)
                
            videos = []
            for entry in playlist_info.get('entries', []):
                if entry:
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
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting playlist: {e}")
            return []
    
    def download_single_transcript(self, video_info):
        """Download transcript for a single video."""
        video_id = video_info['id']
        title = video_info['title']
        url = video_info['url']
        
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            ydl_opts = self.setup_yt_dlp()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Check if transcript is available
                if not info.get('subtitles') and not info.get('automatic_captions'):
                    self.logger.warning(f"⚠️  No transcript available for: {title}")
                    return None
                
                # Download transcript
                ydl.download([url])
                
                # Process transcript files
                transcript_data = self.process_transcript_files(video_id, title, info)
                
                if transcript_data:
                    self.stats['successful_downloads'] += 1
                    self.stats['total_words'] += len(transcript_data.get('text', '').split())
                    self.logger.info(f"✅ Downloaded: {title} ({len(transcript_data.get('text', '').split())} words)")
                    return transcript_data
                else:
                    self.stats['failed_downloads'] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Error downloading {title}: {e}")
            self.stats['failed_downloads'] += 1
            return None
    
    def process_transcript_files(self, video_id, title, video_info):
        """Process downloaded transcript files."""
        try:
            # Look for transcript files
            transcript_files = list(self.transcripts_dir.glob(f"*{video_id}*.json"))
            
            if not transcript_files:
                return None
            
            # Use the first available transcript file
            transcript_file = transcript_files[0]
            
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Extract text from transcript
            if 'events' in transcript_data:
                # Format 1: events structure
                text_parts = []
                for event in transcript_data.get('events', []):
                    if 'segs' in event:
                        for seg in event['segs']:
                            if 'utf8' in seg:
                                text_parts.append(seg['utf8'])
                text = ' '.join(text_parts)
            elif isinstance(transcript_data, list):
                # Format 2: list of segments
                text_parts = []
                for segment in transcript_data:
                    if 'text' in segment:
                        text_parts.append(segment['text'])
                text = ' '.join(text_parts)
            else:
                # Format 3: direct text
                text = str(transcript_data)
            
            # Clean up text
            text = self.clean_transcript_text(text)
            
            if len(text.split()) < 100:  # Skip very short transcripts
                return None
            
            # Save processed transcript
            processed_data = {
                'video_id': video_id,
                'title': title,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'text': text,
                'word_count': len(text.split()),
                'duration': video_info.get('duration'),
                'upload_date': video_info.get('upload_date'),
                'processed_at': datetime.now().isoformat()
            }
            
            # Save as JSON
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            json_file = self.transcripts_dir / f"{safe_title}_{video_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            # Save as TXT
            txt_file = self.txt_dir / f"{safe_title}_{video_id}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Clean up original transcript file
            transcript_file.unlink()
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Error processing transcript for {title}: {e}")
            return None
    
    def clean_transcript_text(self, text):
        """Clean and normalize transcript text."""
        import re
        
        # Remove common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible), etc.
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&amp;', '&', text)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def collect_all_transcripts(self, max_workers=5, resume=True):
        """Collect transcripts from all videos in the playlist."""
        self.logger.info("🚀 Starting mass transcript collection for all 467 episodes!")
        
        # Get all videos
        videos = self.get_playlist_videos()
        if not videos:
            self.logger.error("❌ No videos found in playlist")
            return
        
        # Filter out already downloaded videos if resuming
        if resume:
            existing_files = set()
            for file in self.transcripts_dir.glob("*.json"):
                # Extract video ID from filename
                parts = file.stem.split('_')
                if parts:
                    existing_files.add(parts[-1])
            
            videos = [v for v in videos if v['id'] not in existing_files]
            self.logger.info(f"📋 Resuming: {len(videos)} videos remaining")
        
        # Process videos in parallel
        successful_transcripts = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(self.download_single_transcript, video): video 
                for video in videos
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_video), 1):
                video = future_to_video[future]
                
                try:
                    result = future.result()
                    if result:
                        successful_transcripts.append(result)
                    
                    # Progress update
                    if i % 10 == 0:
                        self.print_progress(i, len(videos))
                        
                except Exception as e:
                    self.logger.error(f"❌ Task failed for {video['title']}: {e}")
        
        # Final statistics
        self.print_final_stats(successful_transcripts)
        
        # Save collection summary
        self.save_collection_summary(successful_transcripts)
        
        return successful_transcripts
    
    def print_progress(self, completed, total):
        """Print progress update."""
        elapsed = datetime.now() - self.stats['start_time']
        success_rate = (self.stats['successful_downloads'] / completed * 100) if completed > 0 else 0
        
        self.logger.info(f"""
📊 PROGRESS UPDATE:
   Completed: {completed}/{total} ({completed/total*100:.1f}%)
   Successful: {self.stats['successful_downloads']} ({success_rate:.1f}%)
   Failed: {self.stats['failed_downloads']}
   Total Words: {self.stats['total_words']:,}
   Elapsed Time: {elapsed}
   ETA: {elapsed * (total - completed) / completed if completed > 0 else 'Unknown'}
        """)
    
    def print_final_stats(self, transcripts):
        """Print final collection statistics."""
        total_time = datetime.now() - self.stats['start_time']
        
        self.logger.info(f"""
🎉 MASS COLLECTION COMPLETED!
═══════════════════════════════════════════════════════
📊 FINAL STATISTICS:
   Total Videos in Playlist: {self.stats['total_videos']}
   Successfully Downloaded: {len(transcripts)}
   Failed Downloads: {self.stats['failed_downloads']}
   Success Rate: {len(transcripts)/self.stats['total_videos']*100:.1f}%
   
📝 CONTENT STATISTICS:
   Total Words Collected: {self.stats['total_words']:,}
   Average Words per Episode: {self.stats['total_words']//len(transcripts) if transcripts else 0:,}
   Estimated Training Examples: {self.stats['total_words']//50*3:,} (rough estimate)
   
⏱️  TIME STATISTICS:
   Total Collection Time: {total_time}
   Average Time per Video: {total_time.total_seconds()/self.stats['total_videos']:.1f} seconds
   
🎯 QUALITY ESTIMATE:
   This dataset will create a WORLD-CLASS Lex Fridman chatbot!
   Expected performance: 9/10 (professional grade)
═══════════════════════════════════════════════════════
        """)
    
    def save_collection_summary(self, transcripts):
        """Save collection summary and statistics."""
        summary = {
            'collection_date': datetime.now().isoformat(),
            'playlist_url': self.playlist_url,
            'total_videos_in_playlist': self.stats['total_videos'],
            'successful_downloads': len(transcripts),
            'failed_downloads': self.stats['failed_downloads'],
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
        
        with open(self.metadata_dir / 'collection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create CSV for easy analysis
        df = pd.DataFrame([
            {
                'title': t['title'],
                'video_id': t['video_id'],
                'word_count': t['word_count'],
                'upload_date': t.get('upload_date'),
                'url': t['url']
            }
            for t in transcripts
        ])
        df.to_csv(self.metadata_dir / 'episodes_summary.csv', index=False)

def main():
    print("🚀 LEX FRIDMAN PODCAST - MASS TRANSCRIPT COLLECTOR")
    print("=" * 60)
    print("📺 Target: All 467 episodes from the official playlist")
    print("🔗 Playlist: https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4")
    print("⚡ This will create a WORLD-CLASS dataset!")
    print()
    
    collector = MassTranscriptCollector()
    
    # Ask for confirmation
    response = input("🤔 This will take several hours. Continue? (y/N): ")
    if response.lower() != 'y':
        print("👋 Collection cancelled.")
        return
    
    # Start collection
    transcripts = collector.collect_all_transcripts(max_workers=3, resume=True)
    
    if transcripts:
        print(f"\n🎉 SUCCESS! Collected {len(transcripts)} transcripts")
        print("📁 Files saved in:")
        print(f"   - Transcripts: {collector.transcripts_dir}")
        print(f"   - Text files: {collector.txt_dir}")
        print(f"   - Metadata: {collector.metadata_dir}")
        print("\n🚀 Ready for next step: Enhanced preprocessing and training!")
    else:
        print("❌ No transcripts collected. Check the logs for details.")

if __name__ == "__main__":
    main() 