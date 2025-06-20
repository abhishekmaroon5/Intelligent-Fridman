import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm

try:
    from yt_dlp import YoutubeDL
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
    from youtube_transcript_api._errors import NoTranscriptFound
except ImportError as e:
    print(f"Missing required packages. Please install them using:")
    print("pip install yt-dlp youtube-transcript-api")
    exit(1)

class LexFridmanTranscriptDownloader:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        # Lex Fridman's channel URL - using the direct channel URL
        self.channel_url = "https://www.youtube.com/@lexfridman"
        
    def ensure_output_dir(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "transcripts"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "txt_files"), exist_ok=True)
        
    def sanitize_filename(self, name: str) -> str:
        """Clean filename for safe saving."""
        # Remove invalid characters and limit length
        clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
        return clean_name[:100]  # Limit to 100 characters
    
    def get_video_ids_from_channel(self, max_videos: int = 400) -> List[Dict]:
        """Get video IDs and metadata from Lex Fridman's channel."""
        print("🔍 Fetching video information from Lex Fridman's channel...")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlist_items': f'1-{max_videos}',  # Limit number of videos
        }
        
        videos_info = []
        try:
            with YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(self.channel_url, download=False)
                
                if 'entries' in result:
                    print(f"📺 Found {len(result['entries'])} videos in channel")
                    
                    for entry in result['entries']:
                        if entry and 'id' in entry:
                            video_info = {
                                'video_id': entry['id'],
                                'title': entry.get('title', 'Unknown Title'),
                                'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                'duration': entry.get('duration', 0),
                                'upload_date': entry.get('upload_date', ''),
                                'view_count': entry.get('view_count', 0),
                                'description': entry.get('description', '')[:500] if entry.get('description') else ''
                            }
                            videos_info.append(video_info)
                else:
                    print("❌ No entries found in channel")
                    
        except Exception as e:
            print(f"❌ Error fetching channel videos: {e}")
            return []
            
        return videos_info
    
    def get_detailed_video_info(self, video_id: str) -> Optional[Dict]:
        """Get detailed information for a specific video."""
        try:
            ydl_opts = {'quiet': True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                return {
                    'video_id': video_id,
                    'title': info.get('title', 'Unknown Title'),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'channel': info.get('channel', ''),
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                }
        except Exception as e:
            print(f"   ⚠️  Error getting detailed info for {video_id}: {e}")
            return None
    
    def save_transcript(self, video_info: Dict) -> bool:
        """Download and save transcript for a video."""
        video_id = video_info['video_id']
        title = video_info['title']
        
        try:
            # Try to get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine transcript text
            full_text = "\n".join([entry['text'] for entry in transcript])
            word_count = len(full_text.split())
            
            # Create safe filename
            safe_title = self.sanitize_filename(title)
            
            # Save as TXT file
            txt_filename = os.path.join(self.output_dir, "txt_files", f"{safe_title}_{video_id}.txt")
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
                f.write(f"Word Count: {word_count}\n")
                f.write("-" * 50 + "\n\n")
                f.write(full_text)
            
            # Save as JSON with metadata
            json_data = {
                **video_info,
                'transcript': full_text,
                'segments': transcript,
                'word_count': word_count,
                'segment_count': len(transcript),
                'collected_at': datetime.now().isoformat()
            }
            
            json_filename = os.path.join(self.output_dir, "transcripts", f"{video_id}.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved: {title[:60]}... ({word_count} words)")
            return True
            
        except NoTranscriptFound:
            print(f"❌ No transcript: {title[:60]}...")
            return False
        except TranscriptsDisabled:
            print(f"❌ Transcripts disabled: {title[:60]}...")
            return False
        except Exception as e:
            print(f"⚠️  Error for {title[:60]}...: {e}")
            return False
    
    def download_all_transcripts(self, max_videos: int = 100):
        """Main method to download all transcripts."""
        print("🚀 Starting Lex Fridman transcript download...")
        
        # Get video information
        videos_info = self.get_video_ids_from_channel(max_videos)
        
        if not videos_info:
            print("❌ No videos found. Please check the channel URL.")
            return [], {}
        
        print(f"📺 Processing {len(videos_info)} videos...\n")
        
        successful_transcripts = []
        failed_videos = []
        
        # Process each video with progress bar
        for i, video_info in enumerate(tqdm(videos_info, desc="Downloading transcripts")):
            print(f"\n📹 Video {i+1}/{len(videos_info)}: {video_info['title'][:50]}...")
            
            # Get detailed info
            detailed_info = self.get_detailed_video_info(video_info['video_id'])
            if detailed_info:
                video_info.update(detailed_info)
            
            # Try to save transcript
            if self.save_transcript(video_info):
                successful_transcripts.append(video_info)
            else:
                failed_videos.append(video_info)
        
        # Create summary
        success_rate = (len(successful_transcripts) / len(videos_info) * 100) if videos_info else 0
        total_words = sum(t.get('word_count', 0) for t in successful_transcripts if 'word_count' in t)
        
        summary = {
            'total_videos_processed': len(videos_info),
            'successful_transcripts': len(successful_transcripts),
            'failed_videos': len(failed_videos),
            'success_rate': success_rate,
            'total_words_collected': total_words,
            'collection_date': datetime.now().isoformat(),
            'channel_url': self.channel_url
        }
        
        # Save combined JSON
        if successful_transcripts:
            combined_file = os.path.join(self.output_dir, "all_transcripts.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(successful_transcripts, f, indent=2, ensure_ascii=False)
            
            # Save CSV metadata
            df = pd.DataFrame([{
                'video_id': t['video_id'],
                'title': t['title'],
                'word_count': t.get('word_count', 0),
                'upload_date': t.get('upload_date', ''),
                'view_count': t.get('view_count', 0),
                'duration': t.get('duration', 0)
            } for t in successful_transcripts])
            
            csv_file = os.path.join(self.output_dir, "transcripts_metadata.csv")
            df.to_csv(csv_file, index=False)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "collection_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print results
        print(f"\n🎉 Download Complete!")
        print(f"📊 Results: {len(successful_transcripts)}/{len(videos_info)} successful ({success_rate:.1f}%)")
        print(f"📝 Total words: {total_words:,}")
        print(f"💾 Files saved to:")
        print(f"   - {self.output_dir}/txt_files/ (individual .txt files)")
        print(f"   - {self.output_dir}/transcripts/ (individual .json files)")
        print(f"   - {self.output_dir}/all_transcripts.json (combined)")
        print(f"   - {self.output_dir}/transcripts_metadata.csv (metadata)")
        
        if successful_transcripts:
            print(f"\n📚 Sample episodes collected:")
            for i, t in enumerate(successful_transcripts[:5]):
                words = t.get('word_count', 0)
                print(f"   {i+1}. {t['title'][:55]}... ({words} words)")
        
        return successful_transcripts, summary

def test_single_video():
    """Test with a recent video first."""
    print("🧪 Testing with Lex Fridman channel access...")
    
    downloader = LexFridmanTranscriptDownloader()
    
    # Test getting videos from channel
    test_videos = downloader.get_video_ids_from_channel(max_videos=5)
    
    if test_videos:
        print(f"✅ Found {len(test_videos)} videos")
        test_video = test_videos[0]
        print(f"   Testing with: {test_video['title'][:50]}...")
        
        # Test transcript download
        success = downloader.save_transcript(test_video)
        if success:
            print("✅ Transcript download successful!")
            return True
        else:
            print("❌ Transcript download failed")
            return False
    else:
        print("❌ Could not access channel videos")
        return False

def main():
    """Main execution function."""
    print("🚀 Lex Fridman Transcript Downloader")
    print("=" * 50)
    
    # Test first
    if test_single_video():
        print("\n" + "=" * 50)
        print("🎯 Starting full download...")
        
        downloader = LexFridmanTranscriptDownloader()
        
        # Ask user for number of videos
        try:
            max_videos = int(input("\nHow many videos to download? (default 50): ") or "50")
        except ValueError:
            max_videos = 50
        
        print(f"\n📥 Downloading transcripts from up to {max_videos} videos...")
        downloader.download_all_transcripts(max_videos=max_videos)
        
    else:
        print("❌ Initial test failed. Please check your internet connection.")

if __name__ == "__main__":
    main() 