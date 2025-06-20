import os
import json
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    import googleapiclient.discovery
except ImportError as e:
    print(f"Missing required packages. Please install them using:")
    print("pip install youtube-transcript-api google-api-python-client")
    exit(1)

class LexFridmanTranscriptCollector:
    def __init__(self, output_dir: str = "data", api_key: Optional[str] = None):
        self.channel_id = "UCSHZKyawb77ixDdsGog4iWA"  # Lex Fridman's channel ID
        self.output_dir = output_dir
        self.api_key = api_key
        self.ensure_output_dir()
        
        # Known video IDs from recent episodes (as fallback)
        self.sample_video_ids = [
            "0Rnq1NpHdmw",  # Recent episode
            "gmgktw3HMcY",  # Another recent episode
            "Ff4fRgnuFgQ",  # Another episode
            # Add more as needed
        ]
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "transcripts"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        
    def get_channel_videos_from_search(self, max_videos: int = 400) -> List[Dict]:
        """Get video IDs using YouTube search."""
        print("🔍 Searching for Lex Fridman videos...")
        
        videos = []
        
        # Method 1: Use requests to get channel page (simple scraping)
        try:
            import re
            
            # Get the channel page
            channel_url = f"https://www.youtube.com/channel/{self.channel_id}/videos"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(channel_url, headers=headers)
            if response.status_code == 200:
                # Extract video IDs from the page content
                video_id_pattern = r'"videoId":"([a-zA-Z0-9_-]{11})"'
                video_ids = re.findall(video_id_pattern, response.text)
                
                # Remove duplicates while preserving order
                unique_video_ids = []
                seen = set()
                for vid_id in video_ids:
                    if vid_id not in seen:
                        unique_video_ids.append(vid_id)
                        seen.add(vid_id)
                
                print(f"Found {len(unique_video_ids)} video IDs from channel page")
                
                # Get metadata for each video
                for i, video_id in enumerate(unique_video_ids[:max_videos]):
                    video_info = self.get_video_metadata(video_id)
                    if video_info:
                        videos.append(video_info)
                        print(f"✅ Got metadata for video {i+1}: {video_info['title'][:50]}...")
                    
                    time.sleep(0.5)  # Rate limiting
                    
        except Exception as e:
            print(f"Error scraping channel page: {e}")
            
        # Fallback: Use sample video IDs if nothing was found
        if not videos:
            print("🔄 Using fallback method with known video IDs...")
            for video_id in self.sample_video_ids:
                video_info = self.get_video_metadata(video_id)
                if video_info:
                    videos.append(video_info)
                    
        return videos
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Get video metadata using oembed API."""
        try:
            # Use YouTube's oembed API to get basic info
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'video_id': video_id,
                    'title': data.get('title', 'Unknown Title'),
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'author_name': data.get('author_name', 'Unknown Author'),
                    'thumbnail_url': data.get('thumbnail_url', ''),
                    'collected_at': datetime.now().isoformat()
                }
            else:
                print(f"⚠️  Could not get metadata for video {video_id}")
                return None
                
        except Exception as e:
            print(f"Error getting metadata for {video_id}: {e}")
            return None
    
    def get_transcript(self, video_id: str) -> Optional[Dict]:
        """Get transcript for a specific video."""
        try:
            # Try to get the transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine all transcript segments into one text
            full_transcript = " ".join([entry['text'] for entry in transcript_list])
            
            return {
                'video_id': video_id,
                'transcript': full_transcript,
                'segments': transcript_list,
                'word_count': len(full_transcript.split()),
                'segment_count': len(transcript_list),
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Could not retrieve transcript for video {video_id}: {str(e)}")
            return None
    
    def collect_all_transcripts(self, max_videos: int = 400):
        """Main method to collect all transcripts."""
        print("🚀 Starting transcript collection for Lex Fridman's channel...")
        
        # Get video information
        videos = self.get_channel_videos_from_search(max_videos)
        
        if not videos:
            print("❌ No videos found. Please check the channel ID or try again later.")
            return [], {}
            
        print(f"📺 Found {len(videos)} videos to process")
        
        successful_transcripts = []
        failed_videos = []
        
        # Use tqdm for progress bar
        for i, video in enumerate(tqdm(videos, desc="Collecting transcripts")):
            print(f"\n📹 Processing video {i+1}/{len(videos)}: {video['title'][:50]}...")
            
            # Get transcript
            transcript_data = self.get_transcript(video['video_id'])
            
            if transcript_data:
                # Combine video metadata with transcript
                combined_data = {**video, **transcript_data}
                successful_transcripts.append(combined_data)
                
                # Save individual transcript file
                transcript_file = os.path.join(
                    self.output_dir, 
                    "transcripts", 
                    f"{video['video_id']}.json"
                )
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)
                    
                print(f"✅ Transcript saved for: {video['title'][:50]}... ({transcript_data['word_count']} words)")
            else:
                failed_videos.append(video)
                print(f"❌ Failed to get transcript for: {video['title'][:50]}...")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        # Calculate success rate safely
        success_rate = (len(successful_transcripts) / len(videos) * 100) if videos else 0
        
        # Save summary
        summary = {
            'total_videos_processed': len(videos),
            'successful_transcripts': len(successful_transcripts),
            'failed_videos': len(failed_videos),
            'success_rate': success_rate,
            'collection_date': datetime.now().isoformat(),
            'failed_video_ids': [v['video_id'] for v in failed_videos],
            'total_words_collected': sum(t.get('word_count', 0) for t in successful_transcripts)
        }
        
        # Save combined transcripts
        if successful_transcripts:
            combined_file = os.path.join(self.output_dir, "all_transcripts.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(successful_transcripts, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "collection_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Create CSV for easy analysis
        if successful_transcripts:
            df = pd.DataFrame([{
                'video_id': t['video_id'],
                'title': t['title'],
                'word_count': t.get('word_count', 0),
                'segment_count': t.get('segment_count', 0),
                'author_name': t.get('author_name', 'Unknown')
            } for t in successful_transcripts])
            
            csv_file = os.path.join(self.output_dir, "transcripts_metadata.csv")
            df.to_csv(csv_file, index=False)
        
        print(f"\n🎉 Collection complete!")
        print(f"📊 Successfully collected {len(successful_transcripts)} transcripts")
        print(f"📈 Success rate: {summary['success_rate']:.1f}%")
        print(f"📝 Total words collected: {summary['total_words_collected']:,}")
        print(f"💾 Data saved to: {self.output_dir}")
        
        return successful_transcripts, summary

def test_single_video():
    """Test function to collect transcript from a single known video."""
    print("🧪 Testing with a single video...")
    
    collector = LexFridmanTranscriptCollector()
    
    # Test with a known Lex Fridman video ID
    test_video_id = "0Rnq1NpHdmw"  # Replace with a known working video ID
    
    # Get metadata
    metadata = collector.get_video_metadata(test_video_id)
    if metadata:
        print(f"✅ Metadata: {metadata['title']}")
    
    # Get transcript
    transcript = collector.get_transcript(test_video_id)
    if transcript:
        print(f"✅ Transcript: {transcript['word_count']} words")
        return True
    else:
        print("❌ Failed to get transcript")
        return False

def main():
    """Main function to run the transcript collection."""
    # First test with a single video
    if not test_single_video():
        print("❌ Single video test failed. Please check your internet connection and try again.")
        return
    
    print("\n" + "="*50)
    print("🚀 Starting full collection...")
    
    collector = LexFridmanTranscriptCollector()
    
    # Start collection (limit to 50 for initial test)
    transcripts, summary = collector.collect_all_transcripts(max_videos=50)
    
    if transcripts:
        print(f"\n📊 Collection Summary:")
        print(f"- Total transcripts collected: {len(transcripts)}")
        print(f"- Average words per transcript: {sum(t.get('word_count', 0) for t in transcripts) / len(transcripts):.0f}")
        print(f"- Total words collected: {sum(t.get('word_count', 0) for t in transcripts):,}")
        print(f"\n📁 Files created:")
        print(f"- data/all_transcripts.json")
        print(f"- data/transcripts_metadata.csv")
        print(f"- data/collection_summary.json")
        print(f"- data/transcripts/ (individual files)")
    else:
        print("❌ No transcripts were collected. Please check the issues above.")

if __name__ == "__main__":
    main() 