import os
import json
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

# You'll need to install these packages:
# pip install youtube-transcript-api pytube google-api-python-client

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from pytube import Channel
    import googleapiclient.discovery
except ImportError as e:
    print(f"Missing required packages. Please install them using:")
    print("pip install youtube-transcript-api pytube google-api-python-client")
    exit(1)

class LexFridmanTranscriptCollector:
    def __init__(self, output_dir: str = "data"):
        self.channel_url = "https://www.youtube.com/c/lexfridman"
        self.channel_id = "UCSHZKyawb77ixDdsGog4iWA"  # Lex Fridman's channel ID
        self.output_dir = output_dir
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "transcripts"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        
    def get_channel_videos(self, max_videos: int = 400) -> List[Dict]:
        """Get video information from Lex Fridman's channel."""
        print(f"Fetching video information from Lex Fridman's channel...")
        
        try:
            channel = Channel(self.channel_url)
            videos = []
            
            for i, video in enumerate(channel.video_urls):
                if i >= max_videos:
                    break
                    
                try:
                    # Extract video ID from URL
                    video_id = video.split("watch?v=")[1]
                    
                    # Get video metadata using pytube
                    from pytube import YouTube
                    yt = YouTube(video)
                    
                    video_info = {
                        'video_id': video_id,
                        'title': yt.title,
                        'url': video,
                        'publish_date': yt.publish_date.isoformat() if yt.publish_date else None,
                        'duration': yt.length,
                        'view_count': yt.views,
                        'description': yt.description[:500] + "..." if len(yt.description) > 500 else yt.description
                    }
                    
                    videos.append(video_info)
                    print(f"Collected metadata for video {i+1}: {yt.title[:50]}...")
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error getting metadata for video {i+1}: {str(e)}")
                    continue
                    
            return videos
            
        except Exception as e:
            print(f"Error accessing channel: {str(e)}")
            return []
    
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
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Could not retrieve transcript for video {video_id}: {str(e)}")
            return None
    
    def collect_all_transcripts(self, max_videos: int = 400):
        """Main method to collect all transcripts."""
        print("Starting transcript collection for Lex Fridman's channel...")
        
        # Get video information
        videos = self.get_channel_videos(max_videos)
        print(f"Found {len(videos)} videos to process")
        
        successful_transcripts = []
        failed_videos = []
        
        for i, video in enumerate(videos):
            print(f"\nProcessing video {i+1}/{len(videos)}: {video['title'][:50]}...")
            
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
                    
                print(f"✅ Transcript saved for: {video['title'][:50]}...")
            else:
                failed_videos.append(video)
                print(f"❌ Failed to get transcript for: {video['title'][:50]}...")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        # Save summary
        summary = {
            'total_videos_processed': len(videos),
            'successful_transcripts': len(successful_transcripts),
            'failed_videos': len(failed_videos),
            'success_rate': len(successful_transcripts) / len(videos) * 100,
            'collection_date': datetime.now().isoformat(),
            'failed_video_ids': [v['video_id'] for v in failed_videos]
        }
        
        # Save combined transcripts
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
                'publish_date': t['publish_date'],
                'duration': t['duration'],
                'word_count': t['word_count'],
                'view_count': t['view_count']
            } for t in successful_transcripts])
            
            csv_file = os.path.join(self.output_dir, "transcripts_metadata.csv")
            df.to_csv(csv_file, index=False)
        
        print(f"\n🎉 Collection complete!")
        print(f"📊 Successfully collected {len(successful_transcripts)} transcripts")
        print(f"📈 Success rate: {summary['success_rate']:.1f}%")
        print(f"💾 Data saved to: {self.output_dir}")
        
        return successful_transcripts, summary

def main():
    """Main function to run the transcript collection."""
    collector = LexFridmanTranscriptCollector()
    
    # Start collection
    transcripts, summary = collector.collect_all_transcripts(max_videos=400)
    
    print(f"\nCollection Summary:")
    print(f"- Total transcripts collected: {len(transcripts)}")
    print(f"- Average words per transcript: {sum(t['word_count'] for t in transcripts) / len(transcripts):.0f}")
    print(f"- Total words collected: {sum(t['word_count'] for t in transcripts):,}")

if __name__ == "__main__":
    main() 