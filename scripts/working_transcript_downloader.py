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
        self.channel_url = "https://www.youtube.com/@lexfridman"
        
    def ensure_output_dir(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "transcripts"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "txt_files"), exist_ok=True)
        
    def sanitize_filename(self, name: str) -> str:
        """Clean filename for safe saving."""
        clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
        return clean_name[:100]
    
    def get_channel_videos(self, max_videos: int = 100) -> List[Dict]:
        """Extract actual video entries from Lex Fridman's channel."""
        print(f"🔍 Fetching up to {max_videos} videos from Lex Fridman's channel...")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlist_items': f'1-{max_videos}',
        }
        
        all_videos = []
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(self.channel_url, download=False)
                
                if result and 'entries' in result:
                    print(f"📺 Found {len(result['entries'])} channel sections")
                    
                    # Look through each section (e.g., "Videos", "Shorts")
                    for section in result['entries']:
                        if section and section.get('_type') == 'playlist' and 'entries' in section:
                            section_title = section.get('title', 'Unknown Section')
                            videos_in_section = section['entries']
                            
                            print(f"   📂 Section '{section_title}': {len(videos_in_section)} videos")
                            
                            # Extract video information from this section
                            for video_entry in videos_in_section:
                                if video_entry and video_entry.get('id'):
                                    video_info = {
                                        'video_id': video_entry['id'],
                                        'title': video_entry.get('title', 'Unknown Title'),
                                        'url': video_entry.get('url', f"https://www.youtube.com/watch?v={video_entry['id']}"),
                                        'duration': video_entry.get('duration', 0),
                                        'view_count': video_entry.get('view_count', 0),
                                        'description': video_entry.get('description', '')[:500] if video_entry.get('description') else '',
                                        'section': section_title
                                    }
                                    all_videos.append(video_info)
                                    
                                    # Stop if we've reached our limit
                                    if len(all_videos) >= max_videos:
                                        break
                            
                            if len(all_videos) >= max_videos:
                                break
                
                print(f"✅ Extracted {len(all_videos)} videos total")
                return all_videos[:max_videos]
                
        except Exception as e:
            print(f"❌ Error fetching channel videos: {e}")
            return []
    
    def download_transcript(self, video_info: Dict) -> bool:
        """Download and save transcript for a single video."""
        video_id = video_info['video_id']
        title = video_info['title']
        
        try:
            # Try to get transcript with multiple language options
            transcript = None
            for languages in [['en'], ['en-US'], ['en-GB'], None]:
                try:
                    if languages:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                    else:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    break
                except:
                    continue
            
            if not transcript:
                return False
            
            # Process transcript
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
                f.write(f"Duration: {video_info.get('duration', 0)} seconds\n")
                f.write(f"Views: {video_info.get('view_count', 0):,}\n")
                f.write(f"Word Count: {word_count}\n")
                f.write("-" * 80 + "\n\n")
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
            
            print(f"✅ {title[:60]}... ({word_count:,} words)")
            return True
            
        except NoTranscriptFound:
            print(f"❌ No transcript: {title[:60]}...")
            return False
        except TranscriptsDisabled:
            print(f"❌ Disabled: {title[:60]}...")
            return False
        except Exception as e:
            print(f"⚠️  Error: {title[:60]}... - {str(e)}")
            return False
    
    def download_all_transcripts(self, max_videos: int = 50):
        """Main method to download all transcripts."""
        print("🚀 Starting Lex Fridman transcript collection...")
        print("=" * 60)
        
        # Get video list
        videos = self.get_channel_videos(max_videos)
        
        if not videos:
            print("❌ No videos found!")
            return [], {}
        
        print(f"\n📥 Processing {len(videos)} videos...")
        
        successful_downloads = []
        failed_downloads = []
        
        # Download with progress bar
        for i, video_info in enumerate(tqdm(videos, desc="Downloading transcripts")):
            print(f"\n📹 {i+1}/{len(videos)}: {video_info['title'][:50]}...")
            
            if self.download_transcript(video_info):
                successful_downloads.append(video_info)
            else:
                failed_downloads.append(video_info)
        
        # Generate summary
        success_rate = (len(successful_downloads) / len(videos) * 100) if videos else 0
        total_words = sum(
            len(open(os.path.join(self.output_dir, "txt_files", f"{self.sanitize_filename(t['title'])}_{t['video_id']}.txt"), 'r', encoding='utf-8').read().split())
            for t in successful_downloads
            if os.path.exists(os.path.join(self.output_dir, "txt_files", f"{self.sanitize_filename(t['title'])}_{t['video_id']}.txt"))
        )
        
        summary = {
            'total_videos_processed': len(videos),
            'successful_downloads': len(successful_downloads),
            'failed_downloads': len(failed_downloads),
            'success_rate': success_rate,
            'estimated_total_words': total_words,
            'collection_date': datetime.now().isoformat(),
            'channel_url': self.channel_url
        }
        
        # Save combined data
        if successful_downloads:
            # Load all JSON files and combine
            all_transcripts = []
            for video in successful_downloads:
                json_file = os.path.join(self.output_dir, "transcripts", f"{video['video_id']}.json")
                if os.path.exists(json_file):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        all_transcripts.append(json.load(f))
            
            # Save combined file
            combined_file = os.path.join(self.output_dir, "all_transcripts.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_transcripts, f, indent=2, ensure_ascii=False)
            
            # Save metadata CSV
            df = pd.DataFrame([{
                'video_id': t['video_id'],
                'title': t['title'],
                'duration_seconds': t.get('duration', 0),
                'view_count': t.get('view_count', 0),
                'section': t.get('section', 'Unknown'),
                'word_count': t.get('word_count', 0) if 'word_count' in t else 0
            } for t in all_transcripts])
            
            csv_file = os.path.join(self.output_dir, "transcripts_metadata.csv")
            df.to_csv(csv_file, index=False)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "download_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print final results
        print(f"\n🎉 Download Complete!")
        print("=" * 60)
        print(f"📊 Success Rate: {success_rate:.1f}% ({len(successful_downloads)}/{len(videos)})")
        print(f"📝 Estimated Total Words: {total_words:,}")
        print(f"💾 Files saved in: {self.output_dir}/")
        print(f"   📄 Individual TXT files: {len(successful_downloads)}")
        print(f"   📋 Individual JSON files: {len(successful_downloads)}")
        print(f"   🗂️  Combined JSON: all_transcripts.json")
        print(f"   📊 Metadata CSV: transcripts_metadata.csv")
        
        if successful_downloads:
            print(f"\n📚 Sample collected episodes:")
            for i, video in enumerate(successful_downloads[:5]):
                duration_min = video.get('duration', 0) // 60
                views_k = video.get('view_count', 0) // 1000
                print(f"   {i+1}. {video['title'][:55]}... ({duration_min}min, {views_k}K views)")
        
        return successful_downloads, summary

def main():
    """Main execution function."""
    print("🚀 Lex Fridman Transcript Downloader")
    print("Using yt_dlp + YouTube Transcript API")
    print("=" * 60)
    
    downloader = LexFridmanTranscriptDownloader()
    
    # Get user input for number of videos
    try:
        max_videos = input("\nHow many videos to download? (press Enter for 30): ").strip()
        max_videos = int(max_videos) if max_videos else 30
    except ValueError:
        max_videos = 30
    
    print(f"\n🎯 Downloading transcripts from up to {max_videos} videos...")
    
    # Start download
    successful_downloads, summary = downloader.download_all_transcripts(max_videos)
    
    if successful_downloads:
        print(f"\n✅ Success! Ready to proceed to next steps:")
        print(f"   📂 Data collected in: {downloader.output_dir}/")
        print(f"   📈 Next: Tokenization and preprocessing")
        print(f"   🤖 Then: Model fine-tuning with your GPUs!")
    else:
        print(f"\n❌ No transcripts were collected. Please check:")
        print(f"   🌐 Internet connection")
        print(f"   📺 Video availability and transcript access")

if __name__ == "__main__":
    main() 