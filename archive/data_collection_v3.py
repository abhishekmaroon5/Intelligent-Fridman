import os
import json
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
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
        
        # Known Lex Fridman video IDs that should have transcripts
        self.known_lex_videos = [
            "H-fMhWf3dcE",  # Lex Fridman #1 - Max Tegmark
            "yNAh6kwQwi8",  # Lex Fridman #2 - François Chollet
            "dBBMpS9AFHM",  # Lex Fridman #3 - Steven Pinker
            "CLQPS8kj8J8",  # Lex Fridman #4 - Jim Keller
            "NuLhzD6xUvc",  # Lex Fridman #5 - Vladimir Vapnik
            "7A8BK-lkjRs",  # Lex Fridman #6 - Chris Lattner
            "Dv7hjcLR56A",  # Lex Fridman #7 - Ian Goodfellow
            "dRAd6b_5UvQ",  # Lex Fridman #8 - Tuomas Sandholm
            "DKXaC6wEhbE",  # Lex Fridman #9 - Stuart Russell
            "iJ1HcX1zJWI",  # Lex Fridman #10 - Yoshua Bengio
            "rRgp_rAJb1Q",  # Lex Fridman #11 - Pieter Abbeel
            "cLv3TTp87jE",  # Lex Fridman #12 - Jeff Atwood
            "aBVGKyOaz8g",  # Lex Fridman #13 - Ilya Sutskever
            "V2u9JdTnJHo",  # Lex Fridman #14 - Greg Brockman
            "6Pv7lWBzJo8",  # Lex Fridman #15 - Jeremy Howard
            # Add more as we verify them
        ]
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "transcripts"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
        
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
        """Get transcript for a specific video with multiple language options."""
        try:
            # Try to get the transcript - first try English, then auto-generated
            transcript_list = None
            
            # Try different language codes
            language_options = ['en', 'en-US', 'en-GB']
            
            for lang in language_options:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    break
                except:
                    continue
            
            # If specific languages fail, try auto-generated
            if not transcript_list:
                try:
                    # Get all available transcripts
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                except:
                    # Try auto-generated
                    try:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-auto'])
                    except:
                        pass
            
            if transcript_list:
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
            else:
                return None
                
        except NoTranscriptFound:
            print(f"❌ No transcript found for video {video_id}")
            return None
        except TranscriptsDisabled:
            print(f"❌ Transcripts disabled for video {video_id}")
            return None
        except Exception as e:
            print(f"❌ Error getting transcript for video {video_id}: {str(e)}")
            return None
    
    def search_lex_videos_manually(self, max_videos: int = 50) -> List[str]:
        """Manually curated list of Lex Fridman video IDs."""
        print("📚 Using manually curated list of Lex Fridman videos...")
        
        # Extended list of known Lex Fridman videos
        all_videos = [
            "H-fMhWf3dcE",  # Max Tegmark
            "yNAh6kwQwi8",  # François Chollet
            "dBBMpS9AFHM",  # Steven Pinker
            "CLQPS8kj8J8",  # Jim Keller
            "NuLhzD6xUvc",  # Vladimir Vapnik
            "7A8BK-lkjRs",  # Chris Lattner
            "Dv7hjcLR56A",  # Ian Goodfellow
            "dRAd6b_5UvQ",  # Tuomas Sandholm
            "DKXaC6wEhbE",  # Stuart Russell
            "iJ1HcX1zJWI",  # Yoshua Bengio
            "rRgp_rAJb1Q",  # Pieter Abbeel
            "cLv3TTp87jE",  # Jeff Atwood
            "aBVGKyOaz8g",  # Ilya Sutskever
            "V2u9JdTnJHo",  # Greg Brockman
            "6Pv7lWBzJo8",  # Jeremy Howard
            "3_bD5l-IGTM",  # Leslie Kaelbling
            "KLj7rE6hqM8",  # George Hotz
            "J5uU9W0ysW4",  # Lisa Feldman Barrett
            "yGGWKbTYJXw",  # Eric Weinstein
            "I-2r1ACjMEw",  # Noam Chomsky
            "gPfriiHBBek",  # Jocko Willink
            "E5Tj11jB4jo",  # Judea Pearl
            "qIGKVIl5Rxs",  # Jeff Hawkins
            "aEBVVzVsNfI",  # Michael Stevens
            "qF7dkxqbpEs",  # Tyler Cowen
            "hk9Xte9YKFE",  # Garry Kasparov
            "rvJLhALOCtI",  # Yann LeCun
            "jJ-OAIh_hzc",  # Jordan Peterson
            "PJ1nZV-HNzY",  # Michael Jordan
            "S1U7y3hU5y8",  # Nick Bostrom
            "8-9THZlIVAE",  # Elon Musk #1
            "dEv99vxKjVI",  # Elon Musk #2
            "smK9dgdTl40",  # Elon Musk #3
            "nAbGfz6QqnI",  # Sam Harris
            "ueAzJzBFCxI",  # Paul Krugman
            "GVZY5qjC_P4",  # Eric Schmidt
            "VuN7qTFG0YQ",  # Andrew Ng
            "IJjaCjfKEy4",  # Rosalind Picard
            "FPJOEPsRCE0",  # Vijay Kumar
            "eLCpDy5J_LY",  # Ben Goertzel
            "rvJLhALOCtI",  # Yann LeCun
            "KjG3dURR5_o",  # Whitney Cummings
            "b_Y_E0yDK8M",  # Sheldon Solomon
            "0z_jIo3FyqE",  # David Ferrucci
        ]
        
        return all_videos[:max_videos]
    
    def collect_all_transcripts(self, max_videos: int = 50):
        """Main method to collect all transcripts."""
        print("🚀 Starting transcript collection for Lex Fridman's channel...")
        
        # Get video IDs
        video_ids = self.search_lex_videos_manually(max_videos)
        print(f"📺 Found {len(video_ids)} video IDs to process")
        
        successful_transcripts = []
        failed_videos = []
        
        # Process each video
        for i, video_id in enumerate(tqdm(video_ids, desc="Collecting transcripts")):
            print(f"\n📹 Processing video {i+1}/{len(video_ids)}: {video_id}")
            
            # Get metadata
            metadata = self.get_video_metadata(video_id)
            if not metadata:
                print(f"❌ Could not get metadata for {video_id}")
                failed_videos.append({'video_id': video_id, 'reason': 'metadata_failed'})
                continue
                
            print(f"   Title: {metadata['title'][:70]}...")
            
            # Get transcript
            transcript_data = self.get_transcript(video_id)
            
            if transcript_data:
                # Combine video metadata with transcript
                combined_data = {**metadata, **transcript_data}
                successful_transcripts.append(combined_data)
                
                # Save individual transcript file
                transcript_file = os.path.join(
                    self.output_dir, 
                    "transcripts", 
                    f"{video_id}.json"
                )
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)
                    
                print(f"✅ Transcript saved: {transcript_data['word_count']} words")
            else:
                failed_videos.append({**metadata, 'reason': 'transcript_failed'})
                print(f"❌ Failed to get transcript")
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
        
        # Calculate success rate safely
        success_rate = (len(successful_transcripts) / len(video_ids) * 100) if video_ids else 0
        
        # Save summary
        summary = {
            'total_videos_processed': len(video_ids),
            'successful_transcripts': len(successful_transcripts),
            'failed_videos': len(failed_videos),
            'success_rate': success_rate,
            'collection_date': datetime.now().isoformat(),
            'failed_video_ids': [v.get('video_id', 'unknown') for v in failed_videos],
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
    print("🧪 Testing with a known Lex Fridman video...")
    
    collector = LexFridmanTranscriptCollector()
    
    # Test with a known Lex Fridman video ID (Max Tegmark episode)
    test_video_id = "H-fMhWf3dcE"
    
    # Get metadata
    metadata = collector.get_video_metadata(test_video_id)
    if metadata:
        print(f"✅ Metadata: {metadata['title']}")
        print(f"   Author: {metadata['author_name']}")
    
    # Get transcript
    transcript = collector.get_transcript(test_video_id)
    if transcript:
        print(f"✅ Transcript: {transcript['word_count']} words")
        print(f"   Sample text: {transcript['transcript'][:100]}...")
        return True
    else:
        print("❌ Failed to get transcript")
        return False

def main():
    """Main function to run the transcript collection."""
    # First test with a single video
    if not test_single_video():
        print("❌ Single video test failed. Continuing with collection anyway...")
    
    print("\n" + "="*50)
    print("🚀 Starting full collection...")
    
    collector = LexFridmanTranscriptCollector()
    
    # Start collection with a reasonable number for initial test
    transcripts, summary = collector.collect_all_transcripts(max_videos=30)
    
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
        
        # Show some example titles
        print(f"\n📚 Sample collected episodes:")
        for i, t in enumerate(transcripts[:5]):
            print(f"   {i+1}. {t['title'][:60]}... ({t.get('word_count', 0)} words)")
            
    else:
        print("❌ No transcripts were collected. Please check the issues above.")

if __name__ == "__main__":
    main() 