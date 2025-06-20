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
except ImportError as e:
    print(f"Missing required packages. Please install them using:")
    print("pip install youtube-transcript-api")
    exit(1)

class LexFridmanTranscriptCollector:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # Verified Lex Fridman video IDs
        self.lex_videos = [
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
        ]
        
    def ensure_output_dir(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "transcripts"), exist_ok=True)
        
    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Get video metadata using oembed API."""
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url, timeout=10)
            
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
            return None
        except Exception as e:
            print(f"   ⚠️  Metadata error for {video_id}: {e}")
            return None
    
    def get_transcript(self, video_id: str) -> Optional[Dict]:
        """Get transcript with fallback language options."""
        try:
            # Try different language options
            for languages in [['en'], ['en-US'], ['en-GB'], None]:
                try:
                    if languages:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                    else:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    
                    # Success! Process the transcript
                    full_transcript = " ".join([entry['text'] for entry in transcript_list])
                    
                    return {
                        'video_id': video_id,
                        'transcript': full_transcript,
                        'segments': transcript_list,
                        'word_count': len(full_transcript.split()),
                        'segment_count': len(transcript_list),
                        'collected_at': datetime.now().isoformat()
                    }
                except:
                    continue
                    
            return None
            
        except Exception as e:
            return None
    
    def collect_transcripts(self, max_videos: int = 30):
        """Main collection method."""
        print("🚀 Starting Lex Fridman transcript collection...")
        
        video_ids = self.lex_videos[:max_videos]
        print(f"📺 Processing {len(video_ids)} videos")
        
        successful_transcripts = []
        failed_videos = []
        
        for i, video_id in enumerate(tqdm(video_ids, desc="Collecting")):
            print(f"\n📹 Video {i+1}/{len(video_ids)}: {video_id}")
            
            # Get metadata
            metadata = self.get_video_metadata(video_id)
            if not metadata:
                failed_videos.append({'video_id': video_id, 'reason': 'metadata_failed'})
                continue
                
            print(f"   📝 {metadata['title'][:60]}...")
            
            # Get transcript
            transcript_data = self.get_transcript(video_id)
            
            if transcript_data:
                # Combine data
                combined_data = {**metadata, **transcript_data}
                successful_transcripts.append(combined_data)
                
                # Save individual file
                transcript_file = os.path.join(self.output_dir, "transcripts", f"{video_id}.json")
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)
                    
                print(f"   ✅ Success: {transcript_data['word_count']} words")
            else:
                failed_videos.append({**metadata, 'reason': 'transcript_failed'})
                print(f"   ❌ No transcript available")
            
            time.sleep(0.5)  # Rate limiting
        
        # Save results
        success_rate = (len(successful_transcripts) / len(video_ids) * 100) if video_ids else 0
        total_words = sum(t.get('word_count', 0) for t in successful_transcripts)
        
        summary = {
            'total_videos_processed': len(video_ids),
            'successful_transcripts': len(successful_transcripts),
            'failed_videos': len(failed_videos),
            'success_rate': success_rate,
            'total_words_collected': total_words,
            'collection_date': datetime.now().isoformat()
        }
        
        # Save combined file
        if successful_transcripts:
            combined_file = os.path.join(self.output_dir, "all_transcripts.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(successful_transcripts, f, indent=2, ensure_ascii=False)
            
            # Save CSV
            df = pd.DataFrame([{
                'video_id': t['video_id'],
                'title': t['title'],
                'word_count': t.get('word_count', 0),
                'author_name': t.get('author_name', 'Unknown')
            } for t in successful_transcripts])
            
            csv_file = os.path.join(self.output_dir, "transcripts_metadata.csv")
            df.to_csv(csv_file, index=False)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "collection_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print results
        print(f"\n🎉 Collection Complete!")
        print(f"📊 Results: {len(successful_transcripts)}/{len(video_ids)} successful ({success_rate:.1f}%)")
        print(f"📝 Total words: {total_words:,}")
        print(f"💾 Saved to: {self.output_dir}/")
        
        if successful_transcripts:
            print(f"\n📚 Sample episodes collected:")
            for i, t in enumerate(successful_transcripts[:5]):
                print(f"   {i+1}. {t['title'][:50]}... ({t.get('word_count', 0)} words)")
        
        return successful_transcripts, summary

def test_single_video():
    """Test with one video first."""
    print("🧪 Testing single video...")
    collector = LexFridmanTranscriptCollector()
    
    # Test Max Tegmark episode
    test_id = "H-fMhWf3dcE"
    metadata = collector.get_video_metadata(test_id)
    
    if metadata:
        print(f"✅ Metadata: {metadata['title']}")
        transcript = collector.get_transcript(test_id)
        if transcript:
            print(f"✅ Transcript: {transcript['word_count']} words")
            print(f"   Sample: {transcript['transcript'][:100]}...")
            return True
    
    print("❌ Test failed")
    return False

def main():
    """Main execution."""
    # Test first
    if test_single_video():
        print("\n" + "="*50)
        collector = LexFridmanTranscriptCollector()
        collector.collect_transcripts(max_videos=25)
    else:
        print("❌ Unable to collect transcripts. Check internet connection.")

if __name__ == "__main__":
    main() 