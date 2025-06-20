from yt_dlp import YoutubeDL
import json

def test_channel_access():
    """Test different ways to access Lex Fridman's channel."""
    
    channel_urls = [
        "https://www.youtube.com/@lexfridman",
        "https://www.youtube.com/c/lexfridman", 
        "https://www.youtube.com/channel/UCSHZKyawb77ixDdsGog4iWA",
        "https://www.youtube.com/user/lexfridman"
    ]
    
    for url in channel_urls:
        print(f"\n🔍 Testing: {url}")
        
        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'playlist_items': '1-10',  # Just get first 10 videos
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(url, download=False)
                
                if result and 'entries' in result:
                    entries = [e for e in result['entries'] if e]  # Filter out None entries
                    print(f"✅ Found {len(entries)} videos")
                    
                    if entries:
                        print("📺 Sample videos:")
                        for i, entry in enumerate(entries[:3]):
                            print(f"   {i+1}. {entry.get('title', 'No title')} (ID: {entry.get('id', 'No ID')})")
                        return url, entries  # Return successful URL and videos
                else:
                    print("❌ No entries found")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return None, []

def test_specific_video():
    """Test with a known Lex Fridman video."""
    print("\n🧪 Testing with a specific recent video...")
    
    # Try a few different recent video IDs
    test_videos = [
        "SQpVUdIZcGM",  # Recent video
        "zmbq0MWZMXQ",  # Another recent one
        "mQV_SmtS_hM",  # Another
    ]
    
    for video_id in test_videos:
        try:
            ydl_opts = {'quiet': True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                print(f"✅ Video found: {info.get('title', 'Unknown title')}")
                print(f"   Channel: {info.get('channel', 'Unknown')}")
                print(f"   Duration: {info.get('duration', 0)} seconds")
                return video_id, info
        except Exception as e:
            print(f"❌ Video {video_id} error: {e}")
    
    return None, None

def main():
    print("🚀 Testing yt_dlp with Lex Fridman content...\n")
    
    # Test channel access
    successful_url, videos = test_channel_access()
    
    if successful_url and videos:
        print(f"\n🎉 Successfully accessed channel: {successful_url}")
        print(f"📊 Found {len(videos)} videos")
        
        # Save the results for reference
        with open("test_videos.json", "w") as f:
            json.dump({
                'channel_url': successful_url,
                'video_count': len(videos),
                'videos': videos[:10]  # Save first 10 for testing
            }, f, indent=2)
        
        print("💾 Saved test results to test_videos.json")
    else:
        print("\n❌ Could not access channel. Trying specific videos...")
        test_specific_video()

if __name__ == "__main__":
    main() 