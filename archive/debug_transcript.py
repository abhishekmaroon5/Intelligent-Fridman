import requests
from youtube_transcript_api import YouTubeTranscriptApi

def test_network():
    """Test basic network connectivity."""
    print("🌐 Testing network connectivity...")
    try:
        response = requests.get("https://www.google.com", timeout=5)
        print(f"✅ Network OK: Status {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Network error: {e}")
        return False

def test_youtube_api():
    """Test YouTube API access."""
    print("📺 Testing YouTube API access...")
    try:
        response = requests.get("https://www.youtube.com", timeout=5)
        print(f"✅ YouTube accessible: Status {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ YouTube access error: {e}")
        return False

def test_oembed(video_id):
    """Test oembed API."""
    print(f"🔗 Testing oembed for video {video_id}...")
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Oembed success: {data.get('title', 'No title')}")
            return True
        else:
            print(f"❌ Oembed failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Oembed error: {e}")
        return False

def test_transcript_api(video_id):
    """Test transcript API with detailed debugging."""
    print(f"📝 Testing transcript API for video {video_id}...")
    
    try:
        # Check available transcripts
        available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        print(f"📋 Available transcripts found:")
        
        for transcript in available_transcripts:
            print(f"   - Language: {transcript.language}")
            print(f"   - Language code: {transcript.language_code}")
            print(f"   - Is generated: {transcript.is_generated}")
        
        # Try to get English transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        print(f"✅ Transcript retrieved: {len(transcript)} segments")
        
        # Show sample
        sample_text = " ".join([entry['text'] for entry in transcript[:3]])
        print(f"📖 Sample text: {sample_text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcript API error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Debugging transcript collection issues...\n")
    
    # Test network
    if not test_network():
        print("❌ Network issues detected. Please check internet connection.")
        return
    
    print()
    
    # Test YouTube access
    if not test_youtube_api():
        print("❌ YouTube access issues detected.")
        return
    
    print()
    
    # Test with a known good video ID
    test_video_id = "H-fMhWf3dcE"  # Max Tegmark episode
    
    # Test oembed
    oembed_success = test_oembed(test_video_id)
    print()
    
    # Test transcript API
    transcript_success = test_transcript_api(test_video_id)
    print()
    
    if oembed_success and transcript_success:
        print("🎉 All tests passed! The system should work.")
    else:
        print("❌ Some tests failed. There may be API restrictions or video availability issues.")
        
        # Try a different video
        print("\n🔄 Trying with a different video...")
        alt_video_id = "dBBMpS9AFHM"  # Steven Pinker episode
        test_oembed(alt_video_id)
        test_transcript_api(alt_video_id)

if __name__ == "__main__":
    main() 