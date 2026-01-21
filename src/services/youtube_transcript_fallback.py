"""
YouTube Transcript Fallback Service

This module provides alternative methods for retrieving YouTube transcripts
when the standard youtube_transcript_api fails.
"""

import asyncio
import logging
import re
import httpx
from typing import Dict, Any, Optional
from ddtrace import tracer

logger = logging.getLogger(__name__)

@tracer.wrap(resource="extract_transcript_from_watch_page")
async def extract_transcript_from_watch_page(video_id: str) -> Dict[str, Any]:
    """
    Alternative transcript extraction by parsing the YouTube watch page.
    This method attempts to extract transcript data from the page source.
    """
    try:
        logger.info(f"Attempting fallback transcript extraction for video: {video_id}")
        
        # Construct the watch URL
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Set up headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            response = await client.get(watch_url)
            
            if response.status_code != 200:
                return {"error": f"Failed to fetch YouTube page: HTTP {response.status_code}"}
            
            page_content = response.text
            
            # Look for transcript data in the page source
            transcript_patterns = [
                # Pattern 1: Look for captions in ytInitialPlayerResponse
                r'"captionTracks":\s*(\[.*?\])',
                # Pattern 2: Look for automatic speech recognition data
                r'"automaticCaptions":\s*({.*?})',
                # Pattern 3: Look for transcript in player config
                r'"transcriptRenderer":\s*({.*?})',
            ]
            
            caption_tracks = None
            for pattern in transcript_patterns:
                match = re.search(pattern, page_content)
                if match:
                    try:
                        import json
                        caption_data = json.loads(match.group(1))
                        caption_tracks = caption_data
                        break
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            if caption_tracks:
                # Try to extract English captions
                english_track = None
                for track in caption_tracks if isinstance(caption_tracks, list) else []:
                    if isinstance(track, dict) and track.get('languageCode', '').startswith('en'):
                        english_track = track
                        break
                
                if english_track and 'baseUrl' in english_track:
                    # Fetch the actual transcript from the caption URL
                    caption_url = english_track['baseUrl']
                    caption_response = await client.get(caption_url)
                    
                    if caption_response.status_code == 200:
                        # Parse the XML transcript
                        transcript_text = parse_transcript_xml(caption_response.text)
                        if transcript_text:
                            return {
                                "transcript": transcript_text,
                                "language": english_track.get('languageCode', 'en'),
                                "method": "fallback_page_extraction"
                            }
            
            # If direct caption extraction fails, try to extract any visible transcript text
            transcript_text = extract_transcript_from_page_text(page_content)
            if transcript_text:
                return {
                    "transcript": transcript_text,
                    "language": "en",
                    "method": "fallback_page_text"
                }
            
            return {"error": "No transcript data found in page source"}
            
    except Exception as e:
        logger.error(f"Error in fallback transcript extraction: {str(e)}", exc_info=True)
        return {"error": f"Fallback transcript extraction failed: {str(e)}"}

def parse_transcript_xml(xml_content: str) -> Optional[str]:
    """Parse transcript XML to extract text content."""
    try:
        # Remove XML tags and extract text content
        import xml.etree.ElementTree as ET
        
        # Clean up the XML content
        xml_content = re.sub(r'&[a-zA-Z0-9#]+;', ' ', xml_content)  # Remove HTML entities
        
        root = ET.fromstring(xml_content)
        
        transcript_parts = []
        for text_elem in root.findall('.//text'):
            text_content = text_elem.text
            if text_content and text_content.strip():
                # Clean up the text
                text_content = re.sub(r'\s+', ' ', text_content.strip())
                transcript_parts.append(text_content)
        
        if transcript_parts:
            return ' '.join(transcript_parts)
        
    except Exception as e:
        logger.warning(f"Failed to parse transcript XML: {str(e)}")
        
        # Fallback: try simple regex extraction
        try:
            # Extract text between XML tags
            text_matches = re.findall(r'<text[^>]*>(.*?)</text>', xml_content, re.DOTALL)
            if text_matches:
                cleaned_texts = []
                for match in text_matches:
                    # Remove HTML tags and clean up
                    clean_text = re.sub(r'<[^>]+>', '', match)
                    clean_text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', clean_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text.strip())
                    if clean_text:
                        cleaned_texts.append(clean_text)
                
                if cleaned_texts:
                    return ' '.join(cleaned_texts)
        except Exception as regex_error:
            logger.warning(f"Regex fallback also failed: {str(regex_error)}")
    
    return None

def extract_transcript_from_page_text(page_content: str) -> Optional[str]:
    """Extract any visible transcript-like content from the page."""
    try:
        # Look for structured transcript data in JavaScript variables
        js_patterns = [
            r'var\s+ytInitialData\s*=\s*({.*?});',
            r'window\["ytInitialData"\]\s*=\s*({.*?});',
            r'ytInitialPlayerResponse\s*=\s*({.*?});',
        ]
        
        for pattern in js_patterns:
            matches = re.findall(pattern, page_content, re.DOTALL)
            for match in matches:
                # Look for transcript-like content in the JSON
                if 'transcript' in match.lower() or 'caption' in match.lower():
                    # This would require more sophisticated JSON parsing
                    # For now, we'll skip this approach
                    pass
        
        return None
        
    except Exception as e:
        logger.warning(f"Failed to extract transcript from page text: {str(e)}")
        return None

@tracer.wrap(resource="get_transcript_with_fallback")
async def get_transcript_with_fallback(video_id: str, primary_method_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try fallback transcript extraction methods if primary method fails.
    """
    if "error" not in primary_method_result:
        # Primary method succeeded, no need for fallback
        return primary_method_result
    
    logger.info(f"Primary transcript method failed for {video_id}, trying fallback methods")
    
    # Try fallback method 1: Page extraction
    fallback_result = await extract_transcript_from_watch_page(video_id)
    
    if "error" not in fallback_result:
        logger.info(f"Fallback transcript extraction succeeded for {video_id}")
        return fallback_result
    
    # Future: Could add more fallback methods here
    # - Third-party transcript APIs
    # - Alternative parsing methods
    # - OCR on video frames (for hardcoded subtitles)
    
    # Return the original error if all methods fail
    error_msg = primary_method_result.get("error", "Unknown error")
    fallback_error = fallback_result.get("error", "Unknown fallback error")
    
    return {
        "error": f"All transcript extraction methods failed. Primary: {error_msg}. Fallback: {fallback_error}"
    }

# Utility function to test the fallback system
async def test_fallback_transcript(video_id: str) -> None:
    """Test the fallback transcript system for a specific video ID."""
    print(f"Testing fallback transcript extraction for: {video_id}")
    
    # Simulate primary method failure
    primary_result = {"error": "Primary method failed (simulated)"}
    
    result = await get_transcript_with_fallback(video_id, primary_result)
    
    if "error" in result:
        print(f"❌ Fallback failed: {result['error']}")
    else:
        print(f"✅ Fallback succeeded!")
        print(f"Method: {result.get('method', 'unknown')}")
        print(f"Language: {result.get('language', 'unknown')}")
        print(f"Transcript length: {len(result.get('transcript', ''))} characters")
        if result.get('transcript'):
            preview = result['transcript'][:200] + "..." if len(result['transcript']) > 200 else result['transcript']
            print(f"Preview: {preview}")

if __name__ == "__main__":
    # Test the fallback system
    async def main():
        test_video_ids = ["YjpBqZnJkic", "7k0UotRheN4"]
        for video_id in test_video_ids:
            await test_fallback_transcript(video_id)
            print("-" * 80)
    
    asyncio.run(main())

