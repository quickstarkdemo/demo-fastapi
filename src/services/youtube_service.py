from openai import AsyncOpenAI, OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
import os
import json
from ddtrace import tracer
import logging
import asyncio
from datetime import datetime
from .notion_service import add_video_summary_to_notion, NotionVideoPayload
from .youtube_transcript_fallback import get_transcript_with_fallback
from pytube import YouTube
import httpx
import re

# Set up logging
logger = logging.getLogger(__name__)

async def process_youtube_video(youtube_url: str, instructions: str = None, save_to_notion: bool = False):
    """Process YouTube video to extract transcript and generate AI summary.
    
    Downloads video transcript, retrieves metadata using pytube, generates
    AI-powered summary using OpenAI, and optionally saves to Notion database.
    
    Args:
        youtube_url (str): Valid YouTube video URL to process.
        instructions (str, optional): Custom instructions for AI summarization.
        save_to_notion (bool, optional): Whether to save results to Notion. Defaults to False.
        
    Returns:
        dict: Contains video_id, transcript, language, summary, and optional metadata.
              If save_to_notion=True, includes notion_page_id on successful save.
              
    Raises:
        Exception: If video ID extraction fails, transcript unavailable,
                  or AI processing encounters errors.
    """
    try:
        # Extract video ID
        video_id = get_video_id(youtube_url)
        if not video_id:
            logger.error(f"Invalid YouTube URL: {youtube_url}")
            return {"error": "Invalid YouTube URL"}

        # Get video details using pytube - run in executor since pytube is synchronous
        video_details = {}
        if save_to_notion:
            loop = asyncio.get_running_loop()
            video_details = await loop.run_in_executor(
                None, lambda: get_youtube_video_details_pytube(youtube_url)
            )
            if "error" in video_details:
                logger.warning(f"Video details warning for {video_id}: {video_details['error']}")
                # We'll continue with the basic metadata that was returned

        logger.info(f"Retrieved video metadata: Title='{video_details.get('title', 'Unknown')}', Channel='{video_details.get('channel', 'Unknown')}'")

        # Get transcript - Run in executor since YouTubeTranscriptApi is synchronous
        loop = asyncio.get_running_loop()
        transcript_result = await loop.run_in_executor(
            None, lambda: get_youtube_transcript(video_id)
        )
        
        # Try fallback transcript methods if primary method failed
        if "error" in transcript_result:
            logger.warning(f"Primary transcript method failed for {video_id}: {transcript_result['error']}")
            logger.info(f"Attempting fallback transcript extraction for {video_id}")
            transcript_result = await get_transcript_with_fallback(video_id, transcript_result)
        
        if "error" in transcript_result:
            logger.error(f"All transcript methods failed for video {video_id}: {transcript_result['error']}")
            return transcript_result

        # Generate summary - use AsyncOpenAI client
        summary_result = await generate_video_summary_async(transcript_result["transcript"], instructions)
        if "error" in summary_result:
            logger.error(f"Summary generation error for video {video_id}: {summary_result['error']}")
            return summary_result
            
        # Save to Notion if requested and we have a summary
        notion_result = {}
        if save_to_notion and "summary" in summary_result and summary_result["summary"]:
            try:
                # Prepare the payload for Notion
                payload_data = {
                    "url": youtube_url,
                    "title": video_details.get("title", f"YouTube Video: {video_id}"),
                    "published_date": video_details.get("published_date", datetime.now().isoformat()),
                    "summary": summary_result["summary"],
                    "channel": video_details.get("channel"),
                    "views": video_details.get("views"),
                }
                
                notion_payload = NotionVideoPayload(**payload_data)
                
                # Add to Notion with transcript as separate parameter
                notion_result = await add_video_summary_to_notion(
                    notion_payload, 
                    views_as_number=True,
                    transcript=transcript_result["transcript"]
                )
                if "error" in notion_result and notion_result["error"]:
                    logger.error(f"Failed to save to Notion: {notion_result['error']}")
            except Exception as e:
                logger.error(f"Error saving to Notion: {str(e)}", exc_info=True)
                notion_result = {"error": f"Error saving to Notion: {str(e)}"}
        
        # Build the response
        response = {
            "video_id": video_id,
            "transcript": transcript_result["transcript"],
            "language": transcript_result["language"],
            "summary": summary_result.get("summary", None),
            "error": summary_result.get("error", None)
        }
        
        # Add video details if available
        if video_details and "title" in video_details:
            response["title"] = video_details["title"]
        if video_details and "published_date" in video_details:
            response["published_date"] = video_details["published_date"]
        if video_details and "channel" in video_details:
            response["channel"] = video_details["channel"]
            
        # Add Notion result if available
        if notion_result:
            if "notion_page_id" in notion_result:
                response["notion_page_id"] = notion_result["notion_page_id"]
            if "error" in notion_result and notion_result["error"]:
                response["notion_error"] = notion_result["error"]
                
        return response
    except Exception as e:
        logger.error(f"Unexpected error in process_youtube_video: {str(e)}", exc_info=True)
        return {"error": f"Server error: {str(e)}"}

@tracer.wrap(resource="get_video_id")
def get_video_id(youtube_url: str) -> str:
    """Extracts the video ID from a YouTube URL with enhanced URL format support."""
    try:
        logger.info(f"Extracting video ID from URL: {youtube_url}")
        
        # Clean the URL first
        youtube_url = youtube_url.strip()
        
        parsed_url = urlparse(youtube_url)
        logger.info(f"Parsed URL - netloc: {parsed_url.netloc}, path: {parsed_url.path}, query: {parsed_url.query}")
        
        # Handle different YouTube URL formats
        if parsed_url.netloc in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
            # Standard YouTube URLs: youtube.com/watch?v=VIDEO_ID
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params and query_params['v'][0]:
                video_id = query_params['v'][0]
                # Clean video ID of any additional parameters (like &si=...)
                video_id = video_id.split('&')[0]  # Remove any additional params
                logger.info(f"Extracted video ID from youtube.com: {video_id}")
                return video_id
                
        elif parsed_url.netloc == 'youtu.be':
            # Shortened URLs: youtu.be/VIDEO_ID
            video_id = parsed_url.path[1:]  # Remove leading /
            if video_id:
                # Clean video ID of any additional parameters
                video_id = video_id.split('?')[0]  # Remove query parameters
                video_id = video_id.split('&')[0]  # Remove additional params
                logger.info(f"Extracted video ID from youtu.be: {video_id}")
                return video_id
        
        # Fallback: try to extract 11-character video ID using regex
        import re
        # YouTube video IDs are typically 11 characters long
        video_id_pattern = r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:\S+)?'
        match = re.search(video_id_pattern, youtube_url)
        if match:
            video_id = match.group(1)
            logger.info(f"Extracted video ID using regex fallback: {video_id}")
            return video_id
        
        logger.warning(f"Could not extract video ID from URL: {youtube_url}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting video ID from {youtube_url}: {str(e)}", exc_info=True)
        return None

@tracer.wrap(resource="get_youtube_video_details_pytube")
def get_youtube_video_details_pytube(youtube_url: str):
    """Fetches title and published date for a YouTube video using pytube (no API key needed)"""
    try:
        logger.info(f"Starting pytube extraction for URL: {youtube_url}")
        
        # Create a YouTube object
        yt = YouTube(youtube_url)
        
        # Log details immediately after extraction
        title = yt.title if hasattr(yt, 'title') and yt.title else f"YouTube Video: {youtube_url.split('v=')[-1].split('&')[0]}"
        publish_date = yt.publish_date.isoformat() if hasattr(yt, 'publish_date') and yt.publish_date else datetime.now().isoformat()
        channel = yt.author if hasattr(yt, 'author') and yt.author else "Unknown Channel"
        views = yt.views if hasattr(yt, 'views') and yt.views else 0
        
        logger.info(f"Pytube extraction results - Title: {title}, Channel: {channel}, Views: {views}")
        
        # Extract the relevant information
        result = {
            "title": title,
            "published_date": publish_date,
            "channel": channel,
            "description": yt.description if hasattr(yt, 'description') and yt.description else "",
            "views": views
        }
        
        # Check if we actually got a proper title (not just "YouTube Video" or empty)
        if not title or title.startswith("YouTube Video:"):
            logger.info("No valid title from pytube, trying fallback method...")
            return get_youtube_metadata_fallback(youtube_url)
            
        logger.info(f"Returning video details: {result}")
        return result
    except Exception as e:
        error_msg = f"Error extracting YouTube metadata: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Try fallback method
        logger.info("Trying fallback metadata extraction method...")
        try:
            return get_youtube_metadata_fallback(youtube_url)
        except Exception as fallback_error:
            logger.error(f"Fallback metadata extraction failed: {str(fallback_error)}", exc_info=True)
            # Final fallback to the video ID in case of any errors
            video_id = youtube_url.split('v=')[-1].split('&')[0] if 'v=' in youtube_url else youtube_url.split('/')[-1].split('?')[0]
            return {
                "title": f"YouTube Video: {video_id}",
                "published_date": datetime.now().isoformat(),
                "channel": "Unknown Channel",
                "description": "",
                "views": 0,
                "error": error_msg
            }

def get_youtube_metadata_fallback(youtube_url: str):
    """Fallback method to get YouTube metadata using httpx and HTML parsing"""
    logger.info(f"Using httpx fallback method for URL: {youtube_url}")
    
    # Extract video ID from URL
    video_id = None
    parsed_url = urlparse(youtube_url)
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        query_params = parse_qs(parsed_url.query)
        if 'v' in query_params:
            video_id = query_params['v'][0]
    elif parsed_url.netloc in ('youtu.be',):
        video_id = parsed_url.path[1:]
        
    if not video_id:
        logger.error(f"Could not extract video ID from URL: {youtube_url}")
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")
    
    # Fetch the video page
    with httpx.Client(timeout=10) as client:
        response = client.get(f"https://www.youtube.com/watch?v={video_id}")
        if response.status_code != 200:
            logger.error(f"Failed to fetch YouTube page: {response.status_code}")
            raise ValueError(f"Failed to fetch YouTube page: {response.status_code}")
            
        html_content = response.text
        
    # Extract metadata from HTML using regex
    title_match = re.search(r'<meta property="og:title" content="([^"]+)"', html_content)
    title = title_match.group(1) if title_match else f"YouTube Video: {video_id}"
    
    channel_match = re.search(r'<link itemprop="name" content="([^"]+)"', html_content)
    channel = channel_match.group(1) if channel_match else "Unknown Channel"
    
    # Try to extract view count - this is more complex as it could be in different formats
    views = 0
    
    # Method 1: Look for "viewCount" in JSON data
    view_count_match = re.search(r'"viewCount":"(\d+)"', html_content)
    if view_count_match:
        try:
            views = int(view_count_match.group(1))
            logger.info(f"Extracted view count from JSON data: {views}")
        except (ValueError, IndexError):
            pass
            
    # Method 2: Look for "view count" text
    if views == 0:
        view_patterns = [
            r'<meta itemprop="interactionCount" content="(\d+)"',
            r'"viewCount\\?":\\?"(\d+)\\?"',
            r'"viewCount":(\d+)',
        ]
        
        for pattern in view_patterns:
            matches = re.search(pattern, html_content)
            if matches:
                try:
                    views = int(matches.group(1))
                    logger.info(f"Extracted view count using pattern {pattern}: {views}")
                    break
                except (ValueError, IndexError):
                    continue
    
    # Format data
    result = {
        "title": title,
        "published_date": datetime.now().isoformat(),  # We don't have accurate date from this method
        "channel": channel,
        "description": "",  # We don't have description from this method
        "views": views
    }
    
    logger.info(f"Fallback extraction results - Title: {title}, Channel: {channel}, Views: {views}")
    return result

@tracer.wrap(resource="get_youtube_transcript")
def get_youtube_transcript(video_id: str):
    """Retrieves the transcript for a YouTube video ID with enhanced error handling."""
    try:
        logger.info(f"Attempting to retrieve transcript for video ID: {video_id}")
        
        # Get list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Log available transcripts for debugging
        available_transcripts = []
        for transcript in transcript_list:
            available_transcripts.append({
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            })
        logger.info(f"Available transcripts for {video_id}: {available_transcripts}")
        
        # Try multiple strategies to get transcript
        transcript = None
        language_used = None
        
        # Strategy 1: Try exact English variants
        for lang_code in ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']:
            try:
                transcript = transcript_list.find_transcript([lang_code])
                language_used = lang_code
                logger.info(f"Found transcript using language code: {lang_code}")
                break
            except NoTranscriptFound:
                continue
        
        # Strategy 2: Try any English transcript (including auto-generated)
        if not transcript:
            try:
                # Look for any transcript that starts with 'en'
                for available_transcript in transcript_list:
                    if available_transcript.language_code.startswith('en'):
                        transcript = available_transcript
                        language_used = available_transcript.language_code
                        logger.info(f"Found English transcript with code: {language_used}")
                        break
            except Exception as e:
                logger.warning(f"Strategy 2 failed: {str(e)}")
        
        # Strategy 3: Try any available transcript and translate if possible
        if not transcript:
            try:
                # Get first available transcript
                available_list = list(transcript_list)
                if available_list:
                    first_transcript = available_list[0]
                    if first_transcript.is_translatable:
                        # Try to translate to English
                        transcript = first_transcript.translate('en')
                        language_used = f"{first_transcript.language_code} (translated to en)"
                        logger.info(f"Using translated transcript from {first_transcript.language_code}")
                    else:
                        # Use original non-English transcript as fallback
                        transcript = first_transcript
                        language_used = first_transcript.language_code
                        logger.info(f"Using non-English transcript: {language_used}")
            except Exception as e:
                logger.warning(f"Strategy 3 failed: {str(e)}")
        
        if not transcript:
            raise NoTranscriptFound("No suitable transcript found after trying all strategies")
        
        # Fetch transcript data
        logger.info(f"Fetching transcript data for {video_id} using language: {language_used}")
        transcript_parts = transcript.fetch()
        
        if not transcript_parts:
            raise Exception("Transcript fetch returned empty data")
        
        # Build full transcript text
        full_transcript = " ".join(part['text'] for part in transcript_parts if 'text' in part)
        
        if not full_transcript.strip():
            raise Exception("Transcript text is empty after processing")
        
        logger.info(f"Successfully retrieved transcript for {video_id} ({len(full_transcript)} characters)")
        
        return {
            "transcript": full_transcript,
            "language": language_used
        }
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        error_msg = f"Transcript not available for video {video_id}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error retrieving transcript for {video_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Add more specific error context
        if "no element found" in str(e).lower():
            error_msg += " (This may be due to region restrictions, video privacy settings, or temporary YouTube API issues)"
        elif "xml" in str(e).lower():
            error_msg += " (XML parsing error - possibly due to malformed response from YouTube)"
        
        return {"error": error_msg}

@tracer.wrap(resource="generate_video_summary_async")
async def generate_video_summary_async(transcript: str, instructions: str = None):
    """Generate a summary of the video transcript using OpenAI's async client"""
    try:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Add tracing information
        span = tracer.current_span()
        if span:
            span.set_tag("transcript_length", len(transcript))
            span.set_tag("has_instructions", instructions is not None)
        
        # Prepare prompt based on instructions
        if instructions:
            prompt = f"""
            Here is a transcript from a YouTube video:
            
            {transcript}
            
            {instructions}
            """
        else:
            prompt = f"""
            Here is a transcript from a YouTube video:
            
            {transcript}
            
            Please provide a concise summary of the main points discussed in this video.
            Include key insights, topics covered, and any important conclusions.
            """
        
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts accurately and concisely."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get the response content
        summary = response.choices[0].message.content
        
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error during AI summarization: {str(e)}", exc_info=True)
        return {"error": f"Error during AI summarization: {str(e)}"}

# Keep the synchronous version for backward compatibility
@tracer.wrap(resource="generate_video_summary")
def generate_video_summary(transcript: str, instructions: str = None):
    """Generate a summary of the video transcript using OpenAI"""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Add tracing information
        span = tracer.current_span()
        if span:
            span.set_tag("transcript_length", len(transcript))
            span.set_tag("has_instructions", instructions is not None)
        
        # Prepare prompt based on instructions
        if instructions:
            prompt = f"""
            Here is a transcript from a YouTube video:
            
            {transcript}
            
            {instructions}
            """
        else:
            prompt = f"""
            Here is a transcript from a YouTube video:
            
            {transcript}
            
            Please provide a concise summary of the main points discussed in this video.
            Include key insights, topics covered, and any important conclusions.
            """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts accurately and concisely."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get the response content
        summary = response.choices[0].message.content
        
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error during AI summarization: {str(e)}")
        return {"error": f"Error during AI summarization: {str(e)}"} 