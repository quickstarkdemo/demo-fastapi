import os
import logging
from notion_client import AsyncClient
from pydantic import BaseModel, HttpUrl
from typing import Optional
from ddtrace import tracer
from dotenv import load_dotenv
import json
import re

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from the project root .env file
APP_ROOT = os.path.join(os.path.dirname(__file__), '..', '..') 
dotenv_path = os.path.join(APP_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    logger.warning(".env file not found at expected location: %s", dotenv_path)

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

if not NOTION_API_KEY:
    logger.warning("NOTION_API_KEY environment variable not set!")
if not NOTION_DATABASE_ID:
    logger.warning("NOTION_DATABASE_ID environment variable not set!")

class NotionVideoPayload(BaseModel):
    """Data structure for creating a Notion page for a YouTube video."""
    url: HttpUrl
    title: str
    published_date: str # Assuming Notion Date property accepts ISO 8601 string
    summary: str
    channel: Optional[str] = None
    views: Optional[int] = None
    # Add other fields if your Notion database has more properties (e.g., video_id)
    # video_id: Optional[str] = None

@tracer.wrap(resource="add_video_summary")
async def add_video_summary_to_notion(payload: NotionVideoPayload, views_as_number: bool = True, transcript: Optional[str] = None) -> dict:
    """Adds a YouTube video summary entry to the configured Notion database.
    
    Args:
        payload: The video data to add to Notion
        views_as_number: Set to True if Views column in Notion is Number type, False if Text type
        transcript: Optional full transcript text to add to the page content
    """
    # Log the incoming payload for debugging
    logger.info(f"Received Notion payload - Title: '{payload.title}', URL: {payload.url}, Channel: '{payload.channel}', Views: {payload.views}")
    
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        error_msg = "Notion API Key or Database ID is not configured."
        logger.error(error_msg)
        return {"error": error_msg}

    notion = AsyncClient(auth=NOTION_API_KEY)
    
    # Map payload fields to Notion database property names and types.
    # IMPORTANT: Adjust these property names ('Title', 'URL', 'Published Date', 'Summary') 
    # to match the EXACT names of the columns in YOUR Notion database.
    properties_data = {
        "Title": {"title": [{"text": {"content": payload.title}}]},
        "URL": {"url": str(payload.url)}, # Ensure URL is passed as string
        "Published Date": {"date": {"start": payload.published_date}}, # Assumes YYYY-MM-DD or ISO 8601
        "Summary": {"rich_text": [{"text": {"content": payload.summary if len(payload.summary) <= 2000 else payload.summary[:1997] + "..."}}]},
    }

    # Add channel and views if they are provided
    if payload.channel:
        properties_data["Channel"] = {"rich_text": [{"text": {"content": payload.channel}}]}
    
    if payload.views is not None:
        # Use number format if views_as_number is True, otherwise use text format
        if views_as_number:
            try:
                # Ensure views is a number
                view_count = int(payload.views) if isinstance(payload.views, (str, int)) else 0
                properties_data["Views"] = {"number": view_count}
                logger.info(f"Adding views as number: {view_count}")
            except (ValueError, TypeError):
                # Fallback to text if conversion fails
                properties_data["Views"] = {"rich_text": [{"text": {"content": str(payload.views)}}]}
                logger.info(f"Fallback: Adding views as text: {payload.views}")
        else:
            properties_data["Views"] = {"rich_text": [{"text": {"content": str(payload.views)}}]}
            logger.info(f"Adding views as text: {payload.views}")

    # Create page content blocks if transcript is provided
    children = []
    transcript_blocks = []
    
    if transcript:
        # Add a heading for the transcript section
        transcript_blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "Full Transcript"}}]
            }
        })
        
        # Format the transcript for better readability
        # Split transcript into paragraphs and add each as a separate block
        transcript_paragraphs = format_transcript(transcript)
        for paragraph in transcript_paragraphs:
            transcript_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": paragraph}}]
                }
            })
    
    # Notion API has limits on the number of blocks per request
    # Start with just the essential blocks and we'll append transcript later if needed
    children = transcript_blocks[:100] if transcript_blocks else []

    # Log the final properties data being sent to Notion
    logger.info(f"Sending Notion properties: {json.dumps(properties_data, default=str)}")
    logger.info(f"Adding {len(children)} blocks to the initial Notion page")

    try:
        span = tracer.current_span()
        if span:
            span.set_tag("notion.database.id", NOTION_DATABASE_ID)
            span.set_tag("video.title", payload.title)
            span.set_tag("video.url", str(payload.url))

        logger.info(f"Making Notion API request to database ID: {NOTION_DATABASE_ID}")
        
        # Create the page with properties and initial children (content blocks)
        create_args = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": properties_data
        }
        
        # Only add children if we have content blocks
        if children:
            create_args["children"] = children
            
        response = await notion.pages.create(**create_args)
        page_id = response.get("id")
        logger.info(f"Successfully added video '{payload.title}' to Notion. Page ID: {page_id}")
        
        # If we have more transcript blocks to add, append them in batches
        if transcript_blocks and len(transcript_blocks) > 100:
            remaining_blocks = transcript_blocks[100:]
            block_batches = [remaining_blocks[i:i+100] for i in range(0, len(remaining_blocks), 100)]
            
            logger.info(f"Adding {len(remaining_blocks)} additional blocks in {len(block_batches)} batches")
            
            # Add each batch of blocks to the page
            for i, batch in enumerate(block_batches):
                try:
                    logger.info(f"Adding batch {i+1}/{len(block_batches)} with {len(batch)} blocks")
                    await notion.blocks.children.append(
                        block_id=page_id,
                        children=batch
                    )
                except Exception as batch_error:
                    logger.error(f"Error adding batch {i+1}: {str(batch_error)}")
                    # Continue with other batches even if one fails
        
        return {"notion_page_id": page_id, "error": None}
    
    except Exception as e:
        # Catching a broad exception, but Notion client might raise specific ones.
        # Consider refining error handling based on notion-client documentation if needed.
        error_msg = f"Failed to add video '{payload.title}' to Notion: {str(e)}"
        logger.error(error_msg, exc_info=True) # Log traceback for debugging
        
        span = tracer.current_span()
        if span:
            span.set_traceback()
            
        return {"error": error_msg}

def format_transcript(transcript: str) -> list[str]:
    """Format the transcript for better readability in Notion.
    
    Args:
        transcript: Raw transcript text
        
    Returns:
        List of paragraphs for Notion blocks, each under 2000 characters (Notion limit)
    """
    # Define the maximum allowed size for a Notion text block
    MAX_BLOCK_SIZE = 1900  # Using 1900 to leave a small buffer
    
    # Clean and normalize the transcript text
    # Replace multiple spaces with single space
    cleaned_text = re.sub(r'\s+', ' ', transcript).strip()
    
    # Detect likely paragraph and speaker breaks
    # Common patterns in transcripts: [Music], speaker names followed by colons, etc.
    speaker_indicators = [
        r'\[.*?\]',              # [Music], [Laughter], etc.
        r'[A-Z][a-z]+:',         # Speaker: pattern
        r'Dr\.\s+[A-Z][a-z]+',   # Dr. Name pattern
        r'Q:',                   # Q: (question) pattern
        r'A:',                   # A: (answer) pattern
    ]
    
    # Create a regex pattern that captures potential speaker changes or paragraph breaks
    speaker_pattern = '|'.join(speaker_indicators)
    
    # First, try to split by potential speaker changes
    paragraphs = []
    
    # Check if the transcript contains any speaker indicators
    if re.search(speaker_pattern, cleaned_text):
        # Split by potential speaker changes
        parts = re.split(f'({speaker_pattern})', cleaned_text)
        
        current_part = ""
        i = 0
        
        while i < len(parts):
            part = parts[i].strip()
            
            # If this is a speaker indicator
            if re.match(speaker_pattern, part):
                # If we have text before this speaker, add it as a paragraph
                if current_part:
                    paragraphs.append(current_part)
                
                # Start a new paragraph with this speaker
                if i + 1 < len(parts):
                    current_part = part + " " + parts[i+1].strip()
                    i += 2
                else:
                    current_part = part
                    i += 1
            else:
                # Regular content
                if current_part:
                    # If adding would exceed block size, create a new paragraph
                    if len(current_part) + len(part) + 1 > MAX_BLOCK_SIZE:
                        paragraphs.append(current_part)
                        current_part = part
                    else:
                        current_part += " " + part
                else:
                    current_part = part
                i += 1
        
        # Add the last paragraph
        if current_part:
            paragraphs.append(current_part)
    else:
        # No speaker indicators found, try to break up by sentences and create logical paragraphs
        
        # First, ensure proper spacing after punctuation to help with sentence detection
        cleaned_text = re.sub(r'(\.|!|\?)([A-Z])', r'\1 \2', cleaned_text)
        
        # Simple heuristic: Break the text into roughly equal-sized paragraphs
        # with a target of ~3-4 sentences per paragraph, but respect MAX_BLOCK_SIZE
        
        # Split into sentences (basic approach)
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        sentences = re.split(sentence_pattern, cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_para = ""
        sentence_count = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the block size
            if len(current_para) + len(sentence) + 1 > MAX_BLOCK_SIZE:
                if current_para:
                    paragraphs.append(current_para)
                
                # If the sentence itself is too long, we need to split it at word boundaries
                if len(sentence) > MAX_BLOCK_SIZE:
                    remaining = sentence
                    while remaining:
                        if len(remaining) <= MAX_BLOCK_SIZE:
                            # Last piece fits
                            current_para = remaining
                            remaining = ""
                        else:
                            # Find a good word boundary to break at
                            break_point = MAX_BLOCK_SIZE
                            while break_point > 0 and remaining[break_point] != ' ':
                                break_point -= 1
                            
                            # If no space found, force a break at max size
                            if break_point == 0:
                                break_point = MAX_BLOCK_SIZE
                            
                            paragraphs.append(remaining[:break_point].strip())
                            remaining = remaining[break_point:].strip()
                else:
                    current_para = sentence
                
                sentence_count = 1
            else:
                # Add to current paragraph
                if current_para:
                    current_para += " " + sentence
                else:
                    current_para = sentence
                
                sentence_count += 1
                
                # Start a new paragraph after 3-4 sentences, unless the current paragraph is very short
                if sentence_count >= 4 and len(current_para) > 300:
                    paragraphs.append(current_para)
                    current_para = ""
                    sentence_count = 0
        
        # Add the last paragraph
        if current_para:
            paragraphs.append(current_para)
    
    # Post-processing to ensure all paragraphs are properly sized
    result = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(para) <= MAX_BLOCK_SIZE:
            result.append(para)
        else:
            # We still need to split this paragraph
            words = para.split(' ')
            current_chunk = ""
            
            for word in words:
                if len(current_chunk) + len(word) + 1 > MAX_BLOCK_SIZE:
                    result.append(current_chunk.strip())
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
            
            if current_chunk:
                result.append(current_chunk.strip())
    
    # Final cleanup: ensure no paragraphs are empty and no paragraph exceeds the limit
    final_result = []
    for para in result:
        if para and len(para) <= MAX_BLOCK_SIZE:
            final_result.append(para)
        elif para:
            # As a last resort, chunk by characters
            for i in range(0, len(para), MAX_BLOCK_SIZE):
                final_result.append(para[i:min(i+MAX_BLOCK_SIZE, len(para))])
    
    return final_result 