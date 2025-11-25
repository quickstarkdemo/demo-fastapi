"""
OpenAI API functions
"""

import os
import logging
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from ddtrace import tracer

from openai import OpenAI
from .services.youtube_service import get_video_id, get_youtube_transcript, generate_video_summary, process_youtube_video
from .services.youtube_batch_service import YouTubeBatchProcessor, ProcessingStrategy, BatchProcessingResult
from .observability.sentry_logging import (
    log_youtube_processing_start, log_youtube_processing_complete,
    log_youtube_batch_processing
)

# Set up logging
logger = logging.getLogger(__name__)

# Load dotenv in the base root refers to application_top
APP_ROOT = os.path.join(os.path.dirname(__file__), '..')
dotenv_path = os.path.join(APP_ROOT, '.env')
load_dotenv(dotenv_path)

OPENAI = os.getenv('OPENAI_API_KEY')
if not OPENAI:
    logger.warning("OPENAI_API_KEY environment variable not set!")
else:
    logger.info("OpenAI API key found.")

client = OpenAI(
    api_key=OPENAI
)

# Create a new router for OpenAI Routes
router_openai = APIRouter(tags=["OpenAI"])

@router_openai.get("/openai-hello")
async def openai_hello():
    """Health check endpoint for OpenAI service.
    
    Simple endpoint to verify the OpenAI service is responding and
    available for processing requests.

    Returns:
        dict: Service status message confirming OpenAI endpoint is accessible.
    """
    return {"message": "You've reached the OpenAI endpoint"}

@router_openai.get("/openai-gen-image/{search}")
@tracer.wrap(service="openai-service", resource="generate_image")
async def openai_gen_image(search: str):
    """Generate an image using OpenAI's DALL-E 3 model.
    
    Creates a 1024x1024 image based on the provided text prompt using
    OpenAI's DALL-E 3 image generation model.

    Args:
        search (str): The text prompt describing the image to generate.

    Returns:
        str: URL of the generated image.
        
    Raises:
        HTTPException: If image generation fails or API is unavailable.
    """ 
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=search,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

class YouTubeRequest(BaseModel):
    """Request model for YouTube video processing.
    
    Attributes:
        url (str): YouTube video URL to process.
        instructions (Optional[str]): Custom instructions for AI summarization.
        save_to_notion (Optional[bool]): Whether to save results to Notion database.
    """
    url: str
    instructions: Optional[str] = None
    save_to_notion: Optional[bool] = False

class BatchYouTubeRequest(BaseModel):
    """Request model for batch YouTube video processing.
    
    Attributes:
        urls (List[str]): List of YouTube video URLs to process.
        strategy (str): Processing strategy - "sequential", "parallel_individual", "batch_combined", or "hybrid".
        instructions (Optional[str]): Custom instructions for AI summarization.
        save_to_notion (Optional[bool]): Whether to save results to Notion database.
        max_parallel (int): Maximum number of parallel processing tasks (default: 3).
    """
    urls: List[str]
    strategy: str = "parallel_individual"
    instructions: Optional[str] = None
    save_to_notion: Optional[bool] = False
    max_parallel: int = 3

@router_openai.post("/summarize-youtube")
@tracer.wrap(service="openai-service", resource="summarize_youtube")
async def summarize_youtube_video(request: YouTubeRequest):
    """Process YouTube video to generate AI-powered summary.
    
    Downloads the video transcript, generates an intelligent summary using OpenAI,
    and optionally saves the results to Notion database for future reference.

    Args:
        request (YouTubeRequest): Contains YouTube URL, custom instructions, 
                                 and Notion save preference.

    Returns:
        dict: Contains video metadata, transcript, and AI-generated summary.
              Includes Notion page ID if saved to database.
        
    Raises:
        HTTPException: If video processing fails, transcript unavailable,
                      or AI summarization encounters errors.
    """
    try:
        # Process video and get result - now async
        result = await process_youtube_video(
            request.url, 
            request.instructions,
            save_to_notion=request.save_to_notion
        )
        
        # Check for errors
        if "error" in result and result["error"]:
            logger.error(f"YouTube processing error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Build response dict with required fields
        response_dict: Dict[str, Any] = {
            "video_id": result["video_id"],
            "transcript": result["transcript"],
            "language": result["language"],
            "summary": result["summary"]
        }
        
        # Add optional fields if they exist in the result
        for field in ["title", "published_date", "notion_page_id"]:
            if field in result:
                response_dict[field] = result[field]
        
        # Add notion error if it exists but the request was successful overall
        if "notion_error" in result:
            response_dict["notion_error"] = result["notion_error"]
        
        # Log successful YouTube processing to Sentry
        log_youtube_processing_complete(
            url=request.url,
            video_id=result["video_id"],
            title=result.get("title", "Unknown"),
            duration=0,  # We don't track duration in individual processing
            save_notion=request.save_to_notion
        )
            
        return response_dict
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full exception for debugging
        logger.error(f"Unexpected error in summarize_youtube_video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router_openai.post("/batch-summarize-youtube")
@tracer.wrap(service="openai-service", resource="batch_summarize_youtube")
async def batch_summarize_youtube_videos(request: BatchYouTubeRequest):
    """Process multiple YouTube videos with various strategies.
    
    Supports different processing strategies to handle context window limitations:
    - sequential: Process videos one by one (safest, slowest)
    - parallel_individual: Process in parallel with individual summaries (recommended)
    - batch_combined: Attempt combined analysis with chunking
    - hybrid: Individual summaries + meta-summary
    
    Args:
        request (BatchYouTubeRequest): Contains URLs, strategy, instructions, and preferences.
    
    Returns:
        dict: Contains processing results, individual video data, and optional meta-summary.
        
    Raises:
        HTTPException: If request validation fails or processing encounters errors.
    """
    try:
        # Validate strategy
        try:
            strategy = ProcessingStrategy(request.strategy)
        except ValueError:
            valid_strategies = [s.value for s in ProcessingStrategy]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy '{request.strategy}'. Valid options: {valid_strategies}"
            )
        
        # Validate URLs list
        if not request.urls:
            raise HTTPException(status_code=400, detail="URLs list cannot be empty")
        
        if len(request.urls) > 20:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Maximum 20 URLs allowed per batch request")
        
        # Validate max_parallel
        if request.max_parallel < 1 or request.max_parallel > 10:
            raise HTTPException(status_code=400, detail="max_parallel must be between 1 and 10")
        
        # Initialize batch processor
        batch_processor = YouTubeBatchProcessor()
        
        # Process the batch
        batch_result: BatchProcessingResult = await batch_processor.process_urls_batch(
            urls=request.urls,
            strategy=strategy,
            instructions=request.instructions,
            save_to_notion=request.save_to_notion,
            max_parallel=request.max_parallel
        )
        
        # Convert result to API response format
        response_data = {
            "strategy_used": batch_result.strategy_used.value,
            "total_videos": batch_result.total_videos,
            "successful_videos": batch_result.successful_videos,
            "failed_videos": batch_result.failed_videos,
            "processing_time": round(batch_result.total_processing_time, 2),
            "results": [
                {
                    "url": result.url,
                    "video_id": result.video_id,
                    "title": result.title,
                    "success": result.success,
                    "summary": result.summary,
                    "processing_time": round(result.processing_time, 2) if result.processing_time else None,
                    "notion_page_id": result.notion_page_id,
                    "error": result.error
                }
                for result in batch_result.results
            ]
        }
        
        # Add meta-summary if available
        if batch_result.meta_summary:
            response_data["meta_summary"] = batch_result.meta_summary
        
        # Add errors if any
        if batch_result.errors:
            response_data["errors"] = batch_result.errors
        
        # Log batch processing results to Sentry
        log_youtube_batch_processing(
            urls_count=batch_result.total_videos,
            strategy=batch_result.strategy_used.value,
            successful=batch_result.successful_videos,
            failed=batch_result.failed_videos,
            total_duration=batch_result.total_processing_time
        )
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full exception for debugging
        logger.error(f"Unexpected error in batch_summarize_youtube_videos: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}") 


@router_openai.post("/save-youtube-to-notion", tags=["Notion"])
async def save_youtube_to_notion(request: YouTubeRequest):
    """
    Save a YouTube video summary to Notion.
    Convenience wrapper that forces `save_to_notion=True` while reusing the core summarization logic.
    """
    modified_request = YouTubeRequest(
        url=request.url,
        instructions=request.instructions,
        save_to_notion=True
    )
    return await summarize_youtube_video(modified_request)
