"""
YouTube Batch Processing Service

Provides multiple strategies for processing multiple YouTube URLs while
managing OpenAI context window limitations.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import tiktoken
from datetime import datetime

from .youtube_service import process_youtube_video, get_video_id
from .notion_service import add_video_summary_to_notion, NotionVideoPayload
from openai import AsyncOpenAI
from ddtrace import tracer

logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """Different strategies for processing multiple YouTube URLs."""
    SEQUENTIAL = "sequential"           # Process one by one, return individual results
    PARALLEL_INDIVIDUAL = "parallel_individual"  # Parallel processing, individual summaries
    BATCH_COMBINED = "batch_combined"   # Attempt combined analysis with chunking
    HYBRID = "hybrid"                   # Individual + meta-summary

@dataclass
class VideoProcessingResult:
    """Result from processing a single video."""
    url: str
    video_id: str
    title: str
    success: bool
    summary: Optional[str] = None
    transcript: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    notion_page_id: Optional[str] = None

@dataclass
class BatchProcessingResult:
    """Result from processing multiple videos."""
    strategy_used: ProcessingStrategy
    total_videos: int
    successful_videos: int
    failed_videos: int
    results: List[VideoProcessingResult]
    meta_summary: Optional[str] = None  # Combined analysis if strategy supports it
    total_processing_time: float = 0.0
    errors: List[str] = None

class YouTubeBatchProcessor:
    """Handles batch processing of YouTube videos with various strategies."""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Context window management
        self.MAX_TOKENS = 120000  # Conservative limit for GPT-4.1-mini
        self.SUMMARY_MAX_TOKENS = 8000  # Reserve for response
        self.EFFECTIVE_MAX_TOKENS = self.MAX_TOKENS - self.SUMMARY_MAX_TOKENS
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    @tracer.wrap(resource="process_urls_batch")
    async def process_urls_batch(
        self,
        urls: List[str],
        strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL_INDIVIDUAL,
        instructions: Optional[str] = None,
        save_to_notion: bool = False,
        max_parallel: int = 3
    ) -> BatchProcessingResult:
        """Main entry point for batch processing YouTube URLs."""
        
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Starting batch processing of {len(urls)} URLs using {strategy.value} strategy")
        
        # Validate URLs first
        valid_urls = []
        invalid_results = []
        
        for url in urls:
            video_id = get_video_id(url)
            if video_id:
                valid_urls.append(url)
            else:
                invalid_results.append(VideoProcessingResult(
                    url=url,
                    video_id="",
                    title="Invalid URL",
                    success=False,
                    error="Invalid YouTube URL format"
                ))
        
        if not valid_urls:
            return BatchProcessingResult(
                strategy_used=strategy,
                total_videos=len(urls),
                successful_videos=0,
                failed_videos=len(urls),
                results=invalid_results,
                total_processing_time=asyncio.get_event_loop().time() - start_time,
                errors=["No valid YouTube URLs provided"]
            )
        
        # Route to appropriate strategy
        if strategy == ProcessingStrategy.SEQUENTIAL:
            results = await self._process_sequential(valid_urls, instructions, save_to_notion)
        elif strategy == ProcessingStrategy.PARALLEL_INDIVIDUAL:
            results = await self._process_parallel_individual(valid_urls, instructions, save_to_notion, max_parallel)
        elif strategy == ProcessingStrategy.BATCH_COMBINED:
            results = await self._process_batch_combined(valid_urls, instructions, save_to_notion, max_parallel)
        elif strategy == ProcessingStrategy.HYBRID:
            results = await self._process_hybrid(valid_urls, instructions, save_to_notion, max_parallel)
        else:
            raise ValueError(f"Unknown processing strategy: {strategy}")
        
        # Add invalid results
        results.results.extend(invalid_results)
        results.total_videos = len(urls)
        results.failed_videos = len(invalid_results) + results.failed_videos
        results.total_processing_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Batch processing completed in {results.total_processing_time:.2f}s: "
                   f"{results.successful_videos}/{results.total_videos} successful")
        
        return results
    
    async def _process_sequential(
        self,
        urls: List[str],
        instructions: Optional[str],
        save_to_notion: bool
    ) -> BatchProcessingResult:
        """Process URLs sequentially (safest, but slowest)."""
        results = []
        
        for i, url in enumerate(urls):
            logger.info(f"Processing video {i+1}/{len(urls)}: {url}")
            
            start_time = asyncio.get_event_loop().time()
            try:
                result_data = await process_youtube_video(url, instructions, save_to_notion)
                
                processing_time = asyncio.get_event_loop().time() - start_time
                
                if "error" in result_data and result_data["error"]:
                    results.append(VideoProcessingResult(
                        url=url,
                        video_id=result_data.get("video_id", ""),
                        title=result_data.get("title", "Unknown"),
                        success=False,
                        error=result_data["error"],
                        processing_time=processing_time
                    ))
                else:
                    results.append(VideoProcessingResult(
                        url=url,
                        video_id=result_data["video_id"],
                        title=result_data.get("title", "Unknown"),
                        success=True,
                        summary=result_data.get("summary"),
                        transcript=result_data.get("transcript"),
                        processing_time=processing_time,
                        notion_page_id=result_data.get("notion_page_id")
                    ))
                
            except Exception as e:
                processing_time = asyncio.get_event_loop().time() - start_time
                logger.error(f"Error processing {url}: {str(e)}")
                results.append(VideoProcessingResult(
                    url=url,
                    video_id=get_video_id(url) or "",
                    title="Error",
                    success=False,
                    error=str(e),
                    processing_time=processing_time
                ))
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchProcessingResult(
            strategy_used=ProcessingStrategy.SEQUENTIAL,
            total_videos=len(urls),
            successful_videos=successful,
            failed_videos=failed,
            results=results
        )
    
    async def _process_parallel_individual(
        self,
        urls: List[str],
        instructions: Optional[str],
        save_to_notion: bool,
        max_parallel: int
    ) -> BatchProcessingResult:
        """Process URLs in parallel with individual summaries."""
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_single_url(url: str) -> VideoProcessingResult:
            async with semaphore:
                start_time = asyncio.get_event_loop().time()
                try:
                    result_data = await process_youtube_video(url, instructions, save_to_notion)
                    processing_time = asyncio.get_event_loop().time() - start_time
                    
                    if "error" in result_data and result_data["error"]:
                        return VideoProcessingResult(
                            url=url,
                            video_id=result_data.get("video_id", ""),
                            title=result_data.get("title", "Unknown"),
                            success=False,
                            error=result_data["error"],
                            processing_time=processing_time
                        )
                    else:
                        return VideoProcessingResult(
                            url=url,
                            video_id=result_data["video_id"],
                            title=result_data.get("title", "Unknown"),
                            success=True,
                            summary=result_data.get("summary"),
                            transcript=result_data.get("transcript"),
                            processing_time=processing_time,
                            notion_page_id=result_data.get("notion_page_id")
                        )
                        
                except Exception as e:
                    processing_time = asyncio.get_event_loop().time() - start_time
                    logger.error(f"Error processing {url}: {str(e)}")
                    return VideoProcessingResult(
                        url=url,
                        video_id=get_video_id(url) or "",
                        title="Error",
                        success=False,
                        error=str(e),
                        processing_time=processing_time
                    )
        
        # Process all URLs in parallel
        logger.info(f"Processing {len(urls)} URLs with max {max_parallel} parallel connections")
        results = await asyncio.gather(*[process_single_url(url) for url in urls])
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchProcessingResult(
            strategy_used=ProcessingStrategy.PARALLEL_INDIVIDUAL,
            total_videos=len(urls),
            successful_videos=successful,
            failed_videos=failed,
            results=results
        )
    
    async def _process_batch_combined(
        self,
        urls: List[str],
        instructions: Optional[str],
        save_to_notion: bool,
        max_parallel: int
    ) -> BatchProcessingResult:
        """Process URLs and attempt combined analysis with intelligent chunking."""
        
        # First get individual processing results (but just metadata, not summaries)
        individual_results = await self._process_parallel_individual(
            urls, None, False, max_parallel  # No individual summaries or Notion saves
        )
        
        # Collect successful transcripts for combined analysis
        successful_results = [r for r in individual_results.results if r.success and r.transcript]
        
        if not successful_results:
            return individual_results
        
        # Try to create a combined summary
        combined_summary = await self._create_combined_summary(
            successful_results, instructions
        )
        
        # If combined analysis succeeded and we need Notion saves, do them now
        if save_to_notion and combined_summary:
            await self._save_combined_results_to_notion(
                successful_results, combined_summary
            )
        
        # Update results with combined summary
        for result in individual_results.results:
            if result.success:
                result.summary = combined_summary
        
        individual_results.strategy_used = ProcessingStrategy.BATCH_COMBINED
        individual_results.meta_summary = combined_summary
        
        return individual_results
    
    async def _process_hybrid(
        self,
        urls: List[str],
        instructions: Optional[str],
        save_to_notion: bool,
        max_parallel: int
    ) -> BatchProcessingResult:
        """Process URLs individually AND create a meta-summary."""
        
        # Get individual results first
        individual_results = await self._process_parallel_individual(
            urls, instructions, save_to_notion, max_parallel
        )
        
        # Create meta-summary from individual summaries
        successful_results = [r for r in individual_results.results if r.success and r.summary]
        
        if len(successful_results) >= 2:
            meta_summary = await self._create_meta_summary(successful_results, instructions)
            individual_results.meta_summary = meta_summary
        
        individual_results.strategy_used = ProcessingStrategy.HYBRID
        return individual_results
    
    async def _create_combined_summary(
        self,
        results: List[VideoProcessingResult],
        instructions: Optional[str]
    ) -> Optional[str]:
        """Create a combined summary from multiple transcripts."""
        
        # Calculate total tokens needed
        total_tokens = 0
        transcript_data = []
        
        for result in results:
            if result.transcript:
                tokens = self.count_tokens(result.transcript)
                transcript_data.append({
                    'title': result.title,
                    'url': result.url,
                    'transcript': result.transcript,
                    'tokens': tokens
                })
                total_tokens += tokens
        
        logger.info(f"Combined transcripts total: {total_tokens} tokens")
        
        if total_tokens <= self.EFFECTIVE_MAX_TOKENS:
            # All transcripts fit, do direct combined analysis
            return await self._analyze_combined_transcripts(transcript_data, instructions)
        else:
            # Need chunking strategy
            return await self._analyze_chunked_transcripts(transcript_data, instructions)
    
    async def _analyze_combined_transcripts(
        self,
        transcript_data: List[Dict],
        instructions: Optional[str]
    ) -> Optional[str]:
        """Analyze all transcripts together (when they fit in context window)."""
        
        try:
            # Prepare the combined prompt
            combined_text = ""
            for i, data in enumerate(transcript_data, 1):
                combined_text += f"\n\n=== VIDEO {i}: {data['title']} ===\n"
                combined_text += f"URL: {data['url']}\n\n"
                combined_text += data['transcript']
            
            if instructions:
                prompt = f"""
You are analyzing multiple YouTube videos together. Here are the transcripts:

{combined_text}

{instructions}

Please provide a comprehensive analysis that identifies:
1. Common themes across videos
2. Key insights from each video
3. How the videos complement each other
4. Overall synthesis of the content
"""
            else:
                prompt = f"""
You are analyzing multiple YouTube videos together. Here are the transcripts:

{combined_text}

Please provide a comprehensive analysis that:
1. Identifies common themes and patterns across all videos
2. Summarizes the key insights from each video
3. Shows how the videos complement or contrast with each other
4. Provides an overall synthesis of the combined content
5. Highlights the most important takeaways from this collection of videos
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes video content and creates insightful summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in combined analysis: {str(e)}")
            return None
    
    async def _analyze_chunked_transcripts(
        self,
        transcript_data: List[Dict],
        instructions: Optional[str]
    ) -> Optional[str]:
        """Analyze transcripts using chunking when they exceed context window."""
        
        logger.info("Using chunked analysis strategy")
        
        try:
            # Strategy: Create individual summaries, then meta-analyze
            individual_summaries = []
            
            for data in transcript_data:
                if data['tokens'] > self.EFFECTIVE_MAX_TOKENS:
                    # Even individual transcript is too long, skip for now
                    # TODO: Implement transcript chunking within single video
                    logger.warning(f"Transcript too long for analysis: {data['title']}")
                    continue
                
                # Get individual summary
                prompt = f"""
Video Title: {data['title']}
URL: {data['url']}

Transcript:
{data['transcript']}

Please provide a detailed summary of this video's main points, key insights, and important conclusions.
"""
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes video content accurately and concisely."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                individual_summaries.append({
                    'title': data['title'],
                    'url': data['url'],
                    'summary': response.choices[0].message.content
                })
            
            # Now create meta-summary from individual summaries
            if individual_summaries:
                return await self._create_meta_summary_from_data(individual_summaries, instructions)
            
        except Exception as e:
            logger.error(f"Error in chunked analysis: {str(e)}")
            return None
    
    async def _create_meta_summary(
        self,
        results: List[VideoProcessingResult],
        instructions: Optional[str]
    ) -> Optional[str]:
        """Create meta-summary from individual video summaries."""
        
        summary_data = [
            {
                'title': r.title,
                'url': r.url,
                'summary': r.summary
            }
            for r in results if r.summary
        ]
        
        return await self._create_meta_summary_from_data(summary_data, instructions)
    
    async def _create_meta_summary_from_data(
        self,
        summary_data: List[Dict],
        instructions: Optional[str]
    ) -> Optional[str]:
        """Create meta-summary from summary data."""
        
        try:
            combined_summaries = ""
            for i, data in enumerate(summary_data, 1):
                combined_summaries += f"\n\n=== VIDEO {i}: {data['title']} ===\n"
                combined_summaries += f"URL: {data['url']}\n\n"
                combined_summaries += data['summary']
            
            if instructions:
                prompt = f"""
Here are summaries from multiple YouTube videos:

{combined_summaries}

{instructions}

Based on these summaries, please provide a meta-analysis that:
1. Identifies overarching themes across all videos
2. Shows connections and relationships between the videos
3. Synthesizes the key insights into a coherent narrative
4. Highlights the most important combined takeaways
"""
            else:
                prompt = f"""
Here are summaries from multiple YouTube videos:

{combined_summaries}

Please create a meta-analysis that:
1. Identifies common themes and patterns across all video summaries
2. Shows how the videos connect or complement each other
3. Synthesizes the key insights into a coherent overall narrative
4. Provides the most important takeaways from this collection of videos
5. Notes any conflicting viewpoints or different perspectives presented
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates insightful meta-analyses from multiple video summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error creating meta-summary: {str(e)}")
            return None
    
    async def _save_combined_results_to_notion(
        self,
        results: List[VideoProcessingResult],
        combined_summary: str
    ) -> None:
        """Save combined analysis results to Notion."""
        
        # For now, save each video individually with the combined summary
        # TODO: Consider creating a "Batch Analysis" page that links to individual videos
        
        for result in results:
            if not result.success:
                continue
                
            try:
                payload_data = {
                    "url": result.url,
                    "title": result.title,
                    "published_date": datetime.now().isoformat(),
                    "summary": combined_summary,  # Use combined summary
                    "channel": "Unknown Channel",
                    "views": 0,
                }
                
                notion_payload = NotionVideoPayload(**payload_data)
                
                notion_result = await add_video_summary_to_notion(
                    notion_payload, 
                    views_as_number=True,
                    transcript=result.transcript
                )
                
                if "notion_page_id" in notion_result:
                    result.notion_page_id = notion_result["notion_page_id"]
                    
            except Exception as e:
                logger.error(f"Error saving {result.url} to Notion: {str(e)}")

# Utility functions for token estimation and management
def estimate_video_tokens(transcript_length: int) -> int:
    """Rough estimation of tokens for a transcript."""
    # Rough estimate: 1 token ≈ 4 characters for English text
    return transcript_length // 4

def can_fit_in_context(transcripts: List[str], max_tokens: int = 120000) -> bool:
    """Check if transcripts can fit in context window."""
    total_chars = sum(len(t) for t in transcripts)
    estimated_tokens = estimate_video_tokens(total_chars)
    return estimated_tokens < max_tokens
