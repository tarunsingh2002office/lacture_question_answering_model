import os
import io
import json
import random
import asyncio
import tempfile
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from typing import Optional
from openai import AsyncOpenAI
from pydub import AudioSegment
from typing import Union, List
from moviepy import VideoFileClip
from reportlab.pdfgen import canvas
from langsmith.run_helpers import trace
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from helper_function.textfile import text1, text2

async def video_to_audio(video_path: Path, output_path: Path) -> Path:
    """Convert video to audio regardless of length"""
    # Validate input path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # Process video in chunks
        with VideoFileClip(video_path) as video:
            audio = video.audio
            audio.write_audiofile(
                output_path,
                codec='mp3',
                bitrate='192k',
                logger=None  # Disable progress bar for cleaner output
            )
        return output_path
    except Exception as e:
        # Clean up partial files on error
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Conversion failed: {str(e)}")
    
async def save_text_to_pdf(
    font_path: Path,
    output_path: Path,
    text_file_path: Path,
    page_width: int = 580,
    page_margin: int = 20,
) -> None:
    try:
        # Read text from the provided text file without blocking the event loop
        text = await asyncio.to_thread(text_file_path.read_text, "utf-8")
        def _generate_pdf():
            # Register font and setup canvas
            pdfmetrics.registerFont(TTFont("Poppins", str(font_path)))
            c = canvas.Canvas(str(output_path), pagesize=letter)
            c.setFont("Poppins", 12)
            # Layout parameters
            x, y = page_margin, 750
            line_height = 20
            words = text.split()
            current_line = []
            # Text layout algorithm
            for word in words:
                test_line = ' '.join(current_line + [word])
                if c.stringWidth(test_line, "Poppins", 12) > (page_width - 2 * page_margin):
                    c.drawString(x, y, ' '.join(current_line))
                    y -= line_height
                    current_line = [word]
                    if y < 50:  # New page check
                        c.showPage()
                        c.setFont("Poppins", 12)
                        y = 750
                else:
                    current_line.append(word)
            # Render remaining text
            if current_line:
                c.drawString(x, y, ' '.join(current_line))
            c.save()
        # Execute blocking PDF generation in thread
        await asyncio.to_thread(_generate_pdf)
    except Exception as err:
        raise Exception(f"something went wrong {err}")

async def split_pdf(input_pdf_path: Path, output_folder: Path) -> int:
    try:
        # Open the PDF file
        reader = await _read_pdf(input_pdf_path)
        total_pages = len(reader.pages)
        
        # Iterate through all pages
        for i, page in enumerate(reader.pages, start=1):
            writer = PdfWriter()
            writer.add_page(page)

            out_path = output_folder / f"page_{i}.pdf"
            # Write the single-page PDF in a thread
            await write_file(out_path, writer)
        return total_pages
    except Exception as err:
        raise Exception(f"something went wrong {err}")
    
async def _read_pdf(path: Path) -> PdfReader:
    try:
        return await asyncio.to_thread(PdfReader, path)
    except Exception as err:
        raise Exception(f"something went wrong {err}")
    
async def write_file(
    path: Path,
    content: Union[bytes, PdfWriter, str, dict],
    encoding: str = "utf-8"
) -> None:
    """
    Unified async file writer that handles:
    - Bytes content (binary mode)
    - PDF Writers (binary mode)
    - Strings (text mode)
    - JSON dictionaries (text mode with JSON formatting)
    """
    try:
        # Binary mode handling
        if isinstance(content, (bytes, PdfWriter)):
            def _sync_write_binary():
                with path.open("wb") as f:
                    if isinstance(content, bytes):
                        f.write(content)
                    else:  # PdfWriter
                        content.write(f)
            await asyncio.to_thread(_sync_write_binary)
        
        # Text mode handling
        else:
            def _sync_write_text():
                with path.open("w", encoding=encoding) as f:
                    if isinstance(content, str):
                        f.write(content)
                    else:  # JSON data
                        json.dump(content, f, indent=4)
            await asyncio.to_thread(_sync_write_text)
    
    except Exception as err:
        raise Exception(f"File write error: {err}")
    
async def audio_to_text(
    path: Path,
    text_file_path: Path,
    hinglish: bool = False,
) -> Path:
    """
    Robust async audio transcription with automatic chunking.
    
    Handles both Whisper-1 and GPT-4o-transcribe limits:
    - Whisper-1: 25MB file limit
    - GPT-4o-transcribe: 25MB file limit + 1500 seconds (25 min) duration limit
    
    Args:
        path: Path to audio file
        text_file_path: Path where transcript will be saved
        hinglish: If True, uses GPT-4o-transcribe with Hinglish prompt
    
    Returns:
        Path to the saved transcript file
    """
    client = AsyncOpenAI()
    
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    # Load audio metadata to check if chunking is needed
    audio = await asyncio.to_thread(AudioSegment.from_file, path)
    duration_seconds = len(audio) / 1000
    
    
    # Ensure output directory exists
    text_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if chunking is needed
    needs_chunking = False
    # GPT-4o-transcribe: 1500s duration limit + 2000 token output limit
    # For Hinglish: use shorter chunks (10 min = 600s) to avoid output token limit
    max_duration = 600 if hinglish else 1400  # Whisper can handle longer chunks
    
    if hinglish:
        # GPT-4o has strict 2000 token output limit - chunk more aggressively
        if duration_seconds > max_duration or file_size_mb > 24:
            needs_chunking = True
    else:
        # Whisper only has 25MB file limit
        if file_size_mb > 24:
            needs_chunking = True
    
    # Calculate estimated cost
    cost_per_minute = 0.006  # Whisper/GPT-4o cost
    estimated_cost = (duration_seconds / 60) * cost_per_minute
    
    # Setup trace
    name = "whisper-translate_audio_chunks" if not hinglish else "gpt-4o-transcribe-translate_audio_chunks_hinglish"
    tags = ["translation", "whisper" if not hinglish else "gpt-4o-transcribe-hinglish"]
    
    async with trace(
        name=name,
        run_type="tool",
        inputs={"file_path": str(path), "duration_seconds": duration_seconds, "file_size_mb": file_size_mb},
        tags=tags,
    ) as run:
        try:
            if not needs_chunking:
                # Process entire file at once
                transcript = await _transcribe_file(client, path, hinglish)
                # Write synchronously to ensure it completes
                _write_transcript_sync(text_file_path, transcript, append=False)
                full_text = transcript
            else:
                # Process in chunks
                full_text = await _transcribe_in_chunks(client, audio, text_file_path, hinglish, max_duration)
            
            run.end(
                outputs={"translation": full_text},
                metadata={"cost_usd": round(estimated_cost, 6)}
            )
        except Exception as e:
            run.end(error=str(e))
            raise
    
    return text_file_path

async def _transcribe_file(
    client: AsyncOpenAI,
    file_path: Path,
    hinglish: bool
) -> str:
    """Transcribe a single audio file using async OpenAI client."""
    # Read file content first
    with open(file_path, "rb") as f:
        file_content = f.read()
    
    # Create a file-like object with the content
    audio_file = BytesIO(file_content)
    audio_file.name = file_path.name  # OpenAI needs a name attribute
    
    try:
        if hinglish:
            response = await client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
                response_format="text",
                prompt=(
                    "Instruction:\n"
                    "1. Translate the entire audio into fluent English.\n"
                    "2. Do NOT leave any Hindi phrases untranslated—except for individual Hindi words.\n"
                    "3. For each Hindi word, output it exactly as spoken in Devanagari script (e.g., 'accha (अच्छा)', 'hai (है)').\n"
                    "4. Keep the Hindi words inline with your English translation; do not transliterate them into Latin.\n"
                    "5. Output only the final English text with inline Hindi words—no extra commentary."
                ),
            )
        else:
            response = await client.audio.translations.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        return response.text if hasattr(response, 'text') else str(response)
    
    except Exception as e:
        raise RuntimeError(f"Transcription API call failed for {file_path.name}: {e}") from e

async def _transcribe_in_chunks(
    client: AsyncOpenAI,
    audio: AudioSegment,
    text_file_path: Path,
    hinglish: bool,
    max_chunk_seconds: int
) -> str:
    """Transcribe audio in chunks and append to file immediately."""
    duration_ms = len(audio)
    chunk_ms = max_chunk_seconds * 1000
    chunk_index = 0
    
    # Clear the file first (synchronously to ensure it happens)
    _write_transcript_sync(text_file_path, "", append=False)
    
    all_transcripts = []  # Keep track for returning full text
    start_ms = 0
    
    while start_ms < duration_ms:
        end_ms = min(start_ms + chunk_ms, duration_ms)
        chunk = audio[start_ms:end_ms]
        chunk_duration = (end_ms - start_ms) / 1000
        
        # Export chunk to temporary file
        temp_file = text_file_path.parent / f"temp_chunk_{chunk_index}.mp3"
        try:
            print(f"Processing chunk {chunk_index}: {chunk_duration:.1f}s...")
            
            # Export chunk (in thread to not block)
            await asyncio.to_thread(
                chunk.export,
                str(temp_file),
                format="mp3",
                bitrate="128k"  # Lower bitrate to stay under 25MB
            )
            
            print(f"  → Chunk {chunk_index} exported, sending to API...")
            
            # Transcribe chunk
            transcript = await _transcribe_file(client, temp_file, hinglish)
            
            print(f"  → Chunk {chunk_index} transcribed, writing to file...")
            
            # IMMEDIATELY write to file SYNCHRONOUSLY (no threading, no delays)
            chunk_text = f"--- CHUNK {chunk_index} ({chunk_duration:.1f}s) ---\n{transcript.strip()}\n\n"
            _write_transcript_sync(text_file_path, chunk_text, append=True)
            
            # Also keep in memory for final return
            all_transcripts.append(transcript.strip())
            
            print(f"✓ Chunk {chunk_index} completed and saved to file!")
            
        except Exception as e:
            # Log error and fail immediately
            error_msg = f"--- CHUNK {chunk_index} FAILED: {str(e)} ---\n\n"
            _write_transcript_sync(text_file_path, error_msg, append=True)
            raise RuntimeError(f"Failed to transcribe chunk {chunk_index}: {e}") from e
        
        finally:
            # Clean up temp file
            if temp_file.exists():
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        chunk_index += 1
        start_ms = end_ms
    
    # Return combined text for the trace output
    return "\n".join(all_transcripts)

def _write_transcript_sync(file_path: Path, content: str, append: bool = False) -> None:
    """
    Synchronous file write that GUARANTEES completion.
    No threading, no async - just direct write.
    """
    mode = "a" if append else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        f.write(content)
        f.flush()  # Flush Python buffer
        os.fsync(f.fileno())  # Force OS to write to disk immediately