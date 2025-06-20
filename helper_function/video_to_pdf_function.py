import os
import io
import json
import asyncio
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
from typing import Union, List
from moviepy import VideoFileClip
from reportlab.pdfgen import canvas
from langsmith.run_helpers import trace
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont

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
    
async def save_text_to_pdf(font_path: Path, output_path: Path, text: str, page_width: int = 580, page_margin: int = 20) -> None:
    try:
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
    
async def audio_to_text(path: Path, hinglish: bool = False) -> str:
    client = OpenAI()
    # Determine file size and total duration
    file_size = path.stat().st_size
    audio = AudioSegment.from_file(path)
    total_duration_sec = audio.duration_seconds

    # Whisper cost: $0.006 per minute
    cost_per_sec = 0.006 / 60

    # Prepare chunks (20MB each)
    chunk_size = 20 * 1024 * 1024  # 20MB
    texts: List[str] = []
    total_cost = 0.0

    name = "whisper-translate_audio_chunks"
    tags = "whisper"
    if hinglish:
        name = "gpt-4o-transcribe-translate_audio_chunks_hinglish"
        tags = "gpt-4o-transcribe-hinglish"
    async with trace(
        name=name,
        run_type="tool",
        inputs={"file_path": str(path)},
        tags=["translation", tags],
        extra=None,
        parent=None,
    ) as run:
        try:
            with path.open("rb") as f:
                idx = 0
                while True:
                    chunk_bytes = f.read(chunk_size)
                    if not chunk_bytes:
                        break

                    # Estimate chunk duration by proportion of total size
                    chunk_duration = total_duration_sec * (len(chunk_bytes) / file_size)
                    chunk_cost = chunk_duration * cost_per_sec
                    total_cost += chunk_cost

                    # Send chunk to Whisper
                    audio_file = io.BytesIO(chunk_bytes)
                    audio_file.name = f"chunk_{idx}.mp3"

                    translation_or_transcription=""
                    
                    if hinglish:
                        translation_or_transcription = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: client.audio.transcriptions.create(
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
                        )
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: client.audio.translations.create(
                                model="whisper-1",
                                file=audio_file,
                            )
                        )
                        translation_or_transcription = result.text
                    
                    texts.append(translation_or_transcription)
                    idx += 1

            result_text = "\n".join(texts)
            run.end(
                outputs={"translation": result_text},
                metadata={"cost_usd": round(total_cost, 6)},
            )
            return result_text

        except Exception as e:
            raise e