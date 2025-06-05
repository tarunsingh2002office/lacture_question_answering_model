import os
import whisper
from fpdf import FPDF
from pathlib import Path
from moviepy import VideoFileClip
from PyPDF2 import PdfReader, PdfWriter

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
    
async def save_text_to_pdf(text: str, output_path: Path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(output_path)

async def split_pdf(input_pdf_path: Path, output_folder: Path) -> int:
    # Open the PDF file
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    
    # Iterate through all pages
    for i in range(total_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])  # Add one page to the writer
        
        # Generate the output file name
        output_file_path = output_folder / f"page_{i + 1}.pdf"
        with output_file_path.open("wb") as output_file:
            writer.write(output_file)
    return total_pages

async def audio_to_text(input_audio_path: Path) -> str:
    model = whisper.load_model("turbo")
    result = model.transcribe(str(input_audio_path))
    return result["text"]