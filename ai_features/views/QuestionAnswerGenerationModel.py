import io
import uuid
import shutil
import zipfile
import asyncio
from core.config import ai_api_secrets
from langchain_openai import ChatOpenAI
from helper_function.runnable_lambda import extract
from fastapi import Request, UploadFile, File, Form
from langchain.schema.runnable import RunnableParallel
# from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.passthrough import RunnableAssign
from helper_function.prompt_templates import question_prompt, summary_prompt
from helper_function.schema_definitions import question_json_schema, summary_json_schema
from helper_function.video_to_pdf_function import video_to_audio, save_text_to_pdf, split_pdf, write_file, audio_to_text

def init_models():
    try:
        # question_generation_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.7)
        # summary_generation_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        question_generation_model = ChatOpenAI(model="o4-mini-2025-04-16") #both the model does not support custom temperature values and only allows the default value of 1
        summary_generation_model = ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
        structured_question_generation_model = question_generation_model.with_structured_output(question_json_schema)
        structured_summary_generation_model = summary_generation_model.with_structured_output(summary_json_schema)
        return structured_summary_generation_model, structured_question_generation_model 
    except Exception as err:
        raise Exception(f"something went wrong {err}") 

async def paths():
    try:
        base_dir = ai_api_secrets.BASE_DIR 
        request_id = str(uuid.uuid4())
        data_dir = base_dir / "data" / request_id
        output_dir = data_dir / "output"
        split_pdf_dir = output_dir / "split_pdf"
        input_pdf_file = data_dir / "input.pdf"
        input_video_file = data_dir / "input.mp4"
        input_audio_file = data_dir / "input.mp3"
        cumulative_detailed_summary_file = output_dir / "cumulative_detailed_summary.txt"
        cumulative_concise_summary_file = output_dir / "cumulative_concise_summary.txt"
        cumulative_question_json_file = output_dir / "cumulative_question.json"
        font_path= base_dir / "font" / "Poppins-Regular.ttf"
        all_paths = {
            "base_dir": base_dir,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "split_pdf_dir": split_pdf_dir,
            "input_pdf_file": input_pdf_file,
            "input_video_file": input_video_file,
            "input_audio_file": input_audio_file,
            "cumulative_detailed_summary_file": cumulative_detailed_summary_file,
            "cumulative_concise_summary_file": cumulative_concise_summary_file,
            "cumulative_question_json_file": cumulative_question_json_file,
            "font_path": font_path
        }
        await asyncio.to_thread(data_dir.mkdir, exist_ok=True, parents=True)
        await asyncio.to_thread(output_dir.mkdir, exist_ok=True, parents=True)
        await asyncio.to_thread(split_pdf_dir.mkdir, exist_ok=True, parents=True)
        return all_paths
    except Exception as err:
        raise Exception(f"something went wrong {err}")
    
async def pdf_loader(pdf_name):
    try:
        loader = PyPDFLoader(pdf_name)
        docs = await loader.aload()
        page_text = docs[0].page_content
        return page_text
    except Exception as err:
        raise Exception(f"something went wrong {err}") 

def chain(structured_summary_generation_model, structured_question_generation_model):
    try: 
        summary_chain = summary_prompt | structured_summary_generation_model
        question_chain = question_prompt | structured_question_generation_model
        chain1 = RunnableAssign(RunnableParallel({"both_summary": summary_chain}))
        chain2 = RunnableAssign(RunnableParallel({"question": question_chain}))
        final_chain = chain1 | extract | chain2
        return final_chain
    except Exception as err:
        raise Exception(f"something went wrong {err}") 
    
async def save_outputs(cumulative_concise_summary, cumulative_detailed_summary, final_json, cumulative_concise_summary_file, cumulative_detailed_summary_file, cumulative_question_json_file):
    
    try: 
        await asyncio.gather(
            write_file(cumulative_concise_summary_file, cumulative_concise_summary),
            write_file(cumulative_detailed_summary_file, cumulative_detailed_summary),
            write_file(cumulative_question_json_file, final_json)
        )
    except Exception as err:
        raise Exception(f"something went wrong {err}")

async def process_single_page(page_num, split_pdf_dir, cumulative_concise_summary, final_chain, number_of_questions):
    try:
        current_page_number = page_num + 1

        pdf_name = split_pdf_dir / f"page_{current_page_number}.pdf"
        page_text = await pdf_loader(pdf_name)
        number_of_questions_in_each_category = number_of_questions/3
        result = await final_chain.ainvoke({
            "page_text": page_text,
            "cumulative_concise_summary": cumulative_concise_summary,
            "number_of_questions": number_of_questions,
            "number_of_questions_in_each_category": number_of_questions_in_each_category
        })

        question = result["question"]
        concise_page_summary=result["concise_page_summary"]
        detailed_page_summary=result["detail_page_summary"]
        concise_page_summary = f"\n\n#### Page {current_page_number} Summary:\n{concise_page_summary}\n"
        detailed_page_summary = f"\n\n#### Page {current_page_number} Summary:\n{detailed_page_summary}\n" 
        question["concise_page_summary"] = concise_page_summary

        return question, concise_page_summary, detailed_page_summary
    except Exception as err:
        raise Exception(f"something went wrong {err}")

def create_zip_sync(all_paths, zip_buffer):
    try:
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                zip_file.write(all_paths["cumulative_concise_summary_file"], arcname="cumulative_concise_summary.txt")
                zip_file.write(all_paths["cumulative_detailed_summary_file"], arcname="cumulative_detailed_summary.txt")
                zip_file.write(all_paths["cumulative_question_json_file"], arcname="cumulative_question.json")
                zip_file.write(all_paths["input_pdf_file"], arcname="input.pdf")
        return zip_buffer
    except Exception as err:
        raise Exception(f"something went wrong {err}")

async def cleanup(all_paths):
    try:
        if all_paths["input_video_file"].exists():
            await asyncio.to_thread(all_paths["input_video_file"].unlink)
        if all_paths["input_audio_file"].exists():
            await asyncio.to_thread(all_paths["input_audio_file"].unlink)
        if all_paths["input_pdf_file"].exists():
            await asyncio.to_thread(all_paths["input_pdf_file"].unlink)
        if all_paths["split_pdf_dir"].exists():
            await asyncio.to_thread(shutil.rmtree, all_paths["split_pdf_dir"])
        if all_paths["output_dir"].exists():
            await asyncio.to_thread(shutil.rmtree, all_paths["output_dir"])
        if all_paths["data_dir"].exists():
            await asyncio.to_thread(shutil.rmtree, all_paths["data_dir"])
    except Exception as err:
        raise Exception(f"something went wrong {err}")

async def QuestionAnswerGenerationModel(request: Request, uploaded_file: UploadFile = File(..., media_type="video/mp4"), number_of_questions: int = Form(...), hinglish: bool = Form(...)):
    try:
        if uploaded_file.content_type != "video/mp4":
            return JSONResponse(content={"message": "Invalid file type. Please upload a video file."}, status_code=400)
        if number_of_questions < 1 or number_of_questions > 21:
            return JSONResponse(content={"message": "Number must be between 1 and 20"}, status_code=400)
        if number_of_questions % 3 !=0:
            return JSONResponse(content={"message": "Number must be divisble by 3"}, status_code=400)
        
        all_paths = await paths()
        file_bytes = await uploaded_file.read()

        await write_file(all_paths["input_video_file"], file_bytes)

        await video_to_audio(all_paths["input_video_file"], output_path=all_paths["input_audio_file"])

        text = await audio_to_text(all_paths["input_audio_file"], hinglish=hinglish)
        await save_text_to_pdf(text=text, output_path=all_paths["input_pdf_file"], font_path=all_paths["font_path"])
        total_pages = await split_pdf(all_paths["input_pdf_file"], all_paths["split_pdf_dir"])
        structured_summary_generation_model, structured_question_generation_model = init_models()
        final_chain = chain(structured_summary_generation_model, structured_question_generation_model)
        cumulative_concise_summary = ""
        cumulative_detailed_summary = ""
        final_json = {}
        for page_num in range(total_pages):
            questions, concise_page_summary, detailed_page_summary = await process_single_page(
                    page_num=page_num,
                    split_pdf_dir=all_paths["split_pdf_dir"],
                    cumulative_concise_summary=cumulative_concise_summary,
                    final_chain=final_chain,
                    number_of_questions=number_of_questions
            )
            final_json[page_num+1] = questions
            cumulative_concise_summary += concise_page_summary
            cumulative_detailed_summary += detailed_page_summary
            await save_outputs(
                cumulative_concise_summary,
                cumulative_detailed_summary,
                final_json,
                all_paths["cumulative_concise_summary_file"],
                all_paths["cumulative_detailed_summary_file"],
                all_paths["cumulative_question_json_file"]
            )
        zip_buffer = io.BytesIO()
        zip_buffer = await asyncio.to_thread(create_zip_sync,all_paths, zip_buffer)
        zip_buffer.seek(0)
        await cleanup(all_paths)
        return StreamingResponse(
            zip_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=results.zip"},
            status_code=200
        )
    except Exception as err:
        required_files = [
            all_paths["cumulative_concise_summary_file"],
            all_paths["cumulative_detailed_summary_file"],
            all_paths["cumulative_question_json_file"],
            all_paths["input_pdf_file"]
        ]
        if all(f.exists() for f in required_files):

            zip_buffer = io.BytesIO()
            zip_buffer = await asyncio.to_thread(create_zip_sync,all_paths, zip_buffer)
            zip_buffer.seek(0)
            await cleanup(all_paths)
            return StreamingResponse(
                zip_buffer,
                media_type="application/x-zip-compressed",
                headers={"Content-Disposition": "attachment; filename=results.zip",
                "err": str(err)},
                status_code=500
            )
        else:
            await cleanup(all_paths)
            return JSONResponse(
                content={"message": "Processing failed", "error": str(err)},
                status_code=500
            )