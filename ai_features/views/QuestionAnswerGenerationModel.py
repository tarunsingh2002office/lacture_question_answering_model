import io
import json
import uuid
import shutil
import zipfile
import asyncio
from core.config import ai_api_secrets
# from langchain_openai import ChatOpenAI
from fastapi import Request, UploadFile, File
from helper_function.runnable_lambda import extract
from langchain.schema.runnable import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.runnables.passthrough import RunnableAssign
from helper_function.prompt_templates import question_prompt, summary_prompt
from helper_function.schema_definitions import question_json_schema, summary_json_schema
from helper_function.video_to_pdf_function import video_to_audio, save_text_to_pdf, split_pdf, audio_to_text

async def init_models():
    try:
        question_generation_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.7)
        summary_generation_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        # question_generation_model = ChatOpenAI(model="o4-mini-2025-04-16") #both the model does not support custom temperature values and only allows the default value of 1
        # summary_generation_model = ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
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
            "cumulative_question_json_file": cumulative_question_json_file
        }
        data_dir.mkdir(exist_ok=True, parents=True)
        output_dir.mkdir(exist_ok=True, parents=True)
        split_pdf_dir.mkdir(exist_ok=True, parents=True)
        return all_paths
    except Exception as err:
        raise Exception(f"something went wrong {err}")
    
async def pdf_loader(pdf_name):
    try:
        loader = PDFMinerLoader(pdf_name)
        docs = await loader.aload()
        page_text = docs[0].page_content
        return page_text
    except Exception as err:
        raise Exception(f"something went wrong {err}") 

async def chain(structured_summary_generation_model, structured_question_generation_model):
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
        with cumulative_concise_summary_file.open("w", encoding="utf-8") as f:
            f.write(cumulative_concise_summary)

        with cumulative_detailed_summary_file.open("w", encoding="utf-8") as f:
            f.write(cumulative_detailed_summary)

        with cumulative_question_json_file.open("w") as f:
            json.dump(final_json, f, indent=4)
        
    except Exception as err:
        raise Exception(f"something went wrong {err}")

async def process_single_page(page_num, split_pdf_dir, cumulative_concise_summary, final_chain):
    try:
        current_page_number = page_num + 1

        pdf_name = split_pdf_dir / f"page_{current_page_number}.pdf"
        page_text = await pdf_loader(pdf_name)

        result = await final_chain.ainvoke({
            "page_text": page_text,
            "cumulative_concise_summary": cumulative_concise_summary
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

async def QuestionAnswerGenerationModel(request: Request, uploaded_file: UploadFile = File(..., media_type="video/mp4")):
    try:
        if uploaded_file.content_type != "video/mp4":
            return JSONResponse(content={"message": "Invalid file type. Please upload a video file."}, status_code=400)
        all_paths = await paths()
        file_bytes = await uploaded_file.read()
        with open(all_paths["input_video_file"], "wb") as out_f:
            out_f.write(file_bytes)
        await video_to_audio(all_paths["input_video_file"], output_path=all_paths["input_audio_file"])
        text = await audio_to_text(all_paths["input_audio_file"])
        await save_text_to_pdf(text, output_path=all_paths["input_pdf_file"])
        total_pages = await split_pdf(all_paths["input_pdf_file"], all_paths["split_pdf_dir"])
        structured_summary_generation_model, structured_question_generation_model = await init_models()
        final_chain = await chain(structured_summary_generation_model, structured_question_generation_model)
        cumulative_concise_summary = ""
        cumulative_detailed_summary = ""
        final_json = {}
        for page_num in range(total_pages):
            questions, concise_page_summary, detailed_page_summary = await process_single_page(
                    page_num,
                    all_paths["split_pdf_dir"],
                    cumulative_concise_summary,
                    final_chain
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
    
    