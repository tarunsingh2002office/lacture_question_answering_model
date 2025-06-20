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
from langchain_community.document_loaders import PDFMinerLoader
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
        font_path= base_dir /"font" / "Poppins-Black.ttf"
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
        loader = PDFMinerLoader(pdf_name)
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
        # text = """
        # Hi guys, my name is Nitish and you're welcome to my YouTube channel. So I'm super happy to announce कि इस YouTube चैनल पे we are starting a new playlist and the topic of the playlist is Agentic AI using g LangGraph. अब honestly ये एक ऐसा topic है जिससे related मुझे countless messages आएं पिछले 3-4 महिनों ं में आपकी site से. आप लोगों ने constantly मुझे ये बोला कि sir please आप LangGraph के उपर एक playlist बनाओ and मैंने अराउन 3-4 महिने पहले decide कर लिया था कि मुझे इस particular topic पे एक बढ़िया playli ist बनानी है and that is why मैं 3-4 महिनों से इस topic के उपर काफी research कर रहा हूँ. मैंने बहुत ti ime ले करके curriculum define किया उसके बाद मैंने काफी preparation की उस curriculum के around to prepa are the content, documentation मैंने बहुत जादा study किया इस पूरे process में और finally अब 3 महिनों क के बाद I am confident enough कि मुझे लगता है कि I can make a playlist on this topic. आज का जो वीडियो ह है वो इस sense में special है कि अगर आप इस playlist को completely follow करना चाहते हो तो आज का वीडियो ो देखना बहुत जरूरी है because आज के वीडियो में मैं आपको अपना complete thought process बताने वाला हूँ क कि इस पूरे playlist को मैंने कैसे plan किया है. मैं इस वीडियो में आपको पूरा का पूरा curriculum भी बताऊ ऊंगा मैं साती सात आपको pre-requisites भी बताऊंगा और इसके अलाभा और जो doubts आपको आ सकते हैं about this s playlist वो सब कुछ हम इस वीडियो में discuss करेंगे. So let's start the video. So पहले एक बार बात करत ते हैं why की? The reason behind starting this playlist. So इस playlist को start करने के पीछे मेरे तीन न primary reasons थे. मैं एक एक करके आपको बताता हूँ. सबसे पहला reason है timing. मुझे ऐसा लगता है कि अ अभी बिलकुल सही time है agentic AI पढ़ने का. Because इस point पे अगर आप कोई भी platform open करो, YouTu ube open करो, Twitter open करो, Instagram open करो, आपको constantly ये term सुनाई देगा, देखने को मिलेग गा. दुनिया भर के जितने भी बड़ी companies हैं, उन बड़ी companies में जितने भी बड़े thought leaders हैं, , हर कोई इस point पे इस particular term को hype up कर रहा है. और मुझे ऐसा लगता है कि इसको hype up करना ा valid भी है. Because this is going to be the next big thing in computer science. सोच के देखो, 2022 म में आपका chat GPT आया और chat GPT के आने के बाद से generative AI का एक completely नया trajectory start t हो गया computer science में. और generative AI tools अब इतने mature हो गया हैं कि उनकी help से सच में ं बहुत powerful agents बनाये जा सकते हैं in the next 5 years. और AI agents जो होंगे, वो future में बहु ुत value create करेंगे. तो दुनिया भर के जितने भी बड़े बड़े leaders हैं, बड़ी बड़ी companies के जो CEOs s हैं और बहुत powerful position में जो लोग हैं, they are able to anticipate कि ये एक ऐसी चीज़ है जो दु ुनिया को बदल के रख सकती है. And that is why इस point पे अगर आप पढ़ते हो कि agentic applications कैसे ब बनाये जाते हैं, तो I feel कि आप ऐसी position में आ जाओगे कि future में आपकी भी बहुत जादा value होगी. त तो timing is the first reason. Reason number two is demand. जैसा मैंने बोला वीडियो के शुरुवात में, पिछ छले 3-4 महिनों में हर तीसरा comment मेरे channel पे यही था कि sir please सब कुछ छोड़ो, आप LangGraph के े उपर एक playlist बनाओ, because industry में बहुत जादा इसके बारे में बात हो रही है. तो आपकी side से एक क बहुत strong demand आया, that was the second reason. And the third reason was build up. अभी तक अगर आपप इस channel पे देखोगे, तो हम बहुत sequentially चीजे cover करते आ रहे हैं, और मेरा mostly ये कोशिश रहतती है कि मैं बहुत organized तरीके से चीजे अच्छे से cover करके आगे बढ़ूं. तो हमने पहले machine learning g किया इस channel पे, उसके बाद हमने deep learning किया, और उसके बाद फिर हमने land chain एकसेटरा स्टार् ्ट किया, generative AI स्टार्ट किया. तो इस point पे I personally feel कि हम लोग इतनी पढ़ाई कर चुके हैं ं कि we are kind of ready to learn and understand land graph and how to build AI agents. So this was reason number 3. So इन तीन reasons की वज़े से I feel कि we are in a position जहांपर हमें इस particular r playlist को start करना चाहिए और इसके बारे में पढ़ना चाहिए. ठीक है? अब आगे बढ़ते हैं और discuss करते हैं कि इस playlist को start करने के पीछे मेरा विजन क्या है? आप कभी भी कोई भी काम करते हो तो उसके पीछे े strong vision होता है. तो मैं आपके साथ अपना विजन share करना चाहता हूँ कि इस playlist के थूँ मैं क्या ा achieve करना चाहता हूँ? तो अगर मैं आपको honestly बताओं जब lang graph market में आया और धीरे धीरे आपक की side से messages मुझे आने लगे कि sir lang graph पढ़ाओ तो the first thing that I did was to go on Yo ouTube and search कि अभी existing क्या content available है lang graph के उपर और मैंने ये notice किया कि दो तरह के content available थे YouTube पे पहला एक ऐसा content जहांपर
        # """
        # print(text)
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