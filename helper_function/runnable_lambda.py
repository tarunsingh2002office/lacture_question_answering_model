from langchain_core.runnables import RunnableLambda

async def extract_function(x):
    return {
        "number_of_questions": x["number_of_questions"],
        "number_of_questions_in_each_category": x["number_of_questions_in_each_category"],
        "page_text": x["page_text"],
        "cumulative_concise_summary": x["cumulative_concise_summary"],
        "detail_page_summary": x["both_summary"]["detail_page_summary"],
        "concise_page_summary": x["both_summary"]["concise_page_summary"]
    }

extract = RunnableLambda(func=extract_function)