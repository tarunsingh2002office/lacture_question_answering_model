from langchain_core.runnables import RunnableLambda

async def extract_function(x):
    return {
        "page_text": x["page_text"],
        "cumulative_concise_summary": x["cumulative_concise_summary"],
        "detail_page_summary": x["both_summary"]["detail_page_summary"],
        "concise_page_summary": x["both_summary"]["concise_page_summary"]
    }

extract = RunnableLambda(func=extract_function)