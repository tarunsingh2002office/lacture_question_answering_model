from langchain_core.runnables import RunnableLambda

async def extract_summary_function(x):
    """Extract summary outputs from the chain"""
    return {
        "detail_page_summary": x["summary_output"]["detail_page_summary"],
        "concise_page_summary": x["summary_output"]["concise_page_summary"]
    }

async def extract_questions_function(x):
    """Extract all model questions for selection"""
    all_questions = {}
    
    # Extract questions from each model
    for key in x.keys():
        if key.endswith("_questions"):
            model_name = key.replace("_questions", "")
            all_questions[model_name] = x[key]
    
    return {
        "all_model_questions": all_questions
    }

# Create runnable lambdas
extract_summary = RunnableLambda(func=extract_summary_function)
extract_questions = RunnableLambda(func=extract_questions_function)
