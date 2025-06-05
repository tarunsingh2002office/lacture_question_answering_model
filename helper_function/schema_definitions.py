question_json_schema = {
    "title": "question_answers_and_explanations",   
    "type": "object",
    "properties": {
        "hard_difficult_questions": {                        
            "type": "array",
            "items": {                        
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Compose a very difficult multiple choice question with 1 correct answer and 3 incorrect answers"
                    },
                    "options":{
                        "type":"array",
                        "items":{
                            "type":"string"
                        },
                        "description":"Compose 4 different answer options for the question with 1 correct answer and 3 incorrect answers"
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "Return the correct answer option complete text"
                    },
                    "answer_explanation": {
                        "type": "string",
                        "description": "Write the detailed explanation for the correct answer"
                    }
                },
                "required": ["question", "options" ,"correct_answer", "answer_explanation"]
            },
            "description": "This array contains all the hard difficulty questions, correct answers and answer explanations",
        },
        "medium_difficult_questions": {                        
            "type": "array",
            "items": {                        
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Compose a medium difficulty multiple choice question with 1 correct answer and 3 incorrect answers"
                    },
                    "options":{
                        "type":"array",
                        "items":{
                            "type":"string"
                        },
                        "description":"Compose 4 different answer options for the question with 1 correct answer and 3 incorrect answers"
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "Return the correct answer to the question"
                    },
                    "answer_explanation": {
                        "type": "string",
                        "description": "Write the detailed explanation for the correct answer"
                    }
                },
                "required": ["question", "options" ,"correct_answer", "answer_explanation"]
            },
            "description": "This array contains all the medium difficulty questions, correct answers and answer explanations",
        },
        "easy_difficult_questions": {                        
            "type": "array",
            "items": {                        
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Compose a easy difficulty multiple choice question with 1 correct answer and 3 incorrect answers"
                    },
                    "options":{
                        "type":"array",
                        "items":{
                            "type":"string"
                        },
                        "description":"Compose 4 different answer options for the question with 1 correct answer and 3 incorrect answers"
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "Return the correct answer to the question"
                    },
                    "answer_explanation": {
                        "type": "string",
                        "description": "Write the detailed explanation for the correct answer"
                    }
                },
                "required": ["question", "options" ,"correct_answer", "answer_explanation"]
            },
            "description": "This array contains all the easy difficulty questions, correct answers and answer explanations",
        },
        "Topic_importance_rating": {
            "type": "string",
            "enum": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "description": """
RUTHLESSLY rate 1-10 CURRENT PAGE IMORTANCE for understanding the complete topic (MUST BE HYPER-CRITICAL):

**10-Point Importance Scale**:
- **10**: The page is absolutely essential; it introduces the main topic or provides a comprehensive overview. Without this page, understanding the topic would be severely compromised.
- **9**: The page introduces a major subtopic or key concept that is critical for understanding the topic.
- **8**: The page provides significant information or explanations that are necessary for a full understanding.
- **7**: The page offers useful details or examples that enhance understanding but are not absolutely necessary.
- **6**: The page contains information that is relevant but could be omitted without greatly affecting understanding.
- **5**: The page has content that is part of the topic but not particularly crucial.
- **4**: The page includes some relevant information but is mostly peripheral.
- **3**: The page has minimal importance; it is largely tangential or a digression.
- **2**: The page has very low importance; it contains little to no relevant information.
- **1**: The page is irrelevant; it does not contribute to understanding the topic at all.

**Anti-Bias Rules**:
- ❌ Never rate based on previous/future pages; consider only the current page's content.
- ❌ No extra points for "setup" or "foreshadowing"; rate based on the actual content provided.
- ❌ Content-heavy ≠ automatically important; focus on the significance of the information, not just the quantity.
- ⛔ If unsure between two numbers, PICK THE LOWER ONE; be conservative in assigning higher ratings.

**Distribution Guidance**:
- **9-10**: Extremely rare, only for the very best pages.
- **7-8**: Very rare, for pages clearly above average.
- **1-6**: Common, with most pages likely in the 4-6 range.
"""
        }
    },
    "required": ["hard_difficult_questions", "medium_difficult_questions", "easy_difficult_questions","Topic_importance_rating"]
}

summary_json_schema = {
    "title": "detailed_page_summary_and_concise_page_summary",
    "type": "object",
    "properties": {
        "detail_page_summary": {
            "type": "string",
            "description": "Comprehensive summary including .."
        },
        "concise_page_summary": {
            "type": "string",
            "description": "Condensed summary focusing ... (3-5 sentences max)"
        }
    },
    "required": ["detail_page_summary", "concise_page_summary"]
}