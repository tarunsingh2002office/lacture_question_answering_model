# Schema for page summary generation
summary_json_schema = {
    "title": "detailed_page_summary_and_concise_page_summary",
    "type": "object",
    "properties": {
        "detail_page_summary": {
            "type": "string",
            "description": "Comprehensive summary including key topics, definitions, explanations, examples, and important details"
        },
        "concise_page_summary": {
            "type": "string",
            "description": "Condensed summary focusing on main points with logical flow (3-5 sentences max)"
        }
    },
    "required": ["detail_page_summary", "concise_page_summary"]
}

# Schema for question generation (used by all models)
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
                        "description": "A very difficult multiple choice question requiring analysis, synthesis, or complex reasoning"
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "4 different answer options (1 correct, 3 incorrect)"
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "The complete text of the correct answer option"
                    },
                    "answer_explanation": {
                        "type": "string",
                        "description": "Detailed explanation for the correct answer"
                    }
                },
                "required": ["question", "options", "correct_answer", "answer_explanation"]
            },
            "description": "Array of hard difficulty questions with answers and explanations"
        },
        "medium_difficult_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A medium difficulty multiple choice question requiring concept application"
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "4 different answer options (1 correct, 3 incorrect)"
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "The complete text of the correct answer option"
                    },
                    "answer_explanation": {
                        "type": "string",
                        "description": "Detailed explanation for the correct answer"
                    }
                },
                "required": ["question", "options", "correct_answer", "answer_explanation"]
            },
            "description": "Array of medium difficulty questions with answers and explanations"
        },
        "easy_difficult_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "An easy difficulty multiple choice question testing direct recall"
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "4 different answer options (1 correct, 3 incorrect)"
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "The complete text of the correct answer option"
                    },
                    "answer_explanation": {
                        "type": "string",
                        "description": "Detailed explanation for the correct answer"
                    }
                },
                "required": ["question", "options", "correct_answer", "answer_explanation"]
            },
            "description": "Array of easy difficulty questions with answers and explanations"
        }
    },
    "required": ["hard_difficult_questions", "medium_difficult_questions", "easy_difficult_questions"]
}

# Schema for cumulative summary generation (combining multiple lectures)
cumulative_summary_json_schema = {
    "title": "combined_lecture_summary",
    "type": "object",
    "properties": {
        "combined_summary": {
            "type": "string",
            "description": "A concise combined summary of all lectures processed so far, prioritizing recent content while maintaining key concepts from earlier lectures. Should be under 2000 words to fit in context windows."
        }
    },
    "required": ["combined_summary"]
}