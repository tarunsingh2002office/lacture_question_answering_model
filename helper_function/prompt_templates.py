from langchain_core.prompts import PromptTemplate

# Template for page summary generation
summary_prompt = PromptTemplate(
    input_variables=["page_text", "cumulative_concise_summary", "number_of_questions", "number_of_questions_in_each_category"],
    template="""
Analyze this educational lecture transcript page with STRICT ADHERENCE TO VISIBLE CONTENT ONLY to generate a DETAILED PAGE SUMMARY and a CONCISE PAGE SUMMARY.

# DETAILED PAGE SUMMARY
Structure your summary under these Markdown headers:

### Key Topics
- List the main subjects or themes introduced on this page
- If none: "No new topics introduced"

### Definitions & Terminology
- Capture any formal definitions, technical terms, or jargon explicitly defined
- Include formal notations if definitions are mathematically presented
- If none: "No definitions or technical terms introduced"

### Explanations & Concepts
- Summarize each concept or idea as explained, using the lecturer's own wording as closely as possible
- Include any step-by-step reasoning, derivations, or pedagogical points
- If none: "No detailed explanations on this page"

### Examples & Illustrations
- Describe any example problems, case studies, analogies, or illustrative anecdotes given
- If none: "No examples or illustrations provided"

### Important Details & Emphases
- Note any crucial facts, caveats, or emphases (e.g., "Remember that...", "It's important not to...")
- If none: "No critical details emphasized"

### Visual Aids
- Descriptions of figures/diagrams using ONLY captions/text references
- If none: "No visual aids referenced"

**Absolute Prohibitions:**
1. Do NOT introduce any information not explicitly in the CURRENT PAGE CONTENT
2. Do NOT infer the lecturer's intent or add personal commentary
3. Do NOT paraphrase examples beyond what is directly stated
4. Do NOT reference anything beyond this page's content
5. Do NOT create summary in language other than English
6. Keep summary concise but comprehensive (max 800 words)

# CONCISE PAGE SUMMARY

**Construction Rules:**
1. MUST connect to previous context using logical connectors:
   - "Building on [previous concept]..."
   - "Next, the lecturer explains..."
   - "Consequently, the focus shifts to..."
   - "Following the discussion on [previous topic]..."
2. If no clear connection: begin with "In this section..." or similar
3. Maintain neutral academic tone (no storytelling voice)
4. Emphasize the flow of ideas (cause→effect or definition→application)
5. ONLY use CURRENT PAGE FACTS + the explicit previous summary provided
6. Maintain consistency with the overall lecture theme
7. Do NOT create summary in language other than English

**Output Requirements:**
- Novel-style academic prose (no lists, no bullet points)
- 3-5 sentences total
- Focus on how this page advances the overall lecture narrative
- Highlight one or two pivotal points from this page
- Emphasize logical flow between topics

# FALLBACK PROTOCOL
For ANY ambiguous or missing information:
1. Acknowledge uncertainty: "Not explicitly stated in transcript"
2. Do NOT attempt to invent details

If the page contains:
- Blank/scanned page with no text → Return both summaries as "Empty page"
- Only title or transition remarks → Return "Transition/overview only"
- Copyright/legal boilerplate → Return "Copyright notice"
- Transition or filler content with no new information → Return "No new information"

# CONTEXT
**Concise summary of all previous pages (for reference only):**
{cumulative_concise_summary}

# CURRENT PAGE CONTENT
{page_text}
"""
)

# Template for question generation by multiple models
question_prompt_multi_model = PromptTemplate(
    input_variables=["lecture_summary", "number_of_questions", "number_of_questions_in_each_category"],
    template="""
Generate exactly {number_of_questions} multiple-choice questions ({number_of_questions_in_each_category} Easy, {number_of_questions_in_each_category} Medium, {number_of_questions_in_each_category} Hard) based SOLELY on the lecture summary provided below.

# STRICT VALIDATION RULES

## 1. Question Basis
- All questions must be in English only
- All questions must be based SOLELY on information in the lecture summary
- If insufficient material exists, generate as many as possible while maintaining Easy:Medium:Hard ratio
- DO NOT use these phrases in questions: "In the transcript", "In the page", "In the document", "In the script", "According to the text", "According to the summary"

## 2. Answer Options
- All options must be in English only
- Each question must have EXACTLY 4 options (1 correct, 3 incorrect)
- DO NOT use option number prefixes like "A.", "B.", "C.", "D.", "A)", "B)", "C)", "D)", "1)", "2)", "3)", "4)"
- DO NOT use banned phrases in options: "In the transcript", "In the page", "In the document", "In the script", "According to the text"

## 3. Diversity
- Cover a wide range of topics from the lecture summary
- Include questions about main topics, key points, explanations, examples, etc.

## 4. Difficulty Levels
- **Easy** ({number_of_questions_in_each_category}): Direct recall or simple recognition (definitions, straightforward facts)
- **Medium** ({number_of_questions_in_each_category}): Apply concepts to modified scenarios or combine two ideas
- **Hard** ({number_of_questions_in_each_category}): Analysis, synthesis, evaluation, complex inferences, multi-step reasoning, or distinguishing subtle nuances

## 5. Answer Explanation
- Provide clear explanation of why the correct answer is right and others are wrong
- Begin each explanation with: "In the lecture...", "As defined...", "According to the instructor...", "From the example given...", or "In the discussion, it is stated that..."
- DO NOT use: "in the transcript", "in the page", "in the document", "in the script", "according to the text"

## 6. Stylistic Constraints
- Use clear, academic language
- Avoid phrasing that implies inference beyond what is written
- DO NOT reference "previous page" or "next page"—only "the summary" or "the lecture"

# LECTURE SUMMARY
{lecture_summary}

Generate the questions now, ensuring strict adherence to all rules above.
"""
)

# Template for selecting best questions from multiple model outputs
question_selection_prompt = PromptTemplate(
    input_variables=["all_model_questions", "lecture_summary", "number_of_questions", "number_of_questions_in_each_category"],
    template="""
You are an expert educational content evaluator. Multiple AI models have generated questions based on a lecture summary. Your task is to select the BEST {number_of_questions} questions ({number_of_questions_in_each_category} Easy, {number_of_questions_in_each_category} Medium, {number_of_questions_in_each_category} Hard).

# SELECTION CRITERIA (Ranked by Priority)

## 1. Accuracy & Relevance (Critical)
- Question must be directly answerable from the lecture summary
- Correct answer must be unambiguously correct
- Incorrect options must be clearly wrong but plausible
- No ambiguity in wording

## 2. Quality of Distractor Options
- Incorrect options should be plausible and test understanding
- Avoid obviously wrong options
- Options should be parallel in structure and length
- No "all of the above" or "none of the above" options

## 3. Clarity & Language
- Clear, concise wording
- No grammatical errors
- Appropriate academic tone
- No banned phrases

## 4. Cognitive Level Appropriateness
- Easy questions: Direct recall, definitions
- Medium questions: Application, analysis
- Hard questions: Synthesis, evaluation, complex reasoning

## 5. Coverage & Diversity
- Select questions covering different topics from the lecture
- Avoid redundant questions testing the same concept
- Balance between theoretical and practical aspects

## 6. Explanation Quality
- Clear, comprehensive explanations
- Explains why correct answer is right
- May explain why key distractors are wrong
- Proper citation format (starting with approved phrases)

# EVALUATION PROCESS

1. **Review all model outputs**: Examine questions from all models
2. **Score each question**: Rate 1-10 on each criterion above
3. **Select best questions**: Choose top {number_of_questions_in_each_category} from each difficulty level
4. **Ensure diversity**: No duplicate concepts, good topic coverage
5. **Final validation**: All selected questions meet ALL validation rules

# MODEL OUTPUTS

{all_model_questions}

# LECTURE SUMMARY (for reference)

{lecture_summary}

# OUTPUT FORMAT

Return EXACTLY {number_of_questions} questions in the required JSON schema format, with:
- {number_of_questions_in_each_category} easy_difficult_questions
- {number_of_questions_in_each_category} medium_difficult_questions
- {number_of_questions_in_each_category} hard_difficult_questions

Select the absolute best questions from all models, ensuring they meet ALL validation criteria.
"""
)

# Template for combining lecture summaries
cumulative_summary_prompt = PromptTemplate(
    input_variables=["previous_lectures_summary", "new_lecture_summary", "lecture_number"],
    template="""
You are tasked with creating a combined summary of multiple lectures. You have:
1. A summary of all previous lectures (Lectures 1 to {lecture_number} minus 1)
2. A new lecture summary (Lecture {lecture_number})

# OBJECTIVE
Create a CONCISE combined summary that:
- Integrates the new lecture content with previous lectures
- Prioritizes recent content (give more weight to Lecture {lecture_number})
- Maintains key foundational concepts from earlier lectures
- Shows logical progression of topics across lectures
- Stays under 2000 words to fit in AI context windows

# CONSTRUCTION RULES

1. **Structure**: Use a narrative flow, NOT bullet points
   - Begin with overarching themes connecting all lectures
   - Progress through major topics chronologically
   - Highlight relationships between concepts across lectures

2. **Content Prioritization**:
   - New lecture content: 50-60% of summary
   - Previous lectures: 40-50% of summary
   - Keep only the most important concepts from earlier lectures
   - Remove redundant information that's been superseded

3. **Connection & Flow**:
   - Use transitional phrases: "Building on the earlier discussion of...", "Expanding from Lecture X...", "This connects to..."
   - Show how concepts evolve across lectures
   - Maintain coherent narrative arc

4. **Compression Guidelines**:
   - For earlier lectures: Keep only core concepts, key definitions, and essential frameworks
   - For recent lecture: Keep detailed explanations and examples
   - Remove: Repetitive information, minor details, tangential discussions

5. **Quality Standards**:
   - Clear, academic prose
   - No loss of critical information
   - Maintains technical accuracy
   - Readable and coherent as a standalone document

# PREVIOUS LECTURES SUMMARY (Lectures 1 to {lecture_number} minus 1)

{previous_lectures_summary}

# NEW LECTURE SUMMARY (Lecture {lecture_number})

{new_lecture_summary}

# OUTPUT

Provide a combined summary that seamlessly integrates all lectures, with emphasis on the most recent content while preserving essential earlier concepts. The summary should be comprehensive yet concise (under 2000 words).
"""
)