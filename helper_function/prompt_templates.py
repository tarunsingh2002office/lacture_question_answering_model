from langchain_core.prompts import PromptTemplate

question_prompt = PromptTemplate(
    input_variables=["detail_page_summary", "cumulative_concise_summary"],
    template="""
Generate up to 30 multiple-choice questions (10 Easy, 10 Medium, 10 Hard) with **four answer options each**, correct answer option (eg. **text of that option**), and a brief answer explanation based on the **detailed page summary**. Use the cumulative concise summary from previous pages only for context, not for generating the questions themselves. Follow these STRICT RULES:

### VALIDATION RULES

1. **Question Basis**  
   - All questions must be based solely on the information provided in the `detail_page_summary`. Do not use any information from the `cumulative_concise_summary` to formulate the questions or their answers, except for context.
   - If there is not enough material to generate all 30 questions, produce as many as possible, but preserve the ratio of Easy:Medium:Hard as closely as possible.
  - In the option text **DO NOT** use banned phrases: "In the transcript," "In the page," "In the document," "In the script," "According to the text," "According to the summary,".

2. **Answer Options Basis**  
   - All questions must only and only have 4 answer options.
   - Each question should have a **1 correct answer option** and **3 incorrect answer options**.
   - In the option text **DO NOT** use banned phrases: "In the transcript," "In the page," "In the document," "In the script," "According to the text".
   - In the option text **DO NOT** add any option number prefix, such as "A.", "B.", "C.", "D." "A)" "B)" "C)" "D)" "1)" "2)" "3)" "4)"

3. **Diversity**
   - Cover a range of topics from the **detailed page summary**, including main topics, key points, explanations, examples, interactive elements, etc.

4. **Difficulty Levels**  
   - **Easy** (10): Direct recall or very simple recognition questions (e.g., definitions, straightforward facts).  
   - **Medium** (10): Require applying a concept to a slightly modified scenario or combining two ideas from the summary.  
   - **Hard** (10): Involve Analysis, synthesis, evaluation, complex inferences, multi‐step reasoning, or distinguishing subtle nuances (e.g., “Which of the following best explains why… based on the lecturer’s reasoning?”).

5. **Answer Explanation**  
   - For each question, provide a clear explanation of why the correct answer is right and why the others are wrong.
   - Each answer explanation, begin with either “In the lecture…” or “As defined…” or “According to the instructor…”  or "From the example given...", or "In the discussion, it is stated that..."
   - DO NOT use banned phrases: “in the transcript,” “in the page,” “in the document,” “in the script,” or “according to the text.”

6. **Stylistic Constraints**  
   - Use clear, academic language.  
   - Avoid phrasing that implies inference beyond what is written (e.g., “It can be assumed that…”).  
   - Do NOT reference “previous page” or “next page”—only refer to “the summary” or “the lecture.”

### CONTEXT
- **Concise Summary of Previous Pages (for reference only):**  
{cumulative_concise_summary}

- **Detailed Summary of Current Page:**  
{detail_page_summary}
""")

summary_prompt = PromptTemplate(
    input_variables=["page_text", "cumulative_concise_summary"],
    template="""
Analyze this educational lecture transcript page with STRICT ADHERENCE TO VISIBLE CONTENT ONLY to generate a DETAILED PAGE SUMMARY and a CONCISE PAGE SUMMARY.

# DETAILED PAGE SUMMARY
- Structure your summary under these Markdown headers:
  ### Key Topics
    - List the main subjects or themes introduced on this page.
    - If none: “No new topics introduced.”
  ### Definitions & Terminology
    - Capture any formal definitions, technical terms, or jargon explicitly defined.
    - Formal notations if "definitions & terminology" are mathematically presented
    - If none: “No definitions or technical terms introduced.”
  ### Explanations & Concepts
    - Summarize each concept or idea as explained, using the lecturer’s own wording as closely as possible.
    - Include any step‐by‐step reasoning, derivations, or pedagogical points.
    - If none: “No detailed explanations on this page.”
  ### Examples & Illustrations
    - Describe any example problems, case studies, analogies, or illustrative anecdotes given.
    - If none: “No examples or illustrations provided.”
  ### Important Details & Emphases
    - Note any crucial facts, caveats, or emphases (e.g., “Remember that…”, “It’s important not to…”).
    - If none: “No critical details emphasized.”
  ### Visual Aids
    - Descriptions of figures/diagrams using ONLY captions/text references.
    - If none: "No visual aids referenced"

- Absolute Prohibitions:
  1. Do NOT introduce any information not explicitly in the CURRENT PAGE CONTENT.
  2. Do NOT infer the lecturer’s intent or add personal commentary.
  3. Do NOT paraphrase examples beyond what is directly stated.
  4. Do NOT reference anything beyond this page’s content.
- Use plain Markdown bullet points within each section.

# CONCISE PAGE SUMMARY
- Construction Rules:
  1. MUST connect to previous context using logical connectors or temporal markers:
     - “Building on [previous concept]…”
     - “Next, the lecturer explains…”
     - “Consequently, the focus shifts to…”
     - "Following the discussion on [previous topic]..."
  2. If no clear connection to earlier pages: begin with "In this section..." or similar.
  3. Maintain neutral academic tone (no storytelling voice).
  4. Emphasize the flow of ideas (cause→effect or definition→application).
  5. ONLY use CURRENT PAGE FACTS + the explicit previous summary provided.
  6. Maintain consistency with the overall lecture theme.

- Output Requirements:
  - Novel‐style academic prose (no lists, no bullet points).
  - 3–5 sentences total.
  - Focus on how this page advances the overall lecture narrative.
  - Highlight one or two pivotal points from this page.
  - Emphasize logical flow between topics

# FALLBACK PROTOCOL
- For ANY ambiguous or missing information:
  1. Acknowledge uncertainty by stating: “Not explicitly stated in transcript.”
  2. Do NOT attempt to invent details.
- If the page contains:
  1. Blank or scanned page with no text → Return both summaries as empty with “Empty page” note.
  2. Only title or transition remarks (no substantive content) → Return both summaries with “Transition/overview only” note.
  3. Copyright/legal boilerplate → Return both summaries with “Copyright notice” note.
  4. Transition or filler content with no new information -> Return empty summaries with "No new information" note

# CONTEXT
- Concise summary of all previous pages (for reference only):  
{cumulative_concise_summary}

# CURRENT PAGE CONTENT
{page_text}
"""
)