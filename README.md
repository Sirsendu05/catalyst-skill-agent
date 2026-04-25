# SkillSense — AI-Powered Skill Assessment Agent

## What it does
SkillSense takes a Job Description and a candidate's resume, then 
conversationally interviews the candidate to assess real proficiency 
on each required skill. It identifies gaps and generates a personalised 
learning plan with curated resources and time estimates.

## Live Demo
[Add your Streamlit URL here after deploying]

## Demo Video
[Add your Loom video link here]

## How to run locally
1. Clone this repo
2. Install requirements: `pip install -r requirements.txt`
3. Add your Anthropic API key to Replit Secrets: `ANTHROPIC_API_KEY=your_key`
4. Run: `streamlit run app.py`

## Architecture
6-stage pipeline:
1. User uploads JD + resume (PDF or text)
2. Parsing — Claude extracts required skills from JD and claimed skills from resume
3. Skill Matcher — maps JD requirements vs resume claims, identifies gaps
4. Conversational Assessment — 2-4 adaptive questions per skill via Claude
5. Scoring + Gap Analysis — 0-10 scores weighted by JD importance
6. Learning Plan — curated free resources with time estimates per skill gap


## Scoring Logic
- Each skill scored 0-10 by the assessment agent
- Gap = claimed score minus assessed score
- Learning plan prioritises by: gap severity × JD importance weight

## Tech Stack
- Frontend: Streamlit
- AI: Claude Haiku via Anthropic API
- PDF parsing: PyPDF2
- Hosting: Replit

## Sample Inputs & Outputs
See the samples/ folder for example JD, resume, and generated report.
