import streamlit as st
import anthropic
import PyPDF2
import json
import io
import re
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkillSense — AI Skill Assessment Agent",
    page_icon="🎯",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stProgress > div > div { background-color: #4F46E5; }
    .skill-badge { padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: 500; }
    .badge-critical { background: #fee2e2; color: #991b1b; }
    .badge-important { background: #fef9c3; color: #92400e; }
    .badge-nice { background: #dcfce7; color: #166534; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Anthropic Client ─────────────────────────────────────────────────────────
def get_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("⚠️ ANTHROPIC_API_KEY not found. Go to Replit Secrets (padlock icon) and add it.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

# ─── Prompt Filler (safe — handles JSON inside values) ───────────────────────
def fill(template, **kwargs):
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result

# ─── Prompts ──────────────────────────────────────────────────────────────────

JD_EXTRACTOR_PROMPT = """You are a precise skill extractor. Given a job description, extract ALL required skills and classify each one.

Return a JSON object in this exact format:
{
  "required_skills": [
    {
      "skill": "Python",
      "importance": "critical",
      "category": "technical",
      "jd_context": "exact phrase from JD mentioning this skill"
    }
  ]
}

Importance levels: "critical" (must-have), "important" (strongly preferred), "nice-to-have"
Categories: "technical", "soft_skill", "domain_knowledge", "tool"

Job Description:
{JD_TEXT}

Return ONLY the JSON. No explanation."""

RESUME_EXTRACTOR_PROMPT = """You are a resume analyzer. Extract all skills a candidate claims to have from their resume.

Return a JSON object in this exact format:
{
  "claimed_skills": [
    {
      "skill": "Python",
      "evidence": "Led Python data pipeline at XYZ Corp",
      "years_mentioned": 3,
      "confidence": "high"
    }
  ]
}

Confidence levels: "high" (explicitly stated with strong evidence), "medium" (mentioned without much detail), "low" (implied or vaguely referenced)

Resume text:
{RESUME_TEXT}

Return ONLY the JSON. No explanation."""

GAP_ANALYZER_PROMPT = """You are a skill gap analyzer. Compare the job requirements against what the candidate claims.

JD Required Skills (JSON): {JD_SKILLS}
Candidate Claimed Skills (JSON): {RESUME_SKILLS}

Return a JSON object:
{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3"],
  "needs_verification": [
    {
      "skill": "Python",
      "reason": "Claimed 5 years experience but no specific projects shown",
      "priority": 1
    }
  ],
  "assessment_order": ["skill_to_assess_first", "skill_to_assess_second"]
}

Prioritise "needs_verification" skills for assessment (skills that appear in both JD and resume but need proof).
Also include top missing critical skills in assessment_order so the agent can probe depth.
Limit assessment_order to a maximum of 5 skills.

Return ONLY the JSON. No explanation."""

ASSESSOR_PROMPT = """You are SkillSense, an expert technical interviewer conducting a friendly but precise skill assessment.

CONTEXT:
- Job Role: {JOB_TITLE}
- Skill being assessed: {CURRENT_SKILL}
- Candidate claimed proficiency: {CLAIMED_LEVEL}
- Conversation so far:
{CONVERSATION_HISTORY}

YOUR RULES:
1. Ask ONE question at a time. Never ask multiple questions in one message.
2. Start with a mid-difficulty question — not too easy, not too hard.
3. If the candidate answers well → increase difficulty in the next question.
4. If the candidate answers poorly or vaguely → ask a simpler follow-up to calibrate.
5. After 2-4 questions, you have enough to score. When ready, say exactly: "ASSESSMENT_COMPLETE" on a new line followed by your score JSON.
6. Be conversational and encouraging. This is not an interrogation.
7. Ask practical, real-world questions — not trivia. Ask "how would you..." not "define..."

SCORING (output when ASSESSMENT_COMPLETE):
{
  "skill": "{CURRENT_SKILL}",
  "assessed_score": 7,
  "claimed_score": 9,
  "gap": 2,
  "evidence": "Candidate understood basic concepts but struggled with async patterns",
  "verdict": "Intermediate — not at the senior level the JD requires"
}

Gap = claimed_score minus assessed_score (positive gap = overstatement, negative = understatement).

If this is the FIRST question for this skill, start with:
"Let's talk about your experience with {CURRENT_SKILL}. [Your question here]"

Current skill to assess: {CURRENT_SKILL}"""

LEARNING_PLAN_PROMPT = """You are a personalised learning advisor. Based on a candidate's skill gaps, create a realistic, actionable learning plan.

Candidate Profile:
- Target Job: {JOB_TITLE}
- Assessed Skills (JSON): {ALL_SCORES}
- Skill Gaps (JSON): {GAPS}
- Available time per week: {HOURS_PER_WEEK} hours

Generate a learning plan in this JSON format:
{
  "summary": "2-sentence overview of where the candidate stands",
  "priority_gaps": [
    {
      "skill": "Docker",
      "current_level": "beginner",
      "target_level": "intermediate",
      "time_estimate_weeks": 3,
      "weekly_hours": 5,
      "resources": [
        {
          "title": "Docker Official Getting Started",
          "url": "https://docs.docker.com/get-started/",
          "type": "documentation",
          "time_hours": 4
        },
        {
          "title": "Docker for Beginners - TechWorld with Nana",
          "url": "https://www.youtube.com/watch?v=pg19Z8LL06w",
          "type": "video",
          "time_hours": 3
        }
      ],
      "adjacent_skills": ["Kubernetes basics", "CI/CD pipelines"],
      "quick_win": "Run your first container in 30 minutes using this exercise: ..."
    }
  ],
  "total_weeks_to_job_ready": 8,
  "motivational_note": "Personalized encouraging message based on their actual strengths"
}

All resources must be real, free, and currently accessible. Return ONLY the JSON. No explanation."""

REPORT_PROMPT = """Summarize this skill assessment session into a clean report.

Candidate: {NAME}
Job Applied For: {JOB_TITLE}
All Assessment Data: {FULL_SESSION_JSON}

Format as a professional report with:
1. Executive Summary (3 sentences)
2. Skills Assessed Table (skill | claimed | assessed | gap | verdict)
3. Top 3 Strengths
4. Top 3 Critical Gaps
5. Recommended Learning Path (week-by-week for first 4 weeks)
6. Hiring Readiness Score: X/100 with explanation

Write in second person ("You demonstrated..."). Be specific, kind, and actionable."""

# ─── PDF Extraction ───────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return ""

# ─── Claude Call ──────────────────────────────────────────────────────────────
def call_claude(prompt, max_tokens=2000):
    client = get_client()
    try:
        response = client.messages.create(
            model="claude-haiku-20240307",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Claude API error: {e}")
        return ""

# ─── Robust JSON Parser ───────────────────────────────────────────────────────
def parse_json(text):
    if not text:
        return None
    text = text.strip()
    for attempt in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r'```(?:json)?\s*([\s\S]*?)```', t).group(1).strip()),
        lambda t: json.loads(re.search(r'(\{[\s\S]*\})', t).group(1)),
    ]:
        try:
            return attempt(text)
        except:
            continue
    return None

# ─── Session State ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "stage": "upload",
        "jd_text": "",
        "resume_text": "",
        "jd_skills": None,
        "resume_skills": None,
        "gap_analysis": None,
        "candidate_name": "",
        "job_title": "",
        "hours_per_week": 10,
        "assessment_queue": [],
        "current_skill": None,
        "conv_history": [],
        "chat_display": [],
        "skill_scores": [],
        "learning_plan": None,
        "report": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_claimed_level(skill):
    if not st.session_state.resume_skills:
        return "not mentioned in resume"
    for s in st.session_state.resume_skills.get("claimed_skills", []):
        if s.get("skill", "").lower() == skill.lower():
            ev = s.get("evidence", "")
            conf = s.get("confidence", "medium")
            return f"{conf} confidence — {ev}" if ev else conf
    return "not mentioned in resume"

def format_history(history):
    if not history:
        return "No conversation yet — this is the first question."
    return "\n".join(
        f"{'Interviewer' if m['role'] == 'assistant' else 'Candidate'}: {m['content']}"
        for m in history
    )

def importance_badge(imp):
    colors = {
        "critical": ("🔴", "badge-critical"),
        "important": ("🟡", "badge-important"),
        "nice-to-have": ("🟢", "badge-nice"),
    }
    icon, cls = colors.get(imp, ("⚪", ""))
    return f"{icon} {imp}"

def confidence_icon(conf):
    return {"high": "✅", "medium": "🔶", "low": "❓"}.get(conf, "⚪")

def resource_icon(rtype):
    return {"video": "🎬", "documentation": "📄", "course": "🎓",
            "book": "📚", "article": "📰"}.get(rtype, "🔗")

# ─── Stage Actions ────────────────────────────────────────────────────────────
def run_parsing():
    with st.spinner("🔍 Extracting skills from Job Description..."):
        jd_resp = call_claude(fill(JD_EXTRACTOR_PROMPT, JD_TEXT=st.session_state.jd_text))
        st.session_state.jd_skills = parse_json(jd_resp)

    with st.spinner("📄 Analysing your resume..."):
        res_resp = call_claude(fill(RESUME_EXTRACTOR_PROMPT, RESUME_TEXT=st.session_state.resume_text))
        st.session_state.resume_skills = parse_json(res_resp)

    if st.session_state.jd_skills and st.session_state.resume_skills:
        st.session_state.stage = "parsed"
        st.rerun()
    else:
        st.error("Parsing failed. Please check your documents and try again.")


def run_gap_analysis():
    with st.spinner("⚖️ Comparing JD requirements against your resume..."):
        prompt = fill(
            GAP_ANALYZER_PROMPT,
            JD_SKILLS=json.dumps(st.session_state.jd_skills),
            RESUME_SKILLS=json.dumps(st.session_state.resume_skills)
        )
        resp = call_claude(prompt)
        st.session_state.gap_analysis = parse_json(resp)

    if st.session_state.gap_analysis:
        queue = st.session_state.gap_analysis.get("assessment_order", [])
        st.session_state.assessment_queue = queue[:5]
        st.session_state.stage = "gap"
        st.rerun()
    else:
        st.error("Gap analysis failed. Please try again.")


def start_assessment():
    if not st.session_state.assessment_queue:
        st.session_state.stage = "learning"
        st.rerun()
        return

    skill = st.session_state.assessment_queue[0]
    st.session_state.current_skill = skill
    st.session_state.conv_history = []
    st.session_state.chat_display = []
    st.session_state.stage = "assessment"

    prompt = fill(
        ASSESSOR_PROMPT,
        JOB_TITLE=st.session_state.job_title,
        CURRENT_SKILL=skill,
        CLAIMED_LEVEL=get_claimed_level(skill),
        CONVERSATION_HISTORY=format_history([])
    )
    with st.spinner(f"Starting assessment for {skill}..."):
        first_q = call_claude(prompt, max_tokens=500)

    st.session_state.conv_history.append({"role": "assistant", "content": first_q})
    st.session_state.chat_display.append({"role": "assistant", "content": first_q})
    st.rerun()


def handle_answer(answer):
    st.session_state.conv_history.append({"role": "user", "content": answer})
    st.session_state.chat_display.append({"role": "user", "content": answer})

    prompt = fill(
        ASSESSOR_PROMPT,
        JOB_TITLE=st.session_state.job_title,
        CURRENT_SKILL=st.session_state.current_skill,
        CLAIMED_LEVEL=get_claimed_level(st.session_state.current_skill),
        CONVERSATION_HISTORY=format_history(st.session_state.conv_history)
    )
    with st.spinner("SkillSense is thinking..."):
        response = call_claude(prompt, max_tokens=600)

    if "ASSESSMENT_COMPLETE" in response:
        parts = response.split("ASSESSMENT_COMPLETE", 1)
        closing_text = parts[0].strip()
        score_json_text = parts[1].strip() if len(parts) > 1 else ""

        score = parse_json(score_json_text)
        if score:
            st.session_state.skill_scores.append(score)

        if closing_text:
            st.session_state.chat_display.append({"role": "assistant", "content": closing_text})

        st.session_state.assessment_queue.pop(0)

        if st.session_state.assessment_queue:
            next_skill = st.session_state.assessment_queue[0]
            st.session_state.current_skill = next_skill
            st.session_state.conv_history = []

            next_prompt = fill(
                ASSESSOR_PROMPT,
                JOB_TITLE=st.session_state.job_title,
                CURRENT_SKILL=next_skill,
                CLAIMED_LEVEL=get_claimed_level(next_skill),
                CONVERSATION_HISTORY=format_history([])
            )
            with st.spinner(f"Moving on to: {next_skill}..."):
                next_q = call_claude(next_prompt, max_tokens=500)

            st.session_state.conv_history.append({"role": "assistant", "content": next_q})
            st.session_state.chat_display.append({
                "role": "assistant",
                "content": f"---\n\n✅ **{st.session_state.skill_scores[-1]['skill']} assessment complete.**\n\n---\n\n**Now let's move on to: {next_skill}**\n\n{next_q}"
            })
        else:
            st.session_state.chat_display.append({
                "role": "assistant",
                "content": "✅ **All skills assessed! Generating your personalised learning plan...**"
            })
            st.session_state.stage = "learning"
    else:
        st.session_state.conv_history.append({"role": "assistant", "content": response})
        st.session_state.chat_display.append({"role": "assistant", "content": response})

    st.rerun()


def generate_learning_plan():
    gaps = [s for s in st.session_state.skill_scores if s.get("gap", 0) > 0]
    prompt = fill(
        LEARNING_PLAN_PROMPT,
        JOB_TITLE=st.session_state.job_title,
        ALL_SCORES=json.dumps(st.session_state.skill_scores),
        GAPS=json.dumps(gaps),
        HOURS_PER_WEEK=str(st.session_state.hours_per_week)
    )
    with st.spinner("📚 Building your personalised learning plan..."):
        resp = call_claude(prompt, max_tokens=3000)
        st.session_state.learning_plan = parse_json(resp)
    st.rerun()


def generate_report():
    session_data = {
        "candidate": st.session_state.candidate_name,
        "job_title": st.session_state.job_title,
        "jd_skills": st.session_state.jd_skills,
        "resume_skills": st.session_state.resume_skills,
        "gap_analysis": st.session_state.gap_analysis,
        "skill_scores": st.session_state.skill_scores,
        "learning_plan": st.session_state.learning_plan,
    }
    prompt = fill(
        REPORT_PROMPT,
        NAME=st.session_state.candidate_name,
        JOB_TITLE=st.session_state.job_title,
        FULL_SESSION_JSON=json.dumps(session_data, indent=2)
    )
    with st.spinner("📝 Writing your full assessment report..."):
        st.session_state.report = call_claude(prompt, max_tokens=2500)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("# 🎯 SkillSense")
st.markdown("*AI-Powered Skill Assessment & Personalised Learning Plan Agent*")
st.caption("Catalyst Hackathon — Deccan AI Experts")
st.divider()

# Progress bar
stage_order = ["upload", "parsed", "gap", "assessment", "learning", "report"]
stage_labels = ["📤 Upload", "🔍 Parse", "⚖️ Gap Analysis", "🎙️ Assessment", "📚 Learning Plan", "📄 Report"]
current_idx = stage_order.index(st.session_state.stage) if st.session_state.stage in stage_order else 0

progress_cols = st.columns(len(stage_labels))
for i, (col, label) in enumerate(zip(progress_cols, stage_labels)):
    with col:
        if i < current_idx:
            st.markdown(f"<small>✅ {label}</small>", unsafe_allow_html=True)
        elif i == current_idx:
            st.markdown(f"<small><b>{label}</b></small>", unsafe_allow_html=True)
        else:
            st.markdown(f"<small style='color:grey'>{label}</small>", unsafe_allow_html=True)

st.progress((current_idx) / (len(stage_order) - 1))
st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":
    st.subheader("Step 1 — Upload Your Documents")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**👤 Your Details**")
        name = st.text_input("Full Name", placeholder="John Doe")
        title = st.text_input("Job Title you're applying for", placeholder="Senior Data Engineer")
        hours = st.slider("Hours you can study per week", 2, 20, 10)

    with col2:
        st.markdown("**📁 Upload Documents**")
        jd_file = st.file_uploader("Job Description (PDF or TXT)", type=["pdf", "txt"])
        resume_file = st.file_uploader("Your Resume / CV (PDF or TXT)", type=["pdf", "txt"])

    with st.expander("📋 Or paste text directly (no PDF needed)"):
        jd_paste = st.text_area("Paste Job Description here", height=150,
                                placeholder="Copy and paste the full job description text...")
        resume_paste = st.text_area("Paste Resume here", height=150,
                                    placeholder="Copy and paste your full resume text...")

    st.markdown("")
    if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
        if not name.strip() or not title.strip():
            st.error("Please enter your name and the job title.")
        else:
            jd_text, resume_text = "", ""

            if jd_file:
                jd_text = extract_pdf_text(jd_file) if jd_file.type == "application/pdf" \
                    else jd_file.read().decode("utf-8")
            elif jd_paste.strip():
                jd_text = jd_paste.strip()

            if resume_file:
                resume_file.seek(0)
                resume_text = extract_pdf_text(resume_file) if resume_file.type == "application/pdf" \
                    else resume_file.read().decode("utf-8")
            elif resume_paste.strip():
                resume_text = resume_paste.strip()

            if not jd_text:
                st.error("Please upload or paste a Job Description.")
            elif not resume_text:
                st.error("Please upload or paste your Resume.")
            else:
                st.session_state.candidate_name = name.strip()
                st.session_state.job_title = title.strip()
                st.session_state.hours_per_week = hours
                st.session_state.jd_text = jd_text
                st.session_state.resume_text = resume_text
                run_parsing()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: PARSED
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "parsed":
    st.subheader("Step 2 — Skills Extracted")
    st.success("Documents parsed successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Skills the JD Requires**")
        jd_skills = st.session_state.jd_skills or {}
        for s in jd_skills.get("required_skills", []):
            imp = s.get("importance", "")
            icon = {"critical": "🔴", "important": "🟡", "nice-to-have": "🟢"}.get(imp, "⚪")
            cat = s.get("category", "")
            ctx = s.get("jd_context", "")
            st.markdown(f"{icon} **{s['skill']}** `{imp}` · *{cat}*")
            if ctx:
                st.caption(f"  → \"{ctx}\"")

    with col2:
        st.markdown("**📄 Skills You Claim (from resume)**")
        res_skills = st.session_state.resume_skills or {}
        for s in res_skills.get("claimed_skills", []):
            conf = s.get("confidence", "")
            icon = confidence_icon(conf)
            ev = s.get("evidence", "")
            st.markdown(f"{icon} **{s['skill']}** · *{conf} confidence*")
            if ev:
                st.caption(f"  → \"{ev}\"")

    st.divider()
    if st.button("▶️ Run Skill Gap Analysis", type="primary", use_container_width=True):
        run_gap_analysis()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "gap":
    st.subheader("Step 3 — Skill Gap Analysis")

    gap = st.session_state.gap_analysis or {}

    col1, col2, col3 = st.columns(3)
    with col1:
        matched = gap.get("matched_skills", [])
        st.metric("Matched Skills", len(matched))
        st.markdown("**✅ You already have these:**")
        for s in matched:
            st.markdown(f"- {s}")

    with col2:
        missing = gap.get("missing_skills", [])
        st.metric("Missing Skills", len(missing))
        st.markdown("**❌ Not on your resume:**")
        for s in missing:
            st.markdown(f"- {s}")

    with col3:
        needs_v = gap.get("needs_verification", [])
        st.metric("Needs Verification", len(needs_v))
        st.markdown("**🔍 Claimed but unproven:**")
        for s in needs_v:
            st.markdown(f"- **{s['skill']}**: {s.get('reason', '')}")

    st.divider()
    queue = st.session_state.assessment_queue
    if queue:
        st.info(
            f"**The agent will now assess {len(queue)} skills conversationally:**\n\n"
            + " → ".join(queue)
            + "\n\nExpect 2–4 questions per skill. Answer honestly — the learning plan works best with accurate scores."
        )
        if st.button("🎙️ Begin Conversational Assessment", type="primary", use_container_width=True):
            start_assessment()
    else:
        st.warning("No skills identified for assessment. Try re-running with more detailed documents.")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: ASSESSMENT (Chat Interface)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "assessment":

    # Sidebar progress
    with st.sidebar:
        st.markdown("### 📊 Assessment Progress")
        total = len(st.session_state.skill_scores) + len(st.session_state.assessment_queue)
        done = len(st.session_state.skill_scores)
        if total > 0:
            st.progress(done / total)
            st.caption(f"{done} of {total} skills done")

        if st.session_state.skill_scores:
            st.markdown("**Completed:**")
            for sc in st.session_state.skill_scores:
                g = sc.get("gap", 0)
                status = "🟢" if g <= 1 else "🟡" if g <= 3 else "🔴"
                st.markdown(f"{status} {sc['skill']} — {sc.get('assessed_score','?')}/10")

        if st.session_state.assessment_queue:
            st.markdown("**Remaining:**")
            for i, s in enumerate(st.session_state.assessment_queue):
                marker = "👉 " if i == 0 else "  "
                st.markdown(f"{marker}{s}")

    # Main chat area
    st.subheader(f"🎙️ Now Assessing: **{st.session_state.current_skill}**")
    st.caption(f"{st.session_state.candidate_name} · Applying for: {st.session_state.job_title}")
    st.divider()

    # Render chat
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Input
    if answer := st.chat_input("Type your answer here and press Enter..."):
        handle_answer(answer)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: LEARNING PLAN
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "learning":
    st.subheader("🎉 Assessment Complete!")
    st.success(f"All skills assessed for **{st.session_state.candidate_name}**")

    # Scores summary table
    st.markdown("### 📊 Your Skill Scores")
    if st.session_state.skill_scores:
        cols = st.columns([2, 1, 1, 1, 3])
        cols[0].markdown("**Skill**")
        cols[1].markdown("**Claimed**")
        cols[2].markdown("**Assessed**")
        cols[3].markdown("**Gap**")
        cols[4].markdown("**Verdict**")
        st.divider()
        for sc in st.session_state.skill_scores:
            c = st.columns([2, 1, 1, 1, 3])
            gap_val = sc.get("gap", 0)
            gap_color = "🟢" if gap_val <= 1 else "🟡" if gap_val <= 3 else "🔴"
            c[0].markdown(f"**{sc.get('skill','?')}**")
            c[1].markdown(f"{sc.get('claimed_score','?')}/10")
            c[2].markdown(f"{sc.get('assessed_score','?')}/10")
            c[3].markdown(f"{gap_color} {gap_val}")
            c[4].markdown(f"*{sc.get('verdict','?')}*")

    st.divider()

    if not st.session_state.learning_plan:
        if st.button("📚 Generate Personalised Learning Plan", type="primary", use_container_width=True):
            generate_learning_plan()
    else:
        plan = st.session_state.learning_plan

        st.markdown("### 📚 Your Personalised Learning Plan")
        st.info(plan.get("summary", ""))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("⏱️ Weeks to Job-Ready", plan.get("total_weeks_to_job_ready", "?"))
        with col2:
            st.markdown(f"💬 *{plan.get('motivational_note', '')}*")

        st.markdown("### 🎯 Skills to Work On")
        for item in plan.get("priority_gaps", []):
            with st.expander(
                f"📖 {item['skill']} — {item.get('time_estimate_weeks','?')} weeks · "
                f"{item.get('weekly_hours','?')}h/week"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Level", item.get("current_level", "?"))
                c2.metric("Target Level", item.get("target_level", "?"))
                c3.metric("Total Hours", item.get("time_estimate_weeks", 0) * item.get("weekly_hours", 0))

                if item.get("quick_win"):
                    st.success(f"⚡ **Quick Win:** {item['quick_win']}")

                st.markdown("**📚 Resources:**")
                for res in item.get("resources", []):
                    icon = resource_icon(res.get("type", ""))
                    st.markdown(
                        f"{icon} [{res['title']}]({res['url']}) "
                        f"· *{res.get('type','')}* · ~{res.get('time_hours','?')}h"
                    )

                if item.get("adjacent_skills"):
                    st.markdown(
                        f"**🔗 Adjacent skills to explore:** "
                        + ", ".join(item["adjacent_skills"])
                    )

        st.divider()
        if not st.session_state.report:
            if st.button("📄 Generate Full Assessment Report", type="primary", use_container_width=True):
                generate_report()
        else:
            st.markdown("### 📄 Full Assessment Report")
            st.markdown(st.session_state.report)
            st.divider()

            report_filename = (
                f"skillsense_report_"
                f"{st.session_state.candidate_name.replace(' ','_')}_"
                f"{datetime.now().strftime('%Y%m%d')}.txt"
            )
            st.download_button(
                label="⬇️ Download Report",
                data=st.session_state.report,
                file_name=report_filename,
                mime="text/plain",
                use_container_width=True
            )

            if st.button("🔄 Start a New Assessment", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("SkillSense · Built for Catalyst Hackathon — Deccan AI Experts · Powered by Claude")
