import streamlit as st
import json
import io
import re
import PyPDF2
from datetime import datetime
from groq import Groq

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkillSense — AI Skill Assessment Agent",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .stProgress > div > div { background-color: #4F46E5; }
</style>
""", unsafe_allow_html=True)

# ─── Groq Client ──────────────────────────────────────────────────────────────
def get_client():
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Add it in Streamlit Secrets.")
        st.stop()
    return Groq(api_key=api_key)

def call_llm(prompt, max_tokens=2000, messages=None):
    client = get_client()
    try:
        msgs = messages if messages else [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=msgs
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API error: {e}")
        return ""

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

RESUME_EXTRACTOR_PROMPT = """You are a resume analyzer. Extract all skills a candidate claims to have.

Return a JSON object:
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

Confidence: "high" (explicitly stated with evidence), "medium" (mentioned without detail), "low" (implied)

Resume text:
{RESUME_TEXT}

Return ONLY the JSON. No explanation."""

GAP_ANALYZER_PROMPT = """You are a skill gap analyzer. Compare job requirements against candidate claims.

JD Required Skills (JSON): {JD_SKILLS}
Candidate Claimed Skills (JSON): {RESUME_SKILLS}

Return a JSON object:
{
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3"],
  "needs_verification": [
    {
      "skill": "Python",
      "reason": "Claimed but needs proof",
      "priority": 1
    }
  ],
  "assessment_order": ["skill_to_assess_first", "skill_to_assess_second"]
}

Limit assessment_order to maximum 4 skills. Return ONLY the JSON. No explanation."""

ASSESSOR_PROMPT = """You are SkillSense, an expert technical interviewer conducting a friendly but precise skill assessment.

CONTEXT:
- Job Role: {JOB_TITLE}
- Skill being assessed: {CURRENT_SKILL}
- Candidate claimed proficiency: {CLAIMED_LEVEL}
- Conversation so far:
{CONVERSATION_HISTORY}

YOUR RULES:
1. Ask ONE question at a time.
2. Start with a mid-difficulty question.
3. If candidate answers well, increase difficulty next time.
4. If candidate answers poorly, ask a simpler follow-up.
5. After 2-4 questions, say exactly "ASSESSMENT_COMPLETE" on a new line then your score JSON.
6. Be conversational and encouraging.
7. Ask practical "how would you..." questions, not definitions.

SCORING JSON format (after ASSESSMENT_COMPLETE):
{
  "skill": "{CURRENT_SKILL}",
  "assessed_score": 7,
  "claimed_score": 8,
  "gap": 1,
  "evidence": "Candidate showed solid understanding",
  "verdict": "Intermediate level — meets role requirements"
}

If FIRST question, start with: "Let's talk about your experience with {CURRENT_SKILL}. [question]"

Current skill to assess: {CURRENT_SKILL}"""

LEARNING_PLAN_PROMPT = """You are a personalised learning advisor. Create a realistic learning plan based on skill gaps.

Candidate Profile:
- Target Job: {JOB_TITLE}
- Assessed Skills: {ALL_SCORES}
- Skill Gaps: {GAPS}
- Available time per week: {HOURS_PER_WEEK} hours

Return this JSON:
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
        }
      ],
      "adjacent_skills": ["Kubernetes basics"],
      "quick_win": "Run your first container in 30 minutes"
    }
  ],
  "total_weeks_to_job_ready": 8,
  "motivational_note": "Encouraging personalised message"
}

All resources must be real and free. Return ONLY the JSON."""

REPORT_PROMPT = """Summarize this skill assessment into a professional report.

Candidate: {NAME}
Job Applied For: {JOB_TITLE}
Assessment Data: {FULL_SESSION_JSON}

Format:
1. Executive Summary (3 sentences)
2. Skills Assessed Table (skill | claimed | assessed | gap | verdict)
3. Top 3 Strengths
4. Top 3 Critical Gaps
5. Recommended Learning Path (week-by-week, 4 weeks)
6. Hiring Readiness Score: X/100 with explanation

Write in second person. Be specific, kind, actionable."""

# ─── Helpers ──────────────────────────────────────────────────────────────────
def fill(template, **kwargs):
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result

def extract_pdf_text(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return ""

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

def init_state():
    defaults = {
        "stage": "upload",
        "jd_text": "", "resume_text": "",
        "jd_skills": None, "resume_skills": None, "gap_analysis": None,
        "candidate_name": "", "job_title": "", "hours_per_week": 10,
        "assessment_queue": [], "current_skill": None,
        "conv_history": [], "chat_display": [], "skill_scores": [],
        "learning_plan": None, "report": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def get_claimed_level(skill):
    if not st.session_state.resume_skills:
        return "not mentioned in resume"
    for s in st.session_state.resume_skills.get("claimed_skills", []):
        if s.get("skill", "").lower() == skill.lower():
            conf = s.get("confidence", "medium")
            ev = s.get("evidence", "")
            return f"{conf} confidence — {ev}" if ev else conf
    return "not mentioned in resume"

def format_history(history):
    if not history:
        return "No conversation yet — this is the first question."
    return "\n".join(
        f"{'Interviewer' if m['role'] == 'assistant' else 'Candidate'}: {m['content']}"
        for m in history
    )

def resource_icon(rtype):
    return {"video": "🎬", "documentation": "📄", "course": "🎓",
            "book": "📚", "article": "📰"}.get(rtype, "🔗")

# ─── Stage Actions ────────────────────────────────────────────────────────────
def run_parsing():
    with st.spinner("🔍 Extracting skills from Job Description..."):
        jd_resp = call_llm(fill(JD_EXTRACTOR_PROMPT, JD_TEXT=st.session_state.jd_text))
        st.session_state.jd_skills = parse_json(jd_resp)
    with st.spinner("📄 Analysing your resume..."):
        res_resp = call_llm(fill(RESUME_EXTRACTOR_PROMPT, RESUME_TEXT=st.session_state.resume_text))
        st.session_state.resume_skills = parse_json(res_resp)
    if st.session_state.jd_skills and st.session_state.resume_skills:
        st.session_state.stage = "parsed"
        st.rerun()
    else:
        st.error("Parsing failed. Please check your documents and try again.")

def run_gap_analysis():
    with st.spinner("⚖️ Comparing JD requirements against your resume..."):
        prompt = fill(GAP_ANALYZER_PROMPT,
                      JD_SKILLS=json.dumps(st.session_state.jd_skills),
                      RESUME_SKILLS=json.dumps(st.session_state.resume_skills))
        resp = call_llm(prompt)
        st.session_state.gap_analysis = parse_json(resp)
    if st.session_state.gap_analysis:
        st.session_state.assessment_queue = st.session_state.gap_analysis.get("assessment_order", [])[:4]
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
    prompt = fill(ASSESSOR_PROMPT,
                  JOB_TITLE=st.session_state.job_title,
                  CURRENT_SKILL=skill,
                  CLAIMED_LEVEL=get_claimed_level(skill),
                  CONVERSATION_HISTORY=format_history([]))
   system_prompt = fill(ASSESSOR_PROMPT,
                  JOB_TITLE=st.session_state.job_title,
                  CURRENT_SKILL=skill,
                  CLAIMED_LEVEL=get_claimed_level(skill),
                  CONVERSATION_HISTORY="")
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please begin the assessment for {skill}."}]
    with st.spinner(f"Starting assessment for {skill}..."):
        first_q = call_llm("", max_tokens=400, messages=msgs)
    st.session_state.conv_history.append({"role": "assistant", "content": first_q})
    st.session_state.chat_display.append({"role": "assistant", "content": first_q})
    st.rerun()

def handle_answer(answer):
    st.session_state.conv_history.append({"role": "user", "content": answer})
    st.session_state.chat_display.append({"role": "user", "content": answer})
  system_prompt = fill(ASSESSOR_PROMPT,
                  JOB_TITLE=st.session_state.job_title,
                  CURRENT_SKILL=st.session_state.current_skill,
                  CLAIMED_LEVEL=get_claimed_level(st.session_state.current_skill),
                  CONVERSATION_HISTORY="")
    msgs = [{"role": "system", "content": system_prompt}] + st.session_state.conv_history
    with st.spinner("SkillSense is thinking..."):
        response = call_llm("", max_tokens=500, messages=msgs)
    if "ASSESSMENT_COMPLETE" in response:
        parts = response.split("ASSESSMENT_COMPLETE", 1)
        closing_text = parts[0].strip()
        score = parse_json(parts[1].strip() if len(parts) > 1 else "")
        if score:
            st.session_state.skill_scores.append(score)
        if closing_text:
            st.session_state.chat_display.append({"role": "assistant", "content": closing_text})
        st.session_state.assessment_queue.pop(0)
        if st.session_state.assessment_queue:
            next_skill = st.session_state.assessment_queue[0]
            st.session_state.current_skill = next_skill
            st.session_state.conv_history = []
            next_prompt = fill(ASSESSOR_PROMPT,
                               JOB_TITLE=st.session_state.job_title,
                               CURRENT_SKILL=next_skill,
                               CLAIMED_LEVEL=get_claimed_level(next_skill),
                               CONVERSATION_HISTORY=format_history([]))
            with st.spinner(f"Moving to: {next_skill}..."):
                next_q = call_llm(next_prompt, max_tokens=400)
            st.session_state.conv_history.append({"role": "assistant", "content": next_q})
            last_score = st.session_state.skill_scores[-1] if st.session_state.skill_scores else {}
            st.session_state.chat_display.append({
                "role": "assistant",
                "content": f"---\n\n✅ **{last_score.get('skill','Previous')} assessment complete.**\n\n---\n\n**Now: {next_skill}**\n\n{next_q}"
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
    prompt = fill(LEARNING_PLAN_PROMPT,
                  JOB_TITLE=st.session_state.job_title,
                  ALL_SCORES=json.dumps(st.session_state.skill_scores),
                  GAPS=json.dumps(gaps),
                  HOURS_PER_WEEK=str(st.session_state.hours_per_week))
    with st.spinner("📚 Building your personalised learning plan..."):
        resp = call_llm(prompt, max_tokens=3000)
        st.session_state.learning_plan = parse_json(resp)
    st.rerun()

def generate_report():
    session_data = {
        "candidate": st.session_state.candidate_name,
        "job_title": st.session_state.job_title,
        "skill_scores": st.session_state.skill_scores,
        "learning_plan": st.session_state.learning_plan,
    }
    prompt = fill(REPORT_PROMPT,
                  NAME=st.session_state.candidate_name,
                  JOB_TITLE=st.session_state.job_title,
                  FULL_SESSION_JSON=json.dumps(session_data))
    with st.spinner("📝 Writing your full assessment report..."):
        st.session_state.report = call_llm(prompt, max_tokens=2500)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎯 SkillSense")
st.markdown("*AI-Powered Skill Assessment & Personalised Learning Plan Agent*")
st.caption("Catalyst Hackathon — Deccan AI Experts")
st.divider()

stage_order = ["upload", "parsed", "gap", "assessment", "learning", "report"]
stage_labels = ["📤 Upload", "🔍 Parse", "⚖️ Gap Analysis", "🎙️ Assessment", "📚 Learning Plan", "📄 Report"]
current_idx = stage_order.index(st.session_state.stage) if st.session_state.stage in stage_order else 0

cols = st.columns(len(stage_labels))
for i, (col, label) in enumerate(zip(cols, stage_labels)):
    with col:
        if i < current_idx:
            st.markdown(f"<small>✅ {label}</small>", unsafe_allow_html=True)
        elif i == current_idx:
            st.markdown(f"<small><b>{label}</b></small>", unsafe_allow_html=True)
        else:
            st.markdown(f"<small style='color:grey'>{label}</small>", unsafe_allow_html=True)
st.progress(current_idx / (len(stage_order) - 1))
st.markdown("")

# ── STAGE 1: UPLOAD ───────────────────────────────────────────────────────────
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
        jd_paste = st.text_area("Paste Job Description here", height=150)
        resume_paste = st.text_area("Paste Resume here", height=150)
    st.markdown("")
    if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
        if not name.strip() or not title.strip():
            st.error("Please enter your name and the job title.")
        else:
            jd_text, resume_text = "", ""
            if jd_file:
                jd_text = extract_pdf_text(jd_file) if jd_file.type == "application/pdf" else jd_file.read().decode("utf-8")
            elif jd_paste.strip():
                jd_text = jd_paste.strip()
            if resume_file:
                resume_file.seek(0)
                resume_text = extract_pdf_text(resume_file) if resume_file.type == "application/pdf" else resume_file.read().decode("utf-8")
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

# ── STAGE 2: PARSED ───────────────────────────────────────────────────────────
elif st.session_state.stage == "parsed":
    st.subheader("Step 2 — Skills Extracted")
    st.success("Documents parsed successfully!")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Skills the JD Requires**")
        for s in (st.session_state.jd_skills or {}).get("required_skills", []):
            icon = {"critical": "🔴", "important": "🟡", "nice-to-have": "🟢"}.get(s.get("importance", ""), "⚪")
            st.markdown(f"{icon} **{s['skill']}** `{s.get('importance','')}` · *{s.get('category','')}*")
            if s.get("jd_context"):
                st.caption(f"  → \"{s['jd_context']}\"")
    with col2:
        st.markdown("**📄 Skills You Claim**")
        for s in (st.session_state.resume_skills or {}).get("claimed_skills", []):
            icon = {"high": "✅", "medium": "🔶", "low": "❓"}.get(s.get("confidence", ""), "⚪")
            st.markdown(f"{icon} **{s['skill']}** · *{s.get('confidence','')} confidence*")
            if s.get("evidence"):
                st.caption(f"  → \"{s['evidence']}\"")
    st.divider()
    if st.button("▶️ Run Skill Gap Analysis", type="primary", use_container_width=True):
        run_gap_analysis()

# ── STAGE 3: GAP ANALYSIS ─────────────────────────────────────────────────────
elif st.session_state.stage == "gap":
    st.subheader("Step 3 — Skill Gap Analysis")
    gap = st.session_state.gap_analysis or {}
    col1, col2, col3 = st.columns(3)
    with col1:
        matched = gap.get("matched_skills", [])
        st.metric("Matched Skills", len(matched))
        st.markdown("**✅ Already have:**")
        for s in matched:
            st.markdown(f"- {s}")
    with col2:
        missing = gap.get("missing_skills", [])
        st.metric("Missing Skills", len(missing))
        st.markdown("**❌ Not on resume:**")
        for s in missing:
            st.markdown(f"- {s}")
    with col3:
        needs_v = gap.get("needs_verification", [])
        st.metric("Needs Verification", len(needs_v))
        st.markdown("**🔍 Claimed but unproven:**")
        for s in needs_v:
            st.markdown(f"- **{s['skill']}**: {s.get('reason','')}")
    st.divider()
    queue = st.session_state.assessment_queue
    if queue:
        st.info(f"**Will assess {len(queue)} skills:** " + " → ".join(queue))
        if st.button("🎙️ Begin Conversational Assessment", type="primary", use_container_width=True):
            start_assessment()

# ── STAGE 4: ASSESSMENT ───────────────────────────────────────────────────────
elif st.session_state.stage == "assessment":
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
                icon = "🟢" if g <= 1 else "🟡" if g <= 3 else "🔴"
                st.markdown(f"{icon} {sc['skill']} — {sc.get('assessed_score','?')}/10")
        if st.session_state.assessment_queue:
            st.markdown("**Remaining:**")
            for i, s in enumerate(st.session_state.assessment_queue):
                st.markdown(f"{'👉 ' if i == 0 else '  '}{s}")
    st.subheader(f"🎙️ Now Assessing: **{st.session_state.current_skill}**")
    st.caption(f"{st.session_state.candidate_name} · {st.session_state.job_title}")
    st.divider()
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
    if answer := st.chat_input("Type your answer here and press Enter..."):
        handle_answer(answer)

# ── STAGE 5: LEARNING PLAN ────────────────────────────────────────────────────
elif st.session_state.stage == "learning":
    st.subheader("🎉 Assessment Complete!")
    st.success(f"All skills assessed for **{st.session_state.candidate_name}**")
    st.markdown("### 📊 Your Skill Scores")
    if st.session_state.skill_scores:
        cols = st.columns([2, 1, 1, 1, 3])
        for c, h in zip(cols, ["**Skill**", "**Claimed**", "**Assessed**", "**Gap**", "**Verdict**"]):
            c.markdown(h)
        st.divider()
        for sc in st.session_state.skill_scores:
            g = sc.get("gap", 0)
            icon = "🟢" if g <= 1 else "🟡" if g <= 3 else "🔴"
            c = st.columns([2, 1, 1, 1, 3])
            c[0].markdown(f"**{sc.get('skill','?')}**")
            c[1].markdown(f"{sc.get('claimed_score','?')}/10")
            c[2].markdown(f"{sc.get('assessed_score','?')}/10")
            c[3].markdown(f"{icon} {g}")
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
            with st.expander(f"📖 {item['skill']} — {item.get('time_estimate_weeks','?')} weeks"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Current", item.get("current_level", "?"))
                c2.metric("Target", item.get("target_level", "?"))
                c3.metric("Hours Total", item.get("time_estimate_weeks", 0) * item.get("weekly_hours", 0))
                if item.get("quick_win"):
                    st.success(f"⚡ **Quick Win:** {item['quick_win']}")
                st.markdown("**📚 Resources:**")
                for res in item.get("resources", []):
                    icon = resource_icon(res.get("type", ""))
                    st.markdown(f"{icon} [{res['title']}]({res['url']}) · ~{res.get('time_hours','?')}h")
                if item.get("adjacent_skills"):
                    st.markdown("**🔗 Adjacent skills:** " + ", ".join(item["adjacent_skills"]))
        st.divider()
        if not st.session_state.report:
            if st.button("📄 Generate Full Assessment Report", type="primary", use_container_width=True):
                generate_report()
        else:
            st.markdown("### 📄 Full Assessment Report")
            st.markdown(st.session_state.report)
            st.divider()
            st.download_button(
                "⬇️ Download Report",
                data=st.session_state.report,
                file_name=f"skillsense_{st.session_state.candidate_name.replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            if st.button("🔄 Start New Assessment", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

st.divider()
st.caption("SkillSense · Catalyst Hackathon — Deccan AI Experts · Powered by Llama 3 via Groq")
