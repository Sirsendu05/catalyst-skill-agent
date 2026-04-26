import streamlit as st
import json
import io
import re
import PyPDF2
from groq import Groq

st.set_page_config(page_title="SkillSense — AI Skill Assessment Agent", page_icon="🎯", layout="wide")

st.markdown("<style>.stProgress > div > div { background-color: #4F46E5; }</style>", unsafe_allow_html=True)

# ── Groq Client ───────────────────────────────────────────────────────────────
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

# ── Prompts ───────────────────────────────────────────────────────────────────
JD_EXTRACTOR_PROMPT = """You are a precise skill extractor. Given a job description, extract ALL required skills.

Return ONLY this JSON format, nothing else:
{
  "required_skills": [
    {
      "skill": "Python",
      "importance": "critical",
      "category": "technical",
      "jd_context": "phrase from JD"
    }
  ]
}

Importance: "critical", "important", "nice-to-have"
Category: "technical", "soft_skill", "domain_knowledge", "tool"

Job Description:
{JD_TEXT}"""

RESUME_EXTRACTOR_PROMPT = """You are a resume analyzer. Extract all skills from this resume.

Return ONLY this JSON format, nothing else:
{
  "claimed_skills": [
    {
      "skill": "Python",
      "evidence": "Built data pipelines at XYZ Corp",
      "years_mentioned": 3,
      "confidence": "high"
    }
  ]
}

Confidence: "high", "medium", "low"

Resume:
{RESUME_TEXT}"""

GAP_ANALYZER_PROMPT = """Compare job requirements vs candidate skills and return ONLY this JSON:
{
  "matched_skills": ["skill1"],
  "missing_skills": ["skill2"],
  "needs_verification": [
    {"skill": "Python", "reason": "Claimed but no project evidence", "priority": 1}
  ],
  "assessment_order": ["Python", "SQL"]
}

Limit assessment_order to 3 skills max.

JD Skills: {JD_SKILLS}
Resume Skills: {RESUME_SKILLS}"""

ASSESSOR_SYSTEM_PROMPT = """You are SkillSense, a friendly but precise technical interviewer.

Job Role: {JOB_TITLE}
Skill being assessed: {CURRENT_SKILL}
Candidate claimed level: {CLAIMED_LEVEL}

RULES:
1. Ask ONE question at a time — never multiple.
2. Start mid-difficulty. Increase if answered well, decrease if answered poorly.
3. After 2-4 exchanges, output ASSESSMENT_COMPLETE on its own line, then this JSON:
{{
  "skill": "{CURRENT_SKILL}",
  "assessed_score": 7,
  "claimed_score": 8,
  "gap": 1,
  "evidence": "what you observed",
  "verdict": "one sentence verdict"
}}
4. Be encouraging and conversational.
5. Ask "how would you..." questions, not definitions.
6. For the very first message, start with: "Let's talk about your experience with {CURRENT_SKILL}. [question]"
"""

LEARNING_PLAN_PROMPT = """Create a personalised learning plan. Return ONLY this JSON:
{{
  "summary": "2-sentence summary",
  "priority_gaps": [
    {{
      "skill": "Docker",
      "current_level": "beginner",
      "target_level": "intermediate",
      "time_estimate_weeks": 3,
      "weekly_hours": 5,
      "resources": [
        {{"title": "Docker Getting Started", "url": "https://docs.docker.com/get-started/", "type": "documentation", "time_hours": 4}}
      ],
      "adjacent_skills": ["Kubernetes"],
      "quick_win": "Run your first container today"
    }}
  ],
  "total_weeks_to_job_ready": 8,
  "motivational_note": "encouraging message"
}}

Target Job: {JOB_TITLE}
Assessed Skills: {ALL_SCORES}
Gaps: {GAPS}
Weekly hours available: {HOURS_PER_WEEK}"""

REPORT_PROMPT = """Write a professional skill assessment report for:
Candidate: {NAME}
Job: {JOB_TITLE}
Data: {FULL_SESSION_JSON}

Include:
1. Executive Summary (3 sentences, second person)
2. Skills Table (skill | claimed/10 | assessed/10 | gap | verdict)
3. Top 3 Strengths
4. Top 3 Critical Gaps
5. Week-by-week learning path (4 weeks)
6. Hiring Readiness Score: X/100 with explanation"""

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    for attempt in [
        lambda t: json.loads(t.strip()),
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
        return "not mentioned"
    for s in st.session_state.resume_skills.get("claimed_skills", []):
        if s.get("skill", "").lower() == skill.lower():
            return f"{s.get('confidence','medium')} confidence — {s.get('evidence','')}"
    return "not mentioned in resume"

def resource_icon(rtype):
    return {"video": "🎬", "documentation": "📄", "course": "🎓", "book": "📚"}.get(rtype, "🔗")

# ── Stage Functions ───────────────────────────────────────────────────────────
def run_parsing():
    with st.spinner("🔍 Extracting skills from Job Description..."):
        jd_resp = call_llm(fill(JD_EXTRACTOR_PROMPT, JD_TEXT=st.session_state.jd_text))
        st.session_state.jd_skills = parse_json(jd_resp)
    with st.spinner("📄 Analysing resume..."):
        res_resp = call_llm(fill(RESUME_EXTRACTOR_PROMPT, RESUME_TEXT=st.session_state.resume_text))
        st.session_state.resume_skills = parse_json(res_resp)
    if st.session_state.jd_skills and st.session_state.resume_skills:
        st.session_state.stage = "parsed"
        st.rerun()
    else:
        st.error("Parsing failed. Please check your documents and try again.")

def run_gap_analysis():
    with st.spinner("⚖️ Analysing skill gaps..."):
        resp = call_llm(fill(GAP_ANALYZER_PROMPT,
                             JD_SKILLS=json.dumps(st.session_state.jd_skills),
                             RESUME_SKILLS=json.dumps(st.session_state.resume_skills)))
        st.session_state.gap_analysis = parse_json(resp)
    if st.session_state.gap_analysis:
        st.session_state.assessment_queue = st.session_state.gap_analysis.get("assessment_order", [])[:3]
        st.session_state.stage = "gap"
        st.rerun()
    else:
        st.error("Gap analysis failed. Please try again.")

def build_assessor_messages(skill, history):
    system = fill(ASSESSOR_SYSTEM_PROMPT,
                  JOB_TITLE=st.session_state.job_title,
                  CURRENT_SKILL=skill,
                  CLAIMED_LEVEL=get_claimed_level(skill))
    msgs = [{"role": "system", "content": system}]
    if not history:
        msgs.append({"role": "user", "content": f"Please begin the assessment for {skill}."})
    else:
        msgs.extend(history)
    return msgs

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
    msgs = build_assessor_messages(skill, [])
    with st.spinner(f"Starting assessment for {skill}..."):
        first_q = call_llm("", max_tokens=400, messages=msgs)
    st.session_state.conv_history.append({"role": "assistant", "content": first_q})
    st.session_state.chat_display.append({"role": "assistant", "content": first_q})
    st.rerun()

def handle_answer(answer):
    st.session_state.conv_history.append({"role": "user", "content": answer})
    st.session_state.chat_display.append({"role": "user", "content": answer})

    msgs = build_assessor_messages(st.session_state.current_skill, st.session_state.conv_history)
    with st.spinner("SkillSense is thinking..."):
        response = call_llm("", max_tokens=500, messages=msgs)

    if "ASSESSMENT_COMPLETE" in response:
        parts = response.split("ASSESSMENT_COMPLETE", 1)
        closing = parts[0].strip()
        score = parse_json(parts[1].strip() if len(parts) > 1 else "")
        if score:
            st.session_state.skill_scores.append(score)
        if closing:
            st.session_state.chat_display.append({"role": "assistant", "content": closing})
        st.session_state.assessment_queue.pop(0)

        if st.session_state.assessment_queue:
            next_skill = st.session_state.assessment_queue[0]
            st.session_state.current_skill = next_skill
            st.session_state.conv_history = []
            next_msgs = build_assessor_messages(next_skill, [])
            with st.spinner(f"Moving to: {next_skill}..."):
                next_q = call_llm("", max_tokens=400, messages=next_msgs)
            st.session_state.conv_history.append({"role": "assistant", "content": next_q})
            last = st.session_state.skill_scores[-1] if st.session_state.skill_scores else {}
            st.session_state.chat_display.append({
                "role": "assistant",
                "content": f"---\n✅ **{last.get('skill','Previous')} done.**\n\n---\n**Next: {next_skill}**\n\n{next_q}"
            })
        else:
            st.session_state.chat_display.append({
                "role": "assistant",
                "content": "✅ **All skills assessed! Click below to generate your learning plan.**"
            })
            st.session_state.stage = "learning"
    else:
        st.session_state.conv_history.append({"role": "assistant", "content": response})
        st.session_state.chat_display.append({"role": "assistant", "content": response})
    st.rerun()

def generate_learning_plan():
    gaps = [s for s in st.session_state.skill_scores if s.get("gap", 0) > 0]
    with st.spinner("📚 Building learning plan..."):
        resp = call_llm(fill(LEARNING_PLAN_PROMPT,
                             JOB_TITLE=st.session_state.job_title,
                             ALL_SCORES=json.dumps(st.session_state.skill_scores),
                             GAPS=json.dumps(gaps),
                             HOURS_PER_WEEK=str(st.session_state.hours_per_week)),
                        max_tokens=3000)
        st.session_state.learning_plan = parse_json(resp)
    st.rerun()

def generate_report():
    with st.spinner("📝 Writing report..."):
        st.session_state.report = call_llm(fill(REPORT_PROMPT,
                                                NAME=st.session_state.candidate_name,
                                                JOB_TITLE=st.session_state.job_title,
                                                FULL_SESSION_JSON=json.dumps({
                                                    "scores": st.session_state.skill_scores,
                                                    "plan": st.session_state.learning_plan
                                                })), max_tokens=2500)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🎯 SkillSense")
st.markdown("*AI-Powered Skill Assessment & Personalised Learning Plan Agent*")
st.caption("Catalyst Hackathon — Deccan AI Experts")
st.divider()

stage_order = ["upload", "parsed", "gap", "assessment", "learning"]
stage_labels = ["📤 Upload", "🔍 Parse", "⚖️ Gap Analysis", "🎙️ Assessment", "📚 Learning Plan"]
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

# ── UPLOAD ────────────────────────────────────────────────────────────────────
if st.session_state.stage == "upload":
    st.subheader("Step 1 — Upload Your Documents")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**👤 Your Details**")
        name = st.text_input("Full Name", placeholder="John Doe")
        title = st.text_input("Job Title applying for", placeholder="Senior Data Engineer")
        hours = st.slider("Study hours per week", 2, 20, 10)
    with col2:
        st.markdown("**📁 Upload Documents**")
        jd_file = st.file_uploader("Job Description (PDF or TXT)", type=["pdf", "txt"])
        resume_file = st.file_uploader("Resume / CV (PDF or TXT)", type=["pdf", "txt"])
    with st.expander("📋 Or paste text directly"):
        jd_paste = st.text_area("Paste Job Description", height=150)
        resume_paste = st.text_area("Paste Resume", height=150)
    st.markdown("")
    if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
        if not name.strip() or not title.strip():
            st.error("Please enter your name and job title.")
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
                st.error("Please provide a Job Description.")
            elif not resume_text:
                st.error("Please provide your Resume.")
            else:
                st.session_state.candidate_name = name.strip()
                st.session_state.job_title = title.strip()
                st.session_state.hours_per_week = hours
                st.session_state.jd_text = jd_text
                st.session_state.resume_text = resume_text
                run_parsing()

# ── PARSED ────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "parsed":
    st.subheader("Step 2 — Skills Extracted")
    st.success("Documents parsed successfully!")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 JD Requires**")
        for s in (st.session_state.jd_skills or {}).get("required_skills", []):
            icon = {"critical": "🔴", "important": "🟡", "nice-to-have": "🟢"}.get(s.get("importance", ""), "⚪")
            st.markdown(f"{icon} **{s['skill']}** `{s.get('importance','')}` · *{s.get('category','')}*")
            if s.get("jd_context"):
                st.caption(f"→ \"{s['jd_context']}\"")
    with col2:
        st.markdown("**📄 You Claim**")
        for s in (st.session_state.resume_skills or {}).get("claimed_skills", []):
            icon = {"high": "✅", "medium": "🔶", "low": "❓"}.get(s.get("confidence", ""), "⚪")
            st.markdown(f"{icon} **{s['skill']}** · *{s.get('confidence','')}*")
            if s.get("evidence"):
                st.caption(f"→ \"{s['evidence']}\"")
    st.divider()
    if st.button("▶️ Run Gap Analysis", type="primary", use_container_width=True):
        run_gap_analysis()

# ── GAP ANALYSIS ──────────────────────────────────────────────────────────────
elif st.session_state.stage == "gap":
    st.subheader("Step 3 — Skill Gap Analysis")
    gap = st.session_state.gap_analysis or {}
    col1, col2, col3 = st.columns(3)
    with col1:
        matched = gap.get("matched_skills", [])
        st.metric("✅ Matched", len(matched))
        for s in matched:
            st.markdown(f"- {s}")
    with col2:
        missing = gap.get("missing_skills", [])
        st.metric("❌ Missing", len(missing))
        for s in missing:
            st.markdown(f"- {s}")
    with col3:
        needs_v = gap.get("needs_verification", [])
        st.metric("🔍 Needs Proof", len(needs_v))
        for s in needs_v:
            st.markdown(f"- **{s['skill']}**: {s.get('reason','')}")
    st.divider()
    queue = st.session_state.assessment_queue
    if queue:
        st.info("**Will assess:** " + " → ".join(queue))
        if st.button("🎙️ Begin Assessment", type="primary", use_container_width=True):
            start_assessment()
    else:
        if st.button("📚 Skip to Learning Plan", type="primary", use_container_width=True):
            st.session_state.stage = "learning"
            st.rerun()

# ── ASSESSMENT ────────────────────────────────────────────────────────────────
elif st.session_state.stage == "assessment":
    with st.sidebar:
        st.markdown("### 📊 Progress")
        total = len(st.session_state.skill_scores) + len(st.session_state.assessment_queue)
        done = len(st.session_state.skill_scores)
        if total > 0:
            st.progress(done / total)
            st.caption(f"{done} of {total} done")
        for sc in st.session_state.skill_scores:
            g = sc.get("gap", 0)
            icon = "🟢" if g <= 1 else "🟡" if g <= 3 else "🔴"
            st.markdown(f"{icon} {sc['skill']} — {sc.get('assessed_score','?')}/10")
        if st.session_state.assessment_queue:
            st.markdown("**Remaining:**")
            for i, s in enumerate(st.session_state.assessment_queue):
                st.markdown(f"{'👉 ' if i == 0 else '  '}{s}")

    st.subheader(f"🎙️ Assessing: **{st.session_state.current_skill}**")
    st.caption(f"{st.session_state.candidate_name} · {st.session_state.job_title}")
    st.divider()
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
    if answer := st.chat_input("Type your answer and press Enter..."):
        handle_answer(answer)

# ── LEARNING PLAN ─────────────────────────────────────────────────────────────
elif st.session_state.stage == "learning":
    st.subheader("🎉 Assessment Complete!")
    st.success(f"Results for **{st.session_state.candidate_name}** — {st.session_state.job_title}")

    if st.session_state.skill_scores:
        st.markdown("### 📊 Skill Scores")
        header = st.columns([2, 1, 1, 1, 3])
        for c, h in zip(header, ["**Skill**", "**Claimed**", "**Assessed**", "**Gap**", "**Verdict**"]):
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
        if st.button("📚 Generate Learning Plan", type="primary", use_container_width=True):
            generate_learning_plan()
    else:
        plan = st.session_state.learning_plan
        st.markdown("### 📚 Personalised Learning Plan")
        st.info(plan.get("summary", ""))
        col1, col2 = st.columns(2)
        col1.metric("⏱️ Weeks to Job-Ready", plan.get("total_weeks_to_job_ready", "?"))
        col2.markdown(f"💬 *{plan.get('motivational_note', '')}*")

        for item in plan.get("priority_gaps", []):
            with st.expander(f"📖 {item['skill']} — {item.get('time_estimate_weeks','?')} weeks · {item.get('weekly_hours','?')}h/week"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Current", item.get("current_level", "?"))
                c2.metric("Target", item.get("target_level", "?"))
                c3.metric("Total Hours", item.get("time_estimate_weeks", 0) * item.get("weekly_hours", 0))
                if item.get("quick_win"):
                    st.success(f"⚡ **Quick Win:** {item['quick_win']}")
                st.markdown("**📚 Resources:**")
                for res in item.get("resources", []):
                    st.markdown(f"{resource_icon(res.get('type',''))} [{res['title']}]({res['url']}) · ~{res.get('time_hours','?')}h")
                if item.get("adjacent_skills"):
                    st.markdown("**🔗 Also learn:** " + ", ".join(item["adjacent_skills"]))

        st.divider()
        if not st.session_state.report:
            if st.button("📄 Generate Full Report", type="primary", use_container_width=True):
                generate_report()
        else:
            st.markdown("### 📄 Full Assessment Report")
            st.markdown(st.session_state.report)
            st.download_button(
                "⬇️ Download Report",
                data=st.session_state.report,
                file_name=f"skillsense_{st.session_state.candidate_name.replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            if st.button("🔄 New Assessment", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

st.divider()
st.caption("SkillSense · Catalyst Hackathon — Deccan AI Experts · Powered by Llama 3.3 via Groq")
