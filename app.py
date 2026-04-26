import streamlit as st
import json, io, re, os, hashlib
from datetime import datetime
from groq import Groq
import PyPDF2

st.set_page_config(page_title="SkillSense", page_icon="🎯", layout="wide")
st.markdown("<style>.stProgress > div > div { background-color: #4F46E5; }</style>", unsafe_allow_html=True)

USERS_FILE = "users.json"
DATA_DIR   = "userdata"
os.makedirs(DATA_DIR, exist_ok=True)

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f: return json.load(f)
    return {}

def save_users(u):
    with open(USERS_FILE,"w") as f: json.dump(u,f,indent=2)

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def user_dir(email):
    p = os.path.join(DATA_DIR, email.replace("@","_at_").replace(".","_"))
    os.makedirs(p, exist_ok=True); return p

def save_session(email, data):
    fname = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(user_dir(email), fname),"w") as f: json.dump(data,f,indent=2)

def load_sessions(email):
    out=[]
    for fn in sorted(os.listdir(user_dir(email)),reverse=True):
        if fn.endswith(".json"):
            with open(os.path.join(user_dir(email),fn)) as f: out.append((fn,json.load(f)))
    return out

def get_client():
    k=st.secrets.get("GROQ_API_KEY","")
    if not k: st.error("GROQ_API_KEY missing"); st.stop()
    return Groq(api_key=k)

def call_llm(prompt="", max_tokens=2000, messages=None):
    try:
        msgs=messages if messages else [{"role":"user","content":prompt}]
        r=get_client().chat.completions.create(model="llama-3.3-70b-versatile",max_tokens=max_tokens,messages=msgs)
        return r.choices[0].message.content
    except Exception as e: st.error(f"API error: {e}"); return ""

JD_P="""Extract ALL skills from this job description. Return ONLY JSON:
{{"required_skills":[{{"skill":"Python","importance":"critical","category":"technical","jd_context":"phrase"}}]}}
Importance: critical/important/nice-to-have. Category: technical/soft_skill/domain_knowledge/tool.
JD: {JD_TEXT}"""

RES_P="""Extract all skills from this resume. Return ONLY JSON:
{{"claimed_skills":[{{"skill":"Python","evidence":"built pipelines","years_mentioned":3,"confidence":"high"}}]}}
Confidence: high/medium/low. Resume: {RESUME_TEXT}"""

GAP_P="""Compare JD vs resume skills. Return ONLY JSON:
{{"matched_skills":["s1"],"missing_skills":["s2"],"needs_verification":[{{"skill":"Python","reason":"no proof","priority":1}}],"assessment_order":["Python","SQL"]}}
Limit assessment_order to 3 skills. JD:{JD_SKILLS} Resume:{RESUME_SKILLS}"""

ASYS="""You are SkillSense, a friendly precise technical interviewer.
Role:{JOB_TITLE} | Skill:{CURRENT_SKILL} | Claimed:{CLAIMED_LEVEL}
RULES: 1)ONE question per message. 2)Start mid-difficulty, adapt. 3)After 2-4 exchanges output ASSESSMENT_COMPLETE on its own line then JSON:
{{"skill":"{CURRENT_SKILL}","assessed_score":7,"claimed_score":8,"gap":1,"evidence":"observation","verdict":"one line"}}
4)Be encouraging. 5)Ask how-would-you questions. First message start: "Let's talk about your experience with {CURRENT_SKILL}. [question]" """

PLAN_P="""Create personalised learning plan. Return ONLY JSON:
{{"summary":"2 sentences","priority_gaps":[{{"skill":"Docker","current_level":"beginner","target_level":"intermediate","time_estimate_weeks":3,"weekly_hours":5,"resources":[{{"title":"Docker Getting Started","url":"https://docs.docker.com/get-started/","type":"documentation","time_hours":4}}],"adjacent_skills":["Kubernetes"],"quick_win":"run first container"}}],"total_weeks_to_job_ready":8,"motivational_note":"message"}}
Job:{JOB_TITLE} Scores:{ALL_SCORES} Gaps:{GAPS} Hours/week:{HOURS_PER_WEEK}"""

RPT_P="""Write professional skill assessment report.
Candidate:{NAME} Job:{JOB_TITLE} Data:{DATA}
Include: 1)Executive Summary (3 sentences, second person) 2)Skills Table (skill|claimed|assessed|gap|verdict) 3)Top 3 Strengths 4)Top 3 Gaps 5)4-week learning path 6)Hiring Readiness X/100"""

CHAT_SYS="""You are SkillSense Assistant, expert career advisor. Help user refine assessment, find resources, explain gaps, or customize learning plan.
User session context: {CTX}. Be helpful, specific, concise."""

def fill(t,**kw):
    for k,v in kw.items(): t=t.replace("{"+k+"}",str(v))
    return t

def parse_json(txt):
    if not txt: return None
    for fn in [lambda t:json.loads(t.strip()),
               lambda t:json.loads(re.search(r'```(?:json)?\s*([\s\S]*?)```',t).group(1).strip()),
               lambda t:json.loads(re.search(r'(\{[\s\S]*\})',t).group(1))]:
        try: return fn(txt)
        except: pass
    return None

def pdf_text(f):
    try:
        r=PyPDF2.PdfReader(io.BytesIO(f.read()))
        return "\n".join(p.extract_text() or "" for p in r.pages).strip()
    except Exception as e: st.error(f"PDF error:{e}"); return ""

def init():
    d={"stage":"upload","jd_text":"","resume_text":"","jd_skills":None,"resume_skills":None,
       "gap":None,"candidate_name":"","job_title":"","hours":10,"queue":[],"cur_skill":None,
       "conv":[],"disp":[],"scores":[],"plan":None,"report":""}
    for k,v in d.items():
        if k not in st.session_state: st.session_state[k]=v

def claimed(skill):
    if not st.session_state.resume_skills: return "not mentioned"
    for s in st.session_state.resume_skills.get("claimed_skills",[]):
        if s.get("skill","").lower()==skill.lower():
            return f"{s.get('confidence','medium')} - {s.get('evidence','')}"
    return "not on resume"

def amsgs(skill,hist):
    sys=fill(ASYS,JOB_TITLE=st.session_state.job_title,CURRENT_SKILL=skill,CLAIMED_LEVEL=claimed(skill))
    m=[{"role":"system","content":sys}]
    if not hist: m.append({"role":"user","content":f"Begin assessment for {skill}."})
    else: m.extend(hist)
    return m

def ri(t): return {"video":"🎬","documentation":"📄","course":"🎓","book":"📚"}.get(t,"🔗")

def ctx():
    return json.dumps({"job":st.session_state.job_title,"candidate":st.session_state.candidate_name,
                       "scores":st.session_state.scores,"stage":st.session_state.stage})

def do_parse():
    with st.spinner("🔍 Reading JD..."): st.session_state.jd_skills=parse_json(call_llm(fill(JD_P,JD_TEXT=st.session_state.jd_text)))
    with st.spinner("📄 Reading resume..."): st.session_state.resume_skills=parse_json(call_llm(fill(RES_P,RESUME_TEXT=st.session_state.resume_text)))
    if st.session_state.jd_skills and st.session_state.resume_skills:
        st.session_state.stage="parsed"; st.rerun()
    else: st.error("Parsing failed.")

def do_gap():
    with st.spinner("⚖️ Analysing gaps..."):
        r=call_llm(fill(GAP_P,JD_SKILLS=json.dumps(st.session_state.jd_skills),RESUME_SKILLS=json.dumps(st.session_state.resume_skills)))
        st.session_state.gap=parse_json(r)
    if st.session_state.gap:
        st.session_state.queue=st.session_state.gap.get("assessment_order",[])[:3]
        st.session_state.stage="gap"; st.rerun()
    else: st.error("Gap analysis failed.")

def do_start():
    if not st.session_state.queue: st.session_state.stage="learning"; st.rerun(); return
    skill=st.session_state.queue[0]
    st.session_state.cur_skill=skill; st.session_state.conv=[]; st.session_state.disp=[]; st.session_state.stage="assessment"
    with st.spinner(f"Starting {skill}..."):
        q=call_llm("",400,amsgs(skill,[]))
    st.session_state.conv.append({"role":"assistant","content":q})
    st.session_state.disp.append({"role":"assistant","content":q})
    st.rerun()

def do_answer(ans):
    st.session_state.conv.append({"role":"user","content":ans})
    st.session_state.disp.append({"role":"user","content":ans})
    with st.spinner("Thinking..."):
        resp=call_llm("",500,amsgs(st.session_state.cur_skill,st.session_state.conv))
    if "ASSESSMENT_COMPLETE" in resp:
        parts=resp.split("ASSESSMENT_COMPLETE",1)
        closing=parts[0].strip(); score=parse_json(parts[1].strip() if len(parts)>1 else "")
        if score: st.session_state.scores.append(score)
        if closing: st.session_state.disp.append({"role":"assistant","content":closing})
        st.session_state.queue.pop(0)
        if st.session_state.queue:
            nxt=st.session_state.queue[0]; st.session_state.cur_skill=nxt; st.session_state.conv=[]
            with st.spinner(f"Moving to {nxt}..."):
                nq=call_llm("",400,amsgs(nxt,[]))
            st.session_state.conv.append({"role":"assistant","content":nq})
            last=st.session_state.scores[-1] if st.session_state.scores else {}
            st.session_state.disp.append({"role":"assistant","content":f"---\n✅ **{last.get('skill','Prev')} done.**\n\n---\n**Next: {nxt}**\n\n{nq}"})
        else:
            st.session_state.disp.append({"role":"assistant","content":"✅ **All done!** Generate your learning plan below."})
            st.session_state.stage="learning"
            if st.session_state.get("logged_in"):
                save_session(st.session_state.user_email,{"job_title":st.session_state.job_title,"candidate_name":st.session_state.candidate_name,"scores":st.session_state.scores,"gap":st.session_state.gap,"ts":datetime.now().isoformat()})
    else:
        st.session_state.conv.append({"role":"assistant","content":resp})
        st.session_state.disp.append({"role":"assistant","content":resp})
    st.rerun()

# ── Auth session init ─────────────────────────────────────────────────────────
for k,v in [("logged_in",False),("user_email",""),("user_name",""),("chat_msgs",[]),("show_chat",False),("show_hist",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("# 🎯 SkillSense")
    st.markdown("*AI-Powered Skill Assessment & Personalised Learning Plan Agent*")
    st.divider()
    _,mid,_=st.columns([1,1.1,1])
    with mid:
        t1,t2=st.tabs(["🔑 Log In","📝 Sign Up"])
        with t1:
            st.markdown("### Welcome back!")
            em=st.text_input("Email",key="li_em",placeholder="you@example.com")
            pw=st.text_input("Password",type="password",key="li_pw")
            if st.button("Log In",type="primary",use_container_width=True):
                users=load_users()
                if em in users and users[em]["password"]==hash_pw(pw):
                    st.session_state.logged_in=True; st.session_state.user_email=em; st.session_state.user_name=users[em]["name"]; st.rerun()
                else: st.error("Invalid email or password.")
            st.divider()
            st.markdown("**Or sign in with Google**")
            g_url=st.secrets.get("GOOGLE_OAUTH_URL","")
            if g_url: st.link_button("🔵 Sign in with Google",g_url,use_container_width=True)
            else: st.info("💡 Google sign-in: add GOOGLE_OAUTH_URL to Streamlit Secrets to enable.")
        with t2:
            st.markdown("### Create your account")
            nn=st.text_input("Full Name",key="su_name",placeholder="John Doe")
            ne=st.text_input("Email",key="su_em",placeholder="you@example.com")
            np=st.text_input("Password",type="password",key="su_pw",placeholder="Min 6 characters")
            np2=st.text_input("Confirm Password",type="password",key="su_pw2")
            if st.button("Create Account",type="primary",use_container_width=True):
                if not all([nn,ne,np,np2]): st.error("Fill all fields.")
                elif len(np)<6: st.error("Password min 6 characters.")
                elif np!=np2: st.error("Passwords don't match.")
                elif "@" not in ne: st.error("Invalid email.")
                else:
                    users=load_users()
                    if ne in users: st.error("Email already registered. Please log in.")
                    else:
                        users[ne]={"name":nn,"password":hash_pw(np),"created":datetime.now().isoformat()}
                        save_users(users)
                        st.session_state.logged_in=True; st.session_state.user_email=ne; st.session_state.user_name=nn
                        st.success("Account created!"); st.rerun()
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
init()

cl,cr=st.columns([3,1])
with cl:
    st.markdown("# 🎯 SkillSense")
    st.caption("Catalyst Hackathon — Deccan AI Experts")
with cr:
    st.markdown(f"<div style='text-align:right;padding-top:8px'>👤 <b>{st.session_state.user_name}</b></div>",unsafe_allow_html=True)
    b1,b2,b3=st.columns(3)
    with b1:
        if st.button("📂",use_container_width=True,help="View History"): st.session_state.show_hist=not st.session_state.show_hist
    with b2:
        if st.button("💬",use_container_width=True,help="AI Chat"): st.session_state.show_chat=not st.session_state.show_chat
    with b3:
        if st.button("🚪",use_container_width=True,help="Logout"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

st.divider()

# ── HISTORY ───────────────────────────────────────────────────────────────────
if st.session_state.show_hist:
    st.subheader("📂 Your Saved Assessments")
    sessions=load_sessions(st.session_state.user_email)
    if not sessions:
        st.info("No saved assessments yet. Complete an assessment to auto-save.")
    for fn,data in sessions:
        ts=fn.replace("session_","").replace(".json","").replace("_"," ")
        with st.expander(f"📋 {data.get('job_title','Role')} — {ts}"):
            st.markdown(f"**Candidate:** {data.get('candidate_name','')} | **Job:** {data.get('job_title','')}")
            for sc in data.get("scores",[]):
                g=sc.get("gap",0); icon="🟢" if g<=1 else "🟡" if g<=3 else "🔴"
                st.markdown(f"- {icon} **{sc.get('skill','')}** — {sc.get('assessed_score','?')}/10 (gap: {g})")
            if st.button("Load this session",key=f"ld_{fn}"):
                st.session_state.scores=data.get("scores",[])
                st.session_state.job_title=data.get("job_title","")
                st.session_state.candidate_name=data.get("candidate_name","")
                st.session_state.gap=data.get("gap",{})
                st.session_state.stage="learning"; st.session_state.show_hist=False; st.rerun()
    st.divider()

# ── CHAT PANEL ────────────────────────────────────────────────────────────────
if st.session_state.show_chat:
    st.subheader("💬 SkillSense AI Assistant")
    st.caption("Ask anything — refine results, get more resources, explain your gaps, customise your plan.")
    chat_box=st.container(height=320)
    with chat_box:
        for m in st.session_state.chat_msgs:
            with st.chat_message(m["role"],avatar="🤖" if m["role"]=="assistant" else "👤"):
                st.markdown(m["content"])
    if ci:=st.chat_input("Ask SkillSense anything...",key="ai_chat"):
        st.session_state.chat_msgs.append({"role":"user","content":ci})
        sys=fill(CHAT_SYS,CTX=ctx())
        msgs=[{"role":"system","content":sys}]+st.session_state.chat_msgs
        with st.spinner("Thinking..."):
            rep=call_llm("",1000,msgs)
        st.session_state.chat_msgs.append({"role":"assistant","content":rep}); st.rerun()
    if st.session_state.chat_msgs:
        if st.button("🗑️ Clear Chat",use_container_width=True):
            st.session_state.chat_msgs=[]; st.rerun()
    st.divider()

# ── PROGRESS BAR ──────────────────────────────────────────────────────────────
stages=["upload","parsed","gap","assessment","learning"]
labels=["📤 Upload","🔍 Parse","⚖️ Gap","🎙️ Assess","📚 Plan"]
idx=stages.index(st.session_state.stage) if st.session_state.stage in stages else 0
cs=st.columns(len(labels))
for i,(c,l) in enumerate(zip(cs,labels)):
    with c:
        if i<idx: st.markdown(f"<small>✅ {l}</small>",unsafe_allow_html=True)
        elif i==idx: st.markdown(f"<small><b>{l}</b></small>",unsafe_allow_html=True)
        else: st.markdown(f"<small style='color:grey'>{l}</small>",unsafe_allow_html=True)
st.progress(idx/(len(stages)-1)); st.markdown("")

# ═══ UPLOAD ══════════════════════════════════════════════════════════════════
if st.session_state.stage=="upload":
    st.subheader(f"Step 1 — Hello {st.session_state.user_name}! Upload Your Documents")
    c1,c2=st.columns(2)
    with c1:
        name=st.text_input("Full Name",value=st.session_state.user_name)
        title=st.text_input("Job Title applying for",placeholder="Senior Data Engineer")
        hours=st.slider("Study hours/week",2,20,10)
    with c2:
        jdf=st.file_uploader("Job Description (PDF/TXT)",type=["pdf","txt"])
        rf=st.file_uploader("Resume/CV (PDF/TXT)",type=["pdf","txt"])
    with st.expander("📋 Or paste text directly"):
        jdp=st.text_area("Paste Job Description",height=130)
        rp=st.text_area("Paste Resume",height=130)
    if st.button("🚀 Start Assessment",type="primary",use_container_width=True):
        if not name or not title: st.error("Enter name and job title.")
        else:
            jd,res="",""
            if jdf: jd=pdf_text(jdf) if jdf.type=="application/pdf" else jdf.read().decode()
            elif jdp.strip(): jd=jdp.strip()
            if rf: rf.seek(0); res=pdf_text(rf) if rf.type=="application/pdf" else rf.read().decode()
            elif rp.strip(): res=rp.strip()
            if not jd: st.error("Provide a Job Description.")
            elif not res: st.error("Provide your Resume.")
            else:
                st.session_state.candidate_name=name; st.session_state.job_title=title
                st.session_state.hours=hours; st.session_state.jd_text=jd; st.session_state.resume_text=res
                do_parse()

# ═══ PARSED ══════════════════════════════════════════════════════════════════
elif st.session_state.stage=="parsed":
    st.subheader("Step 2 — Skills Extracted"); st.success("✅ Documents parsed!")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**📋 JD Requires**")
        for s in (st.session_state.jd_skills or {}).get("required_skills",[]):
            ic={"critical":"🔴","important":"🟡","nice-to-have":"🟢"}.get(s.get("importance",""),"⚪")
            st.markdown(f"{ic} **{s['skill']}** `{s.get('importance','')}` · *{s.get('category','')}*")
            if s.get("jd_context"): st.caption(f'→ "{s["jd_context"]}"')
    with c2:
        st.markdown("**📄 You Claim**")
        for s in (st.session_state.resume_skills or {}).get("claimed_skills",[]):
            ic={"high":"✅","medium":"🔶","low":"❓"}.get(s.get("confidence",""),"⚪")
            st.markdown(f"{ic} **{s['skill']}** · *{s.get('confidence','')}*")
            if s.get("evidence"): st.caption(f'→ "{s["evidence"]}"')
    st.divider()
    if st.button("▶️ Run Gap Analysis",type="primary",use_container_width=True): do_gap()

# ═══ GAP ═════════════════════════════════════════════════════════════════════
elif st.session_state.stage=="gap":
    st.subheader("Step 3 — Skill Gap Analysis")
    g=st.session_state.gap or {}
    c1,c2,c3=st.columns(3)
    with c1:
        m=g.get("matched_skills",[]); st.metric("✅ Matched",len(m))
        for s in m: st.markdown(f"- {s}")
    with c2:
        ms=g.get("missing_skills",[]); st.metric("❌ Missing",len(ms))
        for s in ms: st.markdown(f"- {s}")
    with c3:
        nv=g.get("needs_verification",[]); st.metric("🔍 Needs Proof",len(nv))
        for s in nv: st.markdown(f"- **{s['skill']}**: {s.get('reason','')}")
    st.divider()
    if st.session_state.queue:
        st.info("**Will assess:** "+" → ".join(st.session_state.queue))
        if st.button("🎙️ Begin Assessment",type="primary",use_container_width=True): do_start()
    else:
        if st.button("📚 Go to Learning Plan",type="primary",use_container_width=True):
            st.session_state.stage="learning"; st.rerun()

# ═══ ASSESSMENT ══════════════════════════════════════════════════════════════
elif st.session_state.stage=="assessment":
    with st.sidebar:
        st.markdown("### 📊 Progress")
        tot=len(st.session_state.scores)+len(st.session_state.queue); dn=len(st.session_state.scores)
        if tot>0: st.progress(dn/tot); st.caption(f"{dn} of {tot} done")
        for sc in st.session_state.scores:
            g=sc.get("gap",0); ic="🟢" if g<=1 else "🟡" if g<=3 else "🔴"
            st.markdown(f"{ic} {sc['skill']} — {sc.get('assessed_score','?')}/10")
        if st.session_state.queue:
            st.markdown("**Remaining:**")
            for i,s in enumerate(st.session_state.queue): st.markdown(f"{'👉 ' if i==0 else '  '}{s}")
    st.subheader(f"🎙️ Assessing: **{st.session_state.cur_skill}**")
    st.caption(f"{st.session_state.candidate_name} · {st.session_state.job_title}")
    st.divider()
    for m in st.session_state.disp:
        with st.chat_message(m["role"],avatar="🤖" if m["role"]=="assistant" else "👤"):
            st.markdown(m["content"])
    if a:=st.chat_input("Type your answer and press Enter..."): do_answer(a)

# ═══ LEARNING PLAN ════════════════════════════════════════════════════════════
elif st.session_state.stage=="learning":
    st.subheader("🎉 Assessment Complete!")
    st.success(f"Results for **{st.session_state.candidate_name}** — {st.session_state.job_title}")
    if st.session_state.scores:
        st.markdown("### 📊 Skill Scores")
        hc=st.columns([2,1,1,1,3])
        for c,h in zip(hc,["**Skill**","**Claimed**","**Assessed**","**Gap**","**Verdict**"]): c.markdown(h)
        st.divider()
        for sc in st.session_state.scores:
            g=sc.get("gap",0); ic="🟢" if g<=1 else "🟡" if g<=3 else "🔴"
            c=st.columns([2,1,1,1,3])
            c[0].markdown(f"**{sc.get('skill','?')}**"); c[1].markdown(f"{sc.get('claimed_score','?')}/10")
            c[2].markdown(f"{sc.get('assessed_score','?')}/10"); c[3].markdown(f"{ic} {g}"); c[4].markdown(f"*{sc.get('verdict','?')}*")
    st.divider()
    if not st.session_state.plan:
        if st.button("📚 Generate Learning Plan",type="primary",use_container_width=True):
            gaps=[s for s in st.session_state.scores if s.get("gap",0)>0]
            with st.spinner("📚 Building plan..."):
                r=call_llm(fill(PLAN_P,JOB_TITLE=st.session_state.job_title,ALL_SCORES=json.dumps(st.session_state.scores),GAPS=json.dumps(gaps),HOURS_PER_WEEK=str(st.session_state.hours)),3000)
                st.session_state.plan=parse_json(r)
            st.rerun()
    else:
        p=st.session_state.plan
        st.markdown("### 📚 Personalised Learning Plan")
        st.info(p.get("summary",""))
        c1,c2=st.columns(2)
        c1.metric("⏱️ Weeks to Job-Ready",p.get("total_weeks_to_job_ready","?"))
        c2.markdown(f"💬 *{p.get('motivational_note','')}*")
        for item in p.get("priority_gaps",[]):
            with st.expander(f"📖 {item['skill']} — {item.get('time_estimate_weeks','?')}wk"):
                cc1,cc2,cc3=st.columns(3)
                cc1.metric("Current",item.get("current_level","?")); cc2.metric("Target",item.get("target_level","?"))
                cc3.metric("Total Hours",item.get("time_estimate_weeks",0)*item.get("weekly_hours",0))
                if item.get("quick_win"): st.success(f"⚡ **Quick Win:** {item['quick_win']}")
                st.markdown("**📚 Resources:**")
                for res in item.get("resources",[]): st.markdown(f"{ri(res.get('type',''))} [{res['title']}]({res['url']}) · ~{res.get('time_hours','?')}h")
                if item.get("adjacent_skills"): st.markdown("**🔗 Also explore:** "+", ".join(item["adjacent_skills"]))
        st.divider()
        sc1,sc2=st.columns(2)
        with sc1:
            if st.button("💾 Save Assessment",use_container_width=True):
                save_session(st.session_state.user_email,{"job_title":st.session_state.job_title,"candidate_name":st.session_state.candidate_name,"scores":st.session_state.scores,"gap":st.session_state.gap,"plan":st.session_state.plan,"ts":datetime.now().isoformat()})
                st.success("✅ Saved! View in History (📂).")
        with sc2:
            if not st.session_state.report:
                if st.button("📄 Generate Report",type="primary",use_container_width=True):
                    with st.spinner("📝 Writing report..."):
                        st.session_state.report=call_llm(fill(RPT_P,NAME=st.session_state.candidate_name,JOB_TITLE=st.session_state.job_title,DATA=json.dumps({"scores":st.session_state.scores,"plan":st.session_state.plan})),2500)
                    st.rerun()
        if st.session_state.report:
            st.markdown("### 📄 Full Assessment Report"); st.markdown(st.session_state.report)
            st.download_button("⬇️ Download Report",data=st.session_state.report,file_name=f"skillsense_{st.session_state.candidate_name.replace(' ','_')}.txt",mime="text/plain",use_container_width=True)
        st.divider()
        if st.button("🔄 Start New Assessment",use_container_width=True):
            keep=["logged_in","user_email","user_name","chat_msgs","show_chat","show_hist"]
            for k in [k for k in st.session_state.keys() if k not in keep]: del st.session_state[k]
            st.rerun()

st.divider()
st.caption("SkillSense · Catalyst Hackathon — Deccan AI Experts · Powered by Llama 3.3 via Groq")
