"""
SkillBridge AI - Flask Backend
Powered by Groq (Llama 3.3 70B)
"""

import os
import json
import re
import uuid
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

try:
    import fitz
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from docx import Document
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

app = Flask(__name__, static_folder='.')
CORS(app)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

print("=" * 50)
print("SkillBridge AI")
print(f"API: {'Groq Connected' if client else 'MISSING KEY - check run.py'}")
print(f"PDF: {'Yes' if PDF_OK else 'No'} | DOCX: {'Yes' if DOCX_OK else 'No'}")
print("=" * 50)

SESSIONS = {}
SESSION_TTL_SECONDS = int(os.environ.get("SB_SESSION_TTL_SECONDS", "86400"))  # 24h


def _now():
    return int(time.time())


def _cleanup_sessions():
    t = _now()
    to_delete = [sid for sid, s in SESSIONS.items() if (t - s.get("created_at", t)) > SESSION_TTL_SECONDS]
    for sid in to_delete:
        SESSIONS.pop(sid, None)


def extract_pdf(b):
    if not PDF_OK: return ""
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        return "\n".join(p.get_text() for p in doc).strip()
    except: return ""


def extract_docx(b):
    if not DOCX_OK: return ""
    try:
        import io
        doc = Document(io.BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except: return ""

def extract_cv_text_from_request(req):
    cv_text = ""
    if 'cv_file' in req.files:
        f = req.files['cv_file']
        b = f.read()
        name = (f.filename or "").lower()
        if name.endswith('.pdf'):
            cv_text = extract_pdf(b)
        elif name.endswith(('.docx', '.doc')):
            cv_text = extract_docx(b)
        elif name.endswith('.txt'):
            cv_text = b.decode('utf-8', errors='ignore')
    if not cv_text:
        cv_text = req.form.get('cv_text', '').strip()
    return cv_text


def _parse_json_from_model(raw):
    raw = (raw or "").strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    # Try to isolate the outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _try_basic_json_repairs(raw):
    """
    Best-effort repairs for common model JSON mistakes.
    Returns repaired string.
    """
    s = (raw or "").strip()
    # isolate outer object if possible
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]

    # Replace smart quotes
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")

    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Remove illegal control characters (keep tabs/newlines)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return s


def _call_groq_json(system, prompt, *, temperature=0.2, max_tokens=2000, retries=1):
    """
    Calls Groq and returns parsed JSON, retrying once on malformed JSON.
    """
    last_raw = ""
    last_err = None
    for attempt in range(retries + 1):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt if attempt == 0 else (prompt + "\n\nSTRICT: Output ONLY valid JSON. No trailing commas. No comments. No markdown.")},
            ],
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
        )
        last_raw = response.choices[0].message.content
        try:
            return _parse_json_from_model(last_raw)
        except Exception as e:
            last_err = e
            continue

    # Try local repairs
    try:
        repaired = _try_basic_json_repairs(last_raw)
        return json.loads(repaired)
    except Exception:
        pass

    # Last resort: ask the model to rewrite into valid JSON
    fix_prompt = f"""Rewrite the following content into STRICTLY VALID JSON.

Rules:
- Output ONLY JSON (no markdown/backticks/extra text).
- Preserve the same data/structure as much as possible.
- Fix missing commas, trailing commas, and quoting issues.

CONTENT TO FIX:
{last_raw}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You fix JSON. You output valid JSON only."},
            {"role": "user", "content": fix_prompt},
        ],
        temperature=0.0,
        max_tokens=min(3500, max_tokens),
    )
    fixed_raw = response.choices[0].message.content
    try:
        return _parse_json_from_model(fixed_raw)
    except Exception as e:
        raise ValueError(f"Failed to parse model JSON: {last_err}") from e


def generate_interview_questions(job_desc, job_title="", n_questions=10):
    prompt = f"""You are SkillBridge Interviewer.

Create an interview question set that conversationally assesses real proficiency on each required skill, identifies gaps, and avoids trivia.

TARGET JOB:
{f"Job Title: {job_title}" if job_title else ""}
{f"Job Description: {job_desc}" if job_desc else ""}

Rules:
- Return ONLY JSON (no markdown/backticks/extra text).
- Generate exactly {n_questions} questions.
- Each question must be directly grounded in the job description.
- Questions should test practical understanding (explain, troubleshoot, design, reason, apply).
- Keep questions short and speak to the candidate as "you".

Return JSON with this shape:
{{
  "questions": [
    {{
      "skill": "<skill being tested>",
      "question": "<question text>",
      "why_this_matters": "<1 sentence>",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}"""

    return _call_groq_json(
        "You create interview questions and always output valid JSON only.",
        prompt,
        temperature=0.4,
        max_tokens=2000,
        retries=1,
    )


def extract_required_skills(job_desc, job_title=""):
    prompt = f"""Extract the explicit skills required by the job description.

TARGET JOB:
{f"Job Title: {job_title}" if job_title else ""}
{job_desc}

Rules:
- Return ONLY JSON.
- Include ONLY skills that are explicitly required or clearly stated as requirements in the JD.
- Use short, canonical skill names (e.g. "Python", "Flask", "SQL", "Unit testing", "Docker").
- Do not include soft skills unless explicitly listed as required skills.
- Do not include tools/skills not mentioned in the JD.

Return JSON:
{{"required_skills": ["<skill1>", "<skill2>", "..."]}}"""

    # Pass 1: normal JSON object wrapper
    data = _call_groq_json(
        "You extract required skills and always output valid JSON only.",
        prompt,
        temperature=0.2,
        max_tokens=800,
        retries=2,
    )
    skills = data.get("required_skills") if isinstance(data, dict) else None

    # Pass 2: ask for a pure JSON array of skill strings (smaller + harder to mess up)
    if not skills or not isinstance(skills, list):
        prompt2 = f"""Return ONLY a JSON array of strings: the explicit required skills from this JD. No object wrapper.

JD:
{job_desc}
"""
        data2 = _call_groq_json(
            "You extract required skills and output JSON only.",
            prompt2,
            temperature=0.0,
            max_tokens=600,
            retries=2,
        )
        if isinstance(data2, list):
            skills = data2
        elif isinstance(data2, dict) and isinstance(data2.get("required_skills"), list):
            skills = data2.get("required_skills")
        else:
            skills = []

    # Pass 3: if JD is vague/empty, infer skills from job title (Groq-only, no hardcoding)
    if (not skills or not isinstance(skills, list) or not [s for s in skills if isinstance(s, str) and s.strip()]) and job_title:
        prompt3 = f"""The job description is vague or missing. Infer the most likely required hard skills for this job title.

Job title: {job_title}

Rules:
- Return ONLY a JSON array of strings (no object wrapper).
- Include 10-25 skills/tools/concepts that are typically required for this role.
- Use short, canonical skill names.
- Prefer technical/domain skills over generic soft skills.
- Do NOT include anything unrelated to the role.
"""
        data3 = _call_groq_json(
            "You infer likely required skills for a job title and output JSON only.",
            prompt3,
            temperature=0.2,
            max_tokens=700,
            retries=2,
        )
        if isinstance(data3, list):
            skills = data3
        elif isinstance(data3, dict) and isinstance(data3.get("required_skills"), list):
            skills = data3.get("required_skills")
        else:
            skills = []
    out = []
    seen = set()
    for s in skills:
        if not isinstance(s, str):
            continue
        s2 = s.strip()
        if not s2:
            continue
        k = s2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s2)
    return out


def score_interview_answers(interview_answers, required_skills, job_desc, job_title=""):
    """
    Scores each answer using the 4x25 rubric and maps it to a required skill.
    Returns list of scoring rows with total 0-100.
    """
    prompt = f"""You are grading interview answers to measure real proficiency.

TARGET JOB:
{f"Job Title: {job_title}" if job_title else ""}
{job_desc}

REQUIRED SKILLS (ONLY these are allowed for mapping):
{json.dumps(required_skills, ensure_ascii=False)}

INTERVIEW Q&A (array of objects):
{json.dumps(interview_answers, ensure_ascii=False)}

For EACH item, output a scoring object with:
- required_skill: MUST be exactly one of REQUIRED SKILLS (choose the best match). If none match, pick the closest required skill.
- relevance 0-25
- correctness 0-25
- depth 0-25
- practical_insight 0-25
- total 0-100 (sum)
- evidence: 1-2 sentences why you scored it that way

Rules:
- Nonsense/off-topic => low across all.
- Short but valid => medium.
- Strong + real-world example => high.

Return ONLY JSON:
{{"scoring":[{{"required_skill":"...","question":"...","answer":"...","relevance":0,"correctness":0,"depth":0,"practical_insight":0,"total":0,"evidence":"..."}}]}}"""

    data = _call_groq_json(
        "You grade interview answers and always output valid JSON only.",
        prompt,
        temperature=0.2,
        max_tokens=2500,
        retries=1,
    )
    rows = data.get("scoring") if isinstance(data, dict) else None
    if not rows or not isinstance(rows, list):
        return []
    cleaned = []
    allowed = {s.lower(): s for s in required_skills}
    for r in rows:
        if not isinstance(r, dict):
            continue
        rs = (r.get("required_skill") or "").strip()
        if not rs:
            continue
        # normalize mapping to canonical casing from required_skills when possible
        canon = allowed.get(rs.lower(), rs)
        try:
            rel = int(r.get("relevance", 0))
            cor = int(r.get("correctness", 0))
            dep = int(r.get("depth", 0))
            pra = int(r.get("practical_insight", 0))
        except Exception:
            rel = cor = dep = pra = 0
        rel = max(0, min(25, rel))
        cor = max(0, min(25, cor))
        dep = max(0, min(25, dep))
        pra = max(0, min(25, pra))
        total = rel + cor + dep + pra
        cleaned.append({
            "skill": canon,
            "required_skill": canon,
            "question": (r.get("question") or ""),
            "answer": (r.get("answer") or ""),
            "relevance": rel,
            "correctness": cor,
            "depth": dep,
            "practical_insight": pra,
            "total": total,
            "evidence": (r.get("evidence") or "").strip(),
        })
    return cleaned


def generate_gap_roadmap(gap_skills, scoring_rows, job_desc, job_title=""):
    """
    Generates a roadmap per gap skill (small JSON per call) to avoid huge malformed JSON.
    Always returns one step per input gap skill in the same order.
    """
    def _single(skill, step_num):
        # Pull best available evidence for this skill
        evidence_rows = [r for r in (scoring_rows or []) if (r.get("required_skill") == skill or r.get("skill") == skill)]
        best = 0
        for r in evidence_rows:
            try:
                best = max(best, int(r.get("total", 0)))
            except Exception:
                pass
        prompt = f"""Create a week-by-week learning plan to master ONE skill gap.

TARGET JOB:
{f"Job Title: {job_title}" if job_title else ""}
{job_desc}

SKILL GAP (exact title):
{skill}

INTERVIEW EVIDENCE (0-100 score; lower => longer plan):
Best score for this skill: {best}
Evidence rows: {json.dumps(evidence_rows, ensure_ascii=False)}

Rules:
- Return ONLY JSON.
- Output exactly one object with:
  - why
  - duration (e.g. "4 weeks")
  - weekly_plan: 4-10 weeks, each with 3-6 topics (include practice/projects)
  - resources: 4-8 items (mix course/book/youtube/project/practice), role-relevant

Return JSON:
{{"why":"...","duration":"...","weekly_plan":[{{"week":1,"topics":["..."]}}],"resources":[{{"name":"...","type":"course|book|project|certification|youtube|practice","url":"","free":true,"note":"..."}}]}}"""

        data = _call_groq_json(
            "You create structured learning roadmaps and always output valid JSON only.",
            prompt,
            temperature=0.35,
            max_tokens=1800,
            retries=2,
        )
        if not isinstance(data, dict):
            data = {}
        return {
            "step": step_num,
            "title": skill,
            "why": data.get("why", ""),
            "duration": data.get("duration", ""),
            "weekly_plan": data.get("weekly_plan", []) if isinstance(data.get("weekly_plan", []), list) else [],
            "resources": data.get("resources", []) if isinstance(data.get("resources", []), list) else [],
        }

    out = []
    for i, sk in enumerate(gap_skills or []):
        try:
            out.append(_single(sk, i + 1))
        except Exception:
            # Always return something so UI shows a step
            out.append({
                "step": i + 1,
                "title": sk,
                "why": "Roadmap generation failed for this skill. Try running again.",
                "duration": "",
                "weekly_plan": [],
                "resources": [],
            })
    return out


def generate_support_sections(job_desc, job_title, required_skills, strengths, gaps):
    """
    Generates non-critical sections (career advice, next steps, CV advice, salary insight, alternative roles).
    Keeps response small to reduce JSON errors.
    """
    prompt = f"""Generate helpful supporting sections for a career gap report.

TARGET JOB:
{f"Job Title: {job_title}" if job_title else ""}
{job_desc}

REQUIRED SKILLS:
{json.dumps(required_skills, ensure_ascii=False)}

STRENGTHS:
{json.dumps(strengths, ensure_ascii=False)}

GAPS:
{json.dumps(gaps, ensure_ascii=False)}

Rules:
- Return ONLY JSON.
- Be concise but useful.

Return JSON:
{{
  "career_advice": ["...","...","...","...","..."],
  "cv_advice": ["...","...","...","...","...","..."],
  "next_steps": ["...","...","..."],
  "alternative_roles": [{{"title":"...","match":50,"reason":"..."}}],
  "salary_insight": {{"range":"...","entry":"...","senior":"...","note":"..."}}
}}"""

    data = _call_groq_json(
        "You generate career advice and always output valid JSON only.",
        prompt,
        temperature=0.4,
        max_tokens=1400,
        retries=2,
    )
    if not isinstance(data, dict):
        return {}
    return data


def analyse(cv_text, job_desc, job_title="", interview_answers=None):
    interview_block = ""
    if interview_answers:
        interview_block = f"""
INTERVIEW ANSWERS (MOST IMPORTANT EVIDENCE):
These answers carry 80% of the weight when judging whether the candidate truly knows a skill.
If the CV claims a skill but the answers show weak understanding, treat it as a gap.
If the CV is missing a skill but the answers show strong understanding, you may treat it as a strength (but note the CV should be updated).

{json.dumps(interview_answers, ensure_ascii=False)}
"""

    prompt = f"""You are SkillBridge, a world-class career coach and talent analyst with deep expertise across ALL industries — technology, medicine, law, finance, marketing, design, education, engineering, arts, hospitality, and more.

Analyse this person's CV against the job they want. Be thorough, honest, specific, and genuinely helpful.

HIGHEST PRIORITY SECTIONS:
1) strengths
2) gaps
3) roadmap
Spend most of your effort on these. Be comprehensive (not just 3 items).

CV / RESUME:
{cv_text if cv_text else "No CV provided — give general advice based on the job description only."}

TARGET JOB:
{f"Job Title: {job_title}" if job_title else ""}
{f"Job Description: {job_desc}" if job_desc else ""}
{interview_block}

IMPORTANT WEIGHTING RULES:
- 80% weight: Interview answers (real proficiency evidence).
- 20% weight: CV/resume content.
- Strengths + gaps MUST primarily reflect interview performance, not resume claims.
- The roadmap MUST be based primarily on interview performance (what they can/can't actually do).

INTERVIEW SCORING (MANDATORY):
For EACH interview answer, score it out of 100 using:
1) Relevance (0-25): Does the answer address the question?
2) Correctness (0-25): Are concepts technically correct?
3) Depth (0-25): Is the explanation detailed?
4) Practical Insight (0-25): Does it include real-world example?

Rules:
- If the answer is nonsense/off-topic: all scores low.
- If short but valid: medium scores.
- If strong real-world explanation: high scores.

Use these scores as the main signal for proficiency per skill:
- 0-39 => weak (gap)
- 40-64 => partial (gap / needs improvement)
- 65-84 => good (strength)
- 85-100 => strong (strength)

ROADMAP REQUIREMENTS (MANDATORY):
- Create a roadmap step for EACH gap/weak skill you identify (not just a few).
- If a required skill was NOT answered in the interview, treat it as a GAP and include it in the roadmap.
- Time estimates MUST be based on the interview score (lower score => longer).
- Each roadmap step MUST include a week-by-week plan: weekly topics the user should cover.
- Include MORE resources per step (aim 4-8) and make them relevant to the industry/role.

You MUST respond with ONLY a valid JSON object. No markdown, no backticks, no explanation, no text before or after the JSON. Just the raw JSON object starting with {{ and ending with }}.

{{
  "overall_match": <integer 0-100>,
  "job_title": "<detected or provided job title>",
  "industry": "<detected industry>",
  "summary": "<2-3 sentence honest summary of their candidacy>",

  "interview_scoring": [
    {{
      "skill": "<skill being evaluated>",
      "question": "<the question text>",
      "answer": "<the user's answer (verbatim or lightly cleaned)>",
      "relevance": <integer 0-25>,
      "correctness": <integer 0-25>,
      "depth": <integer 0-25>,
      "practical_insight": <integer 0-25>,
      "total": <integer 0-100>,
      "evidence": "<1-2 sentence justification>"
    }}
  ],

  "strengths": [
    {{"skill": "<skill name>", "level": <integer 0-100>, "note": "<why this is a strength (cite interview score evidence if available)>"}}
  ],

  "gaps": [
    {{"skill": "<missing or weak skill>", "importance": "critical|important|nice-to-have", "note": "<why it matters + what was weak/missing in their interview evidence>"}}
  ],

  "experience_advice": {{
    "years_needed": "<e.g. 2-3 years or Entry level OK>",
    "current_estimate": "<estimate from CV or Unknown>",
    "advice": "<honest advice about experience gap>"
  }},

  "roadmap": [
    {{
      "step": <number>,
      "title": "<skill or area to develop>",
      "why": "<why this step matters>",
      "duration": "<realistic time estimate>",
      "weekly_plan": [
        {{"week": <number>, "topics": ["<topic 1>", "<topic 2>", "<topic 3>"]}}
      ],
      "resources": [
        {{"name": "<resource name>", "type": "course|book|project|certification|youtube|practice", "url": "<real URL if known, else empty string>", "free": <true|false>, "note": "<brief note>"}}
      ]
    }}
  ],

  "career_advice": [
    "<actionable career advice point 1>",
    "<actionable career advice point 2>",
    "<actionable career advice point 3>",
    "<actionable career advice point 4>",
    "<actionable career advice point 5>"
  ],

  "cv_advice": [
    "<specific CV improvement tip 1 — quantify achievements, fix formatting, add keywords etc>",
    "<specific CV improvement tip 2>",
    "<specific CV improvement tip 3>",
    "<specific CV improvement tip 4>",
    "<specific CV improvement tip 5>",
    "<specific CV improvement tip 6>"
  ],

  "alternative_roles": [
    {{"title": "<alternative job title>", "match": <integer 0-100>, "reason": "<why this suits them>"}}
  ],

  "salary_insight": {{
    "range": "<realistic salary range>",
    "entry": "<entry level salary>",
    "senior": "<senior level salary>",
    "note": "<relevant context about salary>"
  }},

  "next_steps": [
    "<concrete immediate action they can take this week>",
    "<second immediate action>",
    "<third immediate action>"
  ]
}}

Be honest but encouraging. Tailor everything specifically to the industry and role — never give generic tech advice for a nursing, law, or arts role. Make resources real and relevant to the specific field."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class career coach. You always respond with valid JSON only — no markdown, no backticks, no extra text. Just raw JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=4000,
        )

        return _parse_json_from_model(response.choices[0].message.content)

    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        print(f"Raw response: {raw[:500]}")
        return {"error": "Failed to parse response. Please try again."}
    except Exception as e:
        print(f"API error: {e}")
        return {"error": str(e)}


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/index.html')
def index_html():
    return send_from_directory('.', 'index.html')

@app.route('/interview.html')
def interview_page():
    return send_from_directory('.', 'interview.html')


@app.route('/api/session/start', methods=['POST'])
def api_session_start():
    try:
        _cleanup_sessions()
        if not client:
            return jsonify({"error": "API key missing. Check run.py"}), 500

        job_desc = request.form.get('job_description', '').strip()
        job_title = request.form.get('job_title', '').strip()
        if not job_desc and not job_title:
            return jsonify({"error": "Please provide a job description or job title."}), 400

        cv_text = extract_cv_text_from_request(request)
        if not cv_text:
            return jsonify({"error": "Please upload a CV/resume file or paste CV text."}), 400

        sid = uuid.uuid4().hex
        SESSIONS[sid] = {
            "created_at": _now(),
            "job_desc": job_desc,
            "job_title": job_title,
            "cv_text": cv_text,
            "questions": None,
            "answers": None,
        }
        return jsonify({"session_id": sid})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/session/questions', methods=['POST'])
def api_session_questions():
    try:
        _cleanup_sessions()
        if not client:
            return jsonify({"error": "API key missing. Check run.py"}), 500

        sid = request.form.get("session_id", "").strip()
        if not sid or sid not in SESSIONS:
            return jsonify({"error": "Invalid or expired session."}), 400

        s = SESSIONS[sid]
        if s.get("questions"):
            return jsonify({"questions": s["questions"]})

        q = generate_interview_questions(s.get("job_desc", ""), s.get("job_title", ""), n_questions=10)
        questions = q.get("questions") if isinstance(q, dict) else None
        if not questions or not isinstance(questions, list):
            return jsonify({"error": "Failed to generate interview questions."}), 500

        s["questions"] = questions
        return jsonify({"questions": questions})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/session/answers', methods=['POST'])
def api_session_answers():
    try:
        _cleanup_sessions()
        payload = request.get_json(silent=True) or {}
        sid = (payload.get("session_id") or "").strip()
        answers = payload.get("answers")
        if not sid or sid not in SESSIONS:
            return jsonify({"error": "Invalid or expired session."}), 400
        if not answers or not isinstance(answers, list):
            return jsonify({"error": "Missing answers."}), 400
        SESSIONS[sid]["answers"] = answers
        return jsonify({"ok": True})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyse_session', methods=['POST'])
def api_analyse_session():
    try:
        _cleanup_sessions()
        if not client:
            return jsonify({"error": "API key missing. Check run.py"}), 500
        payload = request.get_json(silent=True) or {}
        sid = (payload.get("session_id") or "").strip()
        if not sid or sid not in SESSIONS:
            return jsonify({"error": "Invalid or expired session."}), 400
        s = SESSIONS[sid]
        if not s.get("answers"):
            return jsonify({"error": "Interview answers missing. Complete the interview first."}), 400

        job_desc = s.get("job_desc", "")
        job_title = s.get("job_title", "")
        cv_text = s.get("cv_text", "")
        answers = s.get("answers", [])

        required_skills = extract_required_skills(job_desc, job_title)
        if not required_skills:
            return jsonify({"error": "Could not extract required skills from the JD or infer them from the job title. Please provide a clearer job title or a more detailed JD."}), 400

        try:
            scoring_rows = score_interview_answers(answers, required_skills, job_desc, job_title)
        except Exception:
            scoring_rows = []

        # Aggregate per-skill score (max of attempts; missing => 0)
        per_skill_scores = {sk: 0 for sk in required_skills}
        per_skill_evidence = {sk: [] for sk in required_skills}
        for r in scoring_rows:
            sk = r.get("required_skill")
            if sk in per_skill_scores:
                per_skill_scores[sk] = max(per_skill_scores[sk], int(r.get("total", 0)))
                if r.get("evidence"):
                    per_skill_evidence[sk].append(r["evidence"])

        strengths = []
        gaps = []
        for sk in required_skills:
            score = int(per_skill_scores.get(sk, 0))
            ev = " ".join(per_skill_evidence.get(sk, [])[:2]).strip()
            if score >= 65:
                strengths.append({
                    "skill": sk,
                    "level": score,
                    "note": ev or "Strong interview evidence for this required skill."
                })
            else:
                gaps.append({
                    "skill": sk,
                    "importance": "critical",
                    "note": (ev + " " if ev else "") + ("No or weak interview evidence for this required skill." if score == 0 else "Interview shows this skill needs improvement.")
                })

        gap_skills = [g["skill"] for g in gaps]
        roadmap = generate_gap_roadmap(gap_skills, scoring_rows, job_desc, job_title) if gap_skills else []

        # Supporting sections (best-effort; never block strengths/gaps/roadmap)
        extras = {}
        try:
            extras = generate_support_sections(job_desc, job_title, required_skills, strengths, gaps)
        except Exception:
            extras = {}

        # Overall match: average of required skill scores
        overall_match = int(round(sum(per_skill_scores.values()) / max(1, len(per_skill_scores))))

        result = {
            "overall_match": overall_match,
            "job_title": job_title or "Career Gap Analysis",
            "industry": "Detected from JD",
            "summary": "Assessment prioritizes interview answers (80%) over resume (20%) using a 4-part rubric. Strengths and gaps cover every skill explicitly required by the job description.",
            "required_skills": required_skills,
            "interview_scoring": scoring_rows,
            "strengths": strengths,
            "gaps": gaps,
            "experience_advice": {
                "years_needed": "See job description",
                "current_estimate": "Unknown",
                "advice": "Use the roadmap to close gaps skill-by-skill based on interview performance."
            },
            "roadmap": roadmap,
            "career_advice": extras.get("career_advice", []) if isinstance(extras.get("career_advice", []), list) else [],
            "cv_advice": extras.get("cv_advice", []) if isinstance(extras.get("cv_advice", []), list) else [],
            "alternative_roles": extras.get("alternative_roles", []) if isinstance(extras.get("alternative_roles", []), list) else [],
            "salary_insight": extras.get("salary_insight", {"range": "", "entry": "", "senior": "", "note": ""}) if isinstance(extras.get("salary_insight", {}), dict) else {"range": "", "entry": "", "senior": "", "note": ""},
            "next_steps": extras.get("next_steps", []) if isinstance(extras.get("next_steps", []), list) else [],
        }

        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyse', methods=['POST'])
def api_analyse():
    try:
        if not client:
            return jsonify({"error": "API key missing. Check run.py"}), 500

        job_desc  = request.form.get('job_description', '').strip()
        job_title = request.form.get('job_title', '').strip()
        cv_text = extract_cv_text_from_request(request)

        if not job_desc and not job_title:
            return jsonify({"error": "Please provide a job description or job title."}), 400

        # Backwards-compatible endpoint (no interview answers)
        result = analyse(cv_text, job_desc, job_title, interview_answers=None)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "model": "llama-3.3-70b-versatile", "api": "groq"})


if __name__ == '__main__':
    print("\nRunning at http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)