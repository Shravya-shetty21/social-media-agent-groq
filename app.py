
"""
Groq Social Media Agent - Streamlit App
Uses Groq's OpenAI-compatible chat completions API.
"""

import os
import json
import io
import csv
import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import requests

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Groq OpenAI-compatible chat completions endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    st.stop()  # stops app if key missing
    raise SystemExit("ERROR: GROQ_API_KEY missing in .env")

# ---------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------
MASTER_PROMPT_TEMPLATE = """
Generate a {duration}-day social media content calendar for these platforms: {platforms}.
Tone: {tone}.
Audience: {audience}.
Start date: {start_date}.
Caption limit: {char_limit} characters.

Each post must include:
- id
- date
- platform
- post_type (image, reel, story, text, link)
- caption
- hashtags (list)
- image_prompt
- alt_text
- CTA

Brand info:
{brand_info}

Return ONLY a JSON array.
"""

SAFETY_CHECK_PROMPT = """
Check this caption for safety. Respond with:
{{"status":"SAFE" or "UNSAFE", "replacement":""}}

Caption: "{caption}"
"""

# ---------------------------------------------------------
# Groq Model Call (Chat Completions)
# ---------------------------------------------------------
def call_groq(prompt: str, max_tokens: int = 800) -> str:
    """
    Calls Groq's OpenAI-compatible chat completions endpoint.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful social media content generator.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=40)

    if resp.status_code != 200:
        raise RuntimeError(f"Groq Error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ---------------------------------------------------------
# JSON Extraction Helper
# ---------------------------------------------------------
def extract_json(text: str) -> List[Dict[str, Any]]:
    text = text.strip()

    if text.startswith("["):
        return json.loads(text)

    first = text.find("[")
    last = text.rfind("]")

    if first == -1 or last == -1:
        raise ValueError("No JSON array found in response")

    return json.loads(text[first : last + 1])

# ---------------------------------------------------------
# Generate Calendar
# ---------------------------------------------------------
def generate_calendar(
    brand_info: str,
    tone: str,
    audience: str,
    platforms: List[str],
    duration: int,
    start_date: str,
    char_limit: int,
):
    prompt = MASTER_PROMPT_TEMPLATE.format(
        duration=duration,
        platforms=", ".join([p.strip() for p in platforms]),
        tone=tone,
        audience=audience,
        start_date=start_date,
        char_limit=char_limit,
        brand_info=brand_info,
    )

    raw = call_groq(prompt, max_tokens=1400)
    posts = extract_json(raw)

    # Normalization
    for i, p in enumerate(posts, start=1):
        if "id" not in p:
            p["id"] = i
        if isinstance(p.get("hashtags"), str):
            p["hashtags"] = p["hashtags"].split()
        if "hashtags" not in p:
            p["hashtags"] = []

    return posts

# ---------------------------------------------------------
# Safety Checker (optional helper, not wired into UI yet)
# ---------------------------------------------------------
def safety_check(caption: str):
    prompt = SAFETY_CHECK_PROMPT.format(caption=caption.replace('"', '\\"'))

    try:
        result = call_groq(prompt, max_tokens=200)
        data = json.loads(result)
        return data
    except Exception:
        return {"status": "SAFE", "replacement": ""}

# ---------------------------------------------------------
# CSV Export
# ---------------------------------------------------------
def posts_to_csv(posts: List[Dict[str, Any]]):
    output = io.StringIO()
    writer = csv.writer(output)

    header = [
        "id",
        "date",
        "platform",
        "post_type",
        "caption",
        "hashtags",
        "image_prompt",
        "alt_text",
        "CTA",
    ]
    writer.writerow(header)

    for p in posts:
        hashtags = (
            " ".join(p["hashtags"])
            if isinstance(p.get("hashtags"), list)
            else p.get("hashtags", "")
        )
        writer.writerow(
            [
                p.get("id", ""),
                p.get("date", ""),
                p.get("platform", ""),
                p.get("post_type", ""),
                p.get("caption", ""),
                hashtags,
                p.get("image_prompt", ""),
                p.get("alt_text", ""),
                p.get("CTA", ""),
            ]
        )

    return output.getvalue().encode("utf-8")

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Groq Social Media Agent", layout="centered")
st.title("📱 Groq Social Media Agent")

st.caption("Generate multi-day content calendars using Groq (LLM).")

# Keep posts in session state
if "posts" not in st.session_state:
    st.session_state["posts"] = []

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand_name = st.text_input("Brand name", value="My Brand")
        tone = st.selectbox("Tone", ["Friendly", "Professional", "Playful"])
        duration = st.number_input("Duration (days)", min_value=1, max_value=60, value=5)
        char_limit = st.number_input(
            "Caption character limit", min_value=50, max_value=1000, value=220
        )

    with col2:
        audience = st.text_input("Audience", value="Teens & adults")
        platforms_str = st.text_input("Platforms (comma separated)", value="Instagram")
        start_date = st.date_input(
            "Start date", value=datetime.date.today()
        ).isoformat()

    brand_description = st.text_area(
        "Brand description", value="Eco-friendly skincare products."
    )

    submitted = st.form_submit_button("Generate Calendar")

if submitted:
    platforms = [p.strip() for p in platforms_str.split(",") if p.strip()]

    with st.spinner("Calling Groq and generating calendar..."):
        try:
            posts = generate_calendar(
                f"{brand_name}: {brand_description}",
                tone,
                audience,
                platforms,
                int(duration),
                start_date,
                int(char_limit),
            )
            st.session_state["posts"] = posts
            st.success(f"Generated {len(posts)} posts ✅")
        except Exception as e:
            st.error(f"Generation error: {e}")

posts = st.session_state.get("posts", [])

if posts:
    st.subheader(f"Generated Posts ({len(posts)})")

    # Prepare data for table display (hashtags as string)
    display_rows = []
    for p in posts:
        display_rows.append(
            {
                "ID": p.get("id", ""),
                "Date": p.get("date", ""),
                "Platform": p.get("platform", ""),
                "Type": p.get("post_type", ""),
                "Caption": p.get("caption", ""),
                "Hashtags": " ".join(p.get("hashtags", []))
                if isinstance(p.get("hashtags"), list)
                else p.get("hashtags", ""),
                "CTA": p.get("CTA", ""),
            }
        )

    st.dataframe(display_rows, use_container_width=True)

    # Download buttons
    col_j, col_c = st.columns(2)
    with col_j:
        json_bytes = json.dumps(posts, indent=2).encode("utf-8")
        st.download_button(
            label="⬇️ Download JSON",
            data=json_bytes,
            file_name="calendar.json",
            mime="application/json",
        )
    with col_c:
        csv_bytes = posts_to_csv(posts)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="calendar.csv",
            mime="text/csv",
        )
else:
    st.info("Fill the form and click **Generate Calendar** to create posts.")