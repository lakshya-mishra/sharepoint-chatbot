# frontend.py
import streamlit as st
import requests
import base64
import uuid
import pandas as pd

# ==================== CONFIG ====================
BACKEND_URL = "http://127.0.0.1:5000/query"  # Adjust if running elsewhere
FAVICON_PATH = "data/favicon.png"  # Your local favicon
APP_TITLE = "Prism"

# Function to convert image file to Base64
def image_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


favicon_base64 = image_to_base64(FAVICON_PATH)

st.set_page_config(page_title=APP_TITLE, page_icon=FAVICON_PATH, layout="centered")

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    # message schema:
    # { "role": "user" | "bot", "type": "text" | "table",
    #   "content": str (if text), "table": {"columns": [...], "rows": [...] } (if table) }
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # unique per user session

if "refined_queries" not in st.session_state:
    st.session_state.refined_queries = []

if "chip_selected" not in st.session_state:
    st.session_state.chip_selected = None


# ==================== HEADER ====================
st.markdown(
    f"""
     <h1 style='text-align: center;'>
        <img src='data:image/png;base64,{favicon_base64}' width='28' style='vertical-align:middle; margin-right:8px;'/>
        {APP_TITLE}
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:30px; margin-bottom:20px;">
        Explore data with ease with your AI assistant : Prism
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================== HELPERS ====================
def render_bot_bubble_start():
    st.markdown(
        f"""
        <div style="display:flex; justify-content:flex-start; margin:8px 0;">
            <div style="background-color:#F1F0F0; padding:10px 15px; border-radius:12px; 
            max-width:70%; text-align:left; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                <b><img src='data:image/png;base64,{favicon_base64}' width='16' 
                style='vertical-align:middle; margin-right:4px;'/> {APP_TITLE}</b><br>
        """,
        unsafe_allow_html=True,
    )


def render_bot_bubble_end():
    st.markdown("</div></div>", unsafe_allow_html=True)


def append_user_message(text: str):
    st.session_state.messages.append({"role": "user", "type": "text", "content": text})


def append_bot_text(text: str):
    st.session_state.messages.append({"role": "bot", "type": "text", "content": text})


def append_bot_table(columns, rows):
    st.session_state.messages.append(
        {
            "role": "bot",
            "type": "table",
            "table": {"columns": columns, "rows": rows},
        }
    )


def process_backend_response(data: dict):
    """
    Handle new contract:
      - response_type: "text" | "table" | "refine" | "error"
    Fallback to old contract if response_type absent.
    """
    resp_type = data.get("response_type")

    if resp_type == "table":
        table = data.get("table", {})
        columns = table.get("columns", [])
        rows = table.get("rows", [])
        append_bot_table(columns, rows)
        # clear chips if any
        st.session_state.refined_queries = []
        return

    if resp_type == "text":
        text = data.get("response", "⚠️ No answer received.")
        append_bot_text(text)
        st.session_state.refined_queries = []
        return

    if resp_type == "refine":
        options = data.get("refined_queries", [])
        st.session_state.refined_queries = options or []
        append_bot_text(
            "I need a bit more context. Please choose one of the options below:"
        )
        return

    if resp_type == "error":
        err = data.get("error") or data.get("response") or "⚠️ Unknown error."
        append_bot_text(f"⚠️ {err}")
        return

    # ---------- Fallback for older backend responses ----------
    if data.get("response") is not None:
        append_bot_text(str(data["response"]))
        st.session_state.refined_queries = []
        return

    if data.get("refined_queries"):
        st.session_state.refined_queries = data["refined_queries"]
        append_bot_text(
            "I need a bit more context. Please choose one of the options below:"
        )
        return

    append_bot_text("⚠️ I couldn't process your request.")


def call_backend_and_process(query_text: str):
    try:
        response = requests.post(
            BACKEND_URL,
            json={"query": query_text, "session_id": st.session_state.session_id},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            process_backend_response(data)
        else:
            append_bot_text(
                f"⚠️ Backend error {response.status_code}: {response.text}"
            )
    except Exception as e:
        append_bot_text(f"⚠️ Connection error: {str(e)}")


# ==================== CHAT UI ====================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin:8px 0;">
                <div style="background-color:#DCF8C6; padding:10px 15px; border-radius:12px; 
                max-width:70%; text-align:right; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                    <b>🧑 You</b><br>{msg.get("content", "")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Bot bubble: support text and table types
        render_bot_bubble_start()
        if msg.get("type") == "table":
            table = msg.get("table", {})
            columns = table.get("columns", [])
            rows = table.get("rows", [])
            df_disp = pd.DataFrame(rows)
            # Respect column order if provided
            if columns:
                # only keep columns that exist in df to avoid KeyError
                ordered_cols = [c for c in columns if c in df_disp.columns]
                df_disp = df_disp[ordered_cols]
            st.dataframe(df_disp, use_container_width=True)
        else:
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
        render_bot_bubble_end()


# ==================== REFINED QUERY CHIPS ====================
if st.session_state.refined_queries:
    st.markdown("---")
    st.markdown("⚡ I need a bit more context. Did you mean one of these?")

    # Style chips (pill-shaped clickable items)
    chip_css = """
        <style>
        .chip-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
        .chip {
            padding: 8px 15px;
            border-radius: 20px;
            background-color: #e6f2ff;
            color: #004080;
            cursor: pointer;
            font-size: 14px;
            border: 1px solid #99c2ff;
            transition: 0.2s;
        }
        .chip:hover {
            background-color: #cce0ff;
        }
        </style>
    """
    st.markdown(chip_css, unsafe_allow_html=True)

    # Render chips as buttons
    cols = st.columns(len(st.session_state.refined_queries))
    for i, query in enumerate(st.session_state.refined_queries):
        if cols[i].button(query, key=f"chip_{i}"):
            st.session_state.chip_selected = query

# If chip selected → send to backend
if st.session_state.chip_selected:
    query = st.session_state.chip_selected
    append_user_message(query)

    with st.spinner("Thinking..."):
        call_backend_and_process(query)

    st.session_state.chip_selected = None
    # Clear refined queries after selection to avoid duplicates
    st.session_state.refined_queries = []
    st.rerun()


# ==================== INPUT BOX ====================
user_input = st.chat_input("Type your message here...")

if user_input:
    append_user_message(user_input)

    with st.spinner("Thinking..."):
        call_backend_and_process(user_input)

    st.rerun()