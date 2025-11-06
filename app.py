import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import time

# ======================================================
# Safe rerun helper (compatible across Streamlit versions)
# ======================================================
def do_rerun():
    """Safely trigger a Streamlit rerun across Streamlit versions."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass
    st.session_state["_rerun_trigger"] = not st.session_state.get("_rerun_trigger", False)
    st.stop()


# ======================================================
# Load IBM Granite Model Locally
# ======================================================
@st.cache_resource
def load_model():
    model_name = "ibm-granite/granite-3.2-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # only use fp16 on CUDA to avoid CPU float16 issues
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


# ======================================================
# App config and theme
# ======================================================
st.set_page_config(page_title="Personal Finance Chatbot", layout="wide")

st.markdown(
    """
    <style>
    /* ---------- Space Black Theme ---------- */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0a0a0a 0%, #000000 50%, #000000 100%);
        color: #e6eef8;
        height: 100%;
        overflow: auto;
    }

    /* Subtle drifting stars */
    .stApp::before {
        content: "";
        position: fixed;
        left: 0; top: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        background-image:
          radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.10), rgba(255,255,255,0.00)),
          radial-gradient(1px 1px at 30% 40%, rgba(255,255,255,0.08), rgba(255,255,255,0.00)),
          radial-gradient(1px 1px at 60% 10%, rgba(255,255,255,0.07), rgba(255,255,255,0.00)),
          radial-gradient(1px 1px at 80% 70%, rgba(255,255,255,0.06), rgba(255,255,255,0.00));
        opacity: 0.9;
        mix-blend-mode: screen;
        animation: stars-drift 60s linear infinite;
        z-index: -1;
    }

    @keyframes stars-drift {
        from { transform: translateY(0px) translateX(0px) scale(1); }
        to   { transform: translateY(-40px) translateX(20px) scale(1.02); }
    }

    .chat-column {
        max-width: 900px;
        margin: 20px auto;
        padding: 20px;
        border-radius: 12px;
        backdrop-filter: blur(6px) saturate(120%);
    }

    .title-area { display:flex; align-items:center; gap:14px; }
    .logo-circle {
        width:44px; height:44px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        font-size:18px;
        background:linear-gradient(90deg,#3b82f6,#06b6d4);
        color: #021025;
    }

    /* message bubbles */
    .bubble {
        padding:12px 16px;
        border-radius:12px;
        margin:6px 0;
        max-width:80%;
        line-height:1.45;
        opacity:0;
        transform: translateY(4px);
        animation: fadeUp .18s ease forwards;
        word-break: break-word;
    }
    .bubble.assistant {
        background: rgba(255,255,255,0.04);
        align-self:flex-start;
        color: #e6eef8;
    }
    .bubble.user {
        background: linear-gradient(90deg,#06b6d4,#0ea5a1);
        color: #021025;
        align-self:flex-end;
    }

    @keyframes fadeUp { to { opacity: 1; transform: translateY(0); } }

    .messages { display:flex; flex-direction:column; gap:6px; padding:12px; max-height:60vh; overflow:auto; }
    .controls { background: rgba(255,255,255,0.02); padding:12px; border-radius:10px; }
    .muted { color: #9fb3c8; font-size:13px; }
    .example { display:inline-block; padding:6px 10px; margin:4px; border-radius:999px; background: rgba(255,255,255,0.03); cursor:pointer; }
    footer { visibility: hidden; }
    button[role="button"] { transition: transform .08s ease, box-shadow .08s ease; }
    button[role="button"]:active { transform: translateY(1px); }
    .prompt-preview { background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; margin:8px 0; color:#dbeafe; font-size:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# Load model
# ======================================================
with st.spinner("Loading model..."):
    tokenizer, model, device = load_model()

# ======================================================
# Initialize state
# ======================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi — ask me anything about saving, taxes, or investing."}
    ]
# generation knobs defaults
if "temp" not in st.session_state:
    st.session_state["temp"] = 0.7
if "top_p" not in st.session_state:
    st.session_state["top_p"] = 0.9

# ======================================================
# Helper: render messages
# ======================================================
def render_messages():
    messages_container = st.container()
    with messages_container:
        st.markdown("<div class='messages'>", unsafe_allow_html=True)
        for m in st.session_state.chat_history:
            role = m["role"]
            content = m["content"].replace("\n", "<br/>")
            if role == "user":
                st.markdown(
                    f"<div style='display:flex; justify-content:flex-end;'><div class='bubble user'>{content}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='display:flex; justify-content:flex-start;'><div class='bubble assistant'>{content}</div></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# Layout
# ======================================================
left_col, right_col = st.columns([0.78, 0.22])

# ---------------- LEFT ----------------
with left_col:
    st.markdown(
        """
        <div class='chat-column'>
        <div class='title-area'>
            <div class='logo-circle'>•</div>
            <div>
                <h2 style='margin:0'>Personal Finance Chatbot</h2>
                <div class='muted'>Local: IBM Granite 3.2 — concise, responsible finance help</div>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # show messages initially
    render_messages()

    # example prompts
    examples = [
        "How can I save while repaying student loans?",
        "What's a safe investment plan for beginners?",
        "How do I create a monthly budget easily?",
    ]
    st.markdown("<div class='controls'>Try an example:</div>", unsafe_allow_html=True)
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, key=f"ex{i}"):
            # append user message and set input_value so main loop consumes it
            st.session_state.chat_history.append({"role": "user", "content": ex})
            st.session_state["input_value"] = ex

    # ---------------- INPUT FORM with safe submit callback ----------------
    def _submit_callback():
        # capture the text_area value into a safe key and clear the widget value (allowed inside callback)
        st.session_state["input_value"] = st.session_state.get("prompt_text", "").strip()
        st.session_state["prompt_text"] = ""

    with st.form(key="input_form", clear_on_submit=False):
        # label provided for accessibility but collapsed visually
        st.text_area(
            "Message",
            placeholder="Ask me about saving, taxes, or investing...",
            key="prompt_text",
            height=90,
            label_visibility="collapsed",
        )
        cols = st.columns([0.86, 0.14])
        submit = cols[1].form_submit_button("Send", on_click=_submit_callback)

    # If we have an input_value (either from the callback or examples), process it now
    if st.session_state.get("input_value"):
        user_input = st.session_state.pop("input_value")  # consume it once

        # append user message and render immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        render_messages()

        # show prompt preview
        prompt_preview = st.empty()
        prompt_preview.markdown(
            f"<div class='prompt-preview'><strong>Prompt preview</strong><br/>{user_input}</div>",
            unsafe_allow_html=True,
        )

        # compose model prompt
        prompt = (
            f"You are a helpful, responsible financial assistant. Answer concisely and clearly.\n\nUser: {user_input}\nAssistant:"
        )

        # model generation
        with st.spinner("Generating reply..."):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=st.session_state.get("temp", 0.7),
                    top_p=st.session_state.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # extract reply
        if "Assistant:" in decoded:
            reply = decoded.split("Assistant:")[-1].strip()
        else:
            reply = re.sub(re.escape(prompt), "", decoded, flags=re.DOTALL).strip()
        if not reply:
            reply = "Sorry — I couldn't generate an answer right now. Try rephrasing your question."

        # remove preview, append empty assistant placeholder, render
        prompt_preview.empty()
        st.session_state.chat_history.append({"role": "assistant", "content": ""})
        render_messages()

        # typing animation into a temporary slot
        assistant_slot = st.empty()
        displayed = ""
        for ch in reply:
            displayed += ch
            assistant_slot.markdown(
                f"<div style='display:flex; justify-content:flex-start;'><div class='bubble assistant'>{displayed.replace(chr(10), '<br/>')}</div></div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.01)
        # save final reply to history and re-render
        st.session_state.chat_history[-1]["content"] = reply
        render_messages()

    st.markdown("<div class='muted'>Answers are informational only — not professional financial advice.</div>", unsafe_allow_html=True)


# ---------------- RIGHT ----------------
with right_col:
    st.markdown("<div class='controls'>", unsafe_allow_html=True)
    st.markdown("**Conversation**")

    if st.button("Clear conversation"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hi — ask me anything about saving, taxes, or investing."}
        ]
        do_rerun()

    # Provide download button directly (not gated by a separate button) for smoother UX
    export_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
    st.download_button("Export conversation (.txt)", data=export_text, file_name="conversation.txt")

    st.markdown("---")
    st.markdown("**Model Settings**")
    st.markdown(f"Model: `{model.__class__.__name__}`  ")
    st.markdown(f"Device: `{device}`  ")

    temp = st.slider("Temperature", 0.0, 1.0, value=st.session_state.get("temp", 0.7), step=0.05)
    top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, value=st.session_state.get("top_p", 0.9), step=0.05)
    st.session_state["temp"] = temp
    st.session_state["top_p"] = top_p

    st.markdown("---")
    st.markdown("**Quick Tips**")
    st.markdown("- Ask concise financial questions.  \n- Include numbers for personalized replies.  \n- Try: ‘Create a 3-month budget plan’.")
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("Reset UI"):
    do_rerun()
