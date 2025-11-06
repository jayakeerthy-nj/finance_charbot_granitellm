import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# <CHANGE> Configure page with wide layout and custom theme
st.set_page_config(
    page_title="💰 Personal Finance Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Personal Finance Chatbot powered by IBM Granite 3.2 Local LLM"
    }
)

# <CHANGE> Add custom CSS for enhanced styling
st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem;
    }
    
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .user-message {
        
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .assistant-message {
        
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .info-box {
        
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .success-box {
       
        border: 1px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# <CHANGE> Load model with better error handling
@st.cache_resource
def load_model():
    try:
        model_name = "ibm-granite/granite-3.2-2b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.float16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None

# <CHANGE> Initialize sidebar with helpful information
with st.sidebar:
    st.markdown("### 📚 Finance Tips")
    st.info("""
    **Quick Finance Tips:**
    - 💡 Save 20% of your income
    - 📊 Diversify investments
    - 🏦 Build emergency fund
    - 💳 Pay credit cards on time
    - 📈 Track your spending
    """)
    
    st.markdown("---")
    
    # <CHANGE> Add chat management controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📊 Summary", use_container_width=True):
            st.session_state.show_summary = True
    
    st.markdown("---")
    
    # <CHANGE> Display model info
    st.markdown("### 🤖 Model Info")
    st.caption("**Model:** IBM Granite 3.2 (2B)")
    st.caption("**Type:** Local Inference")
    device_info = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
    st.caption(f"**Device:** {device_info}")

# <CHANGE> Enhanced main header with gradient styling
st.markdown("""
<div class="chat-container">
    <h1 style="text-align: center; margin: 0;">💬 Personal Finance Chatbot</h1>
    <p style="text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;">Your AI-powered financial advisor powered by IBM Granite 3.2</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False

# Load model
tokenizer, model, device = load_model()

if model is None:
    st.error("⚠️ Unable to load model. Please check your dependencies.")
    st.stop()

# <CHANGE> Add topic suggestions
st.markdown("### 💭 Quick Topics")
col1, col2, col3, col4 = st.columns(4)

topics = {
    col1: ("💰", "Saving Tips"),
    col2: ("📈", "Investing"),
    col3: ("🏦", "Budgeting"),
    col4: ("💳", "Debt Help")
}

for col, (emoji, topic) in topics.items():
    with col:
        if st.button(f"{emoji} {topic}", use_container_width=True, key=topic):
            st.session_state.user_input = f"Tell me about {topic.lower()}"

st.markdown("---")

# <CHANGE> Enhanced chat input with better styling
user_input = st.chat_input(
    "💬 Ask me anything about saving, taxes, investing, or budgeting...",
    key="user_input"
)

if user_input:
    # Add user input to chat
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # <CHANGE> Show loading state with spinner
    with st.spinner("🤔 Thinking..."):
        # <CHANGE> Model inference with improved prompting
        prompt = f"""You are a helpful, knowledgeable financial advisor. Provide clear, actionable advice about personal finance topics including savings, taxes, investing, budgeting, and debt management. Keep your responses concise and practical.

User: {user_input}
Assistant:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
    
    # Add assistant response to chat
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()

# <CHANGE> Improved chat display with better styling
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <b>You:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <b>💡 Financial Advisor:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # <CHANGE> Add statistics about chat
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Messages", len(st.session_state.chat_history))
    
    with col2:
        user_messages = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
        st.metric("Your Questions", user_messages)
    
    with col3:
        avg_length = sum(len(m["content"].split()) for m in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.metric("Avg Response Length", f"{int(avg_length)} words")

# <CHANGE> Add footer with disclaimer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <b>⚠️ Disclaimer:</b> This chatbot provides general financial information only. 
    For serious financial decisions, consult a licensed financial advisor.
</div>
""", unsafe_allow_html=True)

st.caption("🚀 Powered by IBM Granite 3.2 | Local LLM Inference | Privacy First")
