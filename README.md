# finance_charbot_granitellm
# Personal Finance Chatbot (IBM Granite 3.2 LLM)

An AI-powered personal finance assistant built using Streamlit and IBM Granite 3.2 (2B Instruct LLM) for local inference.  
This chatbot helps users make smarter financial decisions by offering advice on saving, budgeting, investing, debt management, and more — all while running securely on your local machine.

---

## Features

- Local AI Model (Privacy First) — Runs entirely on your system using IBM Granite 3.2 2B Instruct.  
- Interactive Chat Interface — Ask questions naturally and get practical, concise financial advice.  
- Custom Space-Themed UI — Modern Streamlit interface with gradient styling and chat bubbles.  
- Quick Finance Tips Sidebar — Built-in reminders for healthy money habits.  
- Chat Summary Metrics — Tracks conversation stats like message count and average response length.  
- Chat Controls — Clear chat history or view summary with one click.  
- Completely Offline — No data sent to external APIs.

---

## Use Cases

### Scenario 1 — Saving While Paying Student Loans
> “How can I save while repaying my student loans?”  
The chatbot helps plan your savings strategy while managing loan repayments and expenses effectively.

### Scenario 2 — Monthly Expense Breakdown
> “Here’s my income and expenses. Can you summarize them?”  
It calculates your total savings, spending distribution, and provides actionable recommendations.

### Scenario 3 — Smart Investment Advice
> “What are some safe investment options for beginners?”  
The chatbot explains diversified, low-risk strategies tailored to your goals.

### Scenario 4 — Debt Repayment Planning
> “How can I pay off my credit card debt faster?”  
It suggests methods like the snowball or avalanche strategy and how to optimize repayment schedules.

---

## Model Information

| Detail | Description |
|:--|:--|
| Model | IBM Granite 3.2 (2B Instruct) |
| Framework | Hugging Face Transformers |
| Inference Type | Local (CPU / GPU Supported) |
| Precision | float16 |
| Pipeline | text-generation |

---

## Installation & Setup

1. Clone the Repository
bash
git clone https://github.com/<your-username>/personal-finance-chatbot.git
cd personal-finance-chatbot

3. Create and Activate Virtual Environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

4. Install Dependencies
pip install -r requirements.txt


requirements.txt:
streamlit
transformers
torch
torchaudio
torchvision

4. Run the Application
streamlit run app.py

