import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("Dr. Tsai", page_icon=":drug:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm Dr. Tsai Chatbot! How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    with st.spinner('Thinking...'):
        try:
            # ✅ 呼叫 agent 並接收完整回應（包含 answer 與 context）
            response = generate_response(message)

            # 如果是 dict 格式就解析出 answer 和 context，否則直接使用 answer
            if isinstance(response, dict):
                answer = response.get("answer", "I'm not sure.")
                context_docs = response.get("context", [])
            else:
                answer = response
                context_docs = []

            # ✅ 顯示主回覆
            write_message('assistant', answer)

            # ✅ 顯示引用來源（如有）
            if context_docs:
                references = "**References:**\n"
                for i, doc in enumerate(context_docs):
                    meta = doc.get("metadata", {})
                    name = meta.get("name", f"Source {i+1}")
                    url = meta.get("source_url")
                    if url:
                        references += f"- [{name}]({url})\n"
                    else:
                        references += f"- {name}\n"

                write_message("assistant", references)

        except Exception as e:
            write_message("assistant", f"❌ Error: {str(e)}")

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle user input
if question := st.chat_input("What is up?"):
    write_message('user', question)
    handle_submit(question)
