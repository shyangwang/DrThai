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
        # ✅ 呼叫 agent 並接收完整回應（包含 answer 與 context）
        response = generate_response(message)

        # ⛳ 取出答案與引用資料
        answer = response.get("answer", "I'm not sure.")
        context_docs = response.get("context", [])

        # ✅ 顯示主回覆
        write_message('assistant', answer)

        # ✅ 顯示引用來源（如有）
        if context_docs:
            references = "**References:**\n"
            for i, doc in enumerate(context_docs):
                # 這裡假設 doc 是 Document 物件（或 dict 有 metadata 欄位）
                meta = getattr(doc, "metadata", doc.get("metadata", {}))
                name = meta.get("name", f"Source {i+1}")
                url = meta.get("source") or meta.get("source_url")

                if url and url.startswith("http"):
                    references += f"- [{name}]({url})\n"
                else:
                    references += f"- {name}\n"

            # ✅ 顯示於 UI 並存進對話記錄
            write_message("assistant", references)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle user input
if question := st.chat_input("What is up?"):
    write_message('user', question)
    handle_submit(question)
