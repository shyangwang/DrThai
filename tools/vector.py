import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from graph import graph  # 你的 Neo4j 連線設定

from langchain_community.vectorstores import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ✅ 1. 建立 OpenAI 向量模型
embedding = OpenAIEmbeddings(model="text-embedding-3-small")  # 或 "text-embedding-ada-002"

# ✅ 2. 建立 Neo4j Vector Index
neo4jvector = Neo4jVector.from_existing_index(
    embedding=embedding,
    graph=graph,
    index_name="entity_vector",
    node_label="PharmConcept",
    text_node_property="description",
    embedding_node_property="descriptionEmbedding",
    retrieval_query="""
RETURN
    node.description AS text,
    score,
    {
        name: node.name,
        type: node.type,
        relatedGenes: [(g:Gene)-[:RELATED_TO]->(node) | g.name],
        relatedDrugs: [(d:Drug)-[:AFFECTS]->(node) | d.name],
        relatedConditions: [(c:MedicalCondition)-[:TREATED_BY|ASSOCIATED_WITH]->(node) | c.name],
        guidelines: [(g:Guideline)-[:RECOMMENDS]->(node) | g.name],
        source: node.source,
        source_url: 'https://yourdomain.com/source/' + toString(node.source)
    } AS metadata
"""
)

# ✅ 3. 建立 Chat 模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ✅ 4. 建立 QA Chain
retriever = neo4jvector.as_retriever()

instructions = (
    "You are a pharmacogenomics assistant. Use the provided context to answer the question. "
    "If the answer is not in the context, say 'I don't know'. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
pharm_retriever = create_retrieval_chain(retriever, question_answer_chain)

# ✅ 5. 回答查詢函式
def get_pharmacogenomics_answer(input):
    return pharm_retriever.invoke({"input": input})

# # ✅ 6. Streamlit 介面
# st.title("💊 Pharmacogenomics Assistant")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if prompt := st.chat_input("請輸入你的問題..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         try:
#             response = get_pharmacogenomics_answer(prompt)
#             answer = response.get("answer", "I'm not sure.")
#             st.markdown(answer)

#             # ✅ 顯示引用來源
#             docs = response.get("context", [])
#             if docs:
#                 st.markdown("---")
#                 st.markdown("**References:**")
#                 for i, doc in enumerate(docs):
#                     meta = doc.metadata
#                     name = meta.get("name", f"Source {i+1}")
#                     url = meta.get("source_url")
#                     if url:
#                         st.markdown(f"- [{name}]({url})", unsafe_allow_html=True)
#                     else:
#                         st.markdown(f"- {name}")

#             st.session_state.messages.append({"role": "assistant", "content": answer})

#         except Exception as e:
#             st.error(f"❌ 發生錯誤：{str(e)}")
