import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from graph import graph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ✅ 明確設定金鑰
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# ✅ 同樣建議在 llm 也傳入金鑰
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]
)

# Neo4j 向量連結
neo4jvector = Neo4jVector.from_existing_index(
    embedding=embedding,
    graph=graph,
    index_name="entity_vector",
    node_label="PharmConcept",
    text_node_property="description",
    embedding_node_property="descriptionEmbedding",
    retrieval_query="""..."""
)

retriever = neo4jvector.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a pharmacogenomics assistant. Context: {context}"),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
pharm_retriever = create_retrieval_chain(retriever, question_answer_chain)

def get_pharmacogenomics_answer(input):
    return pharm_retriever.invoke({"input": input})
