import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from graph import graph

from langchain_community.vectorstores import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ✅ 加入 API 金鑰從 secrets
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# ✅ Neo4j 向量搜尋初始化
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
        relatedConditions: [(c:Condition)-[:TREATED_BY|ASSOCIATED_WITH]->(node) | c.name],
        guideline: [(g:Guideline)-[:RECOMMENDS]->(node) | g.name],
        source: node.source
    } AS metadata
"""
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]  # 建議也明確指定
)

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

def get_pharmacogenomics_answer(input):
    return pharm_retriever.invoke({"input": input})
