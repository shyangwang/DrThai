import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from graph import graph  # 仍保留你的 graph 連線

from langchain_community.vectorstores import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ✅ 改用 OpenAI 的 Embedding 模型
embedding = OpenAIEmbeddings(model="text-embedding-3-small")  # 也可用 "text-embedding-ada-002"

# ✅ 建立 Neo4j 向量索引連結（可共用 HuggingFace/ OpenAI 的 embedding）
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

# ✅ 改為使用 OpenAI Chat 模型作為 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Retriever 與 QA Chain
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

# 查詢函式
def get_pharmacogenomics_answer(input):
    return pharm_retriever.invoke({"input": input})
