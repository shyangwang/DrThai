import streamlit as st
from llm import llm, embeddings
from graph import graph

from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Step 1: 定義向量索引參數
neo4jvector = Neo4jVector.from_existing_index(
    embeddings=embeddings,
    graph=graph,
    index_name="pharmPGxIndex",                  # 🔁 替換為你建好的 pharmacogenomics index 名稱
    node_label="PharmConcept",                   # 🔁 節點類型，可統一命名或用多類型
    text_node_property="description",            # 🔁 存放文本描述欄位
    embedding_node_property="descriptionEmbedding",  # 🔁 嵌入向量欄位
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

# Step 2: 建立 retriever 與 QA chain
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

# Step 3: 提供查詢介面
def get_pharmacogenomics_answer(input):
    return pharm_retriever.invoke({"input": input})
