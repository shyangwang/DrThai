import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from graph import graph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ✅ 使用 OpenAI 的 embedding 模型
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# ✅ 連接 Neo4j 向量索引（假設已建立於 PharmConcept 節點）
neo4jvector = Neo4jVector.from_existing_index(
    embedding=embedding,
    graph=graph,
    index_name="entity_vector",
    node_label="PharmConcept",  # 確保此 Label 實際存在
    text_node_property="description",  # 查詢主體欄位
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
        guidelines: [(gd:Guideline)-[:RECOMMENDS]->(node) | gd.name],
        source: node.source
    } AS metadata
"""
)

# ✅ 使用 OpenAI Chat 模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ✅ 建立 Retriever + QA Chain
retriever = neo4jvector.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a pharmacogenomics assistant. Use the provided context to answer the question. "
     "If the answer is not in the context, say 'I don't know'.\nContext: {context}"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
pharm_retriever = create_retrieval_chain(retriever, qa_chain)

# ✅ 封裝查詢函式
def get_pharmacogenomics_answer(user_input: str):
    return pharm_retriever.invoke({"input": user_input})
