from functools import lru_cache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from graph import graph  # 仍保留你的 graph 連線

# ✅ 延遲初始化 vector store（避免在 import 階段就觸發 embedding API）
@lru_cache()
def get_vectorstore():
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        request_timeout=30,
        max_retries=5
    )

    return Neo4jVector.from_existing_index(
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

# ✅ Chat model 延遲初始化（可與 retriever 多次共用）
@lru_cache()
def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        request_timeout=60,
        max_retries=5
    )

# ✅ 查詢回答函式
def get_pharmacogenomics_answer(input):
    vectorstore = get_vectorstore()
    llm = get_llm()

    retriever = vectorstore.as_retriever()

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

    return pharm_retriever.invoke({"input": input})
