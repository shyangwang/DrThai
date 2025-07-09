import streamlit as st
from llm import llm
from graph import graph

from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer specializing in pharmacogenomics. Your task is to translate user questions into Cypher queries to retrieve relevant information from a pharmacogenomics knowledge graph.

The knowledge graph contains entities such as:
- Genes
- Gene Variants (e.g., SNPs)
- Drugs
- Diseases
- Drug Responses
- Clinical Guidelines
- Clinical Trials
- Lab Biomarkers

Convert the user's question into a Cypher query using only the relationship types and properties provided in the schema.

Instructions:
- Use only the provided relationship types and properties in the schema.
- Do not use any relationship types or properties that are not explicitly included in the schema.
- Do not return entire nodes or any embedding vector properties.
- When querying gene variants, refer to them using standard notation (e.g., rsID or amino acid change if mentioned).
- Ensure that drug names and gene symbols are capitalized appropriately.

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

# Create the Cypher QA chain
from langchain_neo4j import GraphCypherQAChain

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True
)


