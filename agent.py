from llm import llm
from graph import graph

# Create a medical chat chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical expert providing information about Pharmacogenomics."),
        ("human", "{input}"),
    ]
)

medical_chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools
from langchain.tools import Tool
from tools.cypher import cypher_qa

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general medical chat not covered by other tools",
        func=medical_chat.invoke,
    ), 
    Tool.from_function(
        name="Pharmacogenomics Q&A",
        description="Use this tool to answer questions about pharmacogenomics, drug-gene interactions, and related conditions based on knowledge graph search.",
        func=get_pharmacogenomics_answer,
    ),
    Tool.from_function(
        name="Medical information",
        description="Provide information about medical questions using Cypher",
        func = cypher_qa
    )
]

# Create chat history callback
from langchain_neo4j import Neo4jChatMessageHistory

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


# Create the agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_core.prompts import PromptTemplate

agent_prompt = PromptTemplate.from_template("""
You are a medical expert providing information about Pharmacogenomics.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to medicine, diseases, genes or drugs.
Do not answer any questions about private information.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")


agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent

from utils import get_session_id

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']
