import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Load environment variables (assuming you have an .env file with OPENAI_API_KEY)
load_dotenv()

# Create a sample knowledge base text file
with open("knowledge_base.txt", "w") as f:
    f.write("""
LangChain is a framework for developing applications powered by language models.
It enables applications that:
- Are context-aware: connect a language model to sources of context (prompt instructions, few-shot examples, content to ground responses in, etc.)
- Reason: use a language model to reason (chain of thought prompting, self-critique, structured output, etc.)

LangChain provides standard, extendable interfaces and external integrations
for the following modules:
1. Model I/O: Interface with language models
2. Retrieval: Interface with application-specific data
3. Agents: Let chains choose which tools to use given high-level directives
""")

# Load documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Initialize chat model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Approach 1: ConversationalRetrievalChain
def create_chain():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    
    return qa_chain

# Approach 2: Agent-based implementation
def create_agent():
    # Create a retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="knowledge_base_search",
        description="Searches and returns information from the knowledge base about LangChain."
    )
    
    tools = [retriever_tool]
    
    # Create a prompt template with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the provided tools to answer the user's questions about LangChain. If you don't know the answer, say so."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create an agent executor with memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    
    return agent_executor

# Demo of using the chain
chain = create_chain()
response = chain.invoke({"question": "What is LangChain?"})
print("Chain Response:", response["answer"])

# Demo of using the agent
agent = create_agent()
response = agent.invoke({"input": "What is LangChain?"})
print("Agent Response:", response["output"])

# Follow-up question with chain
response = chain.invoke({"question": "What modules does it provide?"})
print("Chain Follow-up Response:", response["answer"])

# Follow-up question with agent
response = agent.invoke({"input": "What modules does it provide?"})
print("Agent Follow-up Response:", response["output"])
