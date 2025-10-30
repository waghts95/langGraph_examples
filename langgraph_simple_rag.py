"""
Simple RAG (Retrieval-Augmented Generation) Example using LangGraph

This example demonstrates a basic RAG workflow:
1. Index documents into a vector store
2. Create a graph with retrieval and generation nodes
3. Process user queries by retrieving relevant documents and generating answers
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


# Define the state structure for our graph
class RAGState(TypedDict):
    """State for the RAG workflow"""
    question: str
    documents: List[Document]
    answer: str


# Sample documents to index
SAMPLE_DOCUMENTS = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain to enable cyclic graphs and persistent state.",
    "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them to generate informed responses.",
    "Vector databases store embeddings of text chunks, allowing for semantic search. When a query comes in, it's embedded and similar vectors are retrieved.",
    "LangChain is a framework for developing applications powered by language models. It provides modular components for building LLM applications.",
    "State graphs in LangGraph allow you to define nodes (functions) and edges (transitions) to create complex workflows with branching logic.",
]


def create_vector_store():
    """Create and populate a vector store with sample documents"""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    # Create Document objects
    docs = [Document(page_content=text) for text in SAMPLE_DOCUMENTS]

    # Split documents
    splits = text_splitter.split_documents(docs)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore


def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents based on the question"""
    print(f"📚 Retrieving documents for: {state['question']}")

    # Get vector store
    vectorstore = state.get('vectorstore')

    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(state['question'])

    print(f"✓ Retrieved {len(documents)} documents")

    return {
        **state,
        "documents": documents
    }


def generate_answer(state: RAGState) -> RAGState:
    """Generate an answer using the retrieved documents"""
    print(f"🤖 Generating answer...")

    # Format documents as context
    context = "\n\n".join([doc.page_content for doc in state['documents']])

    # Create prompt with context
    prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {state['question']}

Answer:"""

    # Generate answer using LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"✓ Answer generated")

    return {
        **state,
        "answer": response.content
    }


def create_rag_graph(vectorstore):
    """Create the RAG workflow graph"""
    # Initialize the graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Compile the graph
    app = workflow.compile()

    return app


def main():
    """Main function to run the RAG example"""
    print("=" * 60)
    print("Simple RAG with LangGraph Example")
    print("=" * 60)

    # Create vector store
    print("\n1. Creating vector store...")
    vectorstore = create_vector_store()
    print("✓ Vector store created and populated")

    # Create RAG graph
    print("\n2. Creating RAG workflow graph...")
    rag_app = create_rag_graph(vectorstore)
    print("✓ Graph created")

    # Example questions
    questions = [
        "What is LangGraph?",
        "How does RAG work?",
        "What are vector databases used for?",
    ]

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"Question {i}: {question}")
        print('=' * 60)

        # Run the graph
        result = rag_app.invoke({
            "question": question,
            "vectorstore": vectorstore,
            "documents": [],
            "answer": ""
        })

        print(f"\n📝 Answer: {result['answer']}")
        print(f"\n📄 Retrieved Documents:")
        for j, doc in enumerate(result['documents'], 1):
            print(f"  {j}. {doc.page_content[:100]}...")


if __name__ == "__main__":
    # Note: Make sure to set your OPENAI_API_KEY environment variable
    # export OPENAI_API_KEY='your-api-key-here'
    main()
