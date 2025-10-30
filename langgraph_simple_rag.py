"""
Simple RAG (Retrieval-Augmented Generation) Example using LangGraph
with Open Source LLMs and GPU acceleration

This example demonstrates a basic RAG workflow:
1. Index documents into a vector store using GPU-accelerated embeddings
2. Create a graph with retrieval and generation nodes
3. Process user queries by retrieving relevant documents and generating answers
   using open source LLMs
"""

from typing import TypedDict, List
import torch
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')


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
    "Machine learning models can be fine-tuned on specific tasks to improve performance. Transfer learning allows models to leverage knowledge from pre-training.",
    "Natural Language Processing (NLP) is a field of AI focused on the interaction between computers and human language. It enables machines to understand, interpret, and generate human text.",
]


def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("⚠️  No GPU detected, using CPU")
    return device


def create_embeddings(device="cuda"):
    """Create GPU-accelerated embeddings model"""
    # Using all-MiniLM-L6-v2 - fast and efficient embedding model
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print(f"✓ Embeddings model loaded on {device}")
    return embeddings


def create_llm(device="cuda"):
    """Create GPU-accelerated open source LLM"""
    # Using Phi-2 - small but capable model from Microsoft (2.7B parameters)
    # Alternative models you can try:
    # - "mistralai/Mistral-7B-Instruct-v0.2" (7B - better quality, needs more VRAM)
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B - faster, less VRAM)
    # - "google/flan-t5-base" (250M - very fast)

    model_name = "microsoft/phi-2"

    print(f"📥 Loading LLM: {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Load model with GPU support
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
    )

    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    print(f"✓ LLM loaded on {device}")
    return llm


def create_vector_store(embeddings):
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

    # Create vector store with GPU-accelerated embeddings
    print("🔄 Creating vector store and generating embeddings...")
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore


def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents based on the question"""
    print(f"\n📚 Retrieving documents for: {state['question']}")

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
    print(f"\n🤖 Generating answer with open source LLM...")

    # Get LLM from state
    llm = state.get('llm')

    # Format documents as context
    context = "\n\n".join([doc.page_content for doc in state['documents']])

    # Create prompt with context - formatted for instruction-following models
    prompt = f"""Below is an instruction that describes a task, along with context that provides further information. Write a response that appropriately answers the question based on the context.

### Context:
{context}

### Question:
{state['question']}

### Answer:"""

    # Generate answer using LLM
    try:
        response = llm.invoke(prompt)
        # Clean up the response
        answer = response.strip()
        # Extract just the answer part if the model repeats the prompt
        if "### Answer:" in answer:
            answer = answer.split("### Answer:")[-1].strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    print(f"✓ Answer generated")

    return {
        **state,
        "answer": answer
    }


def create_rag_graph(vectorstore, llm):
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
    print("=" * 70)
    print("Simple RAG with LangGraph - Open Source LLM Edition (GPU Accelerated)")
    print("=" * 70)

    # Get device
    device = get_device()

    # Create embeddings model
    print("\n1. Loading embeddings model...")
    embeddings = create_embeddings(device)

    # Create vector store
    print("\n2. Creating vector store...")
    vectorstore = create_vector_store(embeddings)
    print("✓ Vector store created and populated")

    # Load LLM
    print("\n3. Loading open source LLM...")
    llm = create_llm(device)

    # Create RAG graph
    print("\n4. Creating RAG workflow graph...")
    rag_app = create_rag_graph(vectorstore, llm)
    print("✓ Graph created")

    # Example questions
    questions = [
        "What is LangGraph?",
        "How does RAG work?",
        "What are vector databases used for?",
    ]

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"Question {i}: {question}")
        print('=' * 70)

        # Run the graph
        result = rag_app.invoke({
            "question": question,
            "vectorstore": vectorstore,
            "llm": llm,
            "documents": [],
            "answer": ""
        })

        print(f"\n📝 Answer: {result['answer']}")
        print(f"\n📄 Retrieved Documents:")
        for j, doc in enumerate(result['documents'], 1):
            print(f"  {j}. {doc.page_content[:100]}...")


if __name__ == "__main__":
    print("\n🔥 GPU-Accelerated RAG with Open Source Models 🔥\n")
    print("This example uses:")
    print("  • Embeddings: all-MiniLM-L6-v2 (sentence-transformers)")
    print("  • LLM: Microsoft Phi-2 (2.7B parameters)")
    print("  • Vector Store: FAISS")
    print("  • Orchestration: LangGraph\n")

    main()
