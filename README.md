# LangGraph Examples

A collection of examples demonstrating various LangGraph patterns and workflows.

## Examples

### 1. Simple RAG (Retrieval-Augmented Generation)
**File:** `langgraph_simple_rag.py`

A complete example of building a RAG system using LangGraph. This example demonstrates:
- Creating a vector store with FAISS
- Building a stateful graph with retrieval and generation nodes
- Processing questions by retrieving relevant context and generating answers

**Key Concepts:**
- State management with TypedDict
- Sequential node execution (retrieve ’ generate)
- Integration with vector stores and LLMs

### 2. Simple Chatbot
**File:** `langgraph_simple_chatbot.ipynb`

A basic chatbot implementation using LangGraph.

### 3. Simple Workflow
**File:** `langgraph_simple_workflow.ipynb`

Demonstrates basic workflow patterns in LangGraph.

## Setup

### Prerequisites
- Python 3.10 or higher
- OpenAI API key (for RAG example)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langGraph_examples
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### Configuration

For examples using OpenAI:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Running the Examples

### RAG Example

Run the Python script:
```bash
python langgraph_simple_rag.py
```

This will:
1. Create a vector store with sample documents
2. Build a RAG workflow graph
3. Process several example questions
4. Display retrieved documents and generated answers

**Expected Output:**
```
Simple RAG with LangGraph Example
====================================

1. Creating vector store...
 Vector store created and populated

2. Creating RAG workflow graph...
 Graph created

Question 1: What is LangGraph?
====================================
=Ú Retrieving documents for: What is LangGraph?
 Retrieved 3 documents
> Generating answer...
 Answer generated

=Ý Answer: [Generated answer based on retrieved context]
```

### Jupyter Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Then open any of the `.ipynb` files in your browser.

## How the RAG Example Works

The RAG example follows this workflow:

```
User Question
     “
[Retrieve Node] ’ Query vector store for relevant documents
     “
[Generate Node] ’ Use LLM to generate answer from context
     “
   Answer
```

**Key Components:**

1. **State Definition**: `RAGState` TypedDict defines the data flowing through the graph
2. **Vector Store**: FAISS stores document embeddings for semantic search
3. **Retrieve Node**: Finds the top-k most relevant documents
4. **Generate Node**: Creates an answer using retrieved context
5. **Graph**: Orchestrates the workflow with LangGraph

## Customization

### Using Your Own Documents

Modify the `SAMPLE_DOCUMENTS` list in `langgraph_simple_rag.py`:

```python
SAMPLE_DOCUMENTS = [
    "Your first document here...",
    "Your second document here...",
    # Add more documents
]
```

### Changing the LLM

Replace the model in the `generate_answer` function:

```python
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

### Adjusting Retrieval

Modify the number of retrieved documents:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve 5 docs
```

## Dependencies

- **langgraph**: Core library for building stateful workflows
- **langchain-openai**: OpenAI integration for LLMs and embeddings
- **langchain-community**: Community integrations (vector stores, loaders)
- **langchain-text-splitters**: Document chunking utilities
- **faiss-cpu**: Vector similarity search library

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAG Concepts](https://python.langchain.com/docs/use_cases/question_answering/)

## License

MIT
