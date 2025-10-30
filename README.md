# LangGraph Examples

A collection of examples demonstrating various LangGraph patterns and workflows with open source models.

## Examples

### 1. GPU-Accelerated RAG (Retrieval-Augmented Generation)
**File:** `langgraph_simple_rag.py`

A complete example of building a RAG system using LangGraph with **100% open source models** and **GPU acceleration**. No API keys required!

**Features:**
- GPU-accelerated embeddings using sentence-transformers
- Open source LLM (Microsoft Phi-2, 2.7B parameters)
- FAISS vector store for fast similarity search
- Stateful workflow orchestration with LangGraph
- Automatic device detection (CUDA GPU / Apple Silicon / CPU)

**Key Concepts:**
- State management with TypedDict
- Sequential node execution (retrieve → generate)
- Integration with HuggingFace transformers
- GPU optimization with torch.float16

### 2. Simple Chatbot
**File:** `langgraph_simple_chatbot.ipynb`

A basic chatbot implementation using LangGraph.

### 3. Simple Workflow
**File:** `langgraph_simple_workflow.ipynb`

Demonstrates basic workflow patterns in LangGraph.

## Setup

### Prerequisites
- Python 3.10 or higher
- **GPU recommended** (NVIDIA CUDA or Apple Silicon)
  - CPU-only mode works but will be slower
  - For NVIDIA: ~6GB VRAM for Phi-2 model
  - Alternatives: Use TinyLlama (1.1B) for lower VRAM

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langGraph_examples
```

2. Install dependencies:

**With GPU (CUDA):**
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
uv sync
```

**With CPU only:**
```bash
uv sync
```

**For Apple Silicon (M1/M2/M3):**
```bash
# PyTorch with MPS support
uv sync
```

## Running the RAG Example

### Quick Start

Simply run:
```bash
python langgraph_simple_rag.py
```

**No API keys needed!** The script will:
1. Detect your GPU (or fallback to CPU)
2. Download models automatically on first run (~5GB total)
3. Create embeddings for sample documents
4. Process example questions through the RAG pipeline

### First Run

On the first execution, models will be downloaded from HuggingFace Hub:
- **all-MiniLM-L6-v2** (embeddings): ~90MB
- **microsoft/phi-2** (LLM): ~5.5GB

These are cached locally for future runs.

### Expected Output

```
🔥 GPU-Accelerated RAG with Open Source Models 🔥

This example uses:
  • Embeddings: all-MiniLM-L6-v2 (sentence-transformers)
  • LLM: Microsoft Phi-2 (2.7B parameters)
  • Vector Store: FAISS
  • Orchestration: LangGraph

======================================================================
Simple RAG with LangGraph - Open Source LLM Edition (GPU Accelerated)
======================================================================
🚀 Using GPU: NVIDIA GeForce RTX 3080

1. Loading embeddings model...
✓ Embeddings model loaded on cuda

2. Creating vector store...
🔄 Creating vector store and generating embeddings...
✓ Vector store created and populated

3. Loading open source LLM...
📥 Loading LLM: microsoft/phi-2...
✓ LLM loaded on cuda

4. Creating RAG workflow graph...
✓ Graph created

======================================================================
Question 1: What is LangGraph?
======================================================================

📚 Retrieving documents for: What is LangGraph?
✓ Retrieved 3 documents

🤖 Generating answer with open source LLM...
✓ Answer generated

📝 Answer: [Generated answer about LangGraph]
```

## How the RAG Example Works

The GPU-accelerated RAG pipeline:

```
User Question
     ↓
[Embeddings Model] → Convert question to vector (GPU)
     ↓
[Vector Store] → Retrieve top-k similar documents (FAISS)
     ↓
[LLM Model] → Generate answer from context (GPU)
     ↓
   Answer
```

**Performance:**
- **GPU (CUDA)**: ~2-5 seconds per question
- **CPU**: ~15-30 seconds per question

## Customization

### Using Different Open Source Models

Edit the `create_llm()` function in `langgraph_simple_rag.py`:

```python
def create_llm(device="cuda"):
    # Choose your model:

    # Fast & lightweight (1.1B params, ~2GB VRAM)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Balanced (2.7B params, ~6GB VRAM) - DEFAULT
    # model_name = "microsoft/phi-2"

    # Better quality (7B params, ~14GB VRAM)
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # Very fast encoder-decoder (250M params, ~1GB VRAM)
    # model_name = "google/flan-t5-base"
```

### Using Different Embedding Models

Edit the `create_embeddings()` function:

```python
embeddings = HuggingFaceEmbeddings(
    # Fast and efficient - DEFAULT
    model_name="all-MiniLM-L6-v2",

    # Higher quality (slower)
    # model_name="sentence-transformers/all-mpnet-base-v2",

    # Multilingual support
    # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
```

### Adding Your Own Documents

Modify the `SAMPLE_DOCUMENTS` list:

```python
SAMPLE_DOCUMENTS = [
    "Your first document here...",
    "Your second document here...",
    # Add more documents
]
```

Or load from files:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("your_document.txt")
documents = loader.load()
```

### Adjusting Retrieval Parameters

```python
# Retrieve more documents for better context
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Use different search types
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 5, "fetch_k": 10}
)
```

### Tuning LLM Generation

Edit the pipeline parameters in `create_llm()`:

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,        # Longer responses
    temperature=0.7,           # More creative (0.0 = deterministic)
    top_p=0.9,                # Nucleus sampling
    repetition_penalty=1.2,   # Reduce repetition
)
```

## GPU Requirements & Recommendations

### NVIDIA GPUs (CUDA)

| Model | VRAM Required | Speed (per question) |
|-------|---------------|---------------------|
| TinyLlama-1.1B | ~2GB | ~1-2 seconds |
| Phi-2 (2.7B) | ~6GB | ~2-5 seconds |
| Mistral-7B | ~14GB | ~5-10 seconds |

### Apple Silicon (M1/M2/M3)

Works with MPS (Metal Performance Shaders) backend. Performance comparable to mid-range NVIDIA GPUs.

### CPU-Only

All models work on CPU but will be significantly slower (10-30x). Recommended for testing only.

## Troubleshooting

### Out of Memory Error

Reduce model size or batch size:
```python
# Use a smaller model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Or reduce max tokens
max_new_tokens=128
```

### CUDA Not Available

Check PyTorch installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

Reinstall PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Models Download Slowly

Models are downloaded from HuggingFace Hub. First run will take time. Use a different mirror if needed:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Dependencies

- **langgraph**: Stateful workflow orchestration
- **langchain-community**: Community integrations (vector stores, embeddings)
- **langchain-text-splitters**: Document chunking utilities
- **faiss-cpu**: Fast vector similarity search
- **torch**: PyTorch for deep learning and GPU acceleration
- **transformers**: HuggingFace transformers library
- **sentence-transformers**: Embedding models
- **accelerate**: Distributed and mixed-precision training

## Learn More

### LangGraph & LangChain
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAG Concepts](https://python.langchain.com/docs/use_cases/question_answering/)

### Open Source Models
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Phi-2 Model Card](https://huggingface.co/microsoft/phi-2)
- [Sentence Transformers](https://www.sbert.net/)

### Vector Databases
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Vector Search Explained](https://www.pinecone.io/learn/vector-database/)

## License

MIT
