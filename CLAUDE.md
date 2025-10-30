# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph examples repository demonstrating LangGraph workflows and chatbot implementations. LangGraph is a framework for building stateful, multi-step applications with language models, featuring state management, nodes (processing functions), edges (flow control), and conditional routing.

## Development Setup

**Package Manager**: This project uses `uv` for dependency management.

**Environment Setup**:
```bash
# Activate the virtual environment (already present in .venv)
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

**Python Version**: Requires Python >= 3.10 (currently using .python-version file)

## Key Dependencies

- `langgraph` (>=1.0.1): Core graph workflow framework
- `langchain-ollama` (>=1.0.0): Integration with Ollama for local LLMs
- `langchain-openai` (>=1.0.1): OpenAI integration
- `ipykernel` (>=7.1.0): Jupyter notebook support

## Running Examples

**Jupyter Notebooks**: Examples are provided as Jupyter notebooks (.ipynb files). Run them using:
```bash
jupyter notebook langgraph_simple_workflow.ipynb
# or
jupyter notebook langgraph_simple_chatbot.ipynb
```

**Basic Python Script**:
```bash
python main.py
```

## Architecture

### LangGraph Core Concepts

1. **StateGraph**: The main graph structure that orchestrates the workflow
2. **State Management**: Uses `TypedDict` to define the structure of data flowing through nodes
3. **Nodes**: Individual processing functions that transform state
4. **Edges**: Connections between nodes (can be conditional or direct)
5. **Memory**: Uses `MemorySaver` checkpointer to maintain conversation history across invocations
6. **Entry Points**: Define where graph execution starts
7. **END**: Special node to terminate the workflow

### Example Workflows

**langgraph_simple_workflow.ipynb**:
- Demonstrates a basic conversational flow: greet → process_name → ask → END
- Shows state management with TypedDict
- Illustrates conditional routing with `route_next` function
- Uses memory to maintain conversation state

**langgraph_simple_chatbot.ipynb**:
- Implements a chatbot using Ollama's llama3.2 model
- Integrates LangGraph with LangChain's ChatOllama
- Provides helper functions: `chat()`, `reset_chat()`, `show_history()`
- Uses message types: SystemMessage, HumanMessage, AIMessage
- Requires Ollama to be running locally: `ollama serve`

### State Structure Patterns

States are typically defined as TypedDicts containing:
- `messages`: List of conversation messages (with role and content)
- Domain-specific fields (e.g., `user_name`, `step`)

### Configuration Pattern

Thread-based conversation management:
```python
config = {"configurable": {"thread_id": "unique_id"}}
app.invoke(state, config)
```

## LLM Requirements

The chatbot example requires:
1. Ollama running locally: `ollama serve`
2. Model installed: `ollama pull llama3.2`

## Project Structure

- `main.py`: Minimal entry point script
- `*.ipynb`: Interactive examples demonstrating LangGraph patterns
- `pyproject.toml`: Project metadata and dependencies
- `.venv/`: Virtual environment with installed packages
