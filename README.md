# 🤖 Advanced GAIA Agents Challenge Solution

A comprehensive solution for the [Hugging Face Agents Course Unit 4 GAIA Challenge](https://huggingface.co/learn/agents-course/unit4/hands-on), featuring advanced multimodal AI agents with dynamic RAG capabilities, quantized models for Kaggle compatibility, and both synchronous/asynchronous execution modes.

## 🌟 Features

### 🧠 Dual Agent Architecture
- **Agent 1 (LlamaIndex)**: Advanced multimodal agent with dynamic knowledge base and hybrid reranking
- **Agent 2 (Smolagents)**: Gemini-powered agent with BM25 retrieval and observability

### Features for Agent 1
### 🎯 Multimodal Capabilities
- **BAAI Visualized Embedding**: BGE-M3 based multimodal embeddings running on cuda:1
- **Pixtral 12B Quantized**: FP8/4-bit quantized vision-language model for resource-constrained environments
- **Hybrid Retrieval**: Text + visual content processing with ColPali and SentenceTransformer reranking

### ⚡ Execution Modes
- **Asynchronous Mode**: Concurrent question processing for maximum speed
- **Kaggle Compatibility**: Optimized for resource-constrained environments

### 🔍 Advanced RAG System
- **Dynamic Knowledge Base**: Automatically updated with web search results
- **Multimodal Parsing**: Handles text, images, PDFs, audio, and video files
- **Smart Reranking**: Hybrid approach combining text and visual rerankers

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│                  APP                        │
│            (Async/Sync Modes)               │
└─────────────────┬───────────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
    ┌────▼────┐       ┌────▼────┐
    │Agent 1  │       │Agent 2  │
    │LlamaIdx │       │Smolagent│
    └────┬────┘       └────┬────┘
         │                 │
    ┌────▼────┐       ┌────▼────┐
    │Dynamic  │       │BM25 +   │
    │RAG +    │       │Langfuse │
    │Hybrid   │       │Observ.  │
    │Rerank   │       │         │
    └─────────┘       └─────────┘
```

## 🚀 Quick Start

### Prerequisites

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/gaia-agents-challenge
cd gaia-agents-challenge
```

2. **Install FlagEmbedding with visual support**:
```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding/research/visual_bge
pip install -e .
cd ../../..
```

3. **Install additional dependencies**:
#### For Agent 1: 
```bash
pip install -r requirements.txt
```
#### For Agent 2: 
```bash
pip install -r requirements2.txt
```


4. **Set environment variables**:
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export HUGGINGFACEHUB_API_TOKEN="your_hf_token"
export LANGFUSE_PUBLIC_KEY="your_langfuse_public_key"  # Optional
export LANGFUSE_SECRET_KEY="your_langfuse_secret_key"  # Optional
```

### Usage

```bash
# LlamaIndex Agent
python agent.py

# Smolagents Agent
python agent2.py
```

## 📁 Project Structure

```
├── agent.py                 # LlamaIndex-based agent with dynamic RAG
├── agent2.py               # Smolagents-based agent with observability
├── appasync.py             # Original async Gradio interface
├── app.py                  # Original sync Gradio interface
├── custom_models.py        # Custom model implementations
├── requirements.txt        # Python dependencies
├── README.md              # This file
```

## 🧪 Testing

### Run Individual Components
```bash
# Test BAAI embedding
python -c "from custom_models import BaaiMultimodalEmbedding; print('BAAI OK')"

# Test Pixtral quantized
python -c "from custom_models import PixtralQuantizedLLM; print('Pixtral OK')"

# Test agents
python agent.py
python agent2.py
```

### Run GAIA Evaluation
```bash
# Through the web interface
python app.py

# Or programmatically
python -c "
from agent2 import GAIAAgent
agent = GAIAAgent()
result = agent.solve_gaia_question({'Question': 'Test question', 'task_id': 'test'})
print(result)
"
```

## 🔧 Customization

### Adding New Models
1. Create a new class in `custom_models.py`
2. Implement the required interfaces
3. Update the agent configuration

### Modifying RAG Behavior
- Edit `DynamicQueryEngineManager` in `agent.py`
- Adjust reranking strategies in `HybridReranker`
- Configure search parameters in `enhanced_web_search_tool`

### UI Customization
- Modify `app_unified.py` for interface changes
- Add new execution modes
- Integrate additional observability tools

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Failures
- Check internet connectivity for model downloads
- Verify HuggingFace token permissions
- Clear model cache: `rm -rf ~/.cache/huggingface/`

#### Visual BGE Import Errors
```bash
# Ensure proper installation
cd FlagEmbedding/research/visual_bge
pip install -e .
```

## 🔗 References

- [GAIA Benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [BGE Models](https://github.com/FlagOpen/FlagEmbedding)
- [Gradio](https://github.com/gradio-app/gradio)