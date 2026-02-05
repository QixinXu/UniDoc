# CogDoc Data Engine

A data synthesis engine for generating high-quality question-answer pairs and thinking patterns from document data. This engine is designed to create training data for document understanding models.

## 🌟 Features

- **Query Synthesis**: Generate QA pairs from document elements (figures, tables, text)
- **Thinking Pattern Synthesis**: Generate chain-of-thought reasoning from QA pairs
- **Multi-modal Support**: Handle both visual and textual content from PDFs
- **Distributed Processing**: Built with Ray for efficient parallel processing
- **Iterative Refinement**: Improve quality through reference-guided refinement

## 📁 Project Structure

```
data_engine/
├── configs/                          # Configuration files
│   ├── query_synthesis.yaml          # Config for query synthesis
│   └── thinking_synthesis.yaml       # Config for thinking synthesis
├── scripts/                          # Executable scripts
│   ├── run_query_synthesis.py        # Run query synthesis pipeline
│   ├── run_thinking_synthesis.py     # Run thinking synthesis pipeline
│   └── start_vllm_server.sh          # Start vLLM inference server
├── src/                              # Source code
│   ├── __init__.py
│   ├── requirements.txt              # Python dependencies
│   ├── core/                         # Core utilities
│   │   ├── __init__.py
│   │   ├── model_client.py           # LLM client wrapper
│   │   └── utils.py                  # Utility functions
│   ├── query_synthesis/              # Query synthesis module
│   │   ├── __init__.py
│   │   ├── prompts.py                # Prompt templates
│   │   ├── anchors.py                # Anchor classes (Image, Table, Text)
│   │   ├── document_parser.py        # PDF document parser
│   │   └── synthesizer.py            # Main synthesizer
│   └── thinking_synthesis/           # Thinking pattern synthesis module
│       ├── __init__.py
│       ├── prompts.py                # Prompt templates
│       ├── processor.py              # Refinement processor
│       └── synthesizer.py            # Main synthesizer
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- Poppler (for PDF processing)

### Installation

1. Install dependencies:
```bash
cd data_engine/src
pip install -r requirements.txt
```

2. Install Poppler for PDF processing:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
apt-get install poppler-utils
```

### Start the Inference Server

Before running synthesis, start the vLLM server:

```bash
cd data_engine/scripts
chmod +x start_vllm_server.sh

# Default: Qwen2-VL-7B-Instruct on port 8000
./start_vllm_server.sh

# Or with custom settings
MODEL_NAME="Qwen/Qwen2-VL-72B-Instruct" PORT=8001 TENSOR_PARALLEL_SIZE=4 ./start_vllm_server.sh
```

### Run Query Synthesis

Generate QA pairs from parsed PDF documents:

```bash
cd data_engine/scripts

# Process figures
python run_query_synthesis.py \
    --config-file ../configs/query_synthesis.yaml \
    --port 8000 \
    --figure

# Process all element types
python run_query_synthesis.py \
    --config-file ../configs/query_synthesis.yaml \
    --port 8000 \
    --figure --table --pure-text
```

### Run Thinking Synthesis

Generate thinking patterns from QA pairs:

```bash
cd data_engine/scripts

# Standard mode with refinement
python run_thinking_synthesis.py \
    --config-file ../configs/thinking_synthesis.yaml \
    --port 8000

# Fast mode (no refinement)
python run_thinking_synthesis.py \
    --config-file ../configs/thinking_synthesis.yaml \
    --port 8000 \
    --no-refinement
```

## 📖 Detailed Usage

### Query Synthesis

The query synthesis module extracts elements from PDF documents and generates QA pairs:

#### Input Format

Your input should be JSON files from a PDF parsing tool. Each JSON should contain:
```json
{
  "data": {
    "nodes": [
      {"type": "afigure", "refer": {"locs": [{"page": 1, "bbox": [0.1, 0.2, 0.9, 0.8]}]}},
      {"type": "atable", "content": "| A | B |...", "refer": {"locs": [...]}},
      {"type": "apara", "content": "...", "refer": {"locs": [...]}}
    ]
  }
}
```

#### Output Format

Generated QA pairs are saved as JSONL:
```json
{
  "pdf_path": "/path/to/document.pdf",
  "figure": "Figure 1",
  "target_image_path": "/path/to/figure.png",
  "related_pages": [1, 2],
  "keypoints": ["Key finding 1", "Key finding 2"],
  "difficulty": "hard",
  "question": "Based on Figure 1, what is the relationship between...",
  "answer": "The figure shows that..."
}
```

### Thinking Synthesis

The thinking synthesis module generates reasoning chains from QA pairs:

#### Input Format

JSONL file with QA pairs:
```json
{
  "question": "What does Figure 1 show?",
  "answer": "A performance comparison",
  "pdf_path": "/path/to/document.pdf",
  "related_pages": [1, 2]
}
```

#### Output Format

Generated thinking patterns:
```json
{
  "question": "What does Figure 1 show?",
  "answer": "A performance comparison",
  "thinking": "Looking at the images provided, I can see... First, I notice that... This leads me to conclude that...",
  "generated_answer": "A performance comparison",
  "refinement_iterations": 2
}
```

## ⚙️ Configuration

### Query Synthesis Config

```yaml
model:
  model_type: "vllm_server"
  model_name: "Qwen/Qwen2-VL-7B-Instruct"
  model_args:
    max_tokens: 2048
    temperature: 0.7

input:
  raw_data_path: "/path/to/parsed/documents"
  max_num_files: 100

output:
  file_prefix: "./output/queries"
  image_dir: "./output/images"
```

### Thinking Synthesis Config

```yaml
model:
  model_type: "vllm_server"
  model_name: "Qwen/Qwen2-VL-7B-Instruct"
  model_args:
    max_tokens: 4096

input:
  file_path: "/path/to/qa_pairs.jsonl"
  max_num_items: 1000

output:
  file_prefix: "./output/thinking"

processing:
  use_refinement: true
  max_iterations: 3
```

## 🔧 API Reference

### Query Synthesis

```python
from query_synthesis import QuerySynthesizer, ImageAnchor, TableAnchor

# Create synthesizer
synthesizer = QuerySynthesizer(
    config_path="configs/query_synthesis.yaml",
    port=8000
)

# Run full pipeline
synthesizer.run(
    process_figures=True,
    process_tables=True,
    process_text=False
)
```

### Thinking Synthesis

```python
from thinking_synthesis import ThinkingSynthesizer, RefinementProcessor

# Create synthesizer
synthesizer = ThinkingSynthesizer(
    config_path="configs/thinking_synthesis.yaml",
    port=8000
)

# Run with refinement
synthesizer.run(
    mode="initial",
    use_refinement=True,
    max_pages=20
)
```

### Core Utilities

```python
from core import LMClient, load_config, image_to_base64

# Load configuration
config = load_config("configs/query_synthesis.yaml")

# Create model client
model = LMClient(
    model_type="vllm_server",
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    port=8000
)

# Generate text
response = model.generate_vl_vllm_server(messages, num_rollouts=1)
```

## 📝 License

This project is open-sourced under the MIT License.

## 🙏 Acknowledgements

This project builds upon:
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- [Ray](https://github.com/ray-project/ray) for distributed computing
- [Transformers](https://github.com/huggingface/transformers) for model handling
