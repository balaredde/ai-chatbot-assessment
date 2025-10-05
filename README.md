# ATG Technical Assignment: Local Command-Line Chatbot

## Using Hugging Face Transformers

A fully functional local chatbot interface using TinyLlama-1.1B-Chat model with conversation memory management and sliding window buffer.

---

## 📋 Overview

This project implements a command-line chatbot with the following features:
- **Local execution**: Runs entirely on your machine (CPU optimized)
- **Conversation memory**: Maintains context using a sliding window buffer (5 turns)
- **Hugging Face integration**: Uses the `transformers` library and `pipeline` API
- **Modular architecture**: Organized into separate, maintainable Python modules
- **Enhanced CLI**: Interactive commands (/exit, /clear, /info, /help)
- **Clean responses**: Intelligent prompt cleaning and response extraction

---

## 🏗️ Project Structure

```
assessment/
├── model_loader.py      # Model and tokenizer loading with pipeline
├── chat_memory.py       # Sliding window conversation buffer
├── interface.py         # CLI loop and user interaction
├── main.py             # Application entry point (alternative)
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── tests/
    ├── test_pipeline.py
    └── test_dialogpt_format.py
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (first run only, to download model)

### Installation

1. **Clone or extract the project**
   ```bash
   cd e:\assessment
   ```

2. **Create a virtual environment** (recommended)
   ```powershell
   python -m venv cb_env
   .\cb_env\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:
   - `torch>=1.9.0` - PyTorch for model computation
   - `transformers>=4.21.0` - Hugging Face transformers library
   - `colorama>=0.4.5` - Colored terminal output
   - `tokenizers>=0.13.0` - Fast tokenization

   **Note**: By default, this installs CPU-only PyTorch. For **GPU acceleration** (3-5x faster):
   ```powershell
   # Check your CUDA version first
   nvidia-smi
   
   # Then install CUDA-enabled PyTorch (example for CUDA 11.8)
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
   See `GPU_SETUP.md` for detailed GPU setup instructions.

4. **First run** (downloads model ~117MB)
   ```powershell
   python interface.py
   ```

---

## 💻 How to Run

### Quick Start
```powershell
# Activate virtual environment
.\cb_env\Scripts\Activate.ps1

# Run the chatbot
python interface.py
```

### Alternative Entry Point
```powershell
python main.py
```

---

## 📖 Usage Examples

### Sample Interaction

```
======================================================================
                    AI CHATBOT INTERFACE
======================================================================

Initializing components...
Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0...
Running on CPU (this may take a moment)...
✓ Model loaded successfully on CPU!

──────────────────────────────────────────────────────────────────────
✓ Running on: CPU
✓ Memory Window: 5 conversation turns
──────────────────────────────────────────────────────────────────────

Available Commands:
  /exit  - Exit the chatbot
  /clear - Clear conversation memory
  /info  - Display memory statistics
  /help  - Show this help message

======================================================================

You: Hello! How are you?
🤖 Thinking...
Bot: Hello! I'm doing well, thank you for asking. How can I assist you today?

You: What is Python?
🤖 Thinking...
Bot: Python is a high-level programming language known for its simplicity and readability.

You: /exit

👋 Goodbye! Thanks for chatting.
```

### Available Commands

- `/exit` - Quit the chatbot gracefully
- `/clear` - Clear conversation memory buffer
- `/info` - Display memory statistics (turns, messages stored)
- `/help` - Show available commands
- Regular text input - Chat with the bot

---

## 🔧 Technical Details

### Model
- **Name**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Type**: Causal language model optimized for chat
- **Parameters**: 1.1 billion parameters
- **Size**: ~2.2GB (compressed download)
- **Training**: Trained on diverse conversational datasets
- **Context**: Handles multi-turn dialogue with 2048 token context window

### Memory Management
- **Implementation**: Python `deque` with fixed `maxlen`
- **Window size**: 5 conversation turns (10 total messages)
- **Behavior**: Automatically discards oldest messages when buffer is full
- **Format**: Stores messages as tuples with role and content
- **Commands**: `/clear` to reset memory, `/info` to view statistics

### Text Generation
- **Method**: Hugging Face `pipeline` API
- **Parameters**:
  - `max_new_tokens=100` - Limit response length
  - `temperature=0.7` - Balance creativity vs. coherence
  - `do_sample=True` - Enable sampling for natural responses
  - `top_p=0.9` - Nucleus sampling for diversity
  - `repetition_penalty=1.15` - Reduce repetitive responses
  - `return_full_text=False` - Return only new generation

## 📦 Module Descriptions

### `model_loader.py`
**Purpose**: Load and manage the Hugging Face model and tokenizer.

**Key Function**:
```python
def load_model(model_name: str) -> (Pipeline, PreTrainedTokenizer)
```
- Downloads model from Hugging Face Hub (cached locally)
- Creates text-generation pipeline
- Configures device (GPU/CPU)
- Returns pipeline and tokenizer

### `chat_memory.py`
**Purpose**: Manage conversation history with sliding window buffer.

**Key Class**: `ChatMemory`
- `__init__(window_size=4)` - Initialize buffer
- `add_message(role, content)` - Add user/bot message
- `get_context()` - Get standard formatted context
- `get_context_dialogpt_format(eos_token)` - Get DialoGPT-formatted context
- `clear()` - Reset conversation history

### `interface.py`
**Purpose**: CLI interface and conversation loop.

**Key Components**:
- `Spinner` class - Loading animation during generation
- `clean_response()` - Post-process model output
- `run_chat()` - Main conversation loop
- Error handling and graceful shutdown

---

## 🧪 Testing

### Run Test Scripts
```powershell
# Test pipeline output format
python test_pipeline.py

# Test DialoGPT conversation format
python test_dialogpt_format.py
```

### Manual Testing
1. Start the chatbot: `python interface.py`
2. Test single-turn: Ask a question, verify response
3. Test multi-turn: Ask follow-up questions, check if context is maintained
4. Test memory: Exceed 4 turns, verify oldest messages are forgotten
5. Test exit: Type `/exit`, verify graceful shutdown

---

## 🎯 Design Decisions

### Why TinyLlama?
- **Optimized for chat**: Specifically trained for conversational AI
- **Good quality**: 1.1B parameters provide coherent responses
- **CPU compatible**: Works on standard hardware
- **Active development**: Well-maintained by Hugging Face community
- **Local execution**: No API calls or internet required (after download)

### Why Sliding Window Memory?
- **Efficiency**: Prevents context from growing unbounded
- **Focus**: Recent messages are most relevant
- **Performance**: Limits token count for faster generation
- **Simplicity**: Easy to implement and understand

### Why Pipeline API?
- **Simplicity**: High-level abstraction over model inference
- **Best practices**: Handles tokenization, generation, decoding automatically
- **Optimization**: Built-in batching and device management
- **Flexibility**: Easy parameter tuning

---

## 🐛 Troubleshooting

### Issue: Model download fails
**Solution**: Check internet connection and disk space (~2.5GB required). The model is downloaded from Hugging Face Hub on first run.

### Issue: Slow responses
**Solution**: 
- Normal on CPU (10-15 seconds per response for 1.1B model)
- First response is slower (model initialization)
- Subsequent responses in same session are faster
- Consider using smaller model or GPU for faster inference

### Issue: Out of memory error
**Solution**:
- Close other applications to free RAM
- Ensure you have at least 4GB available RAM
- Reduce `max_new_tokens` parameter in `interface.py`

### Issue: ImportError or transformers warnings
**Solution**: 
```powershell
pip install -r requirements.txt --upgrade
```

### Issue: Warning about max_new_tokens vs max_length
**Solution**: This is a minor warning and can be ignored. The implementation already uses `max_new_tokens` correctly.

---

## 📊 Performance Notes

### CPU Performance
- **First run**: 2-5 minutes (model download ~2.2GB)
- **Subsequent runs**: 10-15 seconds (model loading)
- **Generation time**: 5-15 seconds per response (CPU dependent)
- **Memory usage**: ~3GB RAM
- **Model size on disk**: ~2.2GB (cached in `~/.cache/huggingface/`)

### GPU Performance (Optional)
- **Generation time**: 1-3 seconds per response ⚡ **5-10x faster!**
- **Memory usage**: ~2GB VRAM
- **Optimizations**: Automatic FP16 (half precision) on compatible GPUs

**Note**: This implementation is optimized for CPU. For GPU support, modify the `device` parameter in `model_loader.py`.

💡 **Tip**: First-time model download requires stable internet. Subsequent runs are fully offline.

---

## 🎥 Demo Video

*(Optional deliverable)*

A 2-3 minute screen recording demonstrating:
1. Project structure walkthrough
2. Running the chatbot
3. Multi-turn conversation example
4. Code explanation and design decisions

---

## 📝 Assignment Deliverables Checklist

- ✅ **Source Code**: Modular Python scripts (`model_loader.py`, `chat_memory.py`, `interface.py`)
- ✅ **README.md**: Comprehensive setup and usage guide (this file)
- ✅ **Sliding Window Memory**: Implemented in `ChatMemory` class
- ✅ **Hugging Face Pipeline**: Used for model loading and generation
- ✅ **CLI Interface**: Continuous input loop with `/exit` command
- ✅ **Conversation Context**: Maintained across multiple turns
- ⬜ **Demo Video**: (Create separately)

---

## 🚀 Future Enhancements

- Add more commands (`/clear`, `/stats`, `/help`)
- Support multiple model choices
- Implement conversation saving/loading
- Add response streaming for better UX
- Fine-tune model on custom dataset
- Add web interface (Gradio/Streamlit)

---

## 👤 Author

**Machine Learning Intern Candidate**  
ATG Technical Assignment  
October 2025

---

## 📄 License

This project is created for educational purposes as part of the ATG technical assessment.

---

**Note**: TinyLlama is a general-purpose chat model and provides balanced conversational responses. For specialized use cases, consider fine-tuning or using domain-specific models.
