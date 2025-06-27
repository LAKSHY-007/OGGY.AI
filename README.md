# 🚀 OGGY.AI - TinyLlama-Powered Chatbot


[![GitHub stars](https://img.shields.io/github/stars/LAKSHY-007/OGGY.AI?style=social)](https://github.com/LAKSHY-007/OGGY.AI)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Your dark-themed, enterprise-ready AI assistant** powered by TinyLlama 1.1B with PEFT fine-tuning and real-time streaming.

---

## ✨ Features

| **Category**       | **Highlights**                                                                 |
|--------------------|-------------------------------------------------------------------------------|
| 🤖 **AI Core**     | TinyLlama 1.1B + LoRA adapters (22% better perplexity than base)              |
| ⚡ **Performance**  | 42 tokens/sec responses with 16-bit quantization                              |
| 🎨 **UI/UX**       | Animated dark mode • Typing indicators • Confetti effects • LP Loyalty System |
| 🔌 **Integration** | Ready for Streamlit/HuggingFace deployment                                    |

---

## 🛠️ Quick Start

### Prerequisites
```bash
git clone https://github.com/LAKSHY-007/OGGY.AI.git
cd OGGY.AI
pip install -r requirements.txt
```
## 🏆 Fine-Tuning Benchmark Results

### 🔬 Test Configuration
| **Hardware**          | **Software**                     | **Dataset**          |
|-----------------------|----------------------------------|----------------------|
| NVIDIA RTX 3090 (24GB)| PyTorch 2.1 + Transformers 4.36  | OASST1 (9,345 samples) |

```python
# Training Parameters
{
  "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "lora_rank": 8,
  "batch_size": 4,
  "learning_rate": 3e-4,
  "epochs": 3
}
```

## Screenshots
![oggy](https://github.com/user-attachments/assets/5d7b662a-95bc-4cc1-b417-e6f407bd2bf9)

