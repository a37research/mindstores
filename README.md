# Minecraft AI Agent with Vision-LLM Integration

![Agent Demo](demo.gif)  
*Agent gathering resources using vision-guided planning*

## 🌟 Features
- **Multimodal Perception**  
  CLIP-powered visual analysis + GPT-4 text understanding
- **Experience Memory**  
  FAISS-accelerated vector database for long-term learning
- **Adaptive Planning**  
  Dynamic task decomposition with failure recovery
- **Real-Time Metrics**  
  Action success tracking & performance analytics
- **Customizable Goals**  
  "Mine diamonds" or "Build a castle" - define any objective

## 🛠️ Prerequisites
- **Hardware**
  - NVIDIA GPU (RTX 3060+ recommended)
  - 16GB+ RAM
  - Java 17+ (for MineDojo)

- **Software**
  - Python 3.9+
  - Minecraft Java Edition
  - OpenAI API key ([Get here](https://platform.openai.com/))

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/minecraft-ai-agent.git
cd minecraft-ai-agent
```
### 2. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate     # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m pip install 'minedojo~=0.2.0'

# Install FAISS (Choose CPU/GPU version)
conda install -c conda-forge faiss-gpu  # NVIDIA users
conda install -c conda-forge faiss-cpu  # CPU-only
```
### 4. Download CLIP Model
```bash
python -c "import clip; clip.load('ViT-B/32', device='cpu')"
```
## Configuration
### 1. API Keys
```bash
OPENAI_API_KEY = "sk-your-key-here"  # Get from OpenAI dashboard
```
### 2. Environment Settings (mindstores-final.py)
```bash
# Key Parameters
MAX_STEPS = 3000             # Max agent iterations
VISION_LABELS = [            # Custom detection targets
    "tree", "ore", "water", 
    "animal", "building", 
    "enemy", "path", "tool"
]
EXPERIENCE_MEMORY = 1000     # Max stored experiences
```
## Usage
### Basic
```bash
python mindstores-final.py "Explore and mine 3 diamonds"

# Alternative goal:
python mindstores-final.py "Build a wooden house near water"
```
### Advanced Options
```bash
python mindstores-final.py \
    --goal "Defeat 2 zombies and craft iron armor" \
    --steps 5000 \           # Max steps
    --vision_quality medium \# [low|medium|high] 
    --log_level DEBUG        # Verbose logging
```
