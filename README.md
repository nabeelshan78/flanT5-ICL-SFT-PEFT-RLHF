# 📚 llms-in-practice: Unlocking the Power of Large Language Models
Welcome to llms-in-practice! This repository serves as a dedicated space for exploring, experimenting, and mastering various applications of Large Language Models (LLMs). As an aspiring AI/ML Engineer, this is where I document my journey into understanding and leveraging the cutting-edge capabilities of LLMs for practical use cases.


# 🎯 Table of Contents
1. [Generative AI Use Case: Dialogue Summarization](#1-generative-ai-use-case-dialogue-summarization)  


---

## 1. Generative AI Use Case: Dialogue Summarization  
**File:** `summarize_dialogue.ipynb`  

This notebook dives deep into the fascinating world of dialogue summarization using Generative AI. We explore how Large Language Models, specifically **FLAN-T5** from Hugging Face, can be leveraged to condense lengthy conversations into concise, meaningful summaries.

### Overview  
The core objective of this project is to understand and implement dialogue summarization. We begin by exploring a base LLM's capabilities without specific instructions and then progressively enhance its performance through various prompt engineering techniques.

**Key Steps Covered:**

- Loading and inspecting the DialogSum dataset from Hugging Face.  
- Initializing a pre-trained FLAN-T5 (base) model and its corresponding tokenizer.  
- Evaluating baseline summarization performance without prompt engineering.  
- Implementing **Zero-Shot** Inference with custom instructions.  
- Exploring the impact of FLAN-T5's official prompt templates.  
- Demonstrating **One-Shot** and **Few-Shot** Inference for **in-context learning**.  
- Understanding the role of Generative Configuration Parameters (‘max_new_tokens‘, ‘do_sample‘, ‘temperature‘) in controlling output.  

---
