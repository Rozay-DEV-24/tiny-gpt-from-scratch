# 🧠 GPT Architecture — From Tokens to Causal Transformers

This repo/notebook is a **hands-on walk-through of GPT-style transformer architectures**. It builds the key components of a small **causal language model (CLM)**—from tokenization and embeddings to **multi-head self‑attention**, **feed‑forward blocks**, **residual connections**, and **causal masking**—and shows how to train/evaluate a tiny model end‑to‑end.

> Built as a learning project to demystify GPT internals and provide a clean, annotated reference you can extend.

---

## 🎯 Learning Goals
- Understand the **data flow** of a GPT-style model: tokens → embeddings → Transformer blocks → next-token logits.
- Implement **core building blocks**:
  - Tokenizer and vocabulary
  - Token + positional **embeddings**
  - **Causal (masked) multi-head self‑attention**
  - **Feed‑Forward (MLP)** sublayers
  - **LayerNorm**, **residual** connections, **dropout**
- Train a small **causal language model** with cross‑entropy loss and evaluate (loss/perplexity).
- Run small **ablations** (e.g., heads, depth, context length) to see their effect on training.

---

## 🗂️ Project Structure
```
Code_File.ipynb     # Main, fully annotated notebook
README.md                  # This file
```
Optional folders you can add:
```
/data/                     # tiny training corpus (e.g., shakespeare.txt)
/checkpoints/              # model checkpoints
```

---

## ⚙️ Tools & Libraries
- **Python 3.10+**
- **PyTorch** (model, training loop)
- **NumPy** (lightweight utilities)
- **Matplotlib** (optional, for loss curves)
- (Optional) **Hugging Face Tokenizers/Transformers** if you want to swap components

> The notebook is framework‑focused (PyTorch), but the ideas generalize to TensorFlow/JAX.

---

## 🔬 What’s Inside the Notebook
1. **Tokenization & Dataset**
   - Build a toy dataset (tiny corpus) and simple tokenizer (char/byte/BPE options discussed).
   - Create `(input_ids, target_ids)` pairs for next‑token prediction.
2. **Model Components**
   - `Embedding` layer (token + positional embeddings)
   - `CausalSelfAttention` (multi‑head attention + **causal mask**)
   - `FeedForward` MLP
   - `TransformerBlock` (Attn → MLP with **residual + LayerNorm**)
   - `GPT` (stack of blocks + LM head)
3. **Training Loop**
   - Cross‑entropy loss
   - **Teacher forcing** for next‑token prediction
   - **AdamW** optimizer, learning‑rate scheduling, gradient clipping
   - Periodic **evaluation** (loss/perplexity) and optional **checkpointing**
4. **Sampling / Generation**
   - Top‑k / top‑p (**nucleus**) sampling helpers
   - Temperature scaling for creative vs. factual generations
5. **Ablations & Tips**
   - Context length, number of heads, depth, hidden size
   - Regularization (dropout, weight decay)
   - Notes on scaling laws and dataset size

---

## 🚀 Quickstart
1. Install dependencies (PyTorch per your CUDA/CPU setup):
   ```bash
   pip install torch numpy matplotlib
   ```
   *(Optional)*
   ```bash
   pip install transformers tokenizers
   ```
2. Open the notebook:
   ```bash
   jupyter notebook "GPT Architecture.ipynb"
   ```
3. (Optional) Put a small text file in `./data/` (e.g., `shakespeare.txt`) and point the dataset loader to it.

---

## 📈 Results & Observations
- Even a **tiny GPT** can learn character/word statistics on a small corpus, showing decreasing **loss** and **perplexity**.
- **Causal masking** is crucial—without it, the model leaks future tokens and fails the CLM objective.
- Model capacity (depth/heads/hidden size) and **context length** significantly affect quality and compute.

> The notebook is intended for learning rather than SOTA performance.

---

## 🧭 Extending This Project
- Swap the simple tokenizer for **Byte‑Pair Encoding** (BPE) or **SentencePiece**.
- Train on a bigger corpus (Wikitext‑2, TinyStories) with **gradient accumulation**.
- Add **mixed precision** (AMP) and **TensorBoard** logging.
- Export to **ONNX** or try **quantization** for edge deployment.

---

## 🧑‍💻 Author
**Rohit Surya** — Model implementation, training loop, documentation.

---

## 📄 License
For educational use.
