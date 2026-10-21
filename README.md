# Attention Is All You Need — Transformer Implementation

**CS-320 Semester Project | Fall 2026**  
**BS Artificial Intelligence (BS-AI 2024F)**  
**National University of Sciences and Technology (NUST), NBC**

## 📌 Project Overview

This project explores the Transformer architecture introduced in the research paper **“Attention Is All You Need”** by Vaswani et al. (NeurIPS 2017).

The Transformer replaces recurrent and convolutional sequence-processing mechanisms with attention-based operations. Its encoder-decoder architecture uses multi-head self-attention, position-wise feed-forward networks, residual connections, layer normalization, and positional encodings.

The objective of this project is to understand, implement, reproduce, and evaluate the Transformer architecture through machine translation, language modelling, and controlled experiments.

We will implement the multi-head attention mechanism and encoder-decoder stack ourselves, compare our implementation against PyTorch's built-in Transformer modules, and document the results through experiments and analysis.

## 🎯 Project Objectives

- Understand the mathematical foundations of scaled dot-product attention.
- Implement multi-head attention and the Transformer encoder-decoder architecture.
- Understand positional encodings, masking, residual connections, and feed-forward layers.
- Reproduce a small-scale machine translation experiment using Multi30k.
- Implement a decoder-only Transformer for character-level language modelling.
- Compare the Transformer against simple baseline models.
- Perform controlled ablation experiments to study the effects of architectural and training parameters.
- Evaluate translation quality, validation loss, and perplexity.
- Develop an interactive demonstration of translation and attention visualization.

## 📚 Research Paper

**Title:** Attention Is All You Need  
**Authors:** Ashish Vaswani et al.  
**Conference:** Advances in Neural Information Processing Systems (NeurIPS), 2017.

**Paper:** [Read the original research paper](https://arxiv.org/abs/1706.03762)

**Reference implementation and explanation:** [The Annotated Transformer — Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/)

## 📊 Datasets

### 1. Multi30k — German to English Translation

Multi30k will be used to reproduce a small-scale machine translation experiment.

- **Task:** German-to-English translation
- **Training data:** Approximately 29,000 sentence pairs
- **Evaluation:** BLEU score
- **Purpose:** Evaluate the encoder-decoder Transformer on a sequence-to-sequence task.

The original paper reports a BLEU score of 27.3 for its base model on WMT 2014 English-to-German. Our Multi30k results will be reported separately because the dataset, language direction, model scale, and experimental setup differ.

### 2. Tiny Shakespeare — Character-Level Language Modelling

Tiny Shakespeare will be used to implement a decoder-only, GPT-style Transformer.

- **Task:** Character-level next-token prediction
- **Model:** Decoder-only Transformer
- **Baseline:** Bigram or n-gram language model
- **Evaluation:** Validation loss and perplexity

The goal is to examine whether attention-based language modelling improves upon simpler statistical language models.

### 3. Optional Extension — English to Urdu Translation

As an optional extension, we plan to investigate English-to-Urdu translation using a subset of OPUS-100 and compare against a simple baseline.

This extension will be included only if time and computational resources permit.

## 🏗️ Transformer Architecture

The original Transformer is an encoder-decoder architecture composed of six encoder layers and six decoder layers in the base configuration.

### Encoder

Each encoder layer contains:

1. Multi-head self-attention
2. Position-wise feed-forward network
3. Residual connections
4. Layer normalization

### Decoder

Each decoder layer contains:

1. Causally masked multi-head self-attention
2. Encoder-decoder attention
3. Position-wise feed-forward network
4. Residual connections
5. Layer normalization

### Core Components

- Scaled dot-product attention
- Multi-head attention
- Token embeddings
- Sinusoidal positional encodings
- Padding and causal masks
- Feed-forward networks
- Output projection and token generation

We will document tensor shapes throughout the architecture and calculate the parameter count of one encoder block.

## 🧪 Experiments and Ablation Studies

We will investigate the effects of changing one factor at a time while keeping the random seed and data split fixed.

Planned experimental factors include:

- Number of attention heads
- Key dimension \(d_k\)
- Number of encoder and decoder layers
- Dropout rate
- Label smoothing
- Sinusoidal versus learned positional encodings
- Number of learning-rate warm-up steps

Results will be organized into a comparison table, with explanations of how each change affects model behaviour and performance.

## ✅ Implementation and Testing Checklist

The implementation will follow these validation checks:

- Verify tensor shapes at each stage.
- Check that the initial loss is approximately \(\ln C\) under an appropriate uniform-logit initialization, where \(C\) is the number of output classes.
- Test whether the model can overfit a single batch.
- Verify that gradients are non-zero.
- Save checkpoints to Google Drive.
- Compare custom attention and encoder-decoder outputs against PyTorch's built-in modules.
- Compare parameter counts.
- Maintain a bug diary documenting implementation problems and fixes.

## 🛠️ Technologies Used

- **Python** — Programming language
- **PyTorch** — Deep learning framework
- **Google Colab** — Notebook development and experiments
