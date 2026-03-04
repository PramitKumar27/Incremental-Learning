# Vision Transformers (ViT) - Fundamentals

**A Complete Guide to Understanding Vision Transformers and How They Differ from CNNs**

---

## Table of Contents
1. [What is a Vision Transformer?](#what-is-a-vision-transformer)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [CNNs vs ViTs: Key Differences](#cnns-vs-vits-key-differences)
4. [Why ViTs for Space Applications?](#why-vits-for-space-applications)
5. [Your Specific Model: ViT-tiny](#your-specific-model-vit-tiny)

---

## What is a Vision Transformer?

### The Big Picture

**Vision Transformer (ViT)** is a deep learning architecture that applies the **Transformer** mechanism (originally from NLP) directly to images.

**Key Innovation:** Instead of using convolutions, ViT treats an image as a **sequence of patches** and processes them using **self-attention**.

### The Core Idea

```
Traditional CNN:
Image → Conv layers → Features → Classification

Vision Transformer:
Image → Patch Embedding → Transformer Blocks → Classification
```

**Analogy:** 
- CNN: Looks at image with a sliding window (like reading with a magnifying glass)
- ViT: Looks at entire image patches simultaneously (like reading the whole page at once)

---

## Architecture Deep Dive

### Step 1: Patch Embedding

**What happens:** Image is divided into fixed-size patches.

```
Input Image: 224×224×3 (height × width × RGB channels)
Patch Size: 16×16
Number of Patches: (224/16)² = 196 patches

Each patch: 16×16×3 = 768 values
```

**Why patches?** Transformers work on sequences. By splitting the image into patches, we create a sequence (196 patches = 196 sequence elements).

**Linear Projection:** Each patch is flattened and projected to embedding dimension.
```
768 values → Linear(768, 192) → 192-dimensional embedding
```

### Step 2: Position Embeddings

**Problem:** Unlike text, image patches don't have inherent order. Patch #1 could be from top-left or bottom-right.

**Solution:** Add **learnable position embeddings** to encode spatial information.

```
Patch Embedding: [192 dimensions]
Position Embedding: [192 dimensions]
Combined: Patch + Position = [192 dimensions]
```

**Example:**
- Patch from top-left gets position embedding #0
- Patch from top-right gets position embedding #13
- Model learns that these positions are spatially distant

### Step 3: Class Token

**Special token** prepended to the sequence:
```
[CLS, Patch_1, Patch_2, ..., Patch_196]
```

**Purpose:** The [CLS] token aggregates information from all patches and is used for final classification.

**Why?** In NLP, [CLS] token represents the entire sentence. Here, it represents the entire image.

### Step 4: Transformer Blocks (The Heart of ViT)

ViT-tiny has **12 transformer blocks**. Each block contains:

#### A. Multi-Head Self-Attention (MSA)

**What it does:** Allows each patch to "attend to" every other patch.

**Mechanism:**
1. **Query (Q), Key (K), Value (V):** Each patch generates these three vectors
2. **Attention Scores:** Compute similarity between patches
   ```
   Attention(Q,K,V) = softmax(Q×K^T / √d_k) × V
   ```
3. **Multi-Head:** Run multiple attention mechanisms in parallel (3 heads in ViT-tiny)

**Example:**
- Patch containing "sky" attends strongly to other "sky" patches
- Patch with "grass" attends to "field" patches
- Global context: Every patch sees every other patch!

**Key Insight:** This is **global** from layer 1. Unlike CNNs where receptive field grows gradually.

#### B. Layer Normalization

**Applied twice per block:**
- Before attention: `norm1`
- Before MLP: `norm2`

**Formula:**
```
normalized = (x - mean(x)) / sqrt(var(x) + eps)
output = normalized × gamma + beta
```

**Purpose:** Stabilize training, prevent activation explosion.

**Critical for our work:** These 384 parameters (gamma, beta) in the final `norm` layer showed 11% failure rate!

#### C. Multi-Layer Perceptron (MLP)

**Structure:**
```
Input (192 dim) → Linear → GELU → Linear → Output (192 dim)
```

**Expansion:** Typically expands by 4× in middle layer:
```
192 → 768 → 192
```

**Purpose:** Non-linear transformation, feature refinement after attention.

#### D. Residual Connections

**Around both attention and MLP:**
```
x_new = x + Attention(LayerNorm(x))
x_final = x_new + MLP(LayerNorm(x_new))
```

**Why crucial?** Provides bypass paths. In our fault injection:
- Early blocks: Few residual paths accumulated
- Late blocks: Many residual paths → more robust!

### Step 5: Classification Head

**After 12 transformer blocks:**
```
[CLS] token → Final Layer Norm → Linear(192, 10 classes) → Softmax
```

**Output:** Probability distribution over 10 EuroSAT classes.

---

## CNNs vs ViTs: Key Differences

### 1. Receptive Field

**CNN:**
```
Layer 1: 3×3 local window
Layer 2: 5×5 local window (accumulated)
Layer 3: 7×7 local window
...
Layer 10: Global view (finally!)
```

**ViT:**
```
Layer 1: GLOBAL view (all 196 patches)
Layer 2: GLOBAL view
...
Layer 12: GLOBAL view
```

**Impact on fault tolerance:**
- CNN: Early layer fault is spatially contained
- ViT: Every layer fault affects entire image globally

### 2. Parameter Distribution

**CNN:**
```
Early layers: Few parameters (small kernels)
Late layers: Many parameters (large feature maps)
```

**ViT:**
```
All blocks: Equal parameters (444,864 each)
Norm/Head: Tiny (384 and 1,930)
```

**Our finding:** Mid-blocks (2-5) most vulnerable, not late blocks!

### 3. Computation Pattern

**CNN:**
```
Convolution: Local sliding window
- Weight sharing across spatial locations
- Translation invariance built-in
```

**ViT:**
```
Attention: Global pairwise comparisons
- Every patch compared with every other patch
- O(n²) complexity where n = number of patches
- Position must be learned (position embeddings)
```

### 4. Inductive Biases

**CNN:**
- **Locality:** Nearby pixels are related
- **Translation Equivariance:** Cat in top-left = cat in bottom-right
- **Hierarchy:** Build features from simple (edges) to complex (objects)

**ViT:**
- **Minimal inductive bias:** Learns everything from data
- **Flexibility:** Can learn non-local relationships
- **Data hungry:** Needs more training data than CNNs

**For fault injection:** ViT's lack of locality means faults propagate differently!

### 5. Normalization

**CNN:**
- **Batch Normalization:** Normalizes per channel across batch
- Channel-wise: Fault in one channel doesn't affect others

**ViT:**
- **Layer Normalization:** Normalizes across all features
- Feature-wise: Fault affects entire representation!

**Our discovery:** This is why `norm` layer is so vulnerable (11% failure rate).

---

## Why ViTs for Space Applications?

### Advantages

**1. State-of-the-Art Performance**
- Matches or exceeds CNNs on many vision tasks
- Better scaling with data and model size

**2. Unified Architecture**
- Same architecture for different tasks (classification, detection, segmentation)
- Easier to deploy and maintain

**3. Interpretability**
- Attention maps show what model "looks at"
- Useful for debugging in space (limited communication bandwidth)

**4. Transfer Learning**
- Pre-trained ViTs generalize well
- Useful when labeled space data is scarce

### Challenges (Our Research Focus!)

**1. Computational Cost**
- O(n²) attention complexity
- Higher memory footprint than CNNs
- Critical for resource-constrained space hardware

**2. Reliability Under Radiation**
- Unknown fault tolerance characteristics
- Our work: First comprehensive study!
- Finding: Different vulnerability patterns than CNNs

**3. Real-time Constraints**
- Slower inference than optimized CNNs
- Must balance accuracy vs speed for autonomous spacecraft

---

## Your Specific Model: ViT-tiny

### Architecture Specifications

```python
Model: vit_tiny_patch16_224
Total Parameters: 5,526,346
FP32 Bits: 176,843,072
```

**Structure:**
```
Input: 224×224×3 image
Patch Size: 16×16
Number of Patches: 196
Embedding Dimension: 192
Number of Heads: 3
Number of Blocks: 12
MLP Hidden Dimension: 768 (4× expansion)
Output Classes: 10
```

**Parameter Breakdown:**
```
Patch Embedding:  147,648 params (2.69%)
Block 0-11:       444,864 params each (8.11% each, 97.2% total)
Final Norm:       384 params (0.01%) ← CRITICAL! 11% failure rate
Classification:   1,930 params (0.04%)
```

### Why ViT-tiny?

**1. Appropriate for Space Hardware**
- 5.5M parameters manageable for embedded systems
- Fast inference (~0.5 sec for 1000 images)

**2. Still Representative**
- Same architecture as larger ViTs (ViT-base, ViT-large)
- Findings generalize to bigger models

**3. Practical Testing**
- Can run comprehensive fault injection campaigns
- 16,538 tests in 46 hours for baseline

### Your Task: EuroSAT Classification

**Dataset:** European Space Agency satellite imagery
- 10 classes: Forest, Highway, Industrial, Pasture, etc.
- 27,000 images total
- 64×64 RGB images (resized to 224×224)

**Your Model Performance:**
- Baseline Accuracy: 97.90%
- Very high performance on this task
- Shows model has learned robust features

**Why This Task?**
- Mission-relevant: Real satellite imagery
- Demonstrates practical space application
- Can measure fault impact on actual mission scenario

---

## Key Takeaways

### ViT in One Sentence
> "Vision Transformers treat images as sequences of patches and use global self-attention to process them, unlike CNNs which use local convolutions."

### Why ViT Fault Tolerance Differs from CNNs
1. **Global attention** → faults propagate immediately to all patches
2. **Strong residuals** → late layers have accumulated bypass paths
3. **Layer normalization** → faults affect entire feature vector
4. **Uniform block structure** → no gradual receptive field growth

### Your Novel Findings
- **Mid-layers most vulnerable** (not late layers like CNNs)
- **Norm layer critical** (11% failure rate, 0.01% of model)
- **Bit-30 dominance** (86% failure rate, 96% of all failures)
- **Sign bit robust** (0% vs 5-15% in CNNs)

---

## Further Reading

**Original Papers:**
- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" - The ViT paper
- Vaswani et al. (2017): "Attention is All You Need" - Original Transformer

**Comparison Studies:**
- Raghu et al. (2021): "Do Vision Transformers See Like CNNs?"
- Bhojanapalli et al. (2021): "Understanding Robustness of Transformers"

**Your Research Context:**
- First comprehensive statistical fault injection study of Vision Transformers
- Reveals architectural differences in reliability vs CNNs
- Informs fault-tolerant AI for space applications

---

**Next Document:** Foundation Papers (Leveugle, Ruospo) - Statistical Fault Injection Theory
