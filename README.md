# Structured Language Modeling

## Motivation

Suppose we could write an algorithm that directly **synthesizes** a language model with an *a-priori* understanding of syntactic and semantic structure, instead of rediscovering those regularities from data.  
Such a model would  

1. remove much of the heavy empirical training loop (fewer GPU hours),  
2. enforce language-first inductive biases in its latent state, and  
3. arguably enable qualitatively different kinds of systematic reasoning than "enslopified" next-word predictors.  

The project goal is to embed as many *general linguistic priors* as possible **inside the architecture itself**.  
We start with a non-prior baseline (mean pooling) and then progressively graft in convolution, recurrent nets, attention, *regular-expression* banks and finally *context-free-grammar* (CFG) structure.

> "Language model'' here is used in the broad pre-2020 sense: *any* computational system that maps linguistic input to some representation—GOFAI, hypersymbolic, RNN, etc.

---

## Input Processing

### Mean Pooling (baseline)

\[
Y=\frac{1}{L}\sum_{t=1}^{L} X_t ,
\]
a sanity-check comparator.

### Convolution (n-gram detector)

1-D convolution with kernel size \(k\) and stride \(s\):
\[
Y_i=\sigma\!\Bigl(\sum_{j=0}^{k-1} W_j\cdot X_{i\cdot s+j}+b\Bigr).
\]
Multiple filters \(F\) are concatenated.  Captures local morphology and word-order patterns.

### RNN

\[
h_t=\sigma(W_{hh}h_{t-1}+W_{xh}x_t+b_h),\qquad
Y=h_L\ \text{or}\ \frac1L\sum_{t}h_t .
\]

### LSTM (long-range dependencies)

Gated updates  
\[
\begin{aligned}
f_t &=\sigma(W_f[h_{t-1},x_t]+b_f),\\
i_t &=\sigma(W_i[h_{t-1},x_t]+b_i),\\
o_t &=\sigma(W_o[h_{t-1},x_t]+b_o),\\
\tilde C_t &=\tanh(W_C[h_{t-1},x_t]+b_C),\\[2pt]
C_t &=f_t\odot C_{t-1}+i_t\odot\tilde C_t,\\
h_t &=o_t\odot\tanh(C_t).
\end{aligned}
\]

### Multi-head Attention

Standard Transformer attention:
\[
\operatorname{MultiHead}(X)=\bigl[\operatorname{head}_1;\dots;\operatorname{head}_h\bigr]W^O,\qquad
\operatorname{head}_i=\operatorname{softmax}\!\Bigl(\tfrac{QK^\top}{\sqrt{d_k}}\Bigr)V .
\]

---

## Regex  — *Finite-state priors*

Let  
• \(V=\{v_1,\dots,v_{N_v}\}\) be the alphabet,  
• input sequence \(X=(x_1,\dots,x_L)\),  
• a bank of \(D\) regular expressions \(\mathcal R=\{r_1,\dots,r_D\}\).  

### Output interface

For every \(r_d\) we return a *match score* \(s_d(X)\).  
Stack them:
\[
Y=\bigl[s_1(X);\;s_2(X);\;\dots;\;s_D(X)\bigr]\in\mathbb R^{D}.
\]

### 1. Discrete optimisation ("directly tweaking the regex")

Objective  
\[
\mathcal L(r_d)=
\operatorname{median}_{X\in\mathcal D}\operatorname{LevDist}\bigl(X,\;\operatorname{Lang}(r_d)\bigr),
\tag{1}
\]
where \(\operatorname{Lang}(r_d)\) is the language accepted by \(r_d\).

Search strategies  

1. **Beam-guided program synthesis** over the regex DSL  
   – tokens = literals, `|`, concatenation, `*`, `+`, `?`, parentheses.  
   – neighbourhood = single-token edits.  
   – keep top-\(B\) candidates by the score \(-\mathcal L\).

2. **Evolutionary algorithms**  
   – population \(P\) of ASTs, mutation = token edit, crossover = subtree swap.  
   – fitness = \(-\mathcal L-\lambda|r|\).

3. **Reinforcement learning**  
   – policy \(\pi_\theta\) generates token sequence; reward = \(-\mathcal L-\beta|r|\).  
   – update via REINFORCE with entropy regularisation.

These methods preserve *hard* symbolic semantics and allow strict constraints (e.g. POSIX compatibility, worst-case run-time bounds).

### 2. Fully differentiable regex layer

#### 2.1 Compilation to DFA-tensor

Compile each \(r_d\) to a DFA \(\mathcal A^{(d)}=(Q^{(d)},V,\delta^{(d)},q_0^{(d)},F^{(d)})\).  
Represent the transition function as a **log-probability tensor**

\[
T^{(d)}\in\mathbb R^{|Q^{(d)}|\times|Q^{(d)}|\times N_v},
\quad
T_{i,j,v}^{(d)}=
\begin{cases}
0   & \text{if } \delta^{(d)}(q_i,v)=q_j,\\
-\infty & \text{otherwise}.
\end{cases}
\]

#### 2.2 Relaxation

Turn \(T^{(d)}\) into *learnable* parameters \(A^{(d)}\):
\[
P^{(d)}*{i,j,v}=\operatorname{softmax}*{v}\bigl(A^{(d)}_{i,j,v}\bigr).
\tag{2}
\]

#### 2.3 Forward pass (log-space WFSA)

Initial state one-hot \(s_0=e_{q_0^{(d)}}\).  
For \(t=1,\dots,L\):
\[
s_t=\operatorname{softmax}\!\Bigl(
\log P^{(d)}[:,:,x_t]\;+\;\log s_{t-1}
\Bigr).
\tag{3}
\]
Acceptance probability  
\[
s_d(X)=\sigma\bigl(w^{(d)\top}s_L\bigr),
\qquad
w^{(d)}=\sum_{q\in F^{(d)}} e_q .
\]

All operations are batched; merging the \(D\) DFAs into one block-diagonal tensor allows GPU kernels with time \(O\bigl(L\,\sum_d |Q^{(d)}|^2\bigr)\).

#### 2.4 Training signals  

If labelled ⇢ cross-entropy on \(s_d\); unlabeled ⇢ contrastive or self-supervised objectives.  
Regularise by KL\((P^{(d)}\Vert T^{(d)})\) to keep transitions sparse.  
Optionally perform periodic "snap-back" → project each \(\arg\max_v P_{i,j,v}\) to 1, others 0.

#### 2.5 Smooth Levenshtein distance alternative  

Dynamic-time-warping recurrence
\[
\widetilde D_{i,j}=\operatorname{LSE}*\tau
\begin{cases}
\widetilde D*{i-1,j-1}+c_{\text{sub}},\\
\widetilde D_{i-1,j  }+c_{\text{del}},\\
\widetilde D_{i  ,j-1}+c_{\text{ins}},
\end{cases}
\]
temperature \(\tau\) gives differentiability; back-prop updates literal embeddings of the regex bank rather than structure.

---

## CFG  — *Context-free priors*

Let  
• non-terminals \(N=\{A_1,\dots,A_{N_{NT}}\}\), terminal set \(V\) as before.  
We parameterise a **probabilistic CFG (PCFG)**.

### 1. Rule tensors

Terminal (unary) rules  
\[
R_1\in\mathbb R^{N_{NT}\times N_v},\qquad
R_1[A,v]=\log P(A\to v).
\]

Binary rules  
\[
R_2\in\mathbb R^{N_{NT}\times N_{NT}\times N_{NT}},\qquad
R_2[A,B,C]=\log P(A\to BC).
\]
Row-wise log-softmax ensures valid probabilities.

### 2. Inside algorithm (tensor form)

Create chart \(\alpha\in\mathbb R^{L\times L\times N_{NT}}\).

Initialisation (\(\ell=1\)):
\[
\alpha_{i,i+1,:}=R_1[:,x_i].
\tag{5}
\]

Recursion for span length \(\ell=2,\dots,L\):
\[
\alpha_{i,j,:}=
\operatorname{LSE}*{k=i+1}^{j-1}
\operatorname{LSE}*{B,C}
\bigl(R_2[:,B,C]+\alpha_{i,k,B}+\alpha_{k,j,C}\bigr).
\tag{6}
\]

Implementation sketch (PyTorch)

```python
for span in range(2, L+1):
    left  = alpha[:, :L-span, :, None]          # (B, L-span, NT, 1)
    right = alpha[:, span:, None, :]            # (B, L-span, 1, NT)
    tmp   = left + right                       # broadcasting k dimension
    score = tmp[None] + R2[:, None, None]      # add rule scores
    alpha[:, :L-span, :] = logsumexp(score, dim=(0,3,4))
```

Complexity \(O(L^3N_{NT}^2)\) yet GPU-friendly; with \(N_{NT}\le 64\) and \(L\le 64\) parses in milliseconds.

Sentence likelihood  
\[
\log p(X)=\alpha_{0,L,S},\quad S\text{ = start symbol}.
\]

### 3. Learning

Losses  
• *Unsupervised* = \(-\log p(X)\).  
• *Supervised* (gold parse \(T^\star\))  
  \[
  \mathcal L=-\sum_{\text{rules }r\in T^\star}\log P(r).
  \]
Gradients flow through the inside chart; auto-diff reproduces inside–outside EM but allows mini-batch SGD.

### 4. Deterministic LL(k) option

LL(k) grammars can be encoded by a **differentiable push-down automaton (PDA)**:

Stack representation \(S_t\in\mathbb R^{d_{\text{depth}}\times d_{\text{vec}}}\).  
Controller RNN emits logits for *push*, *pop*, *replace*, relaxed via Gumbel-Softmax.  
Predictive table \(\Pi\in\mathbb R^{N_{NT}\times N_v^{\,k}\times N_{\text{action}}}\) is learnable.  
Hard stack behaviour can be recovered with straight-through estimation.

### 5. Feature read-outs

1. Root distribution \(\alpha_{0,L,:}\)  →  vector embedding.  
2. Expected rule counts \(E[\text{freq}(A\!\to\!BC|X)]\) via automatic differentiation.  
3. Viterbi tree (argmax in Eq. 6) → encode using tree-LSTM.

---

## Generation

For generation we invert the parsing layers:

0. Raw logits (baseline). Sample from logits directly outputted by the LLM. (Current convention)
1. Regex-weighted logits: sample DFA paths with probability proportional to \(P^{(d)}\), convert to strings.
2. PCFG-wieghted logits: ditto
3. Mixture of priors: LLM outputs all three p_raw, p_re, p_pcfg and mixing coefficients (beta \leftarrow softmax(c_re, c_pcfg)) for each token

---

### Evaluation Task Suite

| # | Task Family | Concrete Task (Metric) | Suggested Dataset(s) | Main Input Bias Probed |
|---|-------------|------------------------|----------------------|------------------------|
| 1 | A. Next-Word Prediction | Token-level LM (perplexity) | Penn Treebank (PTB), WikiText-2 (medium), WikiText-103 (large) | All; baseline comparison |
| 2 |              | Character-level LM | enwik8 | Regex/CNN (character n-grams) |
| 3 |              | Cloze completion for discourse coherence | LAMBADA | Attention, long-range RNN |
| 4 |              | Subject–verb agreement LM accuracy (Linzen et al.) | Linzen 2016 agreement set | CFG/LSTM bias toward long dependency |
| 5 | B. Sequence-Level Classification | Sentiment (accuracy) | SST-2 (small), IMDB (large) | CNN (local phrases) vs hierarchical (CFG) |
| 6 |              | Topic classification | AG News, DBPedia | Mean-pool vs CNN |
| 7 |              | Acceptability judgements | CoLA | CFG bias (grammar well-formedness) |
| 8 | C. Sequence Labelling | POS tagging (token F1) | PTB POS split, Universal Dependencies | RNN/LSTM; regex feature ablation |
| 9 |              | Named Entity Recognition | CoNLL-2003, OntoNotes 5.0 | CNN+regex for orthography, attention for context |
|10 |              | Shallow chunking | CoNLL-2000 | Regex/CNN (phrase boundary cues) |
|11 | D. Structured Prediction | Constituency parsing (exact match) | PTB §23, Berkeley Neural Parser splits | CFG module most critical |
|12 |              | Dependency parsing (LAS/UAS) | UD English-EWT | LSTM/attention long links |
|13 |              | Machine translation (BLEU) | IWSLT14 En–De (small), WMT14 En–De (large) | Attention; regex for sub-word patterns |
|14 |              | Semantic parsing (exact set match) | WikiSQL, ATIS | CFG/regex (formal grammar) |
|15 | E. Pattern / Grammar | Tomita grammars classification | Synthetic Tomita dataset | Regex layer (finite-state) |
|16 |              | Parentheses-matching / counting | ListOps (accuracy) | CFG layer (hierarchy depth) |
|17 |              | Arithmetic expression evaluation | PCFG-generated MathExpr | CFG; long-range attention |
|18 |              | CFQ (generalisation to new compositions) | Google CFQ | CFG; compositional generalisation |
|19 | F. Reasoning / Inference | Natural Language Inference (accuracy) | SNLI (small), MultiNLI, ANLI | Attention + CFG semantics |
|20 |              | Extractive QA (F1/Exact-Match) | SQuAD 1.1 / 2.0 | Attention; regex for span start/end |

### How to Use the Suite

1. **Encoder swap experiment**  
   Keep the downstream head identical, plug each input-processing layer as encoder, train under identical hyper-parameters, then report the above metrics.

2. **Ablation for symbolic priors**  
   For regex/CFG modules run *hard* (discrete) vs *soft* (relaxed) vs *removed* settings; measure delta-performance on tasks #11, #15–18.

3. **Sample efficiency curves**  
   Evaluate each model on 1 %, 10 %, 100 % of training data for tasks #1, #5, #11 to quantify gains from built-in priors.

4. **Generalisation stress tests**  
   • Zero-shot split in CFQ, ListOps length extrapolation, Linzen long-sentence subset.  
   • Compare perplexity/accuracy degradation as sequence length grows.

5. **Compute budget**  
   Provide FLOPs and wall-time for each encoder on PTB (#1) and ListOps (#16) to verify that symbolic priors can sometimes *reduce* training cost.