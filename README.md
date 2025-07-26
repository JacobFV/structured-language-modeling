# Sturctured Language Modeling

## Motivation

Suppose we could write an algorithm that directly *synthesized* a language model with an understanding of syntactic and semantic structures without having to train on data? Such an algorithm would not be much more computationally efficinet, avoiding needlessly extracting already known structures of language from data, but it would also have fundamentally language-first latent structure and dynamics which would likely enable it to think more creatively than enslopified artifacts that next-word prediction produces.

How could we synthesize such a model? By introducing language biases into the architecture itself. In this project, we are giong to try to add as many general priors of language as we can. We will start with the non-prior baseline implementation, then to regex, then try more advanced.

Note: we use the term "language model" here in iits pre-22023-crazed understanding as just *any* model of language. It could be GOFAI, hypersymbolic, RNN, etc. anything.

## Input processing

Here we do various seq -> output experiments that determine how useful various input processing algorithms are.

### Mean Pooling (baseline)

Simple averaging of input embeddings, serving as a baseline comparison.

Y = \Sum(X) / L

where:

- X = input
- Y = output
- L = length of input sequence

### Convolution 

Sliding window pattern detection using learned filters. Each filter applies a learnable kernel across the input sequence to detect local patterns and linguistic features.

For a 1D convolution with kernel size k and stride s:

Y_i = σ(∑_{j=0}^{k-1} W_j · X_{i·s+j} + b)

where:
- X = input sequence of length L
- Y = output feature map
- W = learnable filter weights of size k
- b = bias term
- σ = activation function (typically ReLU or tanh)
- i = output position index

Multiple filters (F total) can be applied in parallel to capture different patterns:

Y = [conv(X, W_0); conv(X, W_1); ...; conv(X, W_{F-1})]

The convolution operation preserves local sequential structure while reducing dimensionality and extracting position-invariant features relevant to language patterns such as n-grams, morphological structures, and syntactic dependencies.

### RNN

Sequential processing with hidden state updates that maintain memory of previous inputs through recurrent connections.

The basic RNN computes:

h_t = σ(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
y_t = W_{hy} h_t + b_y

where:
- x_t = input at time step t
- h_t = hidden state at time step t
- h_0 = initial hidden state (typically zeros)
- W_{hh} = hidden-to-hidden weight matrix
- W_{xh} = input-to-hidden weight matrix  
- W_{hy} = hidden-to-output weight matrix
- b_h, b_y = bias vectors
- σ = activation function (typically tanh)

For sequence processing, the final output is typically:

Y = h_L or Y = mean([h_1, h_2, ..., h_L])

RNNs can theoretically model arbitrary sequential dependencies but suffer from vanishing gradients for long sequences, limiting their ability to capture long-range linguistic dependencies.

### LSTM

Long-term dependency modeling with gated memory cells that selectively retain, forget, and update information across time steps.

The LSTM uses three gates to control information flow:

Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Cell state update:
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
C_t = f_t ∗ C_{t-1} + i_t ∗ C̃_t

Hidden state:
h_t = o_t ∗ tanh(C_t)

where:
- ∗ denotes element-wise multiplication
- [h_{t-1}, x_t] is vector concatenation
- C_t = cell state (long-term memory)
- h_t = hidden state (short-term memory)
- W_f, W_i, W_o, W_C = learned weight matrices
- b_f, b_i, b_o, b_C = bias vectors

The final sequence representation is Y = h_L or a function of all hidden states. LSTMs excel at capturing long-range dependencies crucial for understanding complex syntactic structures and semantic relationships in language.

### Multi-head attention

Parallel computation of weighted relationships between sequence elements, allowing the model to attend to different representational subspaces simultaneously.

For h attention heads, each head computes:

Attention(Q, K, V) = softmax(QK^T / √d_k)V

where:
- Q = queries = XW^Q_i (d_model × d_k)
- K = keys = XW^K_i (d_model × d_k)  
- V = values = XW^V_i (d_model × d_v)
- d_k = d_v = d_model / h (dimension per head)
- W^Q_i, W^K_i, W^V_i = learned projection matrices

Multi-head attention combines all heads:

MultiHead(X) = Concat(head_1, ..., head_h)W^O

where:
- head_i = Attention(XW^Q_i, XW^K_i, XW^V_i)
- W^O = output projection matrix (d_model × d_model)

The final output for sequence classification is typically:

Y = mean(MultiHead(X)) or Y = MultiHead(X)[0] (using CLS token)

Multi-head attention enables the model to simultaneously focus on different aspects of linguistic structure: syntactic relationships, semantic similarity, positional patterns, and long-range dependencies without the sequential bottleneck of RNNs.

### Regex

We apply a set of D unique regular expressions to analyze the input sequence. For each regex pattern and each position in the sequence, we compute a "match distance" - an integer measuring how close that segment is to matching the pattern. A distance of 0 indicates a perfect match, while larger values represent the minimum number of character edits (insertions, deletions, or substitutions) required to match the pattern.

Importantly, we need to be able to learn the regexes such that they minimize the median regex match distance. This requires that we either (1) directly tweek regexs based on the match errors (weighted by their softmax-weighted alignment) or (2) convert regexes into tensor representations and express regex matching in an fully differentiable operation.

For reference, we employ the following variables:

- X = input
- Y = output
- L = length of input sequence
- v_i \in V = alphabet, input vocabulary
- N_v = |V| = input vocab size
- r_i \in R = regexes
- N_r = num regexes, output dims

#### 1. Directly tweeking regexes to minimze match error

TODO

#### 2. Fully differentiable regex matching and optimization

In this implementation, I convert regexes into a representation that can be modified directly by gradient descent. It works as follows:

1. Convert the regex expression into an e-NFA
2. Expand the e-NFA into a DFA = (Q, V, d, q0, F)
3. Convert the DFA into a modified linear recurrent system where the state is represented by a one-hot encoding and the transition function is a transition matrix selected of |V| possible trnaisiton matrices depending on the input token:
    1. Flatten the transition function d into a |V|x|Q|x|Q| tensor (|V| |Q|x|Q| matrices, one corresponding to the transition function for each possible input).
    2. Init with a one-hot encoding of q0
    3. Evaluate the DFA by one-hot encoding the vocab index of the input token and using that to select which transition matrix to apply.
    4. Reduce sum along the 0th axis to get the final one-hot and dot-prod against the final state vector to check for acceptance.

Okay this is very expensive.

Now, perform the regex filter operation for each filter r_i \in R to get

Y = [regex(X, r_0); regex(X, r_1); ...; regex(X, r_n)]

Ofc, since we have converted the regex op into numerical form we can directly run Y=regex(X, [DFA_0; DFA_1; ...; DFA_n]).

### CFG



## Generation

## Thought

## Interaction
