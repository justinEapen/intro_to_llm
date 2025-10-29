# Introduction to Large Language Models (LLMs) - All Assignment Questions

*Complete collection of current year and previous year assignments*

---

## Current Year Assignments

### Week 1 : Assignment 1

**Q1 (1 point) - MCQ**

Which of the following best demonstrates the principle of distributional semantics?

(A) Words that co-occur frequently tend to share semantic properties.

(B) Each word has a unique, fixed meaning regardless of context.

(C) Syntax determines the entire meaning of a sentence.

(D) Distributional semantics is unrelated to word embeddings.

**Answer:** Words that co-occur frequently tend to share semantic properties.

---

**Q2 (1 point) - MCQ**

Which of the following words is  least likely to be polysemous?

(A) Bank

(B) Tree

(C) Gravity

(D) Idea

**Answer:** Gravity

---

**Q3 (1 point) - MCQ**

Consider the following sentence pair: Sentence 1: Riya dropped the glass. Sentence 2: The glass broke.Does Sentence 1 entail Sentence 2?

(A) Yes

(B) No

**Answer:** No

---

**Q4 (0 point) - MCQ**

Which sentence contains a homonym?

(A) He wound the clock before bed.

(B) She tied her hair in a bun.

(C) I can’t bear the noise.

(D) He likes to bat after lunch.

**Answer:** He likes to bat after lunch.

---

**Q5 (1 point) - MCQ**

Which of the following relationships are incorrectly labeled?

(A) Car is a meronym of wheel.

(B) Rose is a hyponym of flower.

(C) Keyboard is a holonym of key.

(D) Tree is a hypernym of oak.

**Answer:** Car is a meronym of wheel.

---

**Q6 (1 point) - MCQ**

_________ studies how context influences the interpretation of meaning.

(A) Syntax

(B) Morphology

(C) Pragmatics

(D) Semantics

**Answer:** Pragmatics

---

**Q7 (1 point) - MCQ**

In the sentence, “After Sita praised Radha, she smiled happily,” who does “she” most likely refer to?

(A) Sita

(B) Radha

(C) Ambiguous

(D) Neither

**Answer:** Ambiguous

---

**Q8 (1 point) - MCQ**

Which of the following statements is true?(i) Word embeddings capture semantic similarity through context.(ii) Morphological analysis is irrelevant in LLMs.(iii) Hypernyms are more specific than hyponyms.

(A) Only (i)

(B) Only (i) and (iii)

(C) Only (ii) and (iii)

(D) All of the above

**Answer:** Only (i)

---

**Q9 (1 point) - MSQ**

What issues can be observed in the following text?On a much-needed #workcation in beautiful Goa. Workin & chillin by d waves!

(A) Idioms

(B) Non-standard English

(C) Tricky Entity Names

(D) Neologisms

**Answer:** Non-standard English, Neologisms

---

**Q10 (1 point) - MCQ**

In semantic role labelling, we determine the semantic role of each argument with respect to the___________ of the sentence.

(A) noun phrase

(B) subject

(C) predicate

(D) adjunct

**Answer:** predicate

---

### Week 2 : Assignment 2

**Q1 (1 point) - MCQ**

Which of the following does not directly affect perplexity?

(A) Vocabulary size

(B) Sentence probability

(C) Number of tokens

(D) Sentence length

**Answer:** Vocabulary size

---

**Q2 (1 point) - MCQ**

Which equation expresses the chain rule for a 4-word sentence?

(A) P(w1, w2, w3, w4) = P(w1) + P(w2|w1) + P(w3|w2) + P(w4|w3)

(B) P(w1, w2, w3, w4) = P(w1) × P(w2|w1) × P(w3|w1, w2) × P(w4|w1, w2, w3)

(C) P(w1, w2, w3, w4) = P(w1) × P(w2|w1) × P(w3|w2) × P(w4|w3)

(D) P(w1, w2, w3, w4) = P(w4|w3) × P(w3|w2) × P(w2|w1) × P(w1)

**Answer:** P(w1, w2, w3, w4) = P(w1) × P(w2|w1) × P(w3|w1, w2) × P(w4|w1, w2, w3)

---

**Q3 (1 point) - MCQ**

Which assumption allows n-gram models to reduce computation?

(A) Bayes Assumption

(B) Chain Rule

(C) Independence Assumption

(D) Markov Assumption

**Answer:** Markov Assumption

---

**Q4 (1 point) - MCQ**

In a trigram language model, which of the following is a correct example of linear interpolation?

(A) P(wi∣wi−2,wi−1)=λ1P(wi∣wi−2,wi−1)

(B) P(wi∣wi−2,wi−1)=λ1P(wi∣wi−2,wi−1)+λ2P(wi∣wi−1)+λ3P(wi)

(C) P(wi∣wi−2,wi−1)=max(P(wi∣wi−2,wi−1),P(wi∣wi−1))

(D) P(wi∣wi−2,wi−1)=P(wi)P(wi−1)/P(wi−2)

**Answer:** P(wi∣wi−2,wi−1)=λ1P(wi∣wi−2,wi−1)+λ2P(wi∣wi−1)+λ3P(wi)

---

**Q5 (1 point) - MCQ**

A trigram model is equivalent to which order Markov model?

(A) 3

(B) 2

(C) 1

(D) 4

**Answer:** 2

---

**Q6 (1 point) - MCQ**

Which smoothing technique leverages the number of unique contexts a word appears in?

(A) Good-Turing

(B) Add-k

(C) Kneser-Ney

(D) Absolute Discounting

**Answer:** Kneser-Ney

---

**Q7 (2 points) - MCQ**

Assuming a bi-gram language model, calculate the probability of the sentence:&lt;s&gt;birds fly in the blue sky&lt;/s&gt;Ignore the unigram probability of P(&lt;s&gt;) in your calculation.

(A) 2/37

(B) 1/27

(C) 0

(D) 1/36

**Answer:** 0

---

**Q8 (2 points) - MCQ**

Assuming a bi-gram language model, calculate the perplexity of the sentence:&lt;s&gt;birds fly in the blue sky&lt;/s&gt;
Please do not consider &lt;s&gt;and &lt;/s&gt;  as words of the sentence.

(A) 271/4

(B) 271/5

(C) 91/6

(D) None of these

**Answer:** None of these

---

### Week 3 : Assignment 3

**Q1 (1 point) - MCQ**

In backpropagation, which method is used to compute the gradients?

(A) Gradient descent

(B) Chain rule of derivatives

(C) Matrix factorization

(D) Linear regression

**Answer:** Chain rule of derivatives

---

**Q2 (1 point) - MCQ**

Which of the following functions is not differentiable at zero?

(A) Sigmoid

(B) Tanh

(C) ReLU

(D) Linear

**Answer:** ReLU

---

**Q3 (1 point) - MCQ**

In the context of regularization, which of the following statements is true?

(A) L2 regularization tends to produce sparse weights

(B) Dropout is applied during inference to improve accuracy

(C) L1 regularization adds the squared weight penalties to the loss function

(D) Dropout prevents overfitting by randomly disabling neurons during training

**Answer:** Dropout prevents overfitting by randomly disabling neurons during training

---

**Q4 (1 point) - MCQ**

Which activation function is least likely to suffer from vanishing gradients?

(A) Tanh

(B) Sigmoid

(C) ReLU

**Answer:** ReLU

---

**Q5 (1 point) - MCQ**

Which of the following equations correctly represents the derivative of the sigmoid function?

(A) σ(x) · (1 + σ(x))

(B) σ(x)²

(C) σ(x) · (1 − σ(x))

(D) 1 / (1 + e^x)

**Answer:** σ(x) · (1 − σ(x))

---

**Q6 (1 point) - MCQ**

What condition must be met for the Perceptron learning algorithm to converge?

(A) Learning rate must be zero

(B) Data must be non-linearly separable

(C) Data must be linearly separable

(D) Activation function must be sigmoid

**Answer:** Data must be linearly separable

---

**Q7 (1 point) - MCQ**

Which of the following logic functions requires a network with at least one hidden layer to model?

(A) AND

(B) OR

(C) NOT

(D) XOR

**Answer:** XOR

---

**Q8 (1 point) - MCQ**

Why is it necessary to include non-linear activation functions between layers in an MLP?

(A) Without them, the network is just a linear function

(B) They prevent overfitting

(C) They allow backpropagation to work

**Answer:** Without them, the network is just a linear function

---

**Q9 (1 point) - MCQ**

What is typically the output activation function for an MLP solving a binary classification task?

(A) Tanh

(B) ReLU

(C) Sigmoid

(D) Softmax

**Answer:** Sigmoid

---

**Q10 (1 point) - MCQ**

Which type of regularization encourages sparsity in the weights?

(A) L1 regularization

(B) L2 regularization

(C) Dropout

(D) Early stopping

**Answer:** L1 regularization

---

### Week 4 : Assignment 4

**Q1 (1 point) - MCQ**

A one-hot vector representation captures semantic similarity between related words like "king" and "queen".

(A) True

(B) False

**Answer:** False

---

**Q2 (1 point) - MCQ**

Which method is used to reduce the dimensionality of a term-context matrix in count-based word representations?

(A) Principal Component Analysis

(B) Matrix Inversion

(C) Singular Value Decomposition (SVD)

(D) Latent Dirichlet Allocation

**Answer:** Singular Value Decomposition (SVD)

---

**Q3 (1 point) - MCQ**

Which property makes tf-idf a better representation than raw term frequency?

(A) It is non-linear

(B) It accounts for the informativeness of words

(C) It penalizes longer documents

(D) It uses hierarchical clustering

**Answer:** It accounts for the informativeness of words

---

**Q4 (1 point) - MCQ**

What is the purpose of using negative sampling in Word2Vec training?

(A) To reduce dimensionality of word vectors

(B) To ensure gradient convergence

(C) To balance class distribution in classification

(D) To simplify softmax computation

**Answer:** To simplify softmax computation

---

**Q5 (1 point) - MCQ**

In skip-gram Word2Vec, the model:

(A) Predicts a word given its context

(B) Predicts the next sentence

(C) Predicts surrounding context words given a target word

(D) Learns n-gram frequencies

**Answer:** Predicts surrounding context words given a target word

---

**Q6 (1 point) - MCQ**

Why does SVD-based word embedding struggle with adding new words to the vocabulary?

(A) It uses online learning

(B) It lacks semantic interpretability

(C) It assumes word order

(D) It is computationally expensive to retrain

**Answer:** It is computationally expensive to retrain

---

**Q7 (1 point) - MCQ**

Which of the following best describes the term “distributional hypothesis” in NLP?

(A) Words with high frequency have greater meaning

(B) Words are defined by their part-of-speech tags

(C) A word’s meaning is characterized by the words around it

(D) Words should be normalized before vectorization

**Answer:** A word’s meaning is characterized by the words around it

---

**Q8 (1 point) - MCQ**

In Word2Vec, similarity between word vectors is computed using Euclidean distance.

(A) True

(B) False

**Answer:** False

---

**Q9 (1 point) - MCQ**

Which method solves the problem of OOV (Out-Of-Vocabulary) words better?

(A) One-hot encoding

(B) CBOW

(C) Skip-gram with subsampling

(D) FastText embedding

**Answer:** FastText embedding

---

**Q10 (1 point) - MCQ**

If the word "economy" occurs 4 times in a corpus, and "growth" appears in a window of 5 words around it 3 times, what is the entry for (economy, growth) in a term-context matrix?

(A) 1

(B) 2

(C) 3

(D) 4

**Answer:** 3

---

### Week 5 : Assignment 5

**Q1 (1 point) - MCQ**

Which of the following best explains the vanishing gradient problem in RNNs?

(A) RNNs lack memory mechanisms for long-term dependencies.

(B) Gradients grow too large during backpropagation.

(C) Gradients shrink exponentially over long sequences.

(D) RNNs cannot process variable-length sequences.

**Answer:** Gradients shrink exponentially over long sequences.

---

**Q2 (1 point) - MCQ**

In an attention mechanism, what does the softmax function ensure?

(A) Normalization of decoder outputs

(B) Stability of gradients during backpropagation

(C) Values lie between -1 and 1

(D) Attention weights sum to 1

**Answer:** Attention weights sum to 1

---

**Q3 (1 point) - MCQ**

Which of the following is true about the difference between a standard RNN and an LSTM?

(A) LSTM does not use any non-linear activation.

(B) LSTM has a gating mechanism to control information flow.

(C) RNNs have fewer parameters than LSTMs because they use convolution.

(D) LSTMs cannot learn long-term dependencies.

**Answer:** LSTM has a gating mechanism to control information flow.

---

**Q4 (1 point) - MCQ**

Which gate in an LSTM is responsible for deciding how much of the cell state to keep?

(A) Forget gate

(B) Input gate

(C) Output gate

(D) Cell candidate gate

**Answer:** Forget gate

---

**Q5 (1 point) - MCQ**

What improvement does attention bring to the basic Seq2Seq model?

(A) Reduces training time

(B) Removes the need for an encoder

(C) Allows access to all encoder states during decoding

(D) Reduces the number of model parameters

**Answer:** Allows access to all encoder states during decoding

---

**Q6 (1 point) - MCQ**

Which of the following is a correct statement about the encoder-decoder architecture?

(A) The encoder generates tokens one at a time.

(B) The decoder summarizes the input sequence.

(C) The decoder generates outputs based on encoder representations and its own prior outputs.

(D) The encoder stores only the first token of the sequence.

**Answer:** The decoder generates outputs based on encoder representations and its own prior outputs.

---

**Q7 (1 point) - MCQ**

What is self-attention in Transformers used for?

(A) To enable sequential computation

(B) To attend to the previous layer’s output

(C) To relate different positions in the same sequence

(D) To enforce fixed-length output

**Answer:** To relate different positions in the same sequence

---

**Q8 (1 point) - MCQ**

Why are RNNs preferred over fixed-window neural models?

(A) They have a smaller parameter size.

(B) They can process sequences of arbitrary length.

(C) They eliminate the need for embedding layers.

(D) None of the above.

**Answer:** They can process sequences of arbitrary length.

---

**Q9 (2 points) - MCQ**

Given the following encoder and decoder hidden states, compute the attention scores. (Use dot product as the scoring function) Encoder hidden states: h1=[7,3], h2=[0,2], h3=[1,4]  Decoder hidden state: s=[0.2,1.5] 

(A) 0.42, 0.02, 0.56

(B) 0.15, 0.53, 0.32

(C) 0.64, 0.18, 0.18

(D) 0.08, 0.91, 0.01

**Answer:** 0.42, 0.02, 0.56

---

### Week 6 : Assignment 6

**Q1 (1 point) - MCQ**

RoPE uses additive embeddings like sinusoidal encoding.

(A) True

(B) False

**Answer:** False

---

**Q2 (1 point) - MCQ**

Which of the following is true about multi-head attention?

(A) It increases model interpretability by using a single set of attention weights

(B) Each head operates on different parts of the input in parallel

(C) It reduces the number of parameters in the model

(D) Heads are averaged before applying the softmax function

**Answer:** Each head operates on different parts of the input in parallel

---

**Q3 (1 point) - MCQ**

What is the role of the residual connection in the Transformer architecture?

(A) Improve gradient flow during backpropagation

(B) Normalize input embeddings

(C) Reduce computational complexity

(D) Prevent overfitting

**Answer:** Improve gradient flow during backpropagation

---

**Q4 (1 point) - MCQ**

The feedforward network in a Transformer block introduces non-linearity between attention layers.

(A) True

(B) False

**Answer:** True

---

**Q5 (1 point) - MCQ**

The sinusoidal positional encoding uses sine for even dimensions and ___ for odd dimensions.

(A) sine

(B) cosine

(C) tangent

(D) None of these

**Answer:** cosine

---

**Q6 (1 point) - MCQ**

Why is positional encoding added to input embeddings in Transformers?

(A) To provide unique values for each word

(B) To indicate the position of tokens since Transformers are non-sequential

(C) To scale embeddings

(D) To avoid vanishing gradients

**Answer:** To indicate the position of tokens since Transformers are non-sequential

---

**Q7 (2 points) - MCQ**

You are given a self-attention layer with input dimension 512, using 8 heads. What is the output dimension per head?

(A) 64

(B) 128

(C) 32

(D) 256

**Answer:** 64

---

**Q8 (2 points) - MCQ**

For a transformer with dmodel = 512, calculate the positional encoding for position p=14 and dimensions 6 and 7 using the sinusoidal formula: 

(A) Option A: PE(14, 6) = 0.8256, PE(14, 7) = 0.5642

(B) Option B: PE(14, 6) = 0.8192, PE(14, 7) = 0.5735

(C) Option C: PE(14, 6) = 0.5642, PE(14, 7) = 0.8256

(D) Option D: PE(14, 6) = 0.8256, PE(14, 7) = 0.5735

**Answer:** Option A: PE(14, 6) = 0.8256, PE(14, 7) = 0.5642

---

### Week 7 : Assignment 7

**Q1 (1 point) - MCQ**

Why can a pre-trained BART model be fine-tuned directly for abstractive summarization?

(A) Its encoder alone is sufficient.

(B) It shares vocabulary with summarization datasets.

(C) It uses a larger context window than BERT.

(D) It already contains a generative decoder trained jointly during pre-training.

**Answer:** It already contains a generative decoder trained jointly during pre-training.

---

**Q2 (2 points) - MSQ**

For pre-training of encoder-decoder models, which statement(s) is/are true?

(A) The encoder attends bidirectionally to its whole input.

(B) The decoder conditions on earlier decoder tokens and encoder outputs.

(C) Unlabelled text is turned into a supervised task via a noising scheme.

(D) Training relies on a next-sentence-prediction loss.

**Answer:** The encoder attends bidirectionally to its whole input., The decoder conditions on earlier decoder tokens and encoder outputs., Unlabelled text is turned into a supervised task via a noising scheme.

---

**Q3 (2 points) - MSQ**

Which attention mask(s) prevent(s) a token from looking at future positions?

(A) Causal mask

(B) Fully-visible mask

(C) Prefix-LM mask

(D) All of the above

(E) None of the above

**Answer:** Causal mask, Prefix-LM mask

---

**Q4 (1 point) - MCQ**

T5 experiments showed that clean and compact pre-training data can outperform a larger but noisier corpus primarily because:

(A) Larger corpora overfit.

(B) Noise forces the model to waste capacity on modelling irrelevant patterns.

(C) Clean data has longer documents.

(D) Compact data allows bigger batches.

**Answer:** Noise forces the model to waste capacity on modelling irrelevant patterns.

---

**Q5 (1 point) - MCQ**

What makes sampling from an auto-regressive language model straightforward?

(A) The model is deterministic.

(B) The vocabulary is small.

(C) Each conditional distribution over the vocabulary is readily normalised and can be sampled token-by-token.

(D) Beam search guarantees optimality.

**Answer:** Each conditional distribution over the vocabulary is readily normalised and can be sampled token-by-token.

---

**Q6 (1 point) - MCQ**

Why does ELMo build its input token representations from a character-level CNN instead of fixed word embeddings?

(A) To reduce training time by sharing parameters

(B) To avoid UNK tokens and generate representations for any string

(C) To compress embeddings to 128 dimensions

(D) To ensure the same vector for a word in every context

**Answer:** To avoid UNK tokens and generate representations for any string

---

**Q7 (2 points) - NUMERIC**

Calculate the total number of parameters in the decoder part of the autoencoder shown in the image: 

[Image of autoencoder diagram]


**Answer:** 87

---

### Week 8 : Assignment 8

**Q1 (1 point) - MCQ**

In standard instruction tuning with a decoder-only LM, which tokens typically contribute to the next-token prediction loss?

(A) Only the prompt tokens

(B) Only the response tokens

(C) Both prompt and response tokens

(D) Neither

(E) loss is computed at the sequence level only

**Answer:** Only the response tokens

---

**Q2 (1 point) - MCQ**

Why can using multiple instruction templates for the same task help?

(A) It only increases the dataset size.

(B) It regularizes the reward model.

(C) It improves generalization by exposing the model to different phrasings of the instruction.

(D) It ensures the same tokenization across tasks.

**Answer:** It improves generalization by exposing the model to different phrasings of the instruction.

---

**Q3 (1 point) - MCQ**

As the model size grows, what happens to prompt length and initialization sensitivity in prompt tuning?

(A) Both matter more.

(B) Both matter less.

(C) Length matters less but initialization matters more.

(D) Initialization matters less but length matters more.

**Answer:** Both matter less.

---

**Q4 (1 point) - MSQ**

Which of the following statement(s) is/are true about the POSIX metric for quantifying prompt sensitivity?

(A) POSIX is independent of the correctness of the generated responses and captures sensitivity as a property independent of correctness

(B) POSIX is a length-normalized metric

(C) POSIX compares the generated responses against the ground-truth to quantify prompt sensitivity

(D) POSIX captures the variance in the log-likelihood of the same response for different input prompt variations

**Answer:** POSIX is independent of the correctness of the generated responses and captures sensitivity as a property independent of correctness, POSIX is a length-normalized metric, POSIX captures the variance in the log-likelihood of the same response for different input prompt variations

---

**Q5 (1 point) - MCQ**

Which statement is true about prompt sensitivity as captured by POSIX?

(A) Larger models always have lower prompt sensitivity than smaller ones.

(B) Larger models always have higher prompt sensitivity than smaller ones.

(C) Prompt sensitivity decreases for models with a parameter count above a certain threshold.

(D) Increasing parameter count does not necessarily reduce prompt sensitivity.

**Answer:** Increasing parameter count does not necessarily reduce prompt sensitivity.

---

**Q6 (1 point) - MCQ**

In training a reward model with pairwise preferences (x, y+, y-), the Bradley-Terry style objective encourages:

(A) Maximizing rθ (x,y- ) - rθ (x,y+ )

(B) Minimizing the entropy of the policy

(C) Maximizing log⁡σ (rθ (x,y+ ) - rθ (x,y- ))

(D) Setting rθ (x,y) equal to the log-probability under πref

**Answer:** Maximizing log⁡σ (rθ (x,y+ ) - rθ (x,y- ))

---

**Q7 (1 point) - MSQ**

Which of the following are recommended while performing REINFORCE-style policy optimization?

(A) Use the log-derivative trick to obtain an unbiased gradient estimator.

(B) Weight token-level log-probs by the advantage function to reduce variance.

(C) Use importance weights and clip them when sampling from a fixed policy.

(D) Avoid any clipping to preserve gradient magnitude.

**Answer:** Use the log-derivative trick to obtain an unbiased gradient estimator., Weight token-level log-probs by the advantage function to reduce variance., Use importance weights and clip them when sampling from a fixed policy.

---

**Q8 (1 point) - MCQ**

Which method combines reward maximization and minimizing KL divergence?

(A) REINFORCE

(B) Monte Carlo Approximation

(C) Proximal Policy Optimization

(D) Constitutional AI

**Answer:** Proximal Policy Optimization

---

**Q9 (1 point) - MCQ**

Which of the following is the reason for performing alignment beyond instruction tuning in LLMs?

(A) Instruction tuning guarantees safety on harmful queries.

(B) Alignment can prevent outputs that a model might otherwise deem correct, but humans find unacceptable.

(C) Alignment is only needed for small models.

(D) Instruction tuning already optimizes a human preference model.

**Answer:** Alignment can prevent outputs that a model might otherwise deem correct, but humans find unacceptable.

---

**Q10 (1 point) - MCQ**

Let πθ   be the probability of choosing token at in state st assigned by the current policy being optimized, πk be that by the old/reference policy and ∈ > 0 be the clip parameter. When the token-level advantage At is positive, PPO-CLIP maximizes which of the following expression at step t?

(A) max ⁡(πθ/πk   , 1 - ∈) At

(B) max ⁡(πk/πθ   ,1-∈) At

(C) min ⁡(πk/πθ   ,1+∈) At

(D) min ⁡(πθ/πk   ,1+∈) At

**Answer:** min ⁡(πθ/πk   ,1+∈) At

---

### Week 9 : Assignment 9

**Q1 (1 point) - MCQ**

In the knowledge-graph training pipeline that models P(o | s, r) with a softmax over all entities, what practical difficulty motivates the use of negative sampling?

(A) The softmax is undefined for KG scores.

(B) The denominator sums over all entities, which is computationally expensive.

(C) The numerator requires the full adjacency list for each relation.

(D) The scores must be normalized per relation rather than globally.

**Answer:** The denominator sums over all entities, which is computationally expensive.

---

**Q2 (1 point) - MSQ**

Which statements correctly characterize the local closed-world assumption in KG training with negative sampling?

(A) Any unobserved triple is treated as false for training purposes.

(B) It is strictly correct because KGs are exhaustive.

(C) It helps training but may mislabel genuinely missing positives as negatives.

(D) It eliminates the need for development/test splits.

**Answer:** Any unobserved triple is treated as false for training purposes., It helps training but may mislabel genuinely missing positives as negatives.

---

**Q3 (1 point) - MCQ**

For discriminative training, why is it infeasible to enforce all constraints f(s, r, o) ≥  m + f(s′,r,o′) over every possible negative triple?

(A) The number of possible facts is O(E2R), overwhelmingly larger than positives.

(B) Because scores cannot be compared across relations.

(C) Because margins must be tuned per entity.

(D) Because negatives are always ambiguous.

**Answer:** The number of possible facts is O(E2R), overwhelmingly larger than positives.

---

**Q4 (1 point) - MCQ**

Which statement best describes score polarity in KG models?

(A) Scores must always be larger for false triples.

(B) Score polarity is fixed by the dataset.

(C) Some models use higher scores for more plausible triples, others use lower, and probabilities/losses can be adapted accordingly.

(D) Polarity only matters for RotatE

**Answer:** Some models use higher scores for more plausible triples, others use lower, and probabilities/losses can be adapted accordingly.

---

**Q5 (1 point) - MCQ**

Compared to semantic interpretation (logical-form execution), a differentiable KGQA system:

(A) Requires a hand-coded logical form for every question.

(B) Cannot be trained end-to-end.

(C) Provides complete interpretability of reasoning steps.

(D) Learns dense question and graph embeddings and uses cross-attention to align them.

**Answer:** Learns dense question and graph embeddings and uses cross-attention to align them.

---

**Q6 (1 point) - MSQ**

Which statements correctly describe filtered evaluation?

(A) It removes candidates that are true facts in train/dev from the ranked list before scoring the test query.

(B) It increases fairness by not penalizing the model for ranking another correct answer that happened to be in training data.

(C) It always decreases MRR.

(D) It affects measures like MRR and MAP.

**Answer:** It removes candidates that are true facts in train/dev from the ranked list before scoring the test query., It increases fairness by not penalizing the model for ranking another correct answer that happened to be in training data., It affects measures like MRR and MAP.

---

**Q7 (1 point) - MCQ**

Which of the following best captures the motivation for KG completion?

(A) KGs are complete, KG completion mainly compresses them.

(B) Manual curation keeps KGs fully up-to-date.

(C) KGs are useful but incomplete, so we learn embeddings and a scoring function to infer missing facts.

(D) KG completion is only for alignment across languages.

**Answer:** KGs are useful but incomplete, so we learn embeddings and a scoring function to infer missing facts.

---

**Q8 (1 point) - MCQ**

Consider pairwise hinge/ReLU loss for discriminative training with margin m: max{ 0, m + f(s’k , r, o’k) − f(s, r, o) }. When does this loss become exactly zero for a given negative (s’k , r, o’k)?

(A) When f(s, r, o) ≥ m + f(s’k , r, o’k)

(B) When f(s, r, o) = f(s’k , r, o’k)

(C) When f(s’k , r, o’k) ≥ m + f(s, r, o)

(D) Only when m = 0

**Answer:** When f(s, r, o) ≥ m + f(s’k , r, o’k)

---

**Q9 (1 point) - MCQ**

Uniform negative sampling can introduce an extra bias unless you do which of the following when forming the sampled denominator?

(A) Exclude the true object o from the denominator.

(B) Normalize scores per relation type.

(C) Sample only from entities not connected to s.

(D) Always include the true object o in the denominator.

**Answer:** Always include the true object o in the denominator.

---

**Q10 (1 point) - MCQ**

Which of the following is the RotatE scoring function?

(A) f(s, r, o) = ‖s+r−o‖²

(B) f(s, r, o) = ‖s⊙r−o‖², where r lies on the unit circle element-wise

(C) f(s, r, o) = sᵀRᵣo with Rᵣ orthonormal

(D) f(s, r, o) = −⟨s, r, o⟩

**Answer:** f(s, r, o) = ‖s⊙r−o‖², where r lies on the unit circle element-wise

---

### Week 10 : Assignment 10

**Q1 (1 point) - MCQ**

How do Prefix Tuning and Adapters differ in terms of where they inject new task-specific parameters in the Transformer architecture?

(A) Prefix Tuning adds new feed-forward networks after every attention block, while Adapters prepend tokens.

(B) Both approaches modify only the final output layer but in different ways.

(C) Prefix Tuning learns trainable “prefix” hidden states at each layer’s input, whereas Adapters insert small bottleneck modules inside the Transformer blocks.

(D) Both approaches rely entirely on attention masks to inject new task-specific knowledge.

**Answer:** Prefix Tuning learns trainable “prefix” hidden states at each layer’s input, whereas Adapters insert small bottleneck modules inside the Transformer blocks.

---

**Q2 (1 point) - MCQ**

The Structure-Aware Intrinsic Dimension (SAID) improves over earlier low-rank adaptation approaches by:

(A) Ignoring the network structure entirely

(B) Learning one scalar per layer for layer-wise scaling

(C) Sharing the same random matrix across all layers

(D) Using adapters within self-attention layers

**Answer:** Learning one scalar per layer for layer-wise scaling

---

**Q3 (1 point) - MSQ**

Which of the following are correct about the extensions of LoRA?

(A) LongLoRA supports inference on longer sequences using global attention

(B) QLoRA supports low-rank adaptation on 4-bit quantized models

(C) DyLoRA automatically selects the optimal rank during training

(D) LoRA+ introduces gradient clipping to stabilize training

**Answer:** QLoRA supports low-rank adaptation on 4-bit quantized models, DyLoRA automatically selects the optimal rank during training

---

**Q4 (1 point) - MCQ**

Which pruning technique specifically removes weights with the smallest absolute values first, potentially followed by retraining to recover accuracy?

(A) Magnitude Pruning

(B) Structured Pruning

(C) Random Pruning

(D) Knowledge Distillation

**Answer:** Magnitude Pruning

---

**Q5 (1 point) - MCQ**

In Post-Training Quantization (PTQ) for LLMs, why is a calibration dataset used?

(A) To precompute the entire attention matrix for all tokens.

(B) To remove outlier dimensions before applying magnitude-based pruning.

(C) To fine-tune the entire model on a small dataset and store the new weights.

(D) To estimate scale factors for quantizing weights and activations under representative data conditions.

**Answer:** To estimate scale factors for quantizing weights and activations under representative data conditions.

---

**Q6 (1 point) - MCQ**

Which best summarizes the function of the unembedding matrix WU?

(A) It merges the queries and keys for each token before final classification.

(B) It converts the final residual vector into vocabulary logits for next-token prediction.

(C) It is used for normalizing the QK and OV circuits so that their norms match.

(D) It acts as a second attention layer that aggregates multiple heads

**Answer:** It converts the final residual vector into vocabulary logits for next-token prediction.

---

**Q7 (1 point) - MCQ**

Which definition best matches an induction head as discovered in certain Transformer circuits?

(A) A head that specifically attends to punctuation tokens to determine sentence boundaries

(B) A feed-forward sub-layer specialized for outputting next-token probabilities for out-of-distribution tokens

(C) A head that looks for previous occurrences of a token A, retrieves the token B that followed it last time, and then predicts B again

(D) A masking head that prevents the model from looking ahead at future tokens

**Answer:** A head that looks for previous occurrences of a token A, retrieves the token B that followed it last time, and then predicts B again

---

**Q8 (1 point) - MCQ**

In mechanistic interpretability, how can we define ‘circuit’?

(A) A data pipeline for collecting training examples in an autoregressive model

(B) A small LSTM module inserted into a Transformer for additional memory

(C) A device external to the neural network used to fine-tune certain parameters after training

(D) A subgraph of the neural network hypothesized to implement a specific function or behaviour

**Answer:** A subgraph of the neural network hypothesized to implement a specific function or behaviour

---

**Q9 (1 point) - MCQ**

Which best describes the role of Double Quantization in QLoRA?

(A) It quantizes the attention weights twice to achieve 1-bit representations.

(B) It reinitializes parts of the model with random bit patterns for improved regularization.

(C) It quantizes the quantization constants themselves for additional memory savings.

(D) It systematically reverts partial quantized weights back to FP16 whenever performance degrades.

**Answer:** It quantizes the quantization constants themselves for additional memory savings.

---

**Q10 (1 point) - MSQ**

Which of the following are true about sequence-level distillation for LLMs?

(A) It trains a student model by matching the teacher’s sequence outputs (e.g., predicted token sequences) rather than just individual token distributions.

(B) It requires storing only the top-1 predictions from the teacher model for each token.

(C) It can be combined with word-level distillation to transfer both local and global knowledge.

(D) It forces the teacher to produce a chain-of-thought explanation for each example.

**Answer:** It trains a student model by matching the teacher’s sequence outputs (e.g., predicted token sequences) rather than just individual token distributions., It can be combined with word-level distillation to transfer both local and global knowledge.

---

### Week 11 : Assignment 11

**Q1 (1 point) - MCQ**

Assume that you build a document–term matrix M (rows: documents; columns: words) and take its thin SVD M = U Σ Vᵀ. Which statement is most accurate for interpreting V in classical Latent Semantic Analysis (LSA)?

(A) Columns of V (and rows of Vᵀ) give low-dimensional word representations that capture co-occurrence similarity.

(B) V gives only document embeddings, words are in U.

(C) V and U are not orthonormal in LSA.

(D) Σ can be ignored without affecting similarity.

**Answer:** Columns of V (and rows of Vᵀ) give low-dimensional word representations that capture co-occurrence similarity.

---

**Q2 (1 point) - MSQ**

Which statements correctly characterize the basic DistMult approach for knowledge graph completion?

(A) Each relation 𝑟 is parameterized by a full D×D matrix that can capture asymmetric relations.

(B) The relation embedding is a diagonal matrix, leading to a multiplicative interaction of entity embeddings.

(C) DistMult struggles with non-symmetric relations because score(s, r, o) = asT Mr ao is inherently symmetric in s and o.

(D) DistMult’s performance is typically tested only on fully symmetric KGs.

**Answer:** The relation embedding is a diagonal matrix, leading to a multiplicative interaction of entity embeddings., DistMult struggles with non-symmetric relations because score(s, r, o) = asT Mr ao is inherently symmetric in s and o.

---

**Q3 (1 point) - MSQ**

Given a doc–term matrix M, what do MᵀM and MMᵀ capture?

(A) MᵀM: word–word co-occurrence similarity across documents

(B) MMᵀ: document–document similarity via shared terms

(C) Both are identity matrices by construction

(D) MᵀM counts how often a word appears in the corpus total

**Answer:** MᵀM: word–word co-occurrence similarity across documents, MMᵀ: document–document similarity via shared terms

---

**Q4 (1 point) - MCQ**

Which best describes the main advantage of using a factorized representation (e.g., DistMult, ComplEx) for large KGs?

(A) It enforces that every relation in the KG be perfectly symmetric.

(B) It ensures each entity is stored as a one-hot vector, simplifying nearest-neighbour queries.

(C) It collapses the entire KG into a single scalar value.

(D) It significantly reduces parameters and enables generalization to unseen triples by capturing low-rank structure.

**Answer:** It significantly reduces parameters and enables generalization to unseen triples by capturing low-rank structure.

---

**Q5 (1 point) - MCQ**

Which statement best describes the reshaping of a 3D KG tensor X ∈R|E|×|R|×|E| into a matrix factorization problem?

(A) One axis remains for subject, one axis remains for object, and relations are combined into a single expanded axis.

(B) The subject dimension is repeated to match the relation dimension, resulting in a 2D matrix.

(C) Each subject–relation pair is collapsed into a single dimension, while objects remain as separate entries.

(D) The entire KG is vectorized into a 1D array and then factorized with an SVD approach.

**Answer:** Each subject–relation pair is collapsed into a single dimension, while objects remain as separate entries.

---

**Q6 (1 point) - MCQ**

SimplE addresses asymmetry by:

(A) Using separate subject and object embeddings per entity and including inverse relations, with an averaged score over the two directions

(B) Constraining relation vectors to unit modulus

(C) Replacing dot-products by max-pooling

(D) Removing inverse relations entirely

**Answer:** Using separate subject and object embeddings per entity and including inverse relations, with an averaged score over the two directions

---

**Q7 (1 point) - MSQ**

Which of the following statements correctly describe hyperbolic (Poincare) embeddings for hierarchical data?

(A) They map nodes onto a disk (or ball) such that large branching factors can be represented with lower distortion than in Euclidean space.

(B) Distance grows slowly near the center and becomes infinite near the boundary, making it naturally suited for tree-like structures.

(C) They require each node to be embedded on the surface of the Poincare disk of radius 1.

(D) They can achieve arbitrarily low distortion embeddings for trees with the same dimension as Euclidean space.

**Answer:** They map nodes onto a disk (or ball) such that large branching factors can be represented with lower distortion than in Euclidean space., Distance grows slowly near the center and becomes infinite near the boundary, making it naturally suited for tree-like structures.

---

**Q8 (1 point) - MSQ**

Why might a partial-order-based approach (like order embeddings) be beneficial for modelling ‘is-a’ relationships compared to purely distance-based approaches?

(A) They explicitly encode the ancestor–descendant relation as a coordinate-wise inequality or containment.

(B) They can represent negative correlations (i.e., sibling vs. ancestor) more easily than distance metrics.

(C) They inherently guarantee transitive closure of the hierarchy in the learned embedding space.

(D) They do not rely on pairwise distances but use a notion of coordinate-wise ordering or interval containment.

**Answer:** They explicitly encode the ancestor–descendant relation as a coordinate-wise inequality or containment., They do not rely on pairwise distances but use a notion of coordinate-wise ordering or interval containment.

---

**Q9 (1 point) - MCQ**

Which statement about box embeddings in hierarchical modelling is most accurate?

(A) Each entity or type is assigned a single real-valued vector, ignoring bounding volumes.

(B) Containment Ix ⊆ Iy all dimensions encodes x≺y .

(C) They rely on spherical distances around a central node to measure tree depth.

(D) They cannot be used to represent set intersections or partial overlap.

**Answer:** Containment Ix ⊆ Iy all dimensions encodes x≺y .

---

**Q10 (1 point) - MSQ**

For order embeddings with axis-aligned open cones:

(A) Represent each item x by apex ux ; encode x ≺ y as ux ≥ uy (element-wise).

(B) Positive loss encourages all dimensions to satisfy the order ; negative loss enforces at least one dimension to violate it.

(C) All cones (and their intersections) have the same measure in this construction.

(D) This makes modeling negative correlation between sibling types difficult.

**Answer:** Represent each item x by apex ux ; encode x ≺ y as ux ≥ uy (element-wise)., Positive loss encourages all dimensions to satisfy the order ; negative loss enforces at least one dimension to violate it.

---

### Week 12 : Assignment 12

**Q1 (1 point) - MCQ**

Which statements correctly characterize “bias” in the context of LLMs?1. Bias can generate objectionable or stereotypical views in model outputs.2. Bias is always intentionally introduced by malicious data curators.3. Bias can cause harmful real-world impacts such as reinforcing discrimination.4. Bias only affects low-resource languages; high-resource languages are unaffected.

(A) 1 and 2

(B) 1 and 3

(C) 2 and 4

(D) 1, 3, and 4

**Answer:** 1 and 3

---

**Q2 (1 point) - MCQ**

The Stereotype Score (ss) refers to:

(A) The frequency with which a language model rejects biased associations.

(B) The measure of how often a model’s predictions are meaningless as opposed to meaningful.

(C) A ratio of positive sentiment to negative sentiment in model outputs.

(D) The proportion of examples in which a model chooses a stereotypical association over an anti-stereotypical one.

**Answer:** The proportion of examples in which a model chooses a stereotypical association over an anti-stereotypical one.

---

**Q3 (1 point) - MCQ**

Which of the following are prominent sources of bias in LLMs? 1. Improper selection of training data leading to skewed distributions.2. Reliance on older datasets causing “temporal bias.”3. Overemphasis on low-resource languages causing “linguistic inversion.”4. Unequal focus on high-resource languages resulting in “cultural bias.”

(A) 1 and 2 only

(B) 2 and 3 only

(C) 1, 2, and 4

(D) 1, 3, and 4

**Answer:** 1, 2, and 4

---

**Q4 (1 point) - MCQ**

In the context of bias mitigation based on adversarial triggers, which best describes the goal of prepending specially chosen tokens to prompts?

(A) To directly fine-tune the model parameters to remove bias

(B) To override all prior knowledge in a model, effectively “resetting” it

(C) To exploit the model’s distributional patterns, thereby neutralizing or flipping biased associations in generated text

(D) To randomly shuffle the tokens so that the model becomes more robust

**Answer:** To exploit the model’s distributional patterns, thereby neutralizing or flipping biased associations in generated text

---

**Q5 (1 point) - MCQ**

Which of the following best describes the “regard” metric?

(A) It is a measure of how well a model can explain its internal decision process.

(B) It is a measurement of a model’s perplexity on demographically sensitive text.

(C) It is the proportion of times a model self-corrects discriminatory language.

(D) It is a classification label reflecting the attitude towards a demographic group in the generated text.

**Answer:** It is a classification label reflecting the attitude towards a demographic group in the generated text.

---

**Q6 (1 point) - MSQ**

Which of the following steps compose the approach for improving response safety via in-context learning?

(A) Retrieving safety demonstrations similar to the user query.

(B) Fine-tuning the model with additional labeled data after generation.

(C) Providing retrieved demonstrations as examples in the prompt to guide the model’s response generation.

(D) Sampling multiple outputs from LLMs and choosing the majority opinion.

**Answer:** Retrieving safety demonstrations similar to the user query., Providing retrieved demonstrations as examples in the prompt to guide the model’s response generation.

---

**Q7 (1 point) - MSQ**

Which statement(s) is/are correct about how high-resource (HRL) vs. low-resource languages (LRL) affect model training?

(A) LRLs typically have higher performance metrics due to smaller population sizes.

(B) HRLs get more data, so the model might overfit to HRL cultural perspectives.

(C) LRLs are often under-represented, leading to potential underestimation of their cultural nuances.

(D) The dominance of HRLs can cause a reinforcing cycle that perpetuates imbalance.

**Answer:** HRLs get more data, so the model might overfit to HRL cultural perspectives., LRLs are often under-represented, leading to potential underestimation of their cultural nuances., The dominance of HRLs can cause a reinforcing cycle that perpetuates imbalance.

---

**Q8 (1 point) - MCQ**

The “Responsible LLM” concept is stated to address:

(A) Only the bias in LLMs

(B) A set of concerns including explainability, fairness, robustness, and security

(C) Balancing training costs with carbon footprint

(D) Implementation of purely rule-based safety filters

**Answer:** A set of concerns including explainability, fairness, robustness, and security

---

**Q9 (1 point) - MCQ**

Within the StereoSet framework, the icat metric specifically refers to:

(A) The ratio of anti-stereotypical associations to neutral associations

(B) The percentage of times a model refuses to generate content deemed hateful

(C) A measure of domain coverage across different demographic groups

(D) A balanced metric capturing both a model’s language modelling ability and the tendency to avoid stereotypical bias

**Answer:** A balanced metric capturing both a model’s language modelling ability and the tendency to avoid stereotypical bias

---

**Q10 (1 point) - MCQ**

Bias due to improper selection of training data typically arises in LLMs when:

(A) Data are selected exclusively from curated, balanced sources with equal representation

(B) The language model sees only real-time social media feeds without any historical texts

(C) The training corpus over-represents some topics or groups, creating a skewed distribution

(D) All data are automatically filtered to remove any demographic markers

**Answer:** The training corpus over-represents some topics or groups, creating a skewed distribution

---

**Total Questions in Current Year:** 112


## Previous Year Assignments

### Week 1 : Assignment 1

**Q1 (1 point) - MCQ**

Based on Distributional Semantics, which of the following statements is/are true? (i) The meaning of a word is defined by its relationship to other words. (ii) The meaning of a word does not rely on its surrounding context.

(A) Both (i) and (ii) are correct

(B) Only (i) is correct

(C) Only (ii) is correct

(D) Neither (i) nor (ii) is correct

**Answer:** Neither (i) nor (ii) is correct

---

**Q2 (1 point) - MSQ**

Which of the following words have multiple senses?

(A) light

(B) order

(C) letter

(D) buffalo

**Answer:** light, order, letter, buffalo

---

**Q3 (1 point) - MCQ**

Consider the following sentences:Sentence 1: Amit forgot to set an alarm last night. Sentence 2: Amit woke up late today.Does Sentence 1 entail Sentence 2?

(A) True

(B) False

**Answer:** False

---

**Q4 (1 point) - MSQ**

What issues can be observed in the following text?On a much-needed #workcation in beautiful Goa. Workin & chillin by d waves!

(A) Idioms

(B) Non-standard English

(C) Tricky Entity Names

(D) Neologisms

**Answer:** Non-standard English, Neologisms

---

**Q5 (1 point) - MCQ**

Consider the following sentences:Sentence 1: The bats flew out of the cave at sunset.Sentence 2: Rohan bought a new bat to practice cricket.Question: Does the word "bat" have the same meaning in both sentences?

(A) Yes

(B) No

**Answer:** No

---

**Q6 (1 point) - MSQ**

Which of the following statements is/are true?

(A) Apple is a hypernym of fruit

(B) Leaf is a meronym of tree

(C) Flower is a holonym of petal.

(D) Parrot is a hyponym of bird.

**Answer:** Leaf is a meronym of tree, Flower is a holonym of petal., Parrot is a hyponym of bird.

---

**Q7 (1 point) - MCQ**

_________ deals with word formation and internal structure of words.

(A) Pragmatics

(B) Discourse

(C) Semantics

(D) Morphology

**Answer:** Morphology

---

**Q8 (1 point) - MSQ**

Consider the following sentences:Sentence 1: Priya told Meera that she had completed the report on time.Sentence 2: Meera was impressed by her dedication.Which of the following statements is/are true?

(A) In Sentence 1, "she" refers to Meera.

(B) In Sentence 1, "she" refers to Priya.

(C) In Sentence 2, "her" refers to Priya.

(D) In Sentence 2, "her" refers to Meera.

**Answer:** In Sentence 1, "she" refers to Priya., In Sentence 2, "her" refers to Priya.

---

**Q9 (1 point) - MCQ**

In semantic role labeling, we determine the semantic role of each argument with respect to the___________ of the sentence.

(A) noun phrase

(B) subject

(C) predicate

(D) adjunct

**Answer:** predicate

---

**Q10 (1 point) - MCQ**

Which of the following statements is/are true?(i) Artificial Intelligence (AI) is a sub-field of Machine Learning.(ii) LLMs are deep neural networks for processing text.(iii) Generative AI (GenAI) involves only Large Language Models (LLMs)

(A) Only (i) and (ii) are correct

(B) Only (ii) is correct

(C) Only (ii) and (iii) are correct

(D) All of (i), (ii), and (iii) are correct

(E) Neither (i), (ii), or (iii) is correct

**Answer:** Only (ii) is correct

---

### Week 2 : Assignment 2

**Q1 (1 point) - MCQ**

A 5-gram model is a ___________ order Markov Model.

(A) Constant

(B) Five

(C) Six

(D) Four

**Answer:** Four

---

**Q2 (1 point) - MCQ**

For a given corpus, the count of occurrence of the unigram "stay" is 300. If the Maximum Likelihood Estimation (MLE) for the bigram "stay curious" is 0.4, what is the count of occurrence of the bigram "stay curious"?

(A) 123

(B) 300

(C) 750

(D) 120

**Answer:** 120

---

**Q3 (1 point) - MSQ**

Which of the following are governing principles for Probabilistic Language Models?

(A) Chain Rule of Probability

(B) Markov Assumption

(C) Fourier Transform

(D) Gradient Descent

**Answer:** Chain Rule of Probability, Markov Assumption

---

**Q4 (2 points) - MCQ**

For Question 4 to 5, consider the following corpus: the sunset is nice people watch the sunset they enjoy the beautiful sunset Assuming a bi-gram language model, calculate the probability of the sentence: people watch the beautiful sunset Ignore the unigram probability of P() in your calculation.

(A) 2/27

(B) 1/27

(C) 2/9

(D) 1/8

**Answer:** 2/27

---

**Q5 (2 points) - MCQ**

Assuming a bi-gram language model, calculate the perplexity of the sentence: people watch the beautiful sunset Do not consider and  in the count of words of the sentence.

(A) 27$^{1/4}$

(B) (27/2)$^{1/4}$

(C) (27/2)$^{1/5}$

(D) 27$^{1/5}$

**Answer:** (27/2)$^{1/5}$

---

**Q6 (1 point) - MCQ**

What is the main intuition behind Kneser-Ney smoothing?

(A) Assign higher probability to frequent words.

(B) Use continuation probability to better model words appearing in a novel context

(C) Normalize probabilities by word length.

(D) Minimize perplexity for unseen words

**Answer:** Use continuation probability to better model words appearing in a novel context

---

**Q7 (1 point) - MCQ**

In perplexity-based evaluation of a language model, what does a lower perplexity score indicate?

(A) Worse model performance

(B) Better language model performance

(C) Increased vocabulary size

(D) More sparse data

**Answer:** Better language model performance

---

**Q8 (1 point) - MSQ**

Which of the following is a limitation of statistical language models like n-grams?

(A) Fixed context size

(B) High memory requirements for large vocabularies

(C) Difficulty in generalizing to unseen data

(D) All of the above

**Answer:** Fixed context size, High memory requirements for large vocabularies, Difficulty in generalizing to unseen data

---

### Week 3 : Assignment 3

**Q1 (1 point) - MCQ**

State whether the following statement is True/False. The Perceptron learning algorithm can solve problems with non-linearly separable data.

(A) True

(B) False

**Answer:** False

---

**Q2 (1 point) - MCQ**

In backpropagation, which method is used to compute the gradients?

(A) Gradient descent

(B) Chain rule of derivatives

(C) Matrix factorization

(D) Linear regression

**Answer:** Chain rule of derivatives

---

**Q3 (1 point) - MCQ**

Which activation function outputs values in the range [-1,1]?

(A) ReLU

(B) Tanh

(C) Sigmoid

(D) Linear

**Answer:** Tanh

---

**Q4 (1 point) - MCQ**

What is the primary goal of regularization in machine learning?

(A) To improve the computational efficiency of the model

(B) To reduce overfitting

(C) To increase the number of layers in a network

(D) To minimize the loss function directly

**Answer:** To reduce overfitting

---

**Q5 (1 point) - MCQ**

Which of the following is a regularization technique where we randomly deactivate neurons during training?

(A) Early stopping

(B) L1 regularization

(C) Dropout

(D) Weight decay

**Answer:** Dropout

---

**Q6 (1 point) - MCQ**

Which activation function has the vanishing gradient problem for large positive or negative inputs?

(A) ReLU

(B) Sigmoid

(C) GELU

(D) Swish

**Answer:** Sigmoid

---

**Q7 (1 point) - MCQ**

Which activation function is defined as: $f(x)=x⋅\sigma(x)$, where $\sigma(x)$ is the sigmoid function?

(A) Swish

(B) ReLU

(C) GELU

(D) SwiGLU

**Answer:** Swish

---

**Q8 (1 point) - MCQ**

What does the backpropagation algorithm compute in a neural network?

(A) Loss function value at each epoch

(B) Gradients of the loss function with respect to weights of the network

(C) Activation values of the output layer

(D) Output of each neuron

**Answer:** Gradients of the loss function with respect to weights of the network

---

**Q9 (1 point) - MCQ**

Which type of regularization encourages sparsity in the weights?

(A) L1 regularization

(B) L2 regularization

(C) Dropout

(D) Early stopping

**Answer:** L1 regularization

---

**Q10 (1 point) - MCQ**

What is the main purpose of using hidden layers in an MLP?

(A) Helps to make the network bigger

(B) Enables us to handle linearly separable data

(C) Learn complex and nonlinear relationships in the data

(D) Minimize the computational complexity

**Answer:** Learn complex and nonlinear relationships in the data

---

### Week 4 : Assignment 4

**Q1 (1 point) - MCQ**

What is the main drawback of representing words as one-hot vectors?

(A) They cannot capture semantic similarity between words.

(B) They are computationally inefficient.

(C) They cannot incorporate word order effectively.

(D) They are not robust to unseen words.

**Answer:** They cannot capture semantic similarity between words.

---

**Q2 (1 point) - MCQ**

What is the key concept underlying Word2Vec?

(A) Ontological semantics

(B) Decompositional semantics

(C) Distributional semantics

(D) Morphological analysis

**Answer:** Distributional semantics

---

**Q3 (1 point) - MCQ**

Why is sub-sampling frequent words beneficial in Word2Vec?

(A) It increases the computational cost.

(B) It helps reduce the noise from high-frequency words.

(C) It helps eliminate redundancy.

(D) It prevents the model from learning embeddings for common words.

**Answer:** It helps reduce the noise from high-frequency words.

---

**Q4 (1 point) - MSQ**

Which word relations cannot be captured by word2vec?

(A) Polysemy

(B) Antonymy

(C) Analogy

(D) All of the these

**Answer:** Polysemy, Antonymy

---

**Q5 (1 point) - MCQ**

For Question 5 to 6, Consider the following word-word matrix: (The matrix from the PDF is simplified for readability as rows/columns of the matrix. Note: The correct answer is **0.641** for this question, which is not in the list of options from the PDF. I will use the accepted answer from the source.)$W_1$: [1, 5, 3, 0, 1, 5, 7]$W_2$: [4, 2, 4, 1, 6, 2, 0]$W_3$: [2, 7, 9, 2, 5, 1, 8]$W_4$: [5, 0, 7, 4, 2, 0, 14]$W_5$: [0, 5, 1, 0, 1, 2, 4]Compute the cosine similarity between $W_2$ and $W_5$.

(A) 0.516

(B) 0.881

(C) 0.705

(D) 0.541

**Answer:** 0.641

---

**Q6 (4 points) - MCQ**

Which word is most similar to $W_4$ based on cosine similarity?

(A) $W_1$

(B) $W_2$

(C) $W_3$

(D) $W_5$

**Answer:** $W_3$

---

**Q7 (1 point) - MCQ**

What is the difference between CBOW and Skip-Gram in Word2Vec?

(A) CBOW predicts the context word given the target word, while Skip-Gram predicts the target word given the context words.

(B) CBOW predicts the target word given the context words, while Skip-Gram predicts the context words given the target word.

(C) CBOW is used for generating word vectors, while Skip-Gram is not.

(D) Skip-Gram uses a thesaurus, while CBOW does not.

**Answer:** CBOW predicts the target word given the context words, while Skip-Gram predicts the context words given the target word.

---

### Week 5 : Assignment 5

**Q1 (1 point) - MCQ**

Which of the following is a disadvantage of Recurrent Neural Networks (RNNs)?

(A) Can only process fixed-length inputs.

(B) Symmetry in how inputs are processed.

(C) Difficulty accessing information from many steps back.

(D) Weights are not reused across timesteps.

**Answer:** Difficulty accessing information from many steps back.

---

**Q2 (1 point) - MCQ**

Why are RNNs preferred over fixed-window neural models?

(A) They have a smaller parameter size.

(B) They can process sequences of arbitrary length.

(C) They eliminate the need for embedding layers.

(D) None of the above.

**Answer:** They can process sequences of arbitrary length.

---

**Q3 (1 point) - MCQ**

What is the primary purpose of the cell state in an LSTM?

(A) Store short-term information.

(B) Control the gradient flow across timesteps.

(C) Store long-term information.

(D) Perform the activation function.

**Answer:** Store long-term information.

---

**Q4 (1 point) - MCQ**

In training an RNN, what technique is used to calculate gradients over multiple timesteps?

(A) Backpropagation through Time (BPTT)

(B) Stochastic Gradient Descent (SGD)

(C) Dropout Regularization

(D) Layer Normalization

**Answer:** Backpropagation through Time (BPTT)

---

**Q5 (2 points) - MCQ**

Consider a simple RNN: Input vector size: 3 Hidden state size: 4 Output vector size: 2 Number of timesteps: 5 How many parameters are there in total, including the bias terms?

(A) 210

(B) 2190

(C) 90

(D) 42

**Answer:** 42

---

**Q6 (1 point) - MCQ**

What is the time complexity for processing a sequence of length 'N' by an RNN, if the input embedding dimension, hidden state dimension, and output vector dimension are all 'd'?

(A) $O(N)$

(B) $O(N d^2)$

(C) $O(N d)$

(D) $O(N^2 d)$

**Answer:** $O(N d^2)$

---

**Q7 (1 point) - MSQ**

Which of the following is true about Seq2Seq models?(i) Seq2Seq models are always conditioned on the source sentence. (ii) The encoder compresses the input sequence into a fixed-size vector representation. (iii) Seq2Seq models cannot handle variable-length sequences.

(A) (i) and (ii)

(B) (i) only

(C) (ii) only

(D) (i), (ii), and (iii)

**Answer:** (i) and (ii)

---

**Q8 (2 points) - MCQ**

Given the following encoder and decoder hidden states, compute the attention scores. (Use dot product as the scoring function)Encoder hidden states: $h_{1}=[1,2]$, $h_{2}=[3,4]$, $h_{3}=[5,6]$Decoder hidden state: $s=[0.5,1]$

(A) 0.00235, 0.04731, 0.9503

(B) 0.0737, 0.287, 0.6393

(C) 0.9503, 0.0137, 0.036

(D) 0.6393, 0.0737, 0.287

**Answer:** 0.00235, 0.04731, 0.9503

---

### Week 6 : Assignment 6

**Q1 (1 point) - MCQ**

What is the key advantage of multi-head attention?

(A) It uses a single attention score for the entire sequence

(B) It allows attending to different parts of the input sequence simultaneously

(C) It eliminates the need for normalization

(D) It reduces the model size

**Answer:** It allows attending to different parts of the input sequence simultaneously

---

**Q2 (1 point) - MCQ**

What is the role of the residual connection in the Transformer architecture?

(A) Improve gradient flow during backpropagation

(B) Normalize input embeddings

(C) Reduce computational complexity

(D) Prevent overfitting

**Answer:** Improve gradient flow during backpropagation

---

**Q3 (1 point) - MCQ**

Which of the following elements addresses the lack of sequence information in self-attention?

(A) Non-linear transformations

(B) Positional encoding

(C) Masked decoding

(D) Residual connections

**Answer:** Positional encoding

---

**Q4 (1 point) - MSQ**

For Rotary Position Embedding (RoPE), which of the following statements are true?

(A) Combines relative and absolute positional information

(B) Applies a multiplicative rotation matrix to encode positions

(C) Eliminates the need for positional encodings

(D) All of the above

**Answer:** Combines relative and absolute positional information, Applies a multiplicative rotation matrix to encode positions

---

**Q5 (2 points) - MCQ**

Consider a sequence of tokens of length 4: $[w_1, w_2, w_3, w_4]$. Using masked self-attention, compute the attention weights for token $w_3$, assuming the unmasked attention scores are: $[5, 2, 1, 3]$

(A) [0.6234, 0.023, 0.3424, 0.0112]

(B) [0.2957, 0.7043, 0, 0]

(C) [0.9362, 0.0466, 0.0171, 0]

(D) [0.5061, 0.437, 0, 0.0569]

**Answer:** [0.9362, 0.0466, 0.0171, 0]

---

**Q6 (1 point) - MCQ**

_________ maps the values of a feature in the range [0, 1].

(A) Standardization

(B) Normalization

(C) Transformation

(D) Scaling

**Answer:** Normalization

---

**Q7 (1 point) - MCQ**

How does masked self-attention help in autoregressive models?

(A) By attending to all tokens, including future ones

(B) By focusing only on past tokens to prevent information leakage

(C) By ignoring positional information in the sequence.

(D) By disabling the attention mechanism entirely.

**Answer:** By focusing only on past tokens to prevent information leakage.

---

**Q8 (2 points) - MCQ**

For a transformer with $d_{model}=512$, calculate the positional encoding for position $p=10$ and dimensions 2 and 3 using the sinusoidal formula: $PE(p, 2i) = \sin(\frac{p}{10000^{2i/d_{model}}}) ; PE(p, 2i+1) = \cos(\frac{p}{10000^{2i/d_{model}}})$

(A) $\sin(\frac{10}{10000^{1/256}})$, $\cos(\frac{10}{10000^{1/256}})$

(B) $\cos(\frac{10}{10000^{1/512}})$, $\sin(\frac{10}{10000^{1/512}})$

(C) $\cos(\frac{10}{10000^{4/512}})$, $\sin(\frac{10}{10000^{7/256}})$

(D) $\sin(\frac{10}{10000^{2/512}})$, $\cos(\frac{10}{10000^{3/512}})$

**Answer:** $\sin(\frac{10}{10000^{1/256}})$, $\cos(\frac{10}{10000^{1/256}})$

---

### Week 7 : Assignment 7

**Q1 (1 point) - MCQ**

Which of the following best describes how ELMo's architecture captures different linguistic properties?

(A) The model explicitly assigns specific linguistic functions to each layer.

(B) The lower layers capture syntactic information, while higher layers capture semantic information.

(C) All layers capture the similar properties.

(D) ELMO uses a fixed, non-trainable weighting scheme for combining layer-wise representations.

**Answer:** The lower layers capture syntactic information, while higher layers capture semantic information.

---

**Q2 (1 point) - MCQ**

BERT and BART models differ in their architectures. While BERT is (i) model, BART is (ii) one. Select the correct choices for (i) and (ii).

(A) i: Decoder-only, ii: Encoder-only

(B) i: Encoder-decoder, ii: Encoder-only

(C) i: Encoder-only, ii: Encoder-decoder

(D) i: Decoder-only, ii: Encoder-decoder

**Answer:** i: Encoder-only, ii: Encoder-decoder

---

**Q3 (1 point) - MCQ**

The pre-training objective for the T5 model is based on:

(A) Next sentence prediction

(B) Masked language modelling

(C) Span corruption and reconstruction

(D) Predicting the next token

**Answer:** Span corruption and reconstruction and reconstruction

---

**Q4 (1 point) - MCQ**

Which of the following datasets was used to pretrain the T5 model?

(A) Wikipedia

(B) Book Corpus

(C) Common Crawl

(D) C4

**Answer:** C4

---

**Q5 (1 point) - MSQ**

Which of the following special tokens are introduced in BERT to handle sentence pairs?

(A) [MASK] and [CLS]

(B) [SEP] and [CLS]

(C) [CLS] and [NEXT]

(D) [SEP] and [MASK]

**Answer:** [SEP] and [CLS]

---

**Q6 (2 points) - MSQ**

ELMo and BERT represent two different pre-training strategies for language models. Which of the following statement(s) about these approaches is/are true?

(A) ELMo uses a bi-directional LSTM to pre-train word representations, while BERT uses a transformer encoder with masked language modeling.

(B) ELMo provides context-independent word representations, whereas BERT provides context-dependent representations

(C) Pre-training of both ELMo and BERT involve next token prediction.

(D) Both ELMo and BERT produce word embeddings that can be fine-tuned for downstream tasks.

**Answer:** ELMo uses a bi-directional LSTM to pre-train word representations, while BERT uses a transformer encoder with masked language modeling., Both ELMo and BERT produce word embeddings that can be fine-tuned for downstream tasks.

---

**Q7 (1 point) - MCQ**

Decoder-only models are essentially trained based on probabilistic language modelling. Which of the following correctly represents the training objective of GPT-style models?

(A) $P(x|y)$ where $x$ is the input sequence and $y$ is the gold output sequence

(B) $P(x:y)$ where $x$ is the input sequence and $y$ is the gold output sequence

(C) $P(w_{t}|W_{1:t-1})$ where $w_t$ represents the token at position $t$, and $W_{1:t-1}$ is the sequence of tokens from position 1 to $t-1$

(D) $P(w_{t}|W_{1:t+1})$ where $w_t$ represents the token at position $t$, and $W_{1:t+1}$ is the sequence of tokens from position 1 to $t+1$

**Answer:** $P(w_{t}|W_{1:t-1})$ where $w_t$ represents the token at position $t$, and $W_{1:t-1}$ is the sequence of tokens from position 1 to $t-1$

---

**Q8 (2 points) - NUMERIC**

In the previous week, we saw the usage of einsum function in numpy as a generalized operation for performing tensor multiplications. Now, consider two matrices: $A=\begin{bmatrix}1&5\\ 3&7\end{bmatrix}$ and $B=\begin{bmatrix}2&-1\\ 4&2\end{bmatrix}$ Then, what is the output of the following numpy operation? `numpy.einsum('ij,ji->', A, B)`

**Answer:** 29

---

### Week 8 : Assignment 8

**Q1 (1 point) - MSQ**

Which factors influence the effectiveness of instruction tuning?

(A) The number of instruction templates used in training.

(B) The tokenization algorithm used by the model.

(C) The diversity of tasks in the fine-tuning dataset.

(D) The order in which tasks are presented during fine-tuning.

**Answer:** The number of instruction templates used in training., The diversity of tasks in the fine-tuning dataset., The order in which tasks are presented during fine-tuning.

---

**Q2 (1 point) - MSQ**

What are key challenges of soft prompts in prompt-based learning?

(A) Forward pass with them is computationally inefficient compared to that with hard prompts.

(B) They require additional training, unlike discrete prompts.

(C) They cannot be interpreted or used effectively by non-expert users.

(D) They require specialized architectures that differ from standard transformers.

**Answer:** They require additional training, unlike discrete prompts., They cannot be interpreted or used effectively by non-expert users.

---

**Q3 (1 point) - MCQ**

Which statement best describes the impact of fine-tuning versus prompting in LLMs?

(A) Fine-tuning is always superior to prompting in generalization tasks.

(B) Prompting requires gradient updates, while fine-tuning does not.

(C) Fine-tuning modifies the model weights permanently, while prompting does not

(D) Prompting performs better on in-domain tasks compared to fine-tuning.

**Answer:** Fine-tuning modifies the model weights permanently, while prompting does not

---

**Q4 (1 point) - MSQ**

Which of the following aspects of the model outputs are captured by POSIX?

(A) Diversity in the responses to intent-preserving prompt variations

(B) Entropy of the distribution of response frequencies

(C) Time required to generate responses for intent-preserving prompt variations

(D) Variance in the log-likelihood of the same response for different input prompt variations

**Answer:** Diversity in the responses to intent-preserving prompt variations, Entropy of the distribution of response frequencies, Variance in the log-likelihood of the same response for different input prompt variations

---

**Q5 (1 point) - MCQ**

Which key mechanism makes Tree-of-Thought (ToT) prompting more effective than Chain-of-Thought (CoT)?

(A) ToT uses reinforcement learning for better generalization.

(B) ToT allows backtracking to explore multiple reasoning paths.

(C) ToT reduces hallucination by using domain-specific heuristics

(D) ToT eliminates the need for manual prompt engineering.

**Answer:** ToT allows backtracking to explore multiple reasoning paths

---

**Q6 (1 point) - MCQ**

What is a key limitation of measuring accuracy alone when evaluating LLMs?

(A) Accuracy is always correlated with model size.

(B) Accuracy cannot be measured on open-ended tasks.

(C) Accuracy is independent of the training dataset size.

(D) Accuracy does not account for prompt sensitivity.

**Answer:** Accuracy does not account for prompt sensitivity.

---

**Q7 (1 point) - MCQ**

Why is instruction tuning not sufficient for aligning large language models?

(A) It does not generalize to unseen tasks.

(B) It cannot prevent models from generating undesired responses.

(C) It reduces model performance on downstream tasks.

(D) It makes models less capable of learning from new data.

**Answer:** It cannot prevent models from generating undesired responses.

---

**Q8 (1 point) - MCQ**

Why is KL divergence minimized in regularized reward maximization?

(A) To maximize the probability of generating high-reward responses.

(B) To make training more computationally efficient.

(C) To prevent the amplification of bias in training data.

(D) To ensure models do not diverge too far from the reference model.

**Answer:** To ensure models do not diverge too far from the reference model.

---

**Q9 (1 point) - MCQ**

What is the primary advantage of using the log-derivative trick in REINFORCE?

(A) Reducing data requirements

(B) Expanding the token vocabulary

(C) Simplifying gradient computation

(D) Improving sampling diversity

**Answer:** Simplifying gradient computation

---

**Q10 (1 point) - MCQ**

Which method combines reward maximization and minimizing KL divergence?

(A) REINFORCE

(B) Monte Carlo Approximation

(C) Proximal Policy Optimization

(D) Constitutional AI

**Answer:** Proximal Policy Optimization

---

### Week 9 : Assignment 9

**Q1 (1 point) - MCQ**

Which of the following statement best describes why knowledge graphs (KGs) are considered more powerful than a traditional relational knowledge base (KB)?

(A) KGs require no schema, whereas KBs must have strict schemas.

(B) KGs store data only in the form of hypergraphs, eliminating redundancy.

(C) KGs allow flexible, graph-based connections and typed edges, enabling richer relationships and inferences compared to KBs.

(D) KGs completely replace the need for textual sources by storing all possible facts.

**Answer:** KGs allow flexible, graph-based connections and typed edges, enabling richer relationships and inferences compared to KBs.

---

**Q2 (1 point) - MSQ**

Entity alignment and relation alignment are crucial between KGs of different languages. Which of the following factors contribute to effective alignment?

(A) Aligning relations solely by their lexical similarity, ignoring semantic context

(B) Transliteration or language-based string matching for entity labels

(C) Ensuring all language aliases are represented identically in each KG

(D) Matching neighbours, or connected entities, across different KGs

**Answer:** Transliteration or language-based string matching for entity labels, Matching neighbours, or connected entities, across different KGs

---

**Q3 (1 point) - MCQ**

In the context of knowledge graph completion (KGC), which statement best describes the role of the scoring function $f(s,r,o)$?

(A) It determines whether two entities refer to the same real-world concept.

(B) It produces a raw confidence score indicating how plausible a triple $(s,r,o)$ is

(C) It explicitly encodes only the subject's embedding, ignoring the relation and object embeddings

(D) It ensures that every negative triple gets a higher score than any positive triple.

**Answer:** It produces a raw confidence score indicating how plausible a triple $(s,r,o)$ is

---

**Q4 (1 point) - MCQ**

One key difference between the differentiable KG approach and the semantic interpretation approach to KGQA is:

(A) Differentiable KG approaches are fully rule-based, while semantic interpretation is purely neural.

(B) Differentiable KG approaches do not require any graph embeddings, relying instead on explicit logical forms.

(C) Semantic interpretation is more transparent or interpretable, whereas differentiable KG is end-to-end trainable but less interpretable.

(D) Both approaches use logical forms, the primary difference is the type of question they can answer.

**Answer:** Semantic interpretation is more transparent or interpretable, whereas differentiable KG is end-to-end trainable but less interpretable.

---

**Q5 (1 point) - MSQ**

Considering the differentiable KG approach, which elements are typically learned jointly when training an end-to-end KGQA model?

(A) The textual question representation (e.g., BERT embeddings)

(B) The graph structure encoding (e.g., GCN or transformer-based graph embeddings)

(C) Predefined logical forms to ensure interpretability

(D) The final answer selection mechanism that identifies which node(s) in the graph satisfy the question

**Answer:** The textual question representation (e.g., BERT embeddings), The graph structure encoding (e.g., GCN or transformer-based graph embeddings), The final answer selection mechanism that identifies which node(s) in the graph satisfy the question

---

**Q6 (1 point) - MCQ**

Uniform negative sampling can have high variance and may require large number of samples. Why is that the case?

(A) Because the margin-based loss cannot converge without big mini-batches.

(B) Because randomly picking negative entities does not guarantee close or challenging negatives, causing unstable training estimates.

(C) Because negative sampling must ensure every possible negative triple is covered.

(D) Because the number of relations in the KG is too large for small number of samples.

**Answer:** Because randomly picking negative entities does not guarantee close or challenging negatives, causing unstable training estimates.

---

**Q7 (1 point) - MCQ**

In testing embedding and score quality for KG completion, mean rank and hits@K are typical metrics. What does hits@K specifically measure in this context?

(A) The percentage of queries for which the correct answer appears in the top-K of the ranked list.

(B) The reciprocal of the rank of the correct answer.

(C) The probability of the correct answer appearing as the highest scored candidate

(D) The margin of the correct triple score relative to all negative triples

**Answer:** The percentage of queries for which the correct answer appears in the top-K of the ranked list.

---

**Q8 (1 point) - MCQ**

In the TransE model, the scoring function for a triple $(s,r,o)$ is typically defined as $f(s,r,o) = \|e_s + e_r - e_o\|_1$ or $\|e_s + e_r - e_o\|_2^2$, where $e_s, e_r, e_o$ are embeddings of the subject, relation, and object, respectively. Which statement best explains what a low value of $f(s,r,o)$ indicates in this context?

(A) That $(s,r,o)$ is an invalid triple according to the learned embeddings.

(B) That $e_s$ and $e_o$ must be orthogonal.

(C) That the relation embedding $e_r$ is zero.

(D) That $(s,r,o)$ has a high likelihood of being a true fact in the knowledge graph.

**Answer:** That $(s,r,o)$ has a high likelihood of being a true fact in the knowledge graph.

---

**Q9 (1 point) - MCQ**

In RotatE, if a relation $r$ is intended to be symmetric, how would that typically manifest in the complex plane?

(A) The relation embedding $e_r$ must always equal zero.

(B) The angle of $e_r$ must be $\pi/2$

(C) The relation embedding $e_r$ is its own inverse (i.e., a $180^\circ$ rotation when squared).

(D) The magnitude of $e_r$ must be greater than 1.

**Answer:** The relation embedding $e_r$ is its own inverse (i.e., a $180^\circ$ rotation when squared).

---

**Q10 (1 point) - MCQ**

Which main advantage do rotation-based models (like RotatE) have over translation-based ones (like TransE) when it comes to complex multi-relational patterns in a KG?

(A) Rotation-based models cannot model any symmetry or inverse patterns, so they are simpler.

(B) Rotation-based models handle a broader set of relation properties (symmetry, anti-symmetry, inverses, composition) more naturally.

(C) Rotation-based models have no hyperparameters to tune, unlike TransE.

(D) Rotation-based models are guaranteed to yield perfect link prediction.

**Answer:** Rotation-based models handle a broader set of relation properties (symmetry, anti-symmetry, inverses, composition) more naturally.

---

### Week 10 : Assignment 10

**Q1 (1 point) - MCQ**

How do Prefix Tuning and Adapters differ in terms of where they inject new task-specific parameters in the Transformer architecture?

(A) Prefix Tuning adds new feed-forward networks after every attention block, while Adapters prepend tokens.

(B) Both approaches modify only the final output layer but in different ways.

(C) Prefix Tuning learns trainable "prefix" hidden states at each layer's input, whereas Adapters insert small bottleneck modules inside the Transformer blocks.

(D) Both approaches rely entirely on attention masks to inject new task-specific knowledge.

**Answer:** Prefix Tuning learns trainable "prefix" hidden states at each layer's input, whereas Adapters insert small bottleneck modules inside the Transformer blocks.

---

**Q2 (1 point) - MCQ**

The Structure-Aware Intrinsic Dimension (SAID) improves over earlier low-rank adaptation approaches by:

(A) Ignoring the network structure entirely

(B) Learning one scalar per layer for layer-wise scaling

(C) Sharing the same random matrix across all layers

(D) Using adapters within self-attention layers

**Answer:** Learning one scalar per layer for layer-wise scaling

---

**Q3 (1 point) - MSQ**

Which of the following are correct about the extensions of LoRA?

(A) LongLoRA supports inference on longer sequences using global attention

(B) QLoRA supports low-rank adaptation on 4-bit quantized models

(C) DyLoRA automatically selects the optimal rank during training

(D) LoRA+ introduces gradient clipping to stabilize training

**Answer:** QLoRA supports low-rank adaptation on 4-bit quantized models, DyLoRA automatically selects the optimal rank during training

---

**Q4 (1 point) - MCQ**

Which pruning technique specifically removes weights with the smallest absolute values first, potentially followed by retraining to recover accuracy?

(A) Magnitude Pruning

(B) Structured Pruning

(C) Random Pruning

(D) Knowledge Distillation

**Answer:** Magnitude Pruning

---

**Q5 (1 point) - MCQ**

In Post-Training Quantization (PTQ) for LLMs, why is a calibration dataset used?

(A) To precompute the entire attention matrix for all tokens.

(B) To remove outlier dimensions before applying magnitude-based pruning.

(C) To fine-tune the entire model on a small dataset and store the new weights.

(D) To estimate scale factors for quantizing weights and activations under representative data conditions.

**Answer:** To estimate scale factors for quantizing weights and activations under representative data conditions.

---

**Q6 (1 point) - MCQ**

Which best summarizes the function of the unembedding matrix $W_U$?

(A) It merges the queries and keys for each token before final classification.

(B) It converts the final residual vector into vocabulary logits for next-token prediction.

(C) It is used for normalizing the QK and OV circuits so that their norms match.

(D) It acts as a second attention layer that aggregates multiple heads.

**Answer:** It converts the final residual vector into vocabulary logits for next-token prediction.

---

**Q7 (1 point) - MCQ**

Which definition best matches an induction head as discovered in certain Transformer circuits?

(A) A head that specifically attends to punctuation tokens to determine sentence boundaries

(B) A feed-forward sub-layer specialized for outputting next-token probabilities for out-of-distribution tokens

(C) A head that looks for previous occurrences of a token A, retrieves the token B that followed it last time, and then predicts B again

(D) A masking head that prevents the model from looking ahead at future tokens

**Answer:** A head that looks for previous occurrences of a token A, retrieves the token B that followed it last time, and then predicts B again

---

**Q8 (1 point) - MCQ**

In mechanistic interpretability, how can we define “circuit”?

(A) A data pipeline for collecting training examples in an autoregressive model

(B) A small LSTM module inserted into a Transformer for additional memory

(C) A device external to the neural network used to fine-tune certain parameters after training

(D) A subgraph of the neural network hypothesized to implement a specific function or behaviour

**Answer:** A subgraph of the neural network hypothesized to implement a specific function or behaviour

---

**Q9 (1 point) - MCQ**

Which best describes the role of Double Quantization in QLoRA?

(A) It quantizes the attention weights twice to achieve 1-bit representations.

(B) It reinitializes parts of the model with random bit patterns for improved regularization.

(C) It quantizes the quantization constants themselves for additional memory savings.

(D) It systematically reverts partial quantized weights back to FP16 whenever performance degrades.

**Answer:** It quantizes the quantization constants themselves for additional memory savings.

---

**Q10 (1 point) - MSQ**

Which of the following are true about sequence-level distillation for LLMs?

(A) It trains a student model by matching the teacher's sequence outputs (e.g., predicted token sequences) rather than just individual token distributions.

(B) It requires storing only the top-1 predictions from the teacher model for each token.

(C) It can be combined with word-level distillation to transfer both local and global knowledge.

(D) It forces the teacher to produce a chain-of-thought explanation for each example.

**Answer:** It trains a student model by matching the teacher's sequence outputs (e.g., predicted token sequences) rather than just individual token distributions., It can be combined with word-level distillation to transfer both local and global knowledge.

---

### Week 11 : Assignment 11

**Q1 (1 point) - MCQ**

What is the main modification that SimplE makes to DistMult-like models to handle asymmetric relations?

(A) Replacing entity embeddings with random fixed vectors

(B) Introducing separate entity embeddings for subject and object roles, along with inverse relations

(C) Restricting the rank of the relation tensor to 1

(D) Using negative sampling for half of the triple set

**Answer:** Introducing separate entity embeddings for subject and object roles, along with inverse relations

---

**Q2 (1 point) - MSQ**

Which statements correctly characterize the basic DistMult approach for knowledge graph completion?

(A) Each relation $r$ is parameterized by a full $D \times D$ matrix that can capture asymmetric relations.

(B) The relation embedding is a diagonal matrix, leading to a multiplicative interaction of entity embeddings.

(C) DistMult struggles with non-symmetric relations because $score(s,r,o) = a_s^T M_r a_o$ is inherently symmetric in $s$ and $o$

(D) DistMult's performance is typically tested only on fully symmetric KGs.

**Answer:** The relation embedding is a diagonal matrix, leading to a multiplicative interaction of entity embeddings., DistMult struggles with non-symmetric relations because $score(s,r,o) = a_s^T M_r a_o$ is inherently symmetric in $s$ and $o$

---

**Q3 (1 point) - MSQ**

Which statements about the ComplEx extension of DistMult are true?

(A) It uses complex-valued embeddings to better capture asymmetric or anti-symmetric relations.

(B) It replaces the multiplication in DistMult with element-wise addition of real-valued vectors.

(C) For a perfectly symmetric relation, one could set the imaginary part of the relation embedding to zero.

(D) ComplEx requires each entity vector to be unit norm in the complex plane.

**Answer:** It uses complex-valued embeddings to better capture asymmetric or anti-symmetric relations., For a perfectly symmetric relation, one could set the imaginary part of the relation embedding to zero.

---

**Q4 (1 point) - MCQ**

Which best describes the main advantage of using a factorized representation (e.g., DistMult, ComplEx) for large KGs?

(A) It enforces that every relation in the KG be perfectly symmetric.

(B) It ensures each entity is stored as a one-hot vector, simplifying nearest-neighbour queries.

(C) It collapses the entire KG into a single scalar value.

(D) It significantly reduces parameters and enables generalization to unseen triples by capturing low-rank structure.

**Answer:** It significantly reduces parameters and enables generalization to unseen triples by capturing low-rank structure.

---

**Q5 (1 point) - MCQ**

Which statement best describes the reshaping of a 3D KG tensor $X \in R^{|E| \times |R| \times |E|}$ into a matrix factorization problem?

(A) One axis remains for subject, one axis remains for object, and relations are combined into a single expanded axis.

(B) The subject dimension is repeated to match the relation dimension, resulting in a 2D matrix.

(C) Each subject-relation pair is collapsed into a single dimension, while objects remain as separate entries.

(D) The entire KG is vectorized into a 1D array and then factorized with an SVD approach.

**Answer:** Each subject-relation pair is collapsed into a single dimension, while objects remain as separate entries.

---

**Q6 (1 point) - MCQ**

Which key property of hierarchical relationships (e.g., is-a, transitivity) motivates the exploration of specialized embedding methods over standard Euclidean KG embeddings?

(A) Symmetry in the relation (A, is-a, B) implying (B, is-a, A)

(B) Frequent presence of cycles in hierarchical graphs

(C) Transitivity in the form (camel, is-a, mammal) and (mammal, is-a, animal) $\implies$ (camel, is-a, animal)

(D) The high dimensionality of the entity embeddings

**Answer:** Transitivity in the form (camel, is-a, mammal) and (mammal, is-a, animal) $\implies$ (camel, is-a, animal)

---

**Q7 (1 point) - MSQ**

Which of the following statements correctly describe hyperbolic (Poincare) embeddings for hierarchical data?

(A) They map nodes onto a disk (or ball) such that large branching factors can be represented with lower distortion than in Euclidean space.

(B) Distance grows slowly near the center and becomes infinite near the boundary, making it naturally suited for tree-like structures.

(C) They require each node to be embedded on the surface of the Poincare disk of radius 1.

(D) They can achieve arbitrarily low distortion embeddings for trees with the same dimension as Euclidean space.

**Answer:** They map nodes onto a disk (or ball) such that large branching factors can be represented with lower distortion than in Euclidean space., Distance grows slowly near the center and becomes infinite near the boundary, making it naturally suited for tree-like structures.

---

**Q8 (1 point) - MSQ**

Why might a partial-order-based approach (like order embeddings) be beneficial for modelling 'is-a' relationships compared to purely distance-based approaches?

(A) They explicitly encode the ancestor-descendant relation as a coordinate-wise inequality or containment.

(B) They can represent negative correlations (i.e., sibling vs. ancestor) more easily than distance metrics.

(C) They inherently guarantee transitive closure of the hierarchy in the learned embedding space.

(D) They do not rely on pairwise distances but use a notion of coordinate-wise ordering or interval containment.

**Answer:** They explicitly encode the ancestor-descendant relation as a coordinate-wise inequality or containment., They do not rely on pairwise distances but use a notion of coordinate-wise ordering or interval containment.

---

**Q9 (1 point) - MCQ**

Which statement about box embeddings in hierarchical modelling is most accurate?

(A) Each entity or type is assigned a single real-valued vector, ignoring bounding volumes.

(B) Containment $I_x \subset I_y$ across all dimensions encodes $x \prec y$.

(C) They rely on spherical distances around a central node to measure tree depth.

(D) They cannot be used to represent set intersections or partial overlap.

**Answer:** Containment $I_x \subset I_y$ across all dimensions encodes $x \prec y$.

---

**Q10 (1 point) - MCQ**

What is a key challenge with axis-aligned open-cone (order) embeddings for hierarchical KG data?

(A) They enforce that all sibling categories have identical cone apices, which causes overlap.

(B) They require symmetrical relationships for all edges.

(C) They do not allow partial orders to be extended to total orders.

(D) The volume (measure) of cones is the same regardless of how "broad" or "narrow" the cone is, making sub-categories indistinguishable by volume.

**Answer:** The volume (measure) of cones is the same regardless of how "broad" or "narrow" the cone is, making sub-categories indistinguishable by volume.

---

### Week 12 : Assignment 12

**Q1 (1 point) - MCQ**

Which statements correctly characterize "bias" in the context of LLMs?1. Bias can generate objectionable or stereotypical views in model outputs.2. Bias is always intentionally introduced by malicious data curators.3. Bias can cause harmful real-world impacts such as reinforcing discrimination.4. Bias only affects low-resource languages; high-resource languages are unaffected.

(A) 1 and 2

(B) 1 and 3

(C) 2 and 4

(D) 1, 3, and 4

**Answer:** 1 and 3

---

**Q2 (1 point) - MCQ**

The Stereotype Score (ss) refers to:

(A) The frequency with which a language model rejects biased associations.

(B) The measure of how often a model's predictions are meaningless as opposed to meaningful.

(C) A ratio of positive sentiment to negative sentiment in model outputs.

(D) The proportion of examples in which a model chooses a stereotypical association over an anti-stereotypical one.

**Answer:** The proportion of examples in which a model chooses a stereotypical association over an anti-stereotypical one.

---

**Q3 (1 point) - MCQ**

Which of the following are prominent sources of bias in LLMs? 1. Improper selection of training data leading to skewed distributions.2. Reliance on older datasets causing "temporal bias."3. Overemphasis on low-resource languages causing "linguistic inversion."4. Unequal focus on high-resource languages resulting in "cultural bias."

(A) 1 and 2 only

(B) 2 and 3 only

(C) 1, 2, and 4

(D) 1, 3, and 4

**Answer:** 1, 2, and 4

---

**Q4 (1 point) - MCQ**

In the context of bias mitigation based on adversarial triggers, which best describes the goal of prepending specially chosen tokens to prompts?

(A) To directly fine-tune the model parameters to remove bias

(B) To override all prior knowledge in a model, effectively "resetting" it

(C) To exploit the model's distributional patterns, thereby neutralizing or flipping biased associations in generated text

(D) To randomly shuffle the tokens so that the model becomes more robust

**Answer:** To exploit the model's distributional patterns, thereby neutralizing or flipping biased associations in generated text

---

**Q5 (1 point) - MCQ**

Which of the following best describes the "regard" metric?

(A) It is a measure of how well a model can explain its internal decision process.

(B) It is a measurement of a model's perplexity on demographically sensitive text.

(C) It is the proportion of times a model self-corrects discriminatory language.

(D) It is a classification label reflecting the attitude towards a demographic group in the generated text

**Answer:** It is a classification label reflecting the attitude towards a demographic group in the generated text

---

**Q6 (1 point) - MSQ**

Which of the following steps compose the approach for improving response safety via in-context learning?

(A) Retrieving safety demonstrations similar to the user query.

(B) Fine-tuning the model with additional labeled data after generation.

(C) Providing retrieved demonstrations as examples in the prompt to guide the model's response generation.

(D) Sampling multiple outputs from LLMs and choosing the majority opinion.

**Answer:** Retrieving safety demonstrations similar to the user query., Providing retrieved demonstrations as examples in the prompt to guide the model's response generation.

---

**Q7 (1 point) - MSQ**

Which statement(s) is/are correct about how high-resource (HRL) vs. low-resource languages (LRL) affect model training?

(A) LRLs typically have higher performance metrics due to smaller population sizes.

(B) HRLs get more data, so the model might overfit to HRL cultural perspectives.

(C) LRLs are often under-represented, leading to potential underestimation of their cultural nuances.

(D) The dominance of HRLs can cause a reinforcing cycle that perpetuates imbalance.

**Answer:** HRLs get more data, so the model might overfit to HRL cultural perspectives., LRLs are often under-represented, leading to potential underestimation of their cultural nuances., The dominance of HRLs can cause a reinforcing cycle that perpetuates imbalance.

---

**Q8 (1 point) - MCQ**

The "Responsible LLM" concept is stated to address:

(A) Only the bias in LLMs

(B) A set of concerns including explainability, fairness, robustness, and security

(C) Balancing training costs with carbon footprint

(D) Implementation of purely rule-based safety filters

**Answer:** A set of concerns including explainability, fairness, robustness, and security

---

**Q9 (1 point) - MCQ**

Within the StereoSet framework, the icat metric specifically refers to:

(A) The ratio of anti-stereotypical associations to neutral associations

(B) The percentage of times a model refuses to generate content deemed hateful

(C) A measure of domain coverage across different demographic groups

(D) A balanced metric capturing both a model's language modelling ability and the tendency to avoid stereotypical bias

**Answer:** A balanced metric capturing both a model's language modelling ability and the tendency to avoid stereotypical bias

---

**Q10 (1 point) - MCQ**

Bias due to improper selection of training data typically arises in LLMs when:

(A) Data are selected exclusively from curated, balanced sources with equal representation

(B) The language model sees only real-time social media feeds without any historical texts

(C) The training corpus over-represents some topics or groups, creating a skewed distribution

(D) All data are automatically filtered to remove any demographic markers

**Answer:** The training corpus over-represents some topics or groups, creating a skewed distribution

---

**Total Questions in Previous Year:** 109

