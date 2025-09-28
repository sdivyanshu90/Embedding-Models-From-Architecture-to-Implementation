# Embedding Models: From Architecture to Implementation

*A short course by [DeepLearning.AI](https://www.deeplearning.ai/) in collaboration with [Vectara](https://vectara.com/)*
**Instructor:** Ofer Mendelevitch, Head of Developer Relations at Vectara

---

## 📖 Acknowledgment

This course, *[Embedding Models: From Architecture to Implementation](https://www.deeplearning.ai/short-courses/embedding-models-from-architecture-to-implementation/)*, is proudly created by **DeepLearning.AI** in collaboration with **Vectara**. Special thanks to **Ofer Mendelevitch**, who shares his expertise on embedding models and their role in modern AI applications.

---

## 📚 Course Topics & Detailed Explanations

### 1. Intro to Embedding Models

Embedding models are mathematical representations that capture the meaning of words, phrases, or sentences in a continuous vector space. Unlike symbolic representations, embeddings encode semantic similarity — meaning words or sentences with related meanings are mapped closer together.

* Early embeddings were based on co-occurrence statistics (e.g., Word2Vec, GloVe).
* Modern embeddings leverage deep learning and transformers, allowing them to represent context dynamically.
* Embeddings form the foundation for semantic search, recommendation systems, natural language understanding, and Retrieval-Augmented Generation (RAG).

---

### 2. Contextualized Token Embeddings

Traditional embeddings assign a single fixed vector to each word. However, the meaning of a word often depends on context. Contextualized embeddings address this by generating representations conditioned on surrounding words.

* Transformer-based models (like BERT) produce embeddings where the same token has different vectors depending on context.
* Example: the word *bank* in *“river bank”* vs. *“financial bank”* yields distinct embeddings.
* This contextualization allows models to understand nuances, disambiguate meanings, and improve performance in downstream tasks such as question answering and semantic search.

---

### 3. Token vs. Sentence Embedding

* **Token Embedding:** Represents individual words or subwords. Useful for fine-grained analysis and token-level tasks like Named Entity Recognition (NER).
* **Sentence Embedding:** Represents entire sentences or documents as single vectors. Useful for tasks that require comparing or ranking text units, such as semantic similarity, retrieval, or classification.
* Sentence embeddings are typically derived from aggregating contextual token embeddings (e.g., pooling techniques) or by training specialized models like Sentence-BERT.
* Choosing between token and sentence embeddings depends on whether the task is local (word-level) or global (sentence/document-level).

---

### 4. Training a Dual Encoder

A dual encoder is an architecture that uses two separate encoders to map inputs into a shared embedding space.

* **Design:** One encoder processes queries (e.g., questions), and the other processes responses (e.g., candidate answers).
* **Objective:** Minimize the distance between embeddings of matching pairs while maximizing the distance for mismatched pairs.
* **Contrastive Loss:** A common training strategy that ensures semantically aligned pairs are closer in vector space.
* **Advantage:** Efficient retrieval — once embeddings are computed, similarity search can be performed via nearest-neighbor lookup.
* This architecture is widely used in information retrieval, semantic search, and retrieval-augmented systems.

---

### 5. Using Embeddings in RAG

Retrieval-Augmented Generation (RAG) combines the power of embedding-based retrieval with generative models.

* **Process:**

  1. Encode queries into embeddings.
  2. Retrieve the most semantically relevant documents using similarity search.
  3. Pass the retrieved context to a language model for grounded generation.
* **Single Encoder vs. Dual Encoder:**

  * A single encoder model uses the same encoder for both query and document, but may not capture asymmetry between the two.
  * A dual encoder allows specialized encoders for queries and responses, often leading to better retrieval quality.
* Embeddings ensure that retrieved information is semantically relevant, enhancing the factual accuracy and usefulness of generated outputs.
