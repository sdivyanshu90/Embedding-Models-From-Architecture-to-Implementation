# %% [markdown]
# Source notebook: Notebook_4.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L5: Inference

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
# Warning control
import warnings
warnings.filterwarnings('ignore')

# %%
# Notebook cell 4
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, \
                         DPRContextEncoder, DPRQuestionEncoder

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> file:</b> To access <code>requirements.txt</code> for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %%
# Notebook cell 6
def cosine_similarity_matrix(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / norms
    similarity_matrix = np.inner(normalized_features, normalized_features)
    rounded_similarity_matrix = np.round(similarity_matrix, 4)
    return rounded_similarity_matrix

# %% [markdown]
# ## Pure similarity

# %%
# Notebook cell 8
answers = [
    "What is the tallest mountain in the world?",
    "The tallest mountain in the world is Mount Everest.",
    "Mount Shasta",
    "I like my hike in the mountains",
    "I am going to a yoga class"
]

question = 'What is the tallest mountain in the world?'

# %%
# Notebook cell 9
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
question_embedding = list(model.encode(question))

sim = []
for answer in answers:
    answer_embedding = list(model.encode(answer))
    sim.append(cosine_similarity_matrix(np.stack([question_embedding, answer_embedding]))[0,1])

print(sim)
best_inx = np.argmax(sim)
print(f"Question = {question}")
print(f"Best answer = {answers[best_inx]}")

# %% [markdown]
# ## Dual-Encoder inference

# %%
# Notebook cell 11
answer_tokenizer = AutoTokenizer \
                   .from_pretrained("./models/facebook/dpr-ctx_encoder-multiset-base")
answer_encoder = DPRContextEncoder \
                   .from_pretrained("./models/facebook/dpr-ctx_encoder-multiset-base")

question_tokenizer = AutoTokenizer \
                   .from_pretrained("./models/facebook/dpr-question_encoder-multiset-base")
question_encoder = DPRQuestionEncoder \
                   .from_pretrained("./models/facebook/dpr-question_encoder-multiset-base")

# %%
# Notebook cell 12
# Compute the question embeddings
question_tokens = question_tokenizer(question, return_tensors="pt")["input_ids"]
question_embedding = question_encoder(question_tokens).pooler_output.flatten().tolist()
print(question_embedding[:10], len(question_embedding))

# %%
# Notebook cell 13
sim = []
for answer in answers:
    answer_tokens = answer_tokenizer(answer, return_tensors="pt")["input_ids"]
    answer_embedding = answer_encoder(answer_tokens).pooler_output.flatten().tolist() 
    sim.append(cosine_similarity_matrix(np.stack([question_embedding, answer_embedding]))[0,1])

print(sim)
best_inx = np.argmax(sim)
print(f"Question = {question}")
print(f"Best answer = {answers[best_inx]}")

# %%
# Notebook cell 14


# %%
# Notebook cell 15


# %%
# Notebook cell 16


# %%
# Notebook cell 17


# %%
# Notebook cell 18

