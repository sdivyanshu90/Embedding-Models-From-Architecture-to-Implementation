# %% [markdown]
# Source notebook: Notebook_3.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L4: Training a Dual Encoder

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
# Warning control
import warnings
warnings.filterwarnings('ignore')

# %%
# Notebook cell 4
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> file:</b> To access <code>requirements.txt</code> for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %% [markdown]
# ## The CrossEntropyLoss 'trick'

# %%
# Notebook cell 7
df = pd.DataFrame(
    [
        [4.3, 1.2, 0.05, 1.07],
        [0.18, 3.2, 0.09, 0.05],
        [0.85, 0.27, 2.2, 1.03],
        [0.23, 0.57, 0.12, 5.1]
    ]
)
data = torch.tensor(df.values, dtype=torch.float32)

# %%
# Notebook cell 8
def contrastive_loss(data):
    target = torch.arange(data.size(0))
    loss = torch.nn.CrossEntropyLoss()(data, target)
    return loss

# %%
# Notebook cell 9
torch.nn.Softmax(dim=1)(data)

# %%
# Notebook cell 10
torch.nn.Softmax(dim=1)(data).sum(dim=1)

# %%
# Notebook cell 11
N = data.size(0)
non_diag_mask = ~torch.eye(N, N, dtype=bool)

for inx in range(3):
    data = torch.tensor(df.values, dtype=torch.float32)
    data[range(N), range(N)] += inx*0.5
    data[non_diag_mask] -= inx*0.02
    print(data)
    print(f"Loss = {contrastive_loss(data)}")

# %% [markdown]
# ## The Encoder module. 

# %%
# Notebook cell 13
class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, output_embed_dim):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=3,
            norm=torch.nn.LayerNorm([embed_dim]),
            enable_nested_tensor=False
        )
        self.projection = torch.nn.Linear(embed_dim, output_embed_dim)

    def forward(self, tokenizer_output):
        x = self.embedding_layer(tokenizer_output['input_ids'])
        x = self.encoder(x, src_key_padding_mask=tokenizer_output['attention_mask'].logical_not())
        cls_embed = x[:,0,:]
        return self.projection(cls_embed)

# %% [markdown]
# ![Diagram 1](DLAI-diagram-2.png)

# %% [markdown]
# ## Training Loop

# %%
# Notebook cell 16
def train_loop(dataset):
    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64
    batch_size = 32

    # define the question/answer encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question_encoder = Encoder(tokenizer.vocab_size, embed_size, 
                               output_embed_size)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size, 
                             output_embed_size)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=True)    
    optimizer = torch.optim.Adam(
        list(question_encoder.parameters()) + list(answer_encoder.parameters()
    ), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    running_loss = []
    for _, data_batch in enumerate(dataloader):

        # Tokenize the question/answer pairs (each is a batch of 32 questions and 32 answers)
        question, answer = data_batch
        question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
        answer_tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)

        # Compute the embeddings: the output is of dim = 32 x 128
        question_embed = question_encoder(question_tok)
        answer_embed = answer_encoder(answer_tok)

        # Compute similarity scores: a 32x32 matrix
        # row[N] reflects similarity between question[N] and answers[0...31]
        similarity_scores = question_embed @ answer_embed.T

        # we want to maximize the values in the diagonal
        target = torch.arange(question_embed.shape[0], dtype=torch.long)
        loss = loss_fn(similarity_scores, target)
        running_loss += [loss.item()]

        # this is where the magic happens
        optimizer.zero_grad()    # reset optimizer so gradients are all-zero
        loss.backward()
        optimizer.step()

    return question_encoder, answer_encoder

# %% [markdown]
# ## Training in multiple Epochs

# %%
# Notebook cell 18
def train(dataset, num_epochs=100):
    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64
    batch_size = 32

    n_iters = len(dataset) // batch_size + 1

    # define the question/answer encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)

    # define the dataloader, optimizer and loss function    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)    
    optimizer = torch.optim.Adam(list(question_encoder.parameters()) + list(answer_encoder.parameters()), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = []
        for idx, data_batch in enumerate(dataloader):

            # Tokenize the question/answer pairs (each is a batc of 32 questions and 32 answers)
            question, answer = data_batch
            question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            answer_tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            if inx == 0 and epoch == 0:
                print(question_tok['input_ids'].shape, answer_tok['input_ids'].shape)

            # Compute the embeddings: the output is of dim = 32 x 128
            question_embed = question_encoder(question_tok)
            answer_embed = answer_encoder(answer_tok)
            if inx == 0 and epoch == 0:
                print(question_embed.shape, answer_embed.shape)

            # Compute similarity scores: a 32x32 matrix
            # row[N] reflects similarity between question[N] and answers[0...31]
            similarity_scores = question_embed @ answer_embed.T
            if inx == 0 and epoch == 0:
                print(similarity_scores.shape)

            # we want to maximize the values in the diagonal
            target = torch.arange(question_embed.shape[0], dtype=torch.long)
            loss = loss_fn(similarity_scores, target)
            running_loss += [loss.item()]
            if idx == n_iters-1:
                print(f"Epoch {epoch}, loss = ", np.mean(running_loss))

            # this is where the magic happens
            optimizer.zero_grad()    # reset optimizer so gradients are all-zero
            loss.backward()
            optimizer.step()

    return question_encoder, answer_encoder

# %% [markdown]
# ## Let's train

# %%
# Notebook cell 20
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath, sep="\t", nrows=300)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]['questions'], self.data.iloc[idx]['answers']

dataset = MyDataset('./shared_data/nq_sample.tsv')
dataset.data.head(5)

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(num_epochs = 5)</code>:</b> The <code>num_epochs</code> is set to <code>5</code> for speedier execution. You may train the model 
# using a higher number of epochs by changing this parameter.</p>

# %%
# Notebook cell 22
qe, ae = train(dataset, num_epochs=5)

# %%
# Notebook cell 23
question = 'What is the tallest mountain in the world?'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=64)
question_emb = qe(question_tok)[0]
print(question_tok)
print(question_emb[:5])

# %%
# Notebook cell 24
answers = [
    "What is the tallest mountain in the world?",
    "The tallest mountain in the world is Mount Everest.",
    "Who is donald duck?"
]
answer_tok = []
answer_emb = []
for answer in answers:
    tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=64)
    answer_tok.append(tok['input_ids'])
    emb = ae(tok)[0]
    answer_emb.append(emb)

print(answer_tok)
print(answer_emb[0][:5])
print(answer_emb[1][:5])
print(answer_emb[2][:5])

# %%
# Notebook cell 25
question_emb @ torch.stack(answer_emb).T

# %%
# Notebook cell 26


# %%
# Notebook cell 27


# %%
# Notebook cell 28


# %%
# Notebook cell 29


# %%
# Notebook cell 30

