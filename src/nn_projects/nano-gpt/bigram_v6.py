import torch
import torch.nn as nn
import torch.nn.functional as F

# v4 with multi head self attention, feedforward, and (transformer )repeated blocks
# nn becomes deeper , they suffers from optimizations
# so will use residual or skip connections
# hyperparameters
batch_size = 32 # how many independent sequences qill we process in parallel
block_size = 16 # what is maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # need low learning rate for self attention head
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_itrs = 200
n_embd = 100
n_layer = 6
n_head = 4
dropout = 0.2 # 20% of neurons would be dropped and model  trains without them
#-------------------
torch.manual_seed(1337)

with open("/input.txt", mode='r', encoding='utf-8') as f:
    text = f.read()

# here are all unique characters occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#create a mapping from characters to integers
stoi = {s: i for i, s in enumerate(chars)}
itos = {i : s for s, i in stoi.items()}
encode = lambda s : [stoi[c] for c in s] #encoder: take a sting, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

#train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n] # first 90% will be train, rest val
val_data = data[n:]

#data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_itrs)
        for k in range(eval_itrs):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BatchNorm1d: #nn.LayerNorm
    def __init__(self, dim, eps = 1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) #batch mean
        xvar = x.var(1, keepdim=True, unbaiased=True) #batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out


class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size) what i have
        q = self.query(x) #(B, T, head_size) what i am looking for

        #compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # scaled attention (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) ##(B, T, head_size) what will be communicated, actual raw value is wrapped around v
        self.out = wei @ v #(B ,T ,T) @ (B, T, C) ---> (B, T, C)
        return self.out

class MutliHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) #regularization method to avoid overfitting
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenate against dim = -1 or channel dim
        out = self.proj(out)
        self.out = self.dropout(out) # project to match skip connections
        return self.out
    

class Feedforward(nn.Module):
    """ a simple linear layer followed by a non-linearlity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
            )
        
    def forward(self, x):
        self.out = self.net(x)
        return self.out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of head we'd like        
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MutliHeadAttention(n_head, head_size)
        self.ffwd = Feedforward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # slight devaiation from original paper, 
        # we are applying normalization earlier to transformation
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x


#super simple bigram model
class BiagramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off logists for the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #encoding identity of these token
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #encoding also postion of these token [0, 8] would also get its own position embedding vector
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        # idx and targets are both (B, T) tensors of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #(B, T, C) batch time channel 4, 8, 65
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = tok_emb + pos_emb #work with broadcasting (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) #(B, T, vocab_size)
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) #-log

        return logits, loss
    
    #we look only at the last character, no history is used
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indicies in current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, _ = self(idx_cond) # here target is None, hence logits have shape B, T, C
            # focus only on the last time step

            logits = logits[:, -1, :] #becomes(B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) w.r.t last dim
            # sample from distributions
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        
        return idx
    

model = BiagramLanguageModel(vocab_size)
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(max_iters):

    # every once in a while evauluate the loss on train and val sets
    if steps % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate from  the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))