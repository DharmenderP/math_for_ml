import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32 # how many independent sequences qill we process in parallel
block_size = 8 # what is maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_itrs = 200
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

#super simple bigram model
class BiagramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off logists for the next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) #(B, T, C) batch time channel 4, 8, 65
        
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
            # get the predictions
            logits, _ = self(idx) # here target is None, hence logits have shape B, T, C
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