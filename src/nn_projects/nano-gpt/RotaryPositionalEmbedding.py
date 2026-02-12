import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, n_embd, max_seq=1024, base=10000):
        super().__init__()
        self.n_embd = n_embd
        self.max_seq = max_seq
        self.base = base

        theta = 1.0 / (base ** (torch.arange(0, n_embd, 2).float() / n_embd))
        # precompute cached positional embedding (cos and sin)
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, theta)
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1) # Concatenate to handle pairs of dimensions
        self.register_buffer('cache', cache)

    def forward(self, x, seq_len=None):
        B, n_head, T, head_size = x.shape
        if seq_len is None:
            seq_len = T
        
        # Ensure cached embeddings are long enough
        if seq_len > self.max_seq:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq}. "
                             "Consider increasing max_seq_len or recomputing frequencies.")
        cos_pos = self.cache[:seq_len, :, 0]#(seq_len, head_size//2)
        sin_pos = self.cache[:seq_len, :, 1] #(seq_len, head_size//2)

        #preparing for broadcasting
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(1) #(1, 1, seq_len, head_size//2)
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(1) #(1, 1, seq_len, head_size//2)

        #apply rope
        x = x.float().reshape(*x.shape[:-1], -1, 2) # (B, n_head, T, pairs, 2)
        out = torch.stack([
            x[..., 0] * cos_pos - x[..., 1] * sin_pos,
            x[..., 0] * sin_pos + x[..., 1] * cos_pos,
            ], dim=-1)
        
        return out

    
# Example Usage:
if __name__ == "__main__":
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64 # Must be even for RoPE

    # Create dummy query/key tensor
    query_key_tensor = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Initialize RoPE
    rope = RotaryPositionalEmbedding(n_embd=head_dim, max_seq=seq_len)

    # Apply RoPE
    rotated_tensor = rope(query_key_tensor)

    print("Original tensor shape:", query_key_tensor.shape)
    print("Rotated tensor shape:", rotated_tensor.shape)
    print("Example of rotated values (first element):", rotated_tensor[0, 0, 0, :4])





    

        





        