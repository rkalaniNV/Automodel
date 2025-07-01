import math
import torch
import torch.nn as nn

class GPTModel(torch.nn.Module):
    def __init__ (self, hidden_dim, vocab_size, num_layers, num_heads, qkv_dim, seq_len, mlp_dim):
        super().__init__()
        # For a linear operator from R^N to R^M, represented by an M-by-N matrix, we use "Kaiming uniform" initialization,
        # which initializes the matrix entries with i.i.d. samples from Uniform(-1/sqrt(N), 1/sqrt(N)).
        def init_linear (W, N):
            std = 1. / math.sqrt(N)
            W.data.uniform_(-std, std)
            return W

        def make_param(*size):
            return init_linear(nn.Parameter(torch.empty(*size)), size[-1])

        self.num_layers = num_layers
        self.register_parameter("W_E", make_param(hidden_dim, vocab_size))
        self.register_parameter("W_QKV", make_param(num_layers, 3, num_heads, qkv_dim, hidden_dim))
        self.register_parameter("W_O", make_param(num_layers, num_heads, hidden_dim, qkv_dim))
        self.register_parameter("W_FC1", make_param(num_layers, mlp_dim, hidden_dim))
        self.register_parameter("W_FC2", make_param(num_layers, hidden_dim, mlp_dim))
        self.register_parameter("W_U", make_param(vocab_size, hidden_dim))

        self.register_buffer("causal_mask", torch.triu(torch.ones(seq_len, seq_len)))

    def forward(self, input_ids, targets=None, position_ids=None, seq_lens=None):
        X = self.W_E[:, input_ids].permute(1, 2, 0)
        for l in range(self.num_layers):
            Q, K, V = torch.einsum("...dD,BSD->...BSd", self.W_QKV[l, ...], X).tensor_split(3, dim=0)
            Z = torch.einsum("...Td,...Sd->...TS", K, Q)
            Z = Z.masked_fill(self.causal_mask == 0, float('-1e20'))
            Z = torch.nn.functional.softmax(Z, dim=2)
            Z = torch.einsum("...Td,...TS->...Sd", V, Z)
            # Z = torch.einsum("HDd,HBSd->BSD...", self.W_O[l, ...], Z)
            Z = torch.einsum("HDd,...HBSd->BSD", self.W_O[l, ...], Z)
            X = X + Z
            Z = torch.einsum("QD,...D->...Q", self.W_FC1[l, ...], X)
            Z = torch.nn.functional.relu(Z).square()
            Z = torch.einsum("DQ,...Q->...D", self.W_FC2[l, ...], Z)
            X = X + Z
        X = torch.einsum("VD,...D->...V", self.W_U, X)
        # Recipe calculates the loss.
        return X
#        if targets is not None:
#            loss = torch.nn.functional.cross_entropy(X.view(-1, X.size(-1)), targets.view(-1))
#        else:
#            loss = None

#        return X, loss
