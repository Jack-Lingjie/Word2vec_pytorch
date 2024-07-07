from torch import nn
import torch
import torch.nn.functional as F
# 模型中w和c分别用不同的embedding，便于训练，最后会进行参数的合并
class SkipGramModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.w_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.c_embedding = nn.Embedding(vocab_size, embedding_dim)
    # def forward_w(self, words):
    #     w_embeds = self.w_embedding(words)
    #     return w_embeds
    # def forward_c(self, contexts):
    #     c_embeds = self.c_embedding(contexts)

    def forward(self, words, pos_contexts, neg_contexts):
        w_embeds = self.w_embedding(words) # (batch_size, embed_size)
        w_embeds = w_embeds.unsqueeze(2) #(batch_size, embed_size, 1)
        pos_embeds = self.c_embedding(pos_contexts) #(batch_size, context_size, embed_size)
        neg_embeds = self.c_embedding(neg_contexts) #(batch_size, context_size, embed_size)

        pos_loss = torch.bmm(pos_embeds, w_embeds).squeeze(2) #(batch_size, context_size)
        neg_loss = torch.bmm(neg_embeds, w_embeds).squeeze(2) #(batch_size, context_size)

        pos_loss = F.logsigmoid(pos_loss).mean(1)
        neg_loss = F.logsigmoid(-neg_loss).mean(1)

        loss = pos_loss + neg_loss
        return -loss
    
    def get_weight(self):
        return self.w_embedding.weight.data.cpu().numpy()




