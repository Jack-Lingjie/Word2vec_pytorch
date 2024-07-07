from data_util import get_dataloader
from model import SkipGramModule
import torch.nn.functional as F
import torch
from torch import optim
import os
from myloger import logger
from tqdm import tqdm
import numpy as np
import json

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

embedding_size = 128
batch_size = 512
num_epoch = 20
context_size = 3
n_negatives = 5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_save_path = "./model_save/word2vec_reuters_model.pth"
embedding_save_path = "./results/result_embed.json"

 
dataset, dataloader, vocab = get_dataloader(batch_size, context_size, n_negatives)
vocab_size = len(vocab)

 
model = SkipGramModule(vocab_size, embedding_size).to(device)
optimizer = optim.Adam(model.parameters(),lr = 0.001)

def train(model, dataloader, optimizer, num_epoch, device):

    logger.info("Start Train")
    model.train()
    loss_list = []
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(dataloader,desc = f"Training Epoch {epoch}"):
            words, contexts, neg_contexts = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss = model(words, contexts, neg_contexts).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_list.append(total_loss)
        logger.info(f"loss:{total_loss:.2f}")
        torch.save(model.state_dict(), os.path.join(model_save_path))
    return loss_list
loss_list = train(model=model, dataloader=dataloader, optimizer=optimizer, num_epoch=num_epoch, device=device)
np.savetxt("train_loss.csv", np.array(loss_list), delimiter=',')

# save word embeddings
embedding_weights = model.get_weight()
word_dict = {word: embedding_weights[idx].tolist() for word, idx in vocab.token_to_idx.items()}

with open(embedding_save_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(word_dict, ensure_ascii=False, indent=4))
