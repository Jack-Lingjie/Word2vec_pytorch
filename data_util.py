from collections import defaultdict
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from myloger import logger

BOS_TOKEN = "<BOS>" # 句首标记
EOS_TOKEN = "<EOS>" # 句尾标记
PAD_TOKEN = "<PAD>" # 填充标记
 
def get_dataloader(batch_size, context_size, n_negatives):
    """get dataset and dataloader"""
    corpus, vocab = load_reuters()
    unigram_dist = get_unigram_distribution(corpus, len(vocab))
    negative_sample_dist = unigram_dist ** 0.75
    negative_sample_dist /= negative_sample_dist.sum()
 
    dataset = SkipGramDataset(corpus, vocab, context_size = context_size, n_negatives = n_negatives, ns_dist = negative_sample_dist)
    dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = True, collate_fn = dataset.collate_fn)
    return dataset, dataloader, vocab
 

def load_reuters():
    """load reuters dataset"""
    logger.info("load reuters")
    from nltk.corpus import reuters
    text = reuters.sents()
    # 把所有词语进行小写处理，也是降低词表的一种方法
    text = [[token.lower() for token in sentence] for sentence in text]
    vocab = Vocab.build(text, reserved_tokens=[BOS_TOKEN,EOS_TOKEN,PAD_TOKEN])
    corpus = [vocab.covert_token_to_idx(sentence) for sentence in text]
    return corpus, vocab


# 编写训练语料中的每个词的出现频率
# unigram 一元分词，把句子分成一个一个的汉字
# bigram 二元分词，把句子从头到尾每两个字组成一个词语
# trigram 三元分词，把句子从头到尾每三个字组成一个词语.
def get_unigram_distribution(corpus,vocab_size):
    logger.info("get unigram distribution")
    token_count = torch.tensor([0]*vocab_size)
    total_count = 0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_count[token] += 1
    unigram_dist = torch.div(token_count.float(), total_count)
    return unigram_dist


# 模型输入（w,context,neg_context); 
class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size, n_negatives, ns_dist):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        for sentence in tqdm(corpus, desc = 'Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1,len(sentence)-1):
                w = sentence[i]
                # 确定上下文左右边界，不够的地方用pad填充
                left_index = max(0,i-context_size)
                right_index = min(len(sentence),i + context_size)
                context = sentence[left_index:i] + sentence[i+1:right_index+1]
                context += [self.pad] * (context_size * 2 - len(context))
                self.data.append((w, context))
        # 正样本和负样本的比例，比如对于一个w有正样本4（context）个,则负采样20(context*n_negative)个
        self.n_negatives = n_negatives
        # 负采样的分布
        self.ns_dist = ns_dist
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index]
    def collate_fn(self, batch_datas):
        words = torch.tensor([batch[0] for batch in batch_datas], dtype = torch.long)# （batch_size）
        contexts = torch.tensor([batch[1] for batch in batch_datas], dtype = torch.long)# (batch_size,context_size)
        batch_size, context_size = contexts.shape
        neg_contexts = []
        for i in range(batch_size):
            # 保证负样本中不包含当前样本中的context，index_fill的三个参数分别表示：
            # 第一个0表示在第一个维度进行填充，原本ns_dist也就是一维的
            # 第二个context[i]表示一个句子的所有词的词表下标
            # 第三个.0表示把第二个参数所有的词表下标对应的获取概率设为0.0
            ns_dist = self.ns_dist.index_fill(0,contexts[i],.0)
            # torch.multinomial,作用是按照给的概率随机提取数组的下标
            # 第一个参数是和目标数组等大的概率数组，里面可以是小数也可以是整数
            # 第二个参数是随机取下标的数量
            # 第三个参数是取得下标是否放回，就随机取的下标是否可以重复, True就是可以重复
            neg_contexts.append(torch.multinomial(ns_dist, context_size * self.n_negatives, replacement = True))
        # 把neg_contexts 沿着维度0重新组合起来
        neg_contexts = torch.stack(neg_contexts, dim = 0)# (batch_size,context_size * n_negatives)
        return words, contexts, neg_contexts


class Vocab:
    def __init__(self, tokens = None):
        self.idx_to_token = list()
        self.token_to_idx = dict()
        
        if tokens is not None:
            if "<unk>" not in tokens:
                tokens.append("<unk>")
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            # 单独取出未知词的下标
            self.unk = self.token_to_idx["<unk>"]
            
    @classmethod
    def build(cls, text, min_freq = 1, reserved_tokens = None):
        logger.info("preprocessing dataset")
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        
        return cls(uniq_tokens)
        
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)
    def covert_token_to_idx(self, tokens):
        # 查找一系列输入词对应的索引
        return [self.token_to_idx[token] for token in tokens]
    def covert_idx_to_token(self, indices):
        # 查找一系列索引值对应的输入
        return [self.idx_to_token[index] for index in indices]