你是一个精通Transformer的专业人士，请详细的解答一下，Bert与GPT同样是应用Transformer构建模型，二者有哪些不同？
BERT (Bidirectional Encoder Representations from Transformers)和GPT (Generative Pretrained Transformer)都是基于Transformer构建的模型，然而他们在设计和目标上有一些重要的差异。
模型训练的方向：GPT是一个自回归模型，只能从左到右或者从右到左进行预测，即在预测下一个单词时，只能利用该单词之前的上下文信息。而BERT是一个自编码模型，其训练是双向的，它可以同时考虑左右两侧的上下文信息。这是二者最本质的区别。
训练目标：BERT的预训练由两部分组成：Masked Language Model (MLM)和Next Sentence Prediction (NSP)。在MLM中，输入中的部分单词被随机遮盖，模型需要预测这些被遮盖的单词；而在NSP中，模型需要判断两个句子是否连续。这样的设计使得BERT能够理解词汇之间和句子之间的关系。而GPT的预训练任务则是一个单纯的语言模型任务，即基于前面的一串词预测下一个词。
模型架构：虽然BERT和GPT都采用了Transformer的架构，但在使用上有所不同。GPT仅使用了Transformer的解码器（Decoder），并且删除了其自注意力机制的masked部分以保证单向性；而BERT使用了Transformer的编码器（Encoder），可以同时考虑到上下文的左右信息。
应用场景：GPT在生成类任务上（如文本生成，对话系统）表现出色，因为它是一个生成性模型，可以生成连贯的语句；而BERT在理解类任务上（如文本分类，命名实体识别，问答系统等）效果更好，因为它能更好的理解句子的语义和上下文关系。
以上都是BERT和GPT之间的主要差异，但应注意的是，这两种模型都在各自的领域中取得了重要的突破，且在许多NLP任务上，都能得到非常好的效果。


请结合实际的Transformer代码讲述一下Bert与GPT的架构差异
为了更好的理解BERT和GPT的架构差异，我们首先来看一下原始的Transformer模型。Transformer主要包括两个部分：编码器（Encoder）和解码器（Decoder）。这两者都是由多个相同的层堆叠起来构成的。

每一层编码器包含两个子层：第一个是多头自注意力机制（Multi-Head Self Attention）；第二个是一个简单的前馈神经网络（Feed Forward Neural Network）。同时，每一层还包括了残差连接（Residual Connection）和层归一化（Layer Normalization）。

解码器也包含两个与编码器相同的子层，以及一个额外的多头注意力机制，它接受编码器输出的向量作为键和值（Key and Value）。

BERT和GPT的主要区别在于，BERT只使用了Transformer的编码器部分，而GPT只使用了Transformer的解码器部分。

```python
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        assert d_model % heads == 0, "Dimension of model should be divisible by number of heads"

        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

# Bert的编码器
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, hidden_dim=2048):
        super().__init__()
        self.attention = MultiHeadAttention(heads, d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attended = self.attention(x, x, x, mask)
        x = self.norm1(attended + x)
        fedforward = self.feed_forward(x)
        return self.norm2(fedforward + x)

# GPT的解码器
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, hidden_dim=2048):
        super().__init__()
        self.attention = MultiHeadAttention(heads, d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        attended = self.attention(x, x, x, trg_mask)
        x = self.norm1(attended + x)
        attended = self.attention(x, e_outputs, e_outputs, src_mask)
        x = self.norm2(attended + x)
        fedforward = self.feed_forward(x)
        return self.norm3(fedforward + x)
```
以上代码仅为了说明BERT和GPT在模型架构上的主要差异，并不是实际的BERT和GPT模型实现。在实际的BERT和GPT模型中，还会包括一些其他的技术，比如BERT中的Masked Language Model和Next Sentence Prediction预训练任务，以及GPT中的自回归语言模型预训练任务等。