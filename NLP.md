​
# Word vectors
"You shall know a word by the company it keeps"

观其伴而知其义



**Embedding**是指将高维、非结构化的数据（如文本、图片、用户行为等）映射到低维的稠密向量空间（通常是一个高维浮点数向量）。这个过程的核心目标是让相似的数据在向量空间中距离更近，从而方便计算机理解、存储和计算数据之间的关系。
​

## Co-Occurrence
共现(?)矩阵(Co-Occurrence matrix)，矩阵的每个值$w_{i,j}$，表示在一个文档中，表示词$w_{i}$在窗口大小$n$内，词$w_{j}$出现的个数，反之亦然。

**Example: Co-Occurrence with Fixed Window of n=1**:

Document 1: "all that glitters is not gold"

Document 2: "all is well that ends well"


|     *    | `<START>` | all | that | glitters | is   | not  | gold  | well | ends | `<END>` |
|----------|-------|-----|------|----------|------|------|-------|------|------|-----|
| `<START>`    | 0     | 2   | 0    | 0        | 0    | 0    | 0     | 0    | 0    | 0   |
| all      | 2     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 0    | 0   |
| that     | 0     | 1   | 0    | 1        | 0    | 0    | 0     | 1    | 1    | 0   |
| glitters | 0     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 0    | 0   |
| is       | 0     | 1   | 0    | 1        | 0    | 1    | 0     | 1    | 0    | 0   |
| not      | 0     | 0   | 0    | 0        | 1    | 0    | 1     | 0    | 0    | 0   |
| gold     | 0     | 0   | 0    | 0        | 0    | 1    | 0     | 0    | 0    | 1   |
| well     | 0     | 0   | 1    | 0        | 1    | 0    | 0     | 0    | 1    | 1   |
| ends     | 0     | 0   | 1    | 0        | 0    | 0    | 0     | 1    | 0    | 0   |
| `<END>`      | 0     | 0   | 0    | 0        | 0    | 0    | 1     | 1    | 0    | 0   |

在NLP中，我们通常加入 `<START>,<END>`用于封装文本

通常情况下，文本内容包含的矩阵很大，因此我们需要进行降维，经常采用**SVD(Singular Value Decomposition， 主成分分析法的一种)**，来选择前$k$个主成分

## Word2vec 目标函数
### 似然函数
$Likelihood = L(\theta) = \prod\limits^{T}\limits_{t=1}\prod\limits_{-m\leq j \leq m } P(w_{t+j} | w_{t}; \theta)，j\neq0$
这个公式表示在给定参数 
𝜃的情况下，计算一个语料库（长度为 𝑇）的似然函数 
𝐿(𝜃)，并且假设每个中心词$w_t$  对其上下文词的生成概率是独立的。
$t$表示中间的单词，
$\prod\limits_{-m\leq j \leq m } P(w_{t+j} | w_{t}; \theta)$表示在$w_t$单词出现的条件下，在窗口大小为$j$的其他单词出现的概率之积。
整个公式表示所有中心词上下文词出现的概率。
我们需要找到一个参数$\theta$使得目标函数最大。

### 目标函数
$J(\theta)$



## TruncatedSVD
[TruncatedSVD（截断奇异值分解）](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)是 sklearn.decomposition.TruncatedSVD 提供的一种 降维方法，主要用于对稀疏矩阵（大部分值为0）或高维数据进行降维，类似于 PCA（主成分分析），但 TruncatedSVD 不需要数据是中心化的，并且可以应用于文本数据。
**数据中心化**：
数据中心化（Mean Centering）是指 将每个特征的均值调整为零，即 对每个特征减去其均值。中心化通常用于数据预处理，特别是在 PCA（主成分分析）等降维算法中。

**中心化 VS. 非中心化**
PCA（主成分分析）需要中心化数据，因为它基于协方差矩阵，而协方差的计算需要均值为零的数据。

**Truncated SVD(截断奇异值分解)** 不要求数据是中心化的，它可以直接应用于非中心化的数据，尤其适用于稀疏矩阵（如文本数据中的 TF-IDF 矩阵）。

**中心化的作用**
消除均值偏移：使得数据围绕零分布，减少均值对分析的影响。

**提高数值稳定性**：减少数值计算中的误差，特别是在协方差计算和矩阵分解时。

**适用于某些线性代数方法**：如 PCA 需要中心化数据来保证主成分的正确性。

`class sklearn.decomposition.TruncatedSVD(n_components=2, *, algorithm='randomized', n_iter=5, n_oversamples=10, power_iteration_normalizer='auto', random_state=None, tol=0.0`


```py
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
np.random.seed(0)
X_dense = np.random.rand(100, 100)
X_dense[:, 2 * np.arange(50)] = 0
X = csr_matrix(X_dense)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)
```


`np.random.seed(0)` 的作用是设置 NumPy 的随机数种子，确保每次运行代码时生成的随机数完全相同。

```py
print(np.random.rand(3))  # 生成 3 个 [0,1) 之间的随机数
每次运行都会得到不同的结果，比如：

[0.5488135  0.71518937 0.60276338]  # 第一次运行
[0.54488318 0.4236548  0.64589411]  # 第二次运行（不同）
```

`np.random.rand(100, 100)` 生成 100 × 100 的随机数矩阵，值在 [0,1] 之间


X_dense[:, 2 * np.arange(50)] = 0

np.arange(50) 生成 [0, 1, ..., 49]。

`2 * np.arange(50) `生成 [0, 2, 4, ..., 98]，即选取了矩阵的偶数列（第 0、2、4...98 列）。


`X = csr_matrix(X_dense)`

csr_matrix(X_dense) 将 X_dense 转换为 稀疏矩阵格式（Compressed Sparse Row，CSR）。

目的：节省存储空间，加快运算速度。


`svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)`

n_components=5：降维到 5 维，即保留前 5 个奇异值对应的特征。

n_iter=7：使用 7 次迭代 提高计算精度（默认为 5）。

random_state=42：设定随机种子，使结果可复现

`svd.fit(X)`
fit(X) 计算 SVD 分解，得到降维后的矩阵和奇异值。

`print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())`

svd.explained_variance_ratio_ 返回前 5 个主成分的解释方差比，即这些主成分能解释多少数据的方差信息。

sum() 计算这 5 维特征一共解释的数据方差占比，衡量降维后数据的保留信息量。




## Skip-gram with softmax
$$ 
P(O = o \mid C = c) = \frac{\exp(u_o^\top v_c)}{\sum_{w \in \text{Vocab}} \exp(u_w^\top v_c)}
$$

**含义解释**

- **O**：目标词（output word），也称上下文词（context word）  
- **C**：中心词（center word）  
- **o**：一个具体的目标词  
- **c**：一个具体的中心词  
- **uₒ**：目标词 *o* 对应的向量（在输出空间的向量）  
- **v𝑐**：中心词 *c* 对应的向量（在输入空间的向量）  
- **Vocab**：整个词汇表（所有词的集合）

---

**这条公式表示什么？**

它表示在给定中心词 *c* 的情况下，词 *o* 作为上下文词（目标词）出现的概率。它使用了 softmax 函数：

- **分子** 是目标词和中心词向量的点积的指数，点积越大，两者相似度越高，值越大。
- **分母** 是所有词和中心词点积的指数和，作为归一化因子。

---

**直观理解**
- 如果$u_o^\top v_c$ 很大，说明目标词和中心词在语义空间中很接近，那么这个词作为上下文词的概率就高。
- 所有目标词的概率加起来为 1，是一个标准的概率分布。
- 但由于分母需要对整个词汇表求和，计算开销很大，因此训练中往往会使用 **Negative Sampling** 或 **Hierarchical Softmax** 来近似这个公式。

为了组织这些向量，我们使用了两个矩阵：

矩阵 V：它的每一列是一个词作为“中心词”的向量 vc。

矩阵 U：它的每一列是一个词作为“上下文词”的向量 uo。

这两个矩阵中都包含了词汇表（Vocabulary）中所有词的向量，也就是说每个词都会在两个矩阵中各有一个表示（一个是中心词向量，一个是上下文词向量）。

总结来说：

**对每个词 w ∈ Vocabulary，我们学习两个向量：vw 和 uw。**

这些向量分别保存在两个矩阵中：**V（保存 vw 向量），U（保存 uw 向量）。**

这个结构的设计有助于模型根据词与词之间的上下文关系，学习出有意义的词向量。

**损失函数**
$$
J_{naive-softmax}(v_c,o,U)=-log(P(o|c))
$$

 ✅ 词向量和损失函数解释

 ✅ `v_c`：中心词向量（center word vector）

- 指词 **c**（也就是当前输入的中心词）的向量表示。
- 来自词向量矩阵 **V**（中心词矩阵）中的第 `c` 列。
- 是当前模型要用来预测上下文词的“输入向量”。

---

 ✅ `o`：上下文词索引（outside word index）

- 表示当前要预测的一个上下文词（outside word）在词汇表中的索引。
- 损失函数的目标是使模型在给定 `v_c` 的情况下，预测出这个索引对应的词（也就是实际出现的上下文词）。

---

 ✅ `U`：上下文词向量矩阵（outside word vectors matrix）

- 是一个形状为 `(embedding_dim, vocab_size)` 的矩阵。
- 第 `o` 列是上下文词 `o` 的向量 \( u_o \)。
- 损失函数中会用到这个矩阵中所有词的向量来计算 softmax 或 negative sampling。

---

🔁 举个例子：

假设：

- 当前中心词是 **"dog"**，它的向量为 $( v_{\text{dog}} = v_c )$
- 当前上下文词是 **"barks"**，它在词表中的编号为 `o = 15`
- 上下文词矩阵是 $U$

那么损失函数：$J(v_c, o, U)$

表示“使用中心词 **'dog'** 的向量 `v_c`，来预测上下文词 **'barks'**（词表中编号为 15）的损失”。


# Dependency Parsing（依存句法分析）
它的目标是：

✅ 找出句子中各个词之间的“依存关系”，明确谁依赖谁，谁是主语、谓语、宾语等。

📘 举个例子：
句子：

"The cat sat on the mat."

通过依存分析，我们可能得到这样的结构：

```nginx
      sat
     /   \
  cat     on
  /         \
The         mat
             |
            the
```
⛓️ 分析说明：
sat 是句子的核心（谓语动词）。

cat 是 sat 的主语。

on 是 sat 的状语成分（介词）。

mat 是 on 的宾语。

The 和 the 分别修饰 cat 和 mat。

🧠 为什么叫 "Dependency"（依存）？
因为我们不是把句子分析成“短语结构”（如NP, VP），而是直接从每个词出发，看它 “依赖”于哪个词（谁是它的语法头）。

比如：

The 依赖于 cat （它是形容词修饰名词）

cat 依赖于 sat （它是主语）

on 依赖于 sat （它是介词短语）

mat 依赖于 on （介词的宾语）

## Greedy Deterministic Transition-Based Parsing
“Greedy Deterministic Transition-Based Parsing”（贪婪确定性转移式句法分析）是 依存句法分析（Dependency Parsing） 中的一种方法。我们分解来解释：

🧠 先理解几个关键词：
✅ 1. Transition-Based Parsing
这是一种逐步构建依存树的方法，思路是：

把解析过程视为一系列“状态”和“操作”（transition）的序列，通过不断转移状态来构建依存关系。

在解析过程中，维护三个部分：

Stack：暂存处理过的词（例如主语、动词等）

Buffer：还没处理的词（通常是输入句子的剩余部分）

Arcs：已建立的依存关系集合（如主谓、动宾等）

然后使用一些动作（比如 Shift、Left-Arc、Right-Arc）来一步一步构建句子的依存结构。

✅ 2. Deterministic（确定性）
表示：在每一步，只选择一个操作（transition），不会考虑多个选项或回溯。

✅ 3. Greedy（贪婪）
表示：每一步都选择当前看起来最优的那个操作，而不管将来会发生什么。
比如当前评分最高的操作是 “Right-Arc”，那就立刻执行，而不会做深度搜索。

📌 综合起来：“Greedy Deterministic Transition-Based Parsing” 就是：
用一种贪婪、一步步执行、没有回退的方式，通过转移系统构建一个句子的依存树结构。

🧱 举个简化例子
句子：“She eats apples”。

初始状态：

Stack: []

Buffer: [She, eats, apples]

Arcs: []

操作过程：
Shift: 把 "She" 推入 Stack
→ Stack: [She] | Buffer: [eats, apples]

Shift: 把 "eats" 推入 Stack
→ Stack: [She, eats] | Buffer: [apples]

Left-Arc (She ← eats)
→ 建立依存关系 eats → She
→ Stack: [eats] | Buffer: [apples] | Arcs: [(eats, She)]

Shift: 把 "apples" 推入 Stack
→ Stack: [eats, apples] | Buffer: []

Right-Arc (eats → apples)
→ Stack: [eats] | Buffer: [] | Arcs: [(eats, She), (eats, apples)]

最终依存树就是：

eats → She

eats → apples

### A2
接下来，我通过讲解a2作业，加深对依存语句解析的理解。
#### 实现`parser_transition.py`
**_init_**
这个文件中包含`PartialParse`这个类，表示一个依存语句，他包含**栈(stack)、语句缓冲区(buffer)、依存关系(dependencies)**三个属性。`buffer`用于暂存还未解析的单词，`stack`用于存储正在解析的单词，`dependencies`用于存储已解析的关系。
初始化时，默认`stack`中已含有元素`ROOT`。

**parse_step**
对当前的`stack`执行一次状态转移操作
其中
✅ LA（Left-Arc）
含义： 将栈顶第二个词（stack[-2]）作为子节点，连到栈顶第一个词（stack[-1]）上，建立一个从 右到左 的依存关系。
操作： 弹出栈顶第二个元素（stack[-2]）。

例子：
假设：
```
Stack: [A, B]
Buffer: [C, D]
```
执行 LA 后，表示：
B ← A（A 依赖于 B）
结果变为：
```
Stack: [B]
Buffer: [C, D]
Dependencies: [(B, A)]
```
✅ RA（Right-Arc）
含义： 将栈顶第一个词（stack[-1]）作为子节点，连到栈顶第二个词（stack[-2]）上，建立一个从 左到右 的依存关系。
操作： 弹出栈顶第一个元素（stack[-1]）。

例子：
假设：
```
Stack: [A, B]
Buffer: [C, D]
```
执行 RA 后，表示：

A → B（B 依赖于 A）

结果变为：
```
Stack: [A]
Buffer: [C, D]
Dependencies: [(A, B)]
```
✅ S（Shift）
将 buffer 的第一个词移动到 stack 的顶部。

#### 实现`minibatch_parse`
**算法思想**

———————————————————————
**Minibatch Dependency Parsing**
———————————————————————
**输入：语句，状态转移模型，批次大小**
```
将语句列表中的每个句子都实例化为一个`PartialPase`，然后得到一个`PartialPase`列表(`partial_parses`)
初始化`unfinished_parses`(=`partial_parses`)，`dependencies`为空
while `unfinished_parse`不为空:
	从`unfinished_parses`取出批次大小的parses作为`minibatch`
	使用模型预测当前`minibatch`下一步的转移操作
	进行转移操作
	更新`unfinished_parses`（如果其中的`parse`栈长度为1（仅剩`ROOT`)或缓冲区为空，将该`parse`移除)
将`partial_parses`的每个`parse`的依存关系导入`dependencies`
```
——————————————————————

#### 训练转移操作预测模型
首先，模型会提取一个向量表示当前的状态。这个特征向量由tokens列表组成（例如：stack的最后一个单词，buffer的第一个单词，stack倒数第二个词的依存关系，在这里这些称为特征n_features)，他们可以由一串整数表示 $w =[w_1,w_2,...,w_m]$，$m$表示特征的个数，对于每个$w_i$,$0 <= w_i < |V|$，表示这个token再词汇表$V$中的索引，然后我们的神经网络会查找每个词的词向量并将他们合在一起，形成一个输入向量，$x = [E_{w1},...,E_{wm}]$，$E_{wi}$表示$d$维的词向量。
然后计算我们的预测
$$
h = ReLU(xW + b_1)\\
l = h U + b_2\\
\hat y = softmax(l)
$$
$h$表示隐藏层，$l$是logits(分类模型在softmax之前的原始得分向量)
损失函数选用交叉熵
$$
J(\theta） = CE(y,\hat y) =  - \sum _{j=1}^{3}y_jlog \hat y_j
$$
因为要预测哪一种转移操作，所以选择逻辑回归模型，3表示3种转移操作

#### 用PyTorch构建模型`ParserModel`
**_ _init_ _**

🔹代码如下：
```python
self.embed_to_hidden_weight = nn.Parameter(torch.empty(self.embed_size * self.n_features, self.hidden_size))
nn.init.xavier_uniform_(self.embed_to_hidden_weight)
```
✅ 第一行：
```python
self.embed_to_hidden_weight = nn.Parameter(torch.empty(self.embed_size * self.n_features, self.hidden_size))
```
含义：
torch.empty(...)：创建一个 未初始化的张量（tensor），它的**形状是(输入维度,输出维度)**。

self.embed_size * self.n_features：

假设每个词的 embedding 大小是 embed_size，比如 50。

模型使用 n_features 个词作为输入（如栈顶、缓冲区前几个等），比如 36 个。

所以输入特征维度就是 50 × 36 = 1800。

self.hidden_size：

是你自己设定的隐藏层神经元个数，比如 200。

nn.Parameter(...)：

告诉 PyTorch：这是一个需要训练的参数，模型在训练时会对它自动进行梯度更新。

✅ 最终：
你定义了一个形状为 (1800, 200) 的权重矩阵，用于将输入特征从 1800 维变换为隐藏层的 200 维。

✅ 第二行：
```python
nn.init.xavier_uniform_(self.embed_to_hidden_weight)
```
含义：
用 Xavier Uniform 初始化方法对刚刚创建的权重进行初始化。

这种初始化方法可以让不同层的输入输出方差一致，有助于神经网络更快收敛、稳定训练。

📌 总结：
你这两行代码的作用是：

定义一个用于 embedding → hidden layer 的权重矩阵，并使用 Xavier 方法初始化它。

```def embedding_lookup(self, w):```
这是类中的一个方法（一般在 ParserModel 类里），用于将索引序列 w 映射成词向量。

w 是一个 2D 张量，形状通常是 [batch_size, n_features]，表示一个 batch 中每个样本的多个特征词（比如栈顶、缓冲区前几个等）在词表中的索引。

```python
x = None
x = self.embeddings[w]
```
self.embeddings 是一个参数矩阵，形状为 [num_words, embed_size]，表示词表中每个词的词向量。

self.embeddings[w]：利用索引 w 在这个矩阵中查找对应词向量。

得到的 x 是形状为 [batch_size, n_features, embed_size] 的三维张量。

每个样本有 n_features 个词，每个词是 embed_size 维的向量。

```python
x = x.view(w.shape[0], -1)
```
.view(w.shape[0], -1)：将三维张量展平为二维张量。
w.shape[0] 是 batch_size。

-1 表示 PyTorch 自动推导剩下的维度，结果就是 [batch_size, n_features * embed_size]。

也就是：把每个样本的多个词向量拼接成一个大向量作为输入。

```python
return x
```
返回形状为 [batch_size, n_features × embed_size] 的张量，用作模型输入。

```
self.dropout = nn.Dropout(self.dropout_prob)
```
self.dropout_prob：表示 dropout 的概率，即 随机将多少比例的神经元“屏蔽”掉（设置为 0）。

例如 dropout_prob = 0.5 表示每次前向传播时，有 50% 的神经元会被随机置为 0（相当于“关闭”）。

nn.Dropout(p)：PyTorch 中的 Dropout 层，p 是丢弃的概率。

在 训练（training）模式下：每次前向传播都会随机遮掉一部分神经元。

在 评估（eval）模式下：Dropout 不生效，所有神经元都参与运算。

✅ 为什么用 Dropout？
Dropout 是一种正则化方法，用于解决神经网络容易“过拟合”的问题。它的原理是：

训练时随机关闭一部分神经元，使模型不会太依赖某些特征。

相当于让模型学习“冗余路径”，提升泛化能力。


**forward**

$$
h = ReLU(xW + b_1)\\
l = h U + b_2\\
\hat y = softmax(l)
$$
```
def forward(self, w):
        logits = None
        logits = F.relu(torch.matmul(self.embedding_lookup(w), self.embed_to_hidden_weight) + self.embed_to_hidden_bias)
        logits = self.dropout(logits)
        logits = torch.matmul(logits, self.hidden_to_logits_weight) + self.hidden_to_logits_bias
        return logits
```

w是tokens,shape为(batch_size, n_features)的张量，
embedding_lookup(w)会从vocabulary中找出相应索引的词向量，得到(batch_size, n_features,embed_size)组成的三维张量，然后通过view将其展开为(batch_size,n_features*embed_size）的张量

```
shape:
x(batch_size,n_features*embed_size）
W(n_features*embed_size,hidden_size)
b_1是size为hidden_size的一维张量，做加法时，会广播的每项加上偏置
```


# RNN
## 损失函数
时间步长为t的损失函数
$$
J^{(t)}(\theta) = \sum_{t=1}^{|V|}y_{t,j}\log \hat y_{t,j}^{(t)}
$$
整个语料库的损失函数
$$
J(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{|V|}y_{t,j}\log \hat y_{t,j}^{(t)}
$$
## Perplexity
困惑度是用来评估语言模型性能的指标，通常用于比较不同语言模型的优劣。
$$
Perplexity = 2^{J(\theta)}
$$

## 优点
- 可以处理变长序列
- 模型的大小不会随着序列长度的增加而增加
- 步骤 t 的计算可以（理论上）使用来自许多步骤之前的信息。
- 每一个时间步都重复使用相同的参数（权重共享）

## 缺点
- 计算很慢，因为他是顺序的，不能并行化
- 由于梯度消失和梯度爆炸等问题，很难从许多步骤中获取信息，不能很好地处理长距离依赖，只有短期记忆

## RNN 中梯度爆炸和梯度消失的原因

RNN 的本质是一个链式结构，在每个时间步 \( t \)，RNN 会根据当前输入 \( x_t \) 和前一时刻的隐藏状态 \( h_{t-1} \) 来更新隐藏状态：

\[
h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b)
\]

训练时使用 BPTT（Backpropagation Through Time），梯度会从最后一个时间步 \( T \) 反向传播回第一个时间步 \( 1 \)，导致梯度连续相乘：

\[
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \cdot \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
\]

---

### 梯度消失（Vanishing Gradient）

- 若每次乘的值 < 1（如 0.5），则：
  \[
  (0.5)^T \rightarrow 0
  \]
- 随着时间步增多，前面层梯度变为 0，导致无法学习长期依赖。

---

### 梯度爆炸（Exploding Gradient）

- 若每次乘的值 > 1（如 1.5），则：
  \[
  (1.5)^T \rightarrow \infty
  \]
- 导致梯度迅速变大，参数剧烈更新甚至溢出。

---

## 解决方法

### 1. 对抗梯度爆炸的方法

- **梯度裁剪（Gradient Clipping）** ✅ 最常用
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

### 2. 对抗梯度消失的方法
- 使用更好的激活函数（ReLU、Leaky ReLU 等）

- 使用门控 RNN 变体（LSTM、GRU）

- 更好的权重初始化（Xavier、He）

- 使用残差连接（Residual Connection）

- 使用 Dropout / LayerNorm 等正则化


## Deep Bidirectional RNN
双向RNN可以解决RNN的梯度消失问题，因为它可以同时考虑过去和未来的信息。

![](https://i-blog.csdnimg.cn/direct/dfef87d1d15e438cbc005b755e620eaa.png)

# Tips
## 1. List Comprehensions
```py
a_list = [1, ‘4’, 9, ‘a’, 0, 4]

squared_ints = [ e**2 for e in a_list if type(e) == types.IntType ]

print squared_ints
# [ 1, 81, 0, 16 ]
```




## 2. Nested(嵌套) Comprehensions
python的矩阵表示
```py
[ [ 1, 0, 0 ],
  [ 0, 1, 0 ],
  [ 0, 0, 1 ] ]
```
```py
[ [ 1 if item_idx == row_idx else 0 for item_idx in range(0, 3) ] for row_idx in range(0, 3) ]
```

## 3. zip
```py
list(zip([1, 2, 3], ['a', 'b', 'c']))
# 输出: [(1, 'a'), (2, 'b'), (3, 'c')]
```

## 4. Set Comprehensions
Given the list:

names = [ 'Bob', 'JOHN', 'alice', 'bob', 'ALICE', 'J', 'Bob' ]
We require the set:

{ 'Bob', 'John', 'Alice' }
Note the new syntax for denoting a set. Members are enclosed in curly braces.

The following set comprehension accomplishes this:
```py
{ name[0].upper() + name[1:].lower() for name in names if len(name) > 1 }
```

## 5. 
```py
mcase = {'a':10, 'b': 34, 'A': 7, 'Z':3}
mcase_frequency = { k.lower() : mcase.get(k.lower(), 0) + mcase.get(k.upper(), 0) for k in mcase.keys() }
```
这是一段字典推导式（dictionary comprehension），它的作用是：

遍历 mcase.keys() 中的所有键。

将键转换为小写，这样 a 和 A 会被归为 a。

累加原字典中小写键和大写键的值：

mcase.get(k.lower(), 0) 获取小写键的值（如果不存在则返回 0）。

mcase.get(k.upper(), 0) 获取大写键的值（如果不存在则返回 0）。

创建一个新的字典 mcase_frequency，其中所有键都变成了小写，且大小写相同的键的值相加。

