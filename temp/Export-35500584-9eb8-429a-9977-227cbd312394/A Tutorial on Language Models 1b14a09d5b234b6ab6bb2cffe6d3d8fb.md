# A Tutorial on Language Models

Created time: January 6, 2023 3:54 PM
Last edited time: January 13, 2023 7:45 PM

# What this is for.

This is a tutorial on transformers and language models, focusing on GPT models. This is a relatively concise tutorial, that assumes that the reader has some background in deep learning, and I use mathematical notation freely. My main motivation here is to learn by writing, but also there is the fact that good transformer tutorials seem to be lacking. Really, for what is arguably the greatest revolution in deep learning in the last ten years, it is ridiculous that there are so few good tutorials or explanations. This is another, very opinionated one, that may be useful to someone.

# Introduction and motivation.

I may write some introduction later, but to be honest, suffices to play a bit with [ChatGPT](https://chat.openai.com/chat) to have enough motivation for knowing about language models. So, if you’re reading this, I will either assume that you are already interested in language models and is just ready to jump to the tutorial, or I will simply let ChatGPT answer that for me:

![Selection_061.png](A%20Tutorial%20on%20Language%20Models%201b14a09d5b234b6ab6bb2cffe6d3d8fb/Selection_061.png)

# Language models and sequence models.

Operationally, a language model can be thought as following the rule “given string $s$ of natural language, output string $w$ with probability $p(w|s)$”. This is what GPT-like models do at the end of the day. So the question is how to do that?

First, we need to map $s$ and $w$ into some suitable structure. For this, we make use of a ***************************“sequencing function”*************************** $T$ that will map strings $s$ to finite sequences $\mathbf{x}$, with elements belonging to a finite set $\{1, \ldots, n_{vocab}-1\}$. For instance, we transform the sequence “I ate a banana” into a sequence $(10, 123, 2, 1230)$. We assume that the map is ***********one-to-one*********** so that we can define an inverse sequencing function $T^{-1}$ mapping finite sequences $\mathbf{y}$ to string $w$. Now, this is a small lie, as we will explain in the sequencing section, but one can think of this as “true enough” for the discussion that follows.

So we transform our problem into an equivalent sequence problem. Consider our index set $\{1, \ldots, n_{vocab}-1\}$, and let $\mathbf{x}$ be a finite sequence in this set. As an example, we let $K=200$ and $\mathbf{x} = (1, 10, 2, 198, 4)$. Our sequence model will output a finite sequence $\mathbf{y}$, also in $[K]$ with probability $p_M(\mathbf{y}|\mathbf{x})$. Now, since we are dealing with sequences, we can use the chain rule of probability to turn our problem into a **********next-item********** prediction as follows: let $\mathbf{y} = (y_1, \ldots, y_m)$ be a $m$-sized sequence, and consider the special index $0$ signaling the end of a sequence. Then, letting $\mathbf{y}^i = (y_1, \ldots, y_i)$, and letting $\mathbf{x}\mathbf{y}$ be the concatenation of two sequences, we find that

$$
p_M(\mathbf{y}|\mathbf{x}) = p_M(0|\mathbf{x}\mathbf{y})\prod_{i=1}^n p_M(y_i|\mathbf{x}\mathbf{y}^{i-1})
$$

Therefore, our model is ***********************only obliged to predict the next token*********************** $y_i$ ********************given a sequence******************** $\mathbf{x}$*.* The chain rule takes care of the rest. 

Finally, given the sequence $\mathbf{y}$, we can unsequence it **********************************to get string $w$. We do that by applying an inverse sequencing function $T^{-1}$ to each element of $\mathbf{y}$. Thus, we have that our language model is both a product of the sequence model $M$ and the sequencing function $T$, and we have $p_{M, T}(w|s) = p_M(T(w)|T(s))$, and we sample $w$ by sequencing $s$, sampling $y$ from $M$ given $s$ by the chain rule above (until reaching the end-of-a-sequence item), and unsequencing $w$.

## Sequencing, tokenization, and indexing.

Now, how the task of transforming a string into a sequence of integers, which we will call **identifiers** or ***ids***, is composed of two parts:

- Splitting the string into words and subwords, which we call ******tokens******, using a *************tokenization************* rule. As in, transforming “I love you” into (”I”, “love”, “you”).
- Using a lookup table to transform a sequence of ******tokens****** into a sequence of identifiers using a lookup table.

The second part is relatively trivial: we just associate each token in our **********vocabulary**********, that is, our list of known tokens, to a unique **********identifier**********, and use that as a lookup table to transform our sequence of tokens into a sequence of identifiers. Now, if some token ends up not being in our vocabulary, we can associate it to some special unknown identifier.

The first part is more complicated, and there are many tokenizations algorithms. A good tutorial can be found [here](https://huggingface.co/docs/transformers/tokenizer_summary). However, for the time being, we can assume that each word and each sentence is a token (which is a valid tokenization). As for mapping a sequence back, we just invert our lookup table, mapping the “unknown” identifier to a blank sentence or to a “<unk>” symbol or something else, and concatenate our tokens according to some rule (which can be the regular grammar rule for joining words and punctuation).

```python
#Here, we create a vocabulary from the works of Shakespeare,
#and use the Spacy tokenizer.
site = "https://www.gutenberg.org/files/100/100-0.txt"
data_string = urllib.request.urlopen(site).read().decode('utf-8')
#Tokenizer our data using Spacy tokenizer, to create a vocabulary
tokenizer = torchtext.data.utils.get_tokenizer('spacy')
tokens = tokenizer(data_string)
#Create a vocabulary from tokens
sorted_tokens_freq = collections.OrderedDict(
                        sorted(collections.Counter(tokens).items(), key = lambda x : x[1], reverse=True)
)
vocab = torchtext.vocab.vocab(sorted_tokens_freq, specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])
#Create a detokenizing function
def detokenizing(tokens):
    words = [(' ' + s if s.isalpha() else s) for s in tokens]
    return ''.join(words)
print(tokenizer("hello my friend"))
print(vocab(tokenizer("hello my friend")))
print(vocab.lookup_tokens([5, 20, 10]))
print(detokenizing(vocab.lookup_tokens([5, 20, 10])))
#['hello', 'my', 'friend']
#[0, 15, 292]
#['the', 'is', 'of']
# the is of
```

## The sequence model.

Now, we are ready to get on to the main part of our work: explaining the GPT architecture. Remember that we want to model a function that, for sequence $\mathbf{x} = (x_1, \ldots, x_n)$, outputs $p(y_{n+1}|\mathbf{x})$, for $y_{n+1} \in n_{vocab} := \{0, 1, \ldots, n_{vocab}-1\}$ (because we are including the end-of-sentence token). However, as we will soon find out, our architecture will also get, paying “almost nothing”, $p(y_{i+1}|\mathbf{x}^i)$ for $1 \leq i < n$. Moreover, in practice, we will limit our sequences to be smaller than a *****************context window***************** of size $n_{ctx}$, so that $n \leq n_{ctx}$. Therefore, our neural network $f_\theta$ will be a function $f_\theta : [n_{vocab}]^{\leq n_ctx} \to \mathbb{R}^{\leq n_{ctx}, n_{vocab}}$, such that, for $\mathbf{x} = (x_1, \ldots, x_n)$, $f_\theta(\mathbf{x})$ is a $n \times n_{vocab}$-sized matrix with elements $f_\theta(\mathbf{x})_{i, j+1} = p(j|\mathbf{x}^i)$.

Before moving to the architecture itself, it is useful to understand ***how*** we are going to train our net. For that, we need to define a loss function and the data that we are using. In fact, this is where some of the scaling magic of sequence models comes from, because the training data we need are *****only training sequences***** $\mathbf{x}$. This is because, for any sequence $\mathbf{x}^i$, with $i < n$, we know what is the correct label for $p(y_{i+1}|\mathbf{x}^i)$, which is $x_{i+1}$ *itself.* So, for sequence $\mathbf{x}$, we can use the negative log-likelihood to define our loss $l$ for parameters $\theta$ as

$$
l(\theta;\mathbf{x}) = -\sum_{i=1}^{n-1} \log p(x_{i+1}|\mathbf{x}^{i}) = -\sum_{i=1}^{n-1} \log f_\theta(\mathbf{x})_{i, x_{i+1}+1} 
$$

Now, to ensure that works, we need to ensure that, when passing $\mathbf{x}$ to $f_\theta$, the output of $f_\theta(\mathbf{x})_{i}$ *******************only depends on******************* $\mathbf{x}^i$, that is, it should not incorporate any information from subsequent tokens. If not, we will be using it to infer $x_{i+1}$ either information from $x_{i+1}$ itself or “future” positions, thus in a sense “cheating”.

Imagining, for now, $f_\theta(\mathbf{x})$ as a “magic transformer box”, the schematics of our language model work like the following figure, which ends up predicting that after the sentence “apple banana banana” with 70% of chance, the sentence ends.

 

![diagram4.png](A%20Tutorial%20on%20Language%20Models%201b14a09d5b234b6ab6bb2cffe6d3d8fb/diagram4.png)

In the following, we apply this “magic transformer box” philosophy to train the GPT model on the complete works of Shakespeare (actually 80% of them).

```python
#Create our train and text dataloader.
def batchify(x, line_size=64):
    tensor_size = x.shape[-1]
    nbatches = tensor_size//line_size
    overhead = line_size - tensor_size%line_size
    if overhead != 0:
        x = torch.nn.functional.pad(x, (0, overhead))
        nbatches += 1
    new_size = x.shape[:-1] + (nbatches, line_size)
    x = x.reshape(*new_size)
    return x
line_size=64
data_tensor = batchify(torch.tensor(vocab(tokens), dtype=torch.long), line_size)
dataset = torch.utils.data.TensorDataset(data_tensor.to(DEVICE))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
#Create our model
model = GPT(len(vocab), line_size-1, dembed=64, nlayers=6, nheads=8, dk=8, dv=8, nhidden=64)
model.to(DEVICE)
model.train()
lr = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, eta_min=1e-5)
nepochs = 100
print_frequency = 100
min_loss = math.inf
#Train our model
for epoch in range(nepochs):
    model.train()
    for step, [x] in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = loss_fn(x, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-1)
        optimizer.step()
    if loss < min_loss:
        min_loss = loss
        best_model = copy.deepcopy(model)
    scheduler1.step()
    scheduler2.step()
    print(f"Epoch : {epoch}, loss : {min_loss}")
#Write an autocomplete function and evaluate
def autocomplete(model, string, maxiter=128):
    base = torch.tensor(vocab(tokenizer(string))).to(DEVICE)
    nmax = model.ntokens
    for _ in range(maxiter):
        out = model(base[-nmax:])[-1, :]
        ind = torch.multinomial(torch.exp(out), 1)
        base = torch.cat([base, ind], axis=-1)
        if ind.item() == 0:
            break
    tokens = vocab.lookup_tokens(base.cpu().tolist())
    return detokenizing(tokens)
print(autocomplete(model, 'QUEEN OF WALES'))
"""
 QUEEN OF WALES
 RICHARD, and KEEPER at Bordeaux.
 I have leisure kept together,
 Which of twelve thousand looks in France,
 Against the French Duke of Norfolk,
 Whose power of Hereford, Donalbain, speed,
 Under your progenitors, charming arms, Edward’s blood,
 And care not his father lives at MALCOLM.
 It may not be discharged truth,
 Where let, Harry to prevent enacted ships
 To pay, and princes if can be bold
 But must suspect ere we bid be alive
 To follow them to the King with passage.
 My noble lord, prince. I have sounded London,
 So many the living time
"""
```

# Opening the “magic transformer box”.

## Notation

From now on, we will be dealing with tensors of order higher than two, since this simplifies a lot of the notation for transformers. Therefore, we will make use of the Einstein notation, that is, statements such as $\mathbf{a}_{ij} \mathbf{b}_j$ equal to $\sum_j \mathbf{a}_{ij} \mathbf{b}_j$. We will refer to elements of a tensor by lowercase bold letters such as $\mathbf{a}$, and the entire tensor b uppercase letters such as $\mathbf{A}$. Moreover, we will use a particular notation to refer to column and row vectors of a matrix. Namely, if $\mathbf{A}$ is a matrix (an order 2 tensor), then $\mathbf{a}_{i \circ}$ refers to the $i$-th row vector of $\mathbf{A}$ and $\mathbf{a}_{\circ j}$ to the $j$-th column vector of of $\mathbf{A}$. Similarly, with $\mathbf{B}$ is an order (3 tensor), $\mathbf{b}_{\circ j k}$ refers to the vector obtained by selecting indexes $j$ and $k$ in the second and third position, $\mathbf{b}_{i \circ \circ}$ to the submatrix by selecting $i$ in the first position, and so on (this implies of that $\mathbf{b}_{\circ \circ \circ} = \mathbf{B}$). Moreover, $\mathbf{W}$ (using some superscript) always refers to some learnable parameter of our layer. Finally, $\mathbf{1}_{\text{condition}}$ refers to the function that has value 1 if the *condition* is satisfied, else it has value 0. For instance, $\mathbf{1}_{i = j}$ is a function of $i, j$ that equals 1 if $i = j$, and $0$ otherwise.

## Embedding

Embedding is nothing more than the operation of transforming sequences into vectors determined by a learnable lookup table. That is, assume we have $n_{vocab}$ items in our vocabulary and want to associate each token with a $d$-dimensional vector. Then we set up a learnable matrix $\mathbf{W}^e$ of size $(n_{vocab}, d)$ and, for sequence $\mathbf{x}$ of size $n$, the embedding of $\mathbf{x}$ is given by a matrix $\mathbf{V}$ of size $(n, d)$, whose $i$-th row is given by $\mathbf{v}_{i \circ} = \mathbf{w}^e_{x_i \circ}$.

Now, we also need some information about the position of $i$ of token $x_i$. In the original transformers paper, this is given by a ******fixed****** vector $p_i$ whose $j$-th value $p_{i, j}$ equals $\sin \frac{i}{10000^{j/d}}$ if $j$ is even, and $\cos \frac{i}{10000^{(j-1)/d}}$ if $j$ is odd, and letting $\mathbf{v}_{i, \circ} = \mathbf{w}^e_{x_i \circ} + p_i$. Now, we can instead use **********learned********** positional embeddings for context window of maximum size $n_{ctx}$, by letting $\mathbf{W}^{pos}$ be a learnable $(n_{ctx}, d)$-sized matrix and letting $\mathbf{v}_{i\circ} = \mathbf{w}^e_{x_i \circ} + \mathbf{w}^{pos}_{i \circ}$. The total number of parameters will be then either $dn_{vocab}$ or $dn_{vocab} + dn_{ctx}$, depending on whether positional embedding are learned or fixed.

## Unembedding

Now, the transformer operation, to be explained in detail below, will take as an input an embedding matrix $\mathbf{V}$ of size $(n, d)$ and output another matrix of $\mathbf{V}’$ of size $(n, d)$. We need then to associate $\mathbf{v}_{i \circ}$ with the probability of $p(x_{i+1}=u|\mathbf{x}^i)$ of the next $(i+1)$-th token having value $u \in [n_{vocab}]$. For this, we make use of a linear layer with weight matrix $\mathbf{W}^{u}$ of size $(d, n_{vocab})$ and bias vector $\mathbf{b}^{u}$ of size $(n_{vocab})$, letting $\mathbf{v}’_{ij} \to \mathbf{v}’_{ik} \mathbf{w}^{u}_{kj} + \mathbf{b}^u_j$.  Finally, we apply the softmax function to each row of the resulting vector, resulting in an output matrix $\mathbf{Y}$ of size $(n, n_{vocab})$, with $\mathbf{y}_{iu} = p(x_{i+1} = u|\mathbf{x}^i)$. The total number of parameters here will be $d n_{vocab} + n_{vocab}$.

## Attention

Now we are able to get to the core of the transformer architecture. To begin with, consider the real matrix $\mathbf{V}$ of dimension $(n, d)$ that results from embedding sequence $\mathbf{x}$ in $d$ dimensions. An attention layer is one that performs the operation $\mathbf{v}_{ij} \to \mathbf{a}_{ik} \mathbf{v}_{kj}$, with $\mathbf{A}$ being of dimension $(n, n)$, such that $\mathbf{a}_{ik} \geq 0$ and $\sum_k \mathbf{a}_{ik} = 1$. That is, the *attention matrix* $\mathbf{A}$ takes to position $i$ the weighted mean $\mathbf{a}_{ik} \mathbf{v}_{k \cdot}$ of embedding vectors at other positions. Now, $\mathbf{A}$ will be given by two other matrices $\mathbf{Q}$ (query matrix) and $\mathbf{C}$ (key matrix) of dimensions $(n, d^q)$ such that $\mathbf{a}_{ik}$ is informally a measure of “similarity” between $\mathbf{q}_{i\circ}$ and $\mathbf{c}_{k\circ}$.  Now, in transformers, this similarity is given by a “score function” $\operatorname{score}(\mathbf{q}_{i \circ}, \mathbf{c}_{k \circ}) = \left<\mathbf{q}_{i\cdot}, \mathbf{c}_{k\cdot}\right>/\sqrt{d^q}$. The idea is that similarity is measured by an inner product, with a normalization factor $\sqrt{d^q}$ being used for stability. Now, given this score function, we simply apply a softmax function through each ****row**** of the score matrix, giving $\mathbf{a}_{ik} = \frac{e^{\operatorname{score}(\mathbf{q}_{i\circ}, \mathbf{c}_{k \circ})}}{\sum_k e^{\operatorname{score}(\mathbf{q}_{i\circ}, \mathbf{c}_{k \circ})}}$.Therefore, attention is an operation $\mathbf{V}' = \operatorname{attn}(\mathbf{V}, \mathbf{Q}, \mathbf{C})$ with $\mathbf{V}’$ being of dimension $(n, d)$.

Now, what are the query and the key matrix? By doing **************self-attention**************, we simply make $\mathbf{Q}$ and $\mathbf{K}$ depend on $\mathbf{V}$ itself from a learnable linear operation on each ***row*** of $\mathbf{V}$, that is, each embedding vector. So we have $\mathbf{q}_{ij} = \mathbf{v}_{il} \mathbf{w}^q_{lj}$ and $\mathbf{c}_{ij} = \mathbf{v}_{il} \mathbf{w}^c_{lj}$, where $\mathbf{W}^q$ and $\mathbf{W}^c$ are both of dimension $d \times d^q$. Moreover, we also make use of two learnable linear operations acting on $\mathbf{V}$ itself, first one defined by a matrix $\mathbf{W}^v$ with dimension $d \times d^v$ and the second one by a matrix $\mathbf{W}^o$ with dimension $d^v \times d$. Therefore, self-attention is an operation $\mathbf{v}_{ij} \to \mathbf{a}_{ik} \mathbf{v}_{il} \mathbf{w}^v_{lm} \mathbf{w}^o_{mj}$, with $\mathbf{a}_{ik} =  \operatorname{softmax}_k \left[ \left<\mathbf{v}_{il} \mathbf{w}^q_{l\cdot}, \mathbf{v}_{kl} \mathbf{w}^c_{l\cdot} \right>/d' \right]$, and we have an operator $\mathbf{V}’ = \operatorname{sattn}(\mathbf{V};\mathbf{W}^q, \mathbf{W}^c, \mathbf{W}^v, \mathbf{W}^o)$.

Now, there is one key piece missing here. Remember that we always want an output corresponding to the $i$-th token to only depend on tokens $k \leq i$. Now, the above formulation does ****not**** ensure that, since we have in general that $\mathbf{a}_{ik}$ can be greater than zero even if $k > i$, so the $i$-th position uses information from positions above. Now, the solution here is to use a ****mask**** in these cases, so that we enforce $\mathbf{a}_{ik} = 0$. One solution is to simply modify the score function $\operatorname{score}(\mathbf{q}_{i \circ}, \mathbf{c}_{k \circ})$ to a ******causal****** score function $\operatorname{mscore}(\mathbf{q}_{i \circ}, \mathbf{c}_{k \circ}) = \operatorname{score}(\mathbf{q}_{i \circ}, \mathbf{c}_{k \circ}) \mathbf{1}_{k \leq i}$, zeroing out elements $k > i$. Leaving the rest as it is, we have an operator $\mathbf{V}’ = \operatorname{msattn}(\mathbf{V};\mathbf{W}^q, \mathbf{W}^c, \mathbf{W}^v, \mathbf{W}^o)$ that satisfy our requirement. Also, in GPT-3, some layers use a *******banded causal mask*******, that is, for some band size $\beta$, we not only zero out elements $k > i$, but also elements that $i - k > \beta$. So, each token only looks at some limited elements before it. In the figure below, we illustrate masked attention, for no mask, a causal mask, and a banded causal mask.

![Attention with causal mask, banded mask, and banded causal mask](A%20Tutorial%20on%20Language%20Models%201b14a09d5b234b6ab6bb2cffe6d3d8fb/attnmask2.png)

Attention with causal mask, banded mask, and banded causal mask

Now, all formulations above apply to a **********************single attention head.********************** However, the insight of multi-head attention is that we can repeat that for $n_{h}$ heads, by simply letting each operation $\operatorname{msattn}$ depending on different learnable parameters, and summing up the resulting vector. So, letting $\mathbf{W}^q, \mathbf{W}^c, \mathbf{W}^v, \mathbf{W}^o$ being order 3 tensors with dimensions $(n_h, d, d^q), (n_h, d, d^q), (n_h, d, d^v), (n_h, d^v, d)$, our multi-head attention is given by

$$
\mathbf{V’} = \operatorname{mhmsattn}(\mathbf{V};\mathbf{W}^q, \mathbf{W}^c, \mathbf{W}^v, \mathbf{W}^o) = \sum_l \operatorname{msattn}(\mathbf{V};\mathbf{w}^q_{l \circ \circ}, \mathbf{w}^c_{l \circ \circ}, \mathbf{w}^v_{l \circ \circ}, \mathbf{w}^o_{l \circ \circ}).
$$

![Masked multi-head attention with two heads.](A%20Tutorial%20on%20Language%20Models%201b14a09d5b234b6ab6bb2cffe6d3d8fb/diagram1.jpg)

Masked multi-head attention with two heads.

An implementation of masked multi-head attention is shown below.

```python
def dot_product_attn(queries, keys, values, mask=None):
    #queries : (..., ntokens, dk)
    #keys : (..., ntokens, dk)
    #values : (..., ntokens, dv)
    #mask : None or str or (..., ntokens, ntokens)

    dk = queries.shape[-1]
    inner_product = torch.einsum('...ij, ...kj -> ...ik', queries, keys)/math.sqrt(dk) #(..., ntokens, ntokens)
    if mask is not None:
        if isinstance(mask, str):
            ntokens = values.shape[-2]
            if mask == 'upper' or mask == 'causal':
                maskbool = torch.triu(torch.ones(ntokens, ntokens), diagonal=1)
                mask = torch.log(1 - maskbool)
            else:
                raise NotImplementedError
        if isinstance(mask, torch.Tensor):
            if mask.dtype == torch.bool:
                mask = torch.log((~mask).to(torch.float))
            inner_product += mask
    weights = torch.softmax(inner_product/math.sqrt(dk), dim=-1) #(..., ntokens, ntokens)
    wvalues = torch.einsum('...ij, ...jk -> ...ik', weights, values)
    return wvalues

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, nheads, dmodel, dk, dv):
        super().__init__()
        self.nheads = nheads
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        #(..., ntokens, dmodel), (nheads, dmodel, dk) -> (..., nheads, ntokens, dv)
        self.q_proj_matrix = torch.nn.Parameter(torch.zeros([nheads, dmodel, dk]))
        self.k_proj_matrix = torch.nn.Parameter(torch.zeros([nheads, dmodel, dk]))
        self.v_proj_matrix = torch.nn.Parameter(torch.zeros([nheads, dmodel, dv]))
        self.o_proj_matrix = torch.nn.Parameter(torch.zeros([nheads, dmodel, dv]))
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.q_proj_matrix)
        torch.nn.init.xavier_uniform_(self.k_proj_matrix)
        torch.nn.init.xavier_uniform_(self.v_proj_matrix)
        torch.nn.init.xavier_uniform_(self.o_proj_matrix)

    def forward(self, queries, keys, values, mask=None):
        #queries : (..., ntokens, dmodel)
        #keys : (..., ntokens, dmodel)
        #values : (..., ntokens, dmodel)
        #mask : None or (..., ntokens, ntokens)
        projected_queries = torch.einsum('...ij, kjm -> ...kim', queries, self.q_proj_matrix) #(..., nheads, ntokens, dk)
        projected_keys = torch.einsum('...ij, kjm -> ...kim', keys, self.k_proj_matrix) #(..., nheads, ntokens, dk)
        projected_values = torch.einsum('...ij, kjm -> ...kim', values, self.v_proj_matrix) #(..., nheads, ntokens, dv)
        new_projected_values = dot_product_attn(projected_queries, projected_keys, projected_values, mask) #(..., nheads, ntokens, dv)
        new_values = torch.einsum('...ijk, ilk -> ...jl', new_projected_values, self.o_proj_matrix) #(..., ntokens, dmodel)
        return new_values
```

# The transformer block.

Now, given multihead masked self-attention, there are some other pieces necessary to complete a full transformer block of our language model. Our language model is then going to be an embedding layer, the transformer block repeated a number of times and a linear output layer. Fortunately, those other pieces are much simpler than the attention block itself, so we go through them relatively quickly. 

Before we move on, it is imperative to notice that *************only attention will exchange information between token positions*************. The rationale here is that we *designed* our masked multi-head attention so that it satisfies the “not use future information” requirement. If our other components were to exchange information between tokens, we would either have to ensure they also satisfied that requirement, or lose that condition.

### The feedforward neural network.

The second main component of a transformer block as **************feedforward neural network acting on a matrix $\mathbf{V}$ of dimension $(n, d)$ ******************that is shared across all positions******************. That is, we have a parameterized function $f_{FNN}: \mathbb{R}^d \to \mathbb{R}^d$ that will act on each individual row $\mathbf{v}_{i \circ}$ of $\mathbf{V}$. This function is a feedforward neural network with a single hidden layer of dimension $d^{f}$. That is, letting $\operatorname{act}$ be an element-wise activation function (that is usually the $\operatorname{relu}$ but can also be the $\operatorname{gelu}$), and $\mathbf{W}^a, \mathbf{W}^b$ be learnable matrices of size $(d, d^f), (d^f, d)$, and $\mathbf{b}^a, \mathbf{b}^b$ learnable vectors of size $(d^f, d)$, we have that

$$
f_{FNN}(\mathbf{V};\mathbf{W}^a, \mathbf{W}^b) = \operatorname{act}(\mathbf{v}_{ik} \mathbf{w}^a_{kl} + \mathbf{b}_l^a)\mathbf{w}^b_{kj} + \mathbf{b}_l^b.
$$

### Layer normalization and dropout

Layer normalization is the operation given by, for each individual row $\mathbf{v}_{i \circ}$ of $\mathbf{V}$, calculating the mean $\mu_i$ and standard deviation $\sigma_i$ of $\mathbf{v}_{i \circ}$, normalize $\mathbf{v}_{i \circ}$ using $\mu_i, \sigma_i$, and apply a learnable affine transformation that is ****shared across all positions****. That is, letting $\mu_i = \frac{1}{d} \sum_{j} \mathbf{v}_{ij}$ and $\sigma_i = \sqrt{\frac{1}{d} \sum_{j} (\mathbf{v}_{ij} - \mu_i)^2}$, and using parameters $\mathbf{W}^{LN, a}, \mathbf{W}^{LN, b}$   of dimension $(d)$, for each item $\mathbf{v}_{ij}$, we apply the transformation $\mathbf{v}_{ij} \to \mathbf{w}^{LN, a}_{j} \frac{\mathbf{v}_{ij} - \mu_i}{\sigma_i + \epsilon} + \mathbf{w}_j^{LN, b}$.

Similarly, the dropout is a standard operation in neural networks in *********training********* that, for each element of the input, set that input to $0$ with probability $p$, and scales the input by a factor of $\frac{1}{1-p}$.

### Putting the block together.

Now, we have all the pieces to create our transformer block. First, we will wrap both the attention block $\operatorname{mhmsattn}$ and the feedforward neural network block $f_{FNN}$ with an layer normalization on the input and a dropout on the output, creating the blocks $\operatorname{block}_1 = \operatorname{drop} \odot \operatorname{mhmsattn} \odot \operatorname{ln}$ and $\operatorname{block}_2 = \operatorname{drop} \odot f_{FNN} \odot \operatorname{ln}$. Now, we will connect these through residual connections with the input $\mathbf{V}$, with the attention block first and the feedforward block second, creating a full transformer block

$$
\operatorname{transblock}(\mathbf{V}) = \mathbf{V} + \operatorname{block}_1(\mathbf{V}) + \operatorname{block}_1(\operatorname{block}_2(\mathbf{V})).
$$

An implementation of the attention block (here called encoder block) is shown below.

```python
class LayerNorm(torch.nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.dmodel = dmodel
        self.a = torch.nn.Parameter(torch.ones(dmodel))
        self.b = torch.nn.Parameter(torch.zeros(dmodel))
        self.eps = 1e-6

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdims=True)
        std = torch.std(x, dim=-1, keepdims=True)
        xnorm = (x-mu)/(std + self.eps)
        y = self.a*x + self.b
        return y

class EncoderLayer(torch.nn.Module):
    def __init__(self, nheads, dmodel, dk=None, dv=None, nhidden=128, pdrop=0.1):
        super().__init__()
        dk = dk if dk is not None else dmodel
        dv = dv if dv is not None else dmodel
        self.mthattention = MultiHeadAttention(nheads, dmodel, dk, dv)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(dmodel, nhidden),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(nhidden, dmodel))
        self.dropout = torch.nn.Dropout(pdrop)
        self.ln1 = LayerNorm(dmodel)
        self.ln2 = LayerNorm(dmodel)
        
    def forward(self, values, mask=None):
        # values = self.mthattention(values, values, values, mask)
        self_attention = lambda values : self.mthattention(values, values, values, mask)
        values = values + self.dropout(self_attention(self.ln1(values)))
        values = values + self.dropout(self.ffn(self.ln2(values)))
        return values
```

![A transformer decoder block](A%20Tutorial%20on%20Language%20Models%201b14a09d5b234b6ab6bb2cffe6d3d8fb/diagram2.jpg)

A transformer decoder block

Finally, it is useful to have in the head how many learnable parameters a transformer block have. Assuming that our multi-head attention has $n_h$ heads, input dimension $d$, value projection dimension $d^v$, and key/query dimension of $d^q$, we find that the number of parameters in multihead attention equals $2 n_h d(d^v + d^q)$. Our feedforward neural network with a $d^f$-sized hidden layer will have will have $2 d d^f + d + d^f$, and each layer normalization will have $2d$ parameters, bringing the total of parameters to $n_{block} = 2n_hd(d^v + d^q) + 2dd^f + d + d^f + 9d$. By default, GPT-3 lets $d$ be a number divisible by $n_h$, $d^v, d^q = d/n_h$ and $d^f = 4d$, thus simplifying our formula to $n_{block} = 12d^2 + 9d$.

## The full sequence model neural network.

Putting it all together is simple: we simply stack sequentially an embedding + positional embedding layer, $n_T$ transformers block, a **************************final layer normalization************************** block, and an unembedding block. Letting our architecture be as following.

![diagram3.jpg](A%20Tutorial%20on%20Language%20Models%201b14a09d5b234b6ab6bb2cffe6d3d8fb/diagram3.jpg)

The total number of parameters will therefore be

$$
n_{block} = 2n_hd(d^v + d^q) + 2dd^f + d + d^f + 4d \\
n_{tr} = n_T n_{block} + 2d \\
n_{embed} = dn_{vocab} + d n_{ctx} \\
n_{unembed} = dn_{vocab} + n_{vocab} \\
n_{params} = n_{tr} + n_{embed} + n_{unembed},
$$

dropping the $d n_{ctx}$ term if fixed positional embedding is used. If we consider the default setting as in GPT-3, we find that

$$
n_{params} = n_T(12d^2 + 9d) + 2d + n_{vocab}(2d + 1) + d n_{ctx}.
$$

A parameter counting code is as such

```python
def subsnone(x, val):
    return x if x is not None else val

def number_of_gpt_parameters(nvocab, dembed, nlayers, nheads, nctx=None, dk=None, dv=None, nhidden=None):
    nhidden = subsnone(nhidden, 4*dembed)
    dk = subsnone(dk, dembed//nheads)
    dv = subsnone(dv, dembed//nheads)
    n_attn = 2*dembed*(dk + dv)*nheads
    n_fnn = 2*dembed*nhidden + dembed + nhidden
    n_ln = 2*dembed
    n_transformer = (n_attn + n_fnn + 2*n_ln)*nlayers + 2*dembed
    n_embed = dembed*nvocab
    if nctx is not None:
        n_embed += nctx*dembed
    n_unembed = dembed*nvocab + nvocab
    return n_embed + n_unembed + n_transformer
```

We can then finally comlpete our code for GPT.

```python
class Encoder(torch.nn.Module):
    """
        Makes an encoder
    """
    def __init__(self, nlayers, nheads, dmodel, dk=None, dv=None, nhidden=128, pdrop=0.1):
        super().__init__()
        self.nlayers = nlayers
        self.layers = torch.nn.ModuleList(EncoderLayer(nheads, dmodel, dk, dv, nhidden, pdrop) for _ in range(nlayers))
        self.ln = LayerNorm(dmodel)

    def forward(self, values, mask=None):
        for layer in self.layers:
            values = layer(values, mask)
        values = self.ln(values)
        return values

class GPT(torch.nn.Module):
    """
        Implements a version of GPT-2

        Parameters:
        -----------
        nvocab: int
            Size of the model vocabulary
        nctx: int
            Maximum size of context window
        dembed: int
            Size of embedding dimension
        nlayers: int
            Number of encoder layers
        dk: Optional[int]
            Dimension of key projection. If None equals dembed.
        dv: Optional[int]
            Dimension of value projection. If None equals dembed
        nhidden: int
            Number of hidden layers in FNN part of encoder
        pdrop: float
            Dropout probability
    """
    def __init__(self, nvocab, nctx, dembed=64, nlayers=4, nheads=4, dk=None, dv=None, nhidden=None, pdrop=0.1):
        super().__init__()
        self.nvocab = nvocab
        self.ntokens = ntokens
        self.embed = torch.nn.Embedding(nvocab, dembed, padding_idx=0)
        self.pos = torch.nn.Parameter(torch.zeros([nctx, dembed]))
        torch.nn.init.xavier_uniform_(self.pos)
        dk = dk if dk is not None else dembed//nheads
        dv = dv if dv is not None else dembed//nheads
        nhidden = nhidden if nhidden is not None else 4*dembed
        self.decoder = Encoder(nlayers, nheads, dembed, dk, dv, nhidden, pdrop)
        self.projection = torch.nn.Linear(dembed, nvocab)
        self.dropout = torch.nn.Dropout(pdrop)
        self.register_buffer('mask', torch.log(1 - torch.triu(torch.ones(nctx, nctx), diagonal=1)))

    def forward(self, tokens, apply_softmax=False):
        """
            tokens : torch.Tensor
                Tensor of tokens, of type torch.long.
        """
        #tokens : (..., ntokens)
        d = tokens.shape[-1]
        pos = self.pos[:d, :]
        mask = self.mask[:d, :d]
        x = self.dropout(self.embed(tokens) + pos) #(..., ntokens, dembed)
        x = self.decoder(x, mask) #(..., ntokens, dembed)
        x = self.projection(x) #(..., ntokens, dmodel)
        if apply_softmax:
            x = torch.softmax(x, dim=-1) #(..., ntokens, dmodel)
        else:
            x = torch.nn.functional.log_softmax(x, dim=-1) #(..., ntokens, dmodel)
        return x
```

# Conclusion.

In real life, it is very unlikely that you will train a language model from scratch. More realistically, you would pick a pretrained model from a place such as [HuggingFace](https://huggingface.co/) and fine-tune it. Or just use it as it is. Or just use [ChatGPT](https://chat.openai.com/chat). Still, it is useful to know what is going on behind the scene.

Is that the end of history? For GPT-3, which is the base of models such as ChatGPT, it is almost so. The main differences are

- *****Scale*****: I cannot stress this enough. GPT-3 is monstrously big. In the [original paper](https://arxiv.org/pdf/2005.14165.pdf), it has a vocabulary size of $n_{vocab} = 50257$, a context window size of $n_{ctx} = 2048$, and the largest model uses $n_T = 96, d=12288, n_h=96$, giving a total number of parameters on the order of 175 billion (more exactly, I estimated $n_{params} = 175223784799$). Moreover, the training data is monstrous, with the batch size for the largest model being 3.8 milllion items, with total token size in training amounting to 300 billions tokens, taken for instance, from the [entire Wikipedia](https://www.wikipedia.org/) or from a monstrous [web crawling database](https://commoncrawl.org/). Training in such a scale is a large engineering endeavor, and the training costs on the order of [millions of dollars](https://heits.digital/articles/gpt3-overview).
- **********Sparse attention mask:**********. Instead of using the full causal mask, GPT-3 alternates between the full causal mask and the ****banded**** causal mask. This allows for efficient computation of the banded causal mask layers.

And that’s it. You can use GPT-3 in the [OpenAI playground](https://beta.openai.com/playground). Now, next we will understand how to get from GPT-3 to ChatGPT. For this, we will need more background of ******************************************reinforcement learning with human feedback****************************************** (RLHF), which is the objective of the next post.