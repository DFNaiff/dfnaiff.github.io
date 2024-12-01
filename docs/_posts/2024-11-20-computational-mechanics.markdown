---
layout: post
title:  "Struggling with Computational Mechanics "
date:   2025-09-13 12:00:00 -0300
categories:
---

In the following, I'll base myself mainly on four papers:

1. [Computational Mechanics: Pattern and Prediction, Structure and Simplicity](https://arxiv.org/abs/cond-mat/9907176), which I'll refer to CMPP.
2. [Software in the natural world: A computational approach to hierarchical emergence](https://arxiv.org/abs/2402.09090), which I'll refer to SWNW.
3. [Spectral Simplicity of Apparent Complexity, Part I: The Nondiagonalizable Metadynamics of Prediction](https://arxiv.org/abs/1705.08042), which I'll refer to SSAC1.
4. [Spectral Simplicity of Apparent Complexity, Part II: Exact Complexities and Complexity Spectra](https://arxiv.org/abs/1706.00883), which I'll refer to SSAC2.

# The problem setup

Following CMPP, we consider stochastic processes that are:

1. Discrete-valued, taking values in a set $\mathcal{A}$.
2. Discrete-time, taking values in $\mathbb{Z}$ (that is, the process is bidirectional).

Therefore, our process will be as

$$
\pastfuture{X} \ldots X_{-2} X_{-1} X_0 X_1 X_2 \ldots.
$$

A realization of this process will be given by equivalent lowercase letters

$$
\pastfuture{x} \ldots x_{-2} x_{-1} x_0 x_1 x_2 \ldots.
$$

We will assume that we have access to the distribution of $\pastfuture{X}$, so we are not talking about learning, at least for now.

For some fixed step $t$, we define the _past_ process and the _future_ process as

$$
\past{X_t}^L = X_{t-L} \ldots X_{t-1} \\
\future{X_t}^L = X_{t+1} \ldots X_{t+L-1} \\
\past{X_t} = \ldots X_{t-2} X_{t-1} \\
\future{X_t} = X_{t} X_{t+1} \ldots,
$$

and we use the equivalent lowercase letters for their realizations. If we have $t=0$, we will drop the subscript.

Following CMPP, we will assume that the process is _stationary_, that is $P(\future{X_t}^L = \future{x}^L) = P(\future{X}^L = \future{x}^L)$. Informally, this tells us that we cannot infer at all in which timesteps we are from observing the realization of the random variables. The paper do mention that the theory can be extended to non-stationary processes, but it is not done there, nor it will be done here.

# What is a predictive theory?

If we think about what a theory should do, we conclude that it must predict the future $\future{X}$ using some relevant information of the past $\past{X}$. Ideally, the best way of doing that is using the entire past, so that the best prediction one can get is

$$
P(\future{X} = \future{x}|\past{X} = \past{x}).
$$

The problem is, this is simply too much. For instance, if our process is a Markov process, we just need to extract from the past $x_{-1}$, and we can throw away everything else. What computational mechanics will try to answer is "what do we need to know about the past?". That is, what are the patterns that we can get from the past that are relevant to predict the future?

We can think of a theory as a partition of the possible pasts $\past{x}$, such that $\past{x}_A$ and $\past{x}_B$ that captures some relevant information about the future $\pastfuture{X}$. That is, a theory $\mathcal{R}$ of the process $\pastfuture{X}$ will be a partition of the set $\past{\mathcal{A}}$ of possible past realizations of $\past{X}$. In other words, theory group together pasts, so that we extract the relevant information from it.


Following CMPP, we call the set of all possible such theories (that is, all partitions of $\past{\athcal{A}}$) the _Occam's pool_.

# How to evaluate theories?

There are good and bad theories out there. If we are to make progress, we need a way of evaluating such theories. Information theory is the tool of doing this.

Ideally, we would consider $H(\future{X})$, however, usually this quantity is infinite, so instead we consider finite windows $H(\future{X}^L)$ as a function of the window size $L$. We can then define the entropy rate and conditional entropy rates

$$
h(\future{X}) = \lim_{L \to \infty} \frac{1}{L} H(\future{X}^L) \\
h(\future{X}|Y) = \lim_{L \to \infty} \frac{1}{L} H(\future{X}^L|Y). \\
$$

Notice that, due to stationarity, we can talk about the entropy $H(X) = H(\future{X}^1)$ of a single random variable of the stochastic process. Also, we have that $h(\future{X}) \leq H(X)$, due to basic joint entropy inequality. A part in CMPP that is confusing to me is the following definition (mixing my own language in calling members of the Occam's pool a theory):

>  We say a theory $\mathcal{R}$ of $\pastfuture{X}$ captures a pattern if $H(\future{S}^L|\mathcal{R}) \leq L H(S)$ for some $L$.

However, from my point of view, this is an utterly trivial statement, because $H(\future{S}^L|\mathcal{R}) \leq H(\future{S}^L) \leq L H(\future{S})$, with the later being equal if and only if the $S_1, \ldots, S_L$ are independent. In any case, I'm ignoring this for now. In any case, a more useful identity is the following:

$$
H(\future{S}|\mathcal{R}) \geq H(\future{S}|\past{S}).
$$

Which is obvious, because, remember, the random variable $\mathcal{R}$ 