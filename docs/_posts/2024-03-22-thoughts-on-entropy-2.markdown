---
layout: post
title:  "Entropy and the world - Part 2"
date:   2024-03-18 12:00:00 -0300
categories:
---

[Part 1]({% link _posts/2024-03-13-thoughts-on-entropy-1.markdown %})

# Preamble

So, in the previous post, we concluded that, if we assume the maximum entropy principle, we arrive at classical thermodynamics. This is the basis of statistical mechanics, and nothing new was said there. Yet, a question remains: why? Why does any of this work? In this part, we will make some heuristic arguments for why this works, which will be arguments sort of agnostic on which physical system we are considering. However, we will get to the actual physics of the actual world in the next part.

# Microstates and macrostates, intuitively.

In the previous post, we considered a finite number of states $\theta_1, \ldots, \theta_m$, to which we applied the maximum entropy principle. However, here, we will have to be more careful. In particular, if we are to derive the maximum entropy principle, we will need to consider two different kinds of states in our problem. Also, notice we are now assuming a finite set of states $m$, in contrast to the previous post. This is intentional, and we do so because most of our arguments are going to be counting arguments.

Before moving on to formal definitions, we explain microstates and macrostates intuitively. Imagine a system of $N$ particles with unit mass, labeled each as $i=1, \ldots, N$. Assume that these particles are moving in a three-dimensional space (for instance, a closed box), following standard Newtonian mechanics. The individual state of each particle $i$ are its position $\mathbf{x}_i$ and velocity $\mathbf{v}_i$, so that its state is given by $\theta_i = (\pmb{x}_i, \pmb{v}_i) \in \mathbb{R}^6$. Our microstate, which in Newtonian mechanics we would call just state, is then the position and velocity of each of these $N$ particles, or, in other words, our microstates $\xi$ are vectors $\theta^{(N)} \in \mathbb{R}^{6N}$.

Now, the truth is, we don't care about the state of each of these particles. We only care about questions like "How many of these particles have kinetic energy above some value $u_0$" or "How many of these particles can I find at distance $r$ from position $x_0$?". Equivalently, we care about the probability (density) $p(\theta) \in \mathcal{P}(\theta)$ that I will find some particle at position $(\pmb{x\_i}, \pmb{v}\_i)$. We will call such probability density $q \in \mathcal{P}(\mathbb{R}^6)$. We will call this probability the macrostate of our system.

Notice that the important bit here is that the dimensionality of the random variable we care about is just $\mathbb{R}^{6}$, which is much less than $\mathbb{R}^{6N}$. Sure, care in fact about the space of probability distributions $p \in \mathcal{P}(\mathbb{R}^6)$, which lives in another type of space, so such comparisons cannot be done so easily. However, assume we divided $\mathbb{R}^{6}$ in a set of $M$ boxes, labeled by $m=1, \ldots, M$, such that our individual states $\theta_i$ are given by "$(\pmb{x}_i, \pmb{v}_i)$ is in box $m$". In other words, our individual state space is now a finite set $\[M\] = \{1, \ldots, M}$, and our microstate space is $\[M\]^N$, whose size is given by $M^N$. Crucially, the possible probabilities can only be given by

$$
\{(q_1 = n_1/N, \ldots, q_m = n_m/N); n_j \geq 0, n_1 + \ldots + n_M = N\}
$$.

Therefore, the possible probabilities are isomorphic to the set of [compositions](https://en.wikipedia.org/wiki/Composition_(combinatorics)) of $N$ in $M$ pieces (including 0). That is, each element of this set consists of a set $\pmb{n}(n_1, \ldots, n_m)$ whose value $n_j$ means "number of microstates in individual state $j$". Naming this set as 

$$
C[N, M] := \{\pmb{n} = (n_1, \ldots, n_m) ; n_j \geq 0, n_1 + \ldots + n_M = N\},
$$

we find through some simple combinatorics that the size of this set is given by

$$
\abs{C[N, M]} = \frac{(N + M - 1)!}{N!(M-1)!},
$$

which is smaller than $M^N$. Therefore, _many microstates will correspond to the same microstate_. This is the crucial setting in which we will make our argument.

Just to formalize better, we will then consider a set $\Theta$ of individual states, whose elements are $\theta$, and, for $N$ copies of the state, we will consider the microstate set as $\Theta^N$, whose elements are given by $\theta^{(N)}$. Finally, the macrostate set will be $\mathcal{P}(\Theta)$, with an element $q \in \mathcal{P}(\Theta)$ being the macrostate of our system. If we consider $\Theta = \[M\]$, we will then consider also $C[N, M]$ as described above, which, for simplicity, we will also call the macrostate of our system, with each macrostate being a set $\pmb{n} = (n_1, \ldots, n_M) \in C[N, M]$.

# A counting argument.

With that in mind, we can imagine that we have $N$ copies of our physical system, each labeled $i=1, \ldots, N$, such that each of these systems can be in one of $M$ possible states, labeled $j=1, \ldots, M$. Each individual state is given by $\theta_i \in \[M\]$, and we let the microstate be $\pmb{\theta} \in \[M\]^N$. Therefore, we can ask for each macrostate $\pmb{n} \in C[N, M]$ the following question:

> How many possible set of macrostates $\pmb{\theta}$ of each individual copy are compatible with a fixed $\pmb{n}$?

We refer to this amount as $W(\pmb{n}; N)$. Again, simple combinatorics give us the answer as

$$
W(\pmb{n};N) = \frac{N!}{n_1! \ldots n_M!}.
$$

Now, if we assume $N$ to be _very_ large, each $\pmb{n}$ with $W(\pmb{n}; N)$ will also have $n_j$ large for every $j$. Therefore, defining $S\_I(\mathbf{n};N) := \frac{1}{N} \log W(\pmb{n};N)$, we can deploy Stirling's approximation to find that 

$$
S\_I(\mathbf{n};N) := \frac{1}{N} \log W(\pmb{n};N) = - \sum_{j} \frac{n_j}{N} \log \frac{n_j}{N}.
$$

Therefore, we find that $S\_I(\mathbf{n};N)$ is the information entropy of the distribution $\{q_j = n_j/N}_{j=1}^M$. So, we find that, when maximizing the information entropy for $\{q_j\}$, for $N$ copies of the physical system, we find the macrostate $\pmb{n} = (n_1, \ldots, n_j)$ that is realizable by most microstates. Moreover, for large $N$, this is overwhelmingly larger, because, for any other $\mathbf{n}'$ with $S\_I(\mathbf{n}';N) < S\_I(\mathbf{n}';N)$, we find that

$$
\frac{W(\pmb{n}';N)}{W(\pmb{n};N)} \approx e^{-N \left(S\_I(\mathbf{n};N) - S\_I(\mathbf{n}';N)\right)}
$$

Therefore, one tentative answer to the question "Why is the maximum entropy principle valid?" can be as follows:

> In general, our system is jumping very rapidly between individual states, so that in practice we are always dealing with an ensemble of $N$ individual states. Maximizing the maximum information entropy is equivalent to finding the macrostate that can be realized by most microstates.

This is the answer by Boltzmann and Gibbs, and through this answer, statistical mechanics can be developed and turned into a rich framework. Yet, although the answer works in practice, it does not work in theory, or at least is not complete, as we will see below.

# A probabilistic counterargument.

The problem here lies in the "jumping very rapidly between individual states" part. Let us be more clear about what is happening here.

Assume that our physical system jumps between individual states at discrete times $t=0, \delta, 2 \delta, \ldots$, such that $\delta$ is a characteristic jumping time. At each time $t$, our system has a latent state $\xi_t$ of a possible set of latent states $\Xi$, such that both:

- Our transition dynamic is Markovian in the latent space, so that

$$
\xi_{t+\delta} \sim p_\xi(\xi_{st+\delta} \mid \xi_t),
$$

- And our individual states $\theta_i$ depends only on the latent state $\xi_t$, such that

$$
\theta_t \sim p_{\theta|\xi}(\theta_t|\xi_t).
$$

Now, for the sake of argument, we assume that our system dynamics reaches a stationary distribution $p_{\xi}(x_t)$, inducing a stationary distribution on the individual state $p_\theta(\theta_t)$. Moreover, we assume that we sample our system in long enough intervals $T$ such that $\theta_{t+T}$ is essentially independent of $\theta_t$ (this is just to avoid details involving hidden Markov chains). Therefore, we can assume that, when sampling $N$ individual states in such way, for each $t=1, \ldots, N$, we have that

$$
\theta_t \sim p(\theta); \quad \theta^N_t \sim \prod p(\theta_i) \\
p(\theta=j) = p_j, \quad j=1, \ldots, M.
$$

Therefore, we find that the probability of each macrostate $p(\pmb{n}) = p(n_1, \ldots, n_M)$ is given the multinomial distribution

$$
p(\pmb{n}) = p(n_1, \ldots, n_M) = W(\pmb{n};N) \exp( \sum_j n_j \log p_j).
$$

By applying again the Stirling's approximation, we find that

$$
\log p(\pmb{n}) = \sum_{j=1}^M \frac{n_j}{N} \left(\log \frac{n_j}{N} - \log p_j\right).
$$

Maximizing $\log p(\pmb{n})$ let us find that

$$
n_j \approx N p_j, \quad j=1, \ldots, M
$$

Here we find a problem. In general, $\{N p_j\}$ is _not_ the macrostate with the maximum possible amount of microstates, unless we postulate that $p_i$ is the maximum information entropy distribution. But we fall in a loop because what we wanted to do was to justify this postulate _in the first place_! Therefore, following Boltzmann's argument to its logical conclusion leaves us with circular reasoning. So, although our problem is solved in practice, we still do not have an understanding of why the maximum entropy principle is thus in the actual world. This a problem. In the next part, I'll follow Grandy's approach and get into the depths of quantum mechanics to try to get a glimpse into the problem.