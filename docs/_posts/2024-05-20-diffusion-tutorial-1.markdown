---
layout: post
title:  "A mathematical tutorial on diffusion models."
date:   2024-07-06 12:00:00 -0300
categories:
---

<h1> Preamble </h1>
In this series, we will give an introduction to diffusion models from a mathematical viewpoint. This post is the first in the series, where we give the very basics of the theory of diffusion models, mainly following [the EDM framework](https://arxiv.org/abs/2206.00364). In the second part, we will build on this theory to implement a diffusion model from scratch, following this modern framework.

<h1> Introduction </h1>

We can state the objective of a generative model can be stated, for unconditional generation, as follows:

> Given some unlabeled data $\mathcal{D} = \\{x\_1, \ldots, x\_D\\}$ sampled from a true distribution $q(x)$ _unknown_ to us, how can we, from $\mathcal{D}$, create a distribution $p(x)$ such that:
>    1. The distribution $p(x)$ is close to $q(x)$ in some relevant sense.
>    2. We can easily sample from $p(x)$.

Notice that we are _not_ looking for any other information from $p(x)$ in principle, except for when it helps us sample from it. In other words, if you are interested in, say, the probability density function of $p(x)$ as a final goal, your problem is different from the one stated above.

In this case of conditional generation, the problem is stated as follows:

> Given some labeled data $\mathcal{D} = \\{(x\_1, y\_1), \ldots, (x\_D, y\_D)\\}$ sampled from a true distribution $q(x,y) = q(x \mid y) q(y)$ _unknown_ to us, how can we, from $\mathcal{D}$, create a family of distributions distribution $p(x \mid y)$ such that:
>    1. The distribution $p(x \mid y)$ is close to $q(x \mid y)$ in some relevant sense for every $y$.
>    2. We can easily sample from $p(x \mid y)$, given some label $y$.

Since "for every $y$" may be too ambitious for a goal, we will substitute it instead for
>    1. The distribution $p(x, y) := p(x \mid y) q(y)$ is close to $q(x, y)$ in some relevant sense.


Notice that, although we are dealing with labeled data here, the problem is conceptually different from supervised learning. In particular, we are not interested in giving a label $y$ for some given $x$, but instead, _given_ $y$, generating samples $x$ such that $y$ is a correct label for $x$. Of course, conditional generation is reduced to unconditional generation when we are dealing with a single null label $y$.

When dealing with neural networks, our task (when considering the more general case of conditional generation) simplifies to:

<ol>
<li> Creating a distribution $p_\theta(x \mid y)$, parameterized by some set $\theta \in \mathbb{R}^M$ of neural network weights, such that we have an algorithm for sampling from $p_\theta(x \mid y)$ when given $\theta$ and a label $y$. </li>

<li> Creating a differentiable loss function $\mathcal{L}(x, y;\theta)$ such that we are led to $p_\theta(x, y) := p_\theta(x \mid y) q(y)$ being approximately equal to when minimizing

$$
L(\theta) := \mathbb{E}_{x \sim q(x)} L(x; \theta)
$$

</li>

<li> Minimizing $L(\theta)$ by stochastic gradient descent using minibatches of $\mathcal{D}$. </li> </ol>

 This is the problem that diffusion models will aim to solve. Diffusion models are not alone in trying to do so, and they are accompanied by techniques such as [Generative Adversarial Networks](https://en.wikipedia.org/wiki/Flow-based_generative_model), [Variational Autoencoders](https://en.wikipedia.org/wiki/Variational_autoencoder), and [Flow-based generative models](https://en.wikipedia.org/wiki/Flow-based_generative_model). It is not the task of this tutorial to explain why diffusion models currently work better than those other models, as I am still confused about this. Therefore, I will instead work on the easier task of just describing diffusion models.

 <h1> Sampling through denoising </h1>

 For now, let us forget about the full problem. Better, let us forget about learning itself. Consider instead the problem of sampling from some distribution $q(x)$. We will assume that $q(x)$ is a probability distribution in $\mathbb{R}^N$, and devise a method of sampling from $q(x)$ which we will call _sampling through denoising_.
 
 Even with access to the probability density function $q(x)$, sampling is not a trivial task, as the [huge](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) [amount of](https://en.wikipedia.org/wiki/Rejection_sampling) [developed](https://en.wikipedia.org/wiki/Importance_sampling) [sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) [techniques](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm), both approximate and exact, can attest. In some senses, our sampling technique will be a particularly ill-suited one, since it will depend on objects whose evaluation will be intractable. However, we will find out that these objects can be _learned_ fairly well by neural networks, and our sampling technique will become very useful in this case.

 <h2> Noised distributions </h2>

For devising our technique, we will first need to define the process of _noising_, in which we create a random variable $X_\sigma$, for $\sigma \geq 0$, through the following process:

1. Sample $X_0 \sim q(x)$.
2. Sample $Z \sim \mathcal{N}(0, I)$, where $\mathcal{N}(0, I)$ is the standard multivariate normal distribution in $\mathbb{R}^N$.
3. Let $X_\sigma := X_0 + \sigma Z$.

This way, the random variable $X_\sigma$ is distributed according to $p(x, \sigma)$, where $p(x; \sigma)$ is a family of probability distributions whose density is given by

$$
p(x; \sigma) := \int_{\mathbb{R}^N} \mathcal{N}(x \mid x_0, \sigma^2 I) q(x_0) dx_0,
$$

where $\mathcal{N}(x \mid x_0, \sigma^2 I)$ is the density of the multivariate normal distribution with mean $x_0$ and covariance $\sigma^2 I$, given by

$$
\mathcal{N}(x \mid x_0, \sigma^2 I) = \left(2 \pi \sigma \right)^{-N/2} \exp \left( -\frac{1}{2 \sigma^2} \norm{x - x_0}^2 \right).
$$

This family of distribution has three fundamental properties. The first two are obvious:

1. $p(x;0) = q(x)$
2. $p(x;\sigma)$ asymptotically converges to $\mathcal{N}(0, \sigma^2 I)$ when $\sigma \to \infty$.

The third key property is more involved, in that it satisfies the _probability flow ODE_, to be defined soon. Before that, we will define the key objects we will be dealing with, which are the _noised score functions_, defined as

$$
s(x, \sigma) := \nabla_x \log p(x;\sigma).
$$

As a spoiler, _these_ are the objects that our neural network will be approximating, since they will be the building blocks of our sampling techniques. To see why, we will take a look at the probability flow ODE.

<h2> The probability flow ODE </h2>

Consider the following differential equation, which we will call (surprise) the probability flow ODE

$$
\frac{d x}{d \sigma} = -\sigma s(x, \sigma),
$$

With $s(x,\sigma)$ defined as above. The probability flow ODE of course defines a deterministic map $x_\sigma = F(x_\tau, \tau, \sigma)$ defined by integrating the above ODE from $\tau$ to $\sigma$, either forward or backward, with initial (or terminal) condition $x_\tau$, resulting in the final value $x_\sigma$.

A major theorem is that this map (under whatever regularity conditions) will have the following property:

> If $X_{\tau}$ is distributed according to $p(x;\tau)$, then $X_\sigma := F(X_{\tau}; \tau, \sigma)$ is distributed according to $p(x;\sigma)$.

We will give a rough demonstration of this result below, but before, let us just see why this is an amazing result. Namely, it gives us a general recipe for sampling a random variable $X_\sigma \sim p(x;\sigma)$, given that we know how to sample from $p(x;\tau)$:

1. Sample $X_\tau$ from $p(x;\tau)$.
2. Solve the probability flow ODE from $\tau$ to $\sigma$ with initial (terminal) condition $X_\tau$, using some numerical method.
3. The solution $X_\sigma$ is sampled from $p(x;\sigma)$.

Why is this a great recipe? Because remember, our original problem is to sample from $q(x) = p(x;0)$. And we _know_ how to (approximately) sample from $p(x;\sigma)$ when $\sigma$ is large, since in this case $p(x;\sigma) \approx \mathcal{N}(0, \sigma^2 I)$, and we know very well how to sample from normal distributions. Therefore, here is a recipe on how to sample from $q(x)$:

1. Sample $X_{\sigma_\max}$ from $\mathcal{N}(0, \sigma_\max^2 I)$, for $\sigma_\max$ large.
2. Solve the probability flow ODE backward from $\sigma_\max$ to $0$, with terminal condition $X_\sigma$, using some numerical method.
3. The solution $X_0$ is sampled from $q(x)$.

We will call this amazing sampling process _denoising_. Well... it would be amazing, _if_ we had access to the noised score functions $s(x, \sigma)$. However, remember that

$$
s(x, \sigma) := \nabla_x \log p(x;\sigma) = \nabla_x \log \int_{\mathbb{R}^N} \mathcal{N}(x \mid x_0,\sigma^2 I) q(x_0) dx_0.
$$

But, we cannot calculate this quantity, because it is a complicated integral depending on $q(x_0)$. Let us face it, calculating this integral is intractable, except if we _maybe_ had some way to sample from $q(x_0)$, which is exactly what we are looking for.

So, why are we studying this method at all? Because of an amazing property of that, although the calculation of the noised score functions $s(x, \sigma)$ are intractable, they can be fairly well _approximated_ by neural networks, through the second key result in which diffusion models depend.

<h2> A sketch of a derivation of the probability flow ODE </h2>

Assume we want to construct a sequence of random variables $X_\sigma$, indexed by $\sigma \geq 0$, with the following properties:

<ul>
<li> $X_\sigma \sim p(x;\sigma)$. </li>
<li> When given some $X_\tau$, $X_\sigma|X_\tau$ is given by evolving _some_ ordinary differential equation

$$
\frac{d x}{d \sigma} = f(x, \sigma),
$$

with initial condition $x(\tau) = X_\tau$.

</li>
</ul>

The second condition above implies that the family of densities $p(x;\sigma)$ should satisfy the [Fokker-Planck equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation)

$$
\pderiv{p(x;\sigma)}{\sigma} = -\nabla_x \cdot \left( f(x, \sigma) p(x;\sigma) \right).
$$

It suffices to show then that $f(x, \sigma) = -\sigma \nabla_x \log p(x;\sigma)$ satisfies the equation above, as follows:

<ol>
<li>
First, we notice that, since $p(x;\sigma) = \int_{\mathbb{R}^N} \mathcal{N}(x \mid x_0, \sigma) q(x_0) dx_0$, then the left side of the Fokker-Planck equation becomes

$$
\int_{\mathbb{R}^N} \pderiv{\mathcal{N}(x \mid x_0, \sigma^2 I)}{\sigma} q(x_0) dx_0
$$

</li>
<li>
Plugging in $f(x, \sigma) = \sigma \nabla_x \log p(x;\sigma)$, and noticing that 

$$
p(x;\sigma) \nabla_x \log p(x;\sigma) = \nabla_x p(x;\sigma),
$$
the right side of the Fokker-Planck equation becomes

$$
\nabla_x \cdot \left( -\sigma \nabla_x p(x;\sigma) \right) = \int_{\mathbb{R}^N} \left( -\sigma \nabla^2_x \mathcal{N}(x \mid x_0, \sigma^2 I) \right) q(x_0) dx_0
$$

</li>
<li>
Since the density $\mathcal{N}(x \mid x_0, \sigma^2 I)$ is given by

$$
\mathcal{N}(x \mid x_0, \sigma^2 I) = \left(2 \pi \sigma \right)^{-N/2} \exp \left( -\frac{1}{2 \sigma^2} \norm{x - x_0}^2 \right),
$$

through straightforward (if somewhat laborious) differentiation, we find that

$$
\pderiv{\mathcal{N}(x \mid x_0, \sigma^2 I)}{\sigma} = \sigma \nabla^2_x \mathcal{N}(x \mid x_0, \sigma^2 I),
$$

thus showing that the RHS and the LHS side of the Fokker-Planck equation are equal if $f(x, \sigma) = -\sigma \nabla_x \log p(x;\sigma)$.
</li>
</ol>


<h1> Approximating the noised score functions </h1>

Good. Now, we need to approximate the score function

$$
s(x, \sigma) = \nabla_x \log p(x; \sigma) = \nabla_x \log \int_{\mathbb{R}^N} \mathcal{N}(x \mid x_0, \sigma^2 I) q(x_0) dx_0.
$$

Let our approximation be a neural network $s_\theta(x, \sigma)$, parameterized by $\theta$. We will first try to do the obvious step: minimize the difference between $s_\theta(x, \sigma)$ and $s(x, \sigma)$, "suitably averaged". Let us see what we would mean by that.

Remember, we are interested in the score function because we want to solve the probability flow ODE

$$
\frac{d x}{d \sigma} = -\sigma s(x, \sigma)
$$

backward with $X_{\sigma_\max} \sim \mathcal{N}(0, \sigma_\max^2 I)$ as terminal condition. In this case, we have that, for each $\sigma$, $X_\sigma \sim p(x; \sigma)$. Therefore, it stands to reason that, for each $\sigma$, we want our solution to be accurate where $p(x; \sigma)$ is concentrated.

Thinking as a regression problem, we then want to minimize the difference between $s_\theta(X_\sigma, \sigma)$ and $s(X_\sigma, \sigma)$ with $X_\sigma \sim p(x; \sigma)$. However, we have that $X_\sigma = X + \sigma Z$, with $X \sim q(x)$, $Z \sim \mathcal{N}(0, I)$, so $X_\sigma \mid X, \sigma \sim \mathcal{N}(X, \sigma^2 I)$. For this to work out, we need also to define a distribution $\lambda(\sigma)$ such that $\sigma \sim p_\sigma(\sigma)$, ideally with the support of $\sigma$ concentrated in $(0, \sigma_{\max})$. Nothing impedes us from adding a loss weight $\lambda^{\text{ideal}}(\sigma)$ for the noise, which will prove useful later.

Finally, we need a way of measuring the distance between $s(x, \sigma)$ and $s_\theta(x, \sigma)$. Since they are both vectors in $\mathbb{R}^N$, the natural way of measuring the distance in the squared Euclidean norm $\norm{s_\theta(x, \sigma) - s(x, \sigma)}^2$. Therefore, we arrive at an ideal loss function for our neural network.

$$
L^{\text{ideal}}(\theta) = \mathbb{E}_{\sigma \sim p_\sigma(\sigma)} \lambda^{\text{ideal}} (\sigma) \mathbb{E}_{X \sim q(x)} \mathbb{E}_{X_\sigma \sim \mathcal{N}(X, \sigma^2 I)} \norm{s_\theta(X_\sigma, \sigma) - s(X_\sigma, \sigma)}^2.
$$

The obvious problem here is that we cannot compute $L^{\text{ideal}}(\theta)$, since we cannot compute $s(x, \sigma)$. However, the second main theorem of diffusion models will come to help us, saying that minimizing $L^{\text{ideal}}(\theta)$ is equivalent to minimizing a much easier loss function.

<h2> The score matching tracking </h2>   

Here is our second main theorem, which will allow us to minimize $L^{\text{ideal}}(\theta)$ without actually computing it.

<blockquote>

We have that $L^{\text{ideal}}(\theta)$ defined as above satisfies

$$
L^{\text{ideal}}(\theta) = L(\theta) + C,
$$

where $L(\theta)$ is given by

$$
\mathbb{E}_{\sigma \sim p_\sigma(\sigma)} \lambda^{\text{ideal}}(\sigma) \mathbb{E}_{X \sim q(x)} \mathbb{E}_{X_\sigma \sim \mathcal{N}(X, \sigma^2 I)} \norm{s_\theta(X_\sigma, \sigma) - \nabla_{X_\sigma} \log p(X_\sigma|X, \sigma)}^2,
$$

and $C$ is a constant independent of $\theta$. Therefore, minimizing $L^{\text{ideal}}(\theta)$ in respect to $\theta$ is equivalent of minimizing $L(\theta)$ in respect to $\theta$.

</blockquote>

Before moving to the (optional) proof, marvel at how much easier this theorem makes our task. Because, unlike $L^{ideal}$, there is no term here that we cannot compute. This is because we have

$$
\nabla_{X_\sigma} \log p(X_\sigma|X, \sigma) = \nabla_{X_\sigma} \log \mathcal{N}(X_\sigma|X, \sigma^2 I) = \frac{X - X_\sigma}{\sigma^2}.
$$

Thus, our loss $L(\theta)$ becomes

$$
\mathbb{E}_{\sigma \sim p_\sigma(\sigma)} \lambda^{\text{ideal}}(\sigma) \mathbb{E}_{X \sim q(x)} \mathbb{E}_{X_\sigma \sim \mathcal{N}(X, \sigma^2 I)} \norm{s_\theta(X_\sigma, \sigma) - \frac{X - X_\sigma}{\sigma^2}}^2.
$$

This suggests a natural reparameterization for $s_\theta(X_\sigma, \sigma)$. We write

$$
s_\theta(X_\sigma, \sigma) = \frac{D_\theta(X_\sigma, \sigma) - X_\sigma}{\sigma^2},
$$

and our loss becomes

$$
\mathbb{E}_{\sigma \sim p_\sigma(\sigma)} \lambda(\sigma) \mathbb{E}_{X \sim q(x)} \mathbb{E}_{X_\sigma \sim \mathcal{N}(X, \sigma^2 I)} \norm{D_\theta(X_\sigma, \sigma) - X}^2,
$$

where we define $\lambda(\sigma) := \sigma^{-2} \lambda^{\text{ideal}}(\sigma)$. Therefore, the interpretation for $D_\theta(X_\sigma, \sigma)$ is clearer: if perfectly trainer, $s_\theta(X_\sigma, \sigma)$ will be the optimal predictor for $X$, in expectation of $(X_\sigma, X, \sigma)$, according to the squared Euclidean norm $\norm{\cdot}^2$. Notice that $D_\theta(X_\sigma, \sigma)$ will only predict well $X$ if $\sigma$ is small, since, for large $\sigma$, there will not be enough information since the sample $X_\sigma$ is too noised. In fact, for $\sigma$ large, we have that $X_\sigma$ approximately follows $\mathcal{N}(0, \sigma^2)$, thus the minimum of $D_\theta(X_\sigma, \sigma)$ will be equal to the minimum of

$$
\mathbb{E}_{X \sim q(x)} \norm{D_\theta(X_\sigma, \sigma) - X}^2,
$$

thus, the optimal prediction will be $D_\theta(X_\sigma, \sigma) = \mathbb{E}_{X \sim q(x)} \left[X\right]$.

<h1> Proof of the score-matching trick </h1>

To prove the validity of the score-matching trick, we state a more general result

<blockquote>
Let $s: \mathbb{R}^n \to \mathbb{R}^n$ be a score function that satisfies [whatever regularity conditions we need for the following calculations to be valid]. Let $q(x_0)$ be a continuous distribution supported on $\mathbb{R}^n$, and $q_\sigma(x_\sigma \mid x_0)$ be a family of conditional distributions, also continuous and supported on $\mathbb{R}^n$, such that

$$
q_\sigma(x_\sigma) = \int_{\mathbb{R}^n} q_\sigma(x_\sigma \mid x_0) q(x_0) dx_0.
$$

Let $L^0[s]$ and $L[s]$ be functionals defined as

$$
L^0[s] := \mathbb{E}_{q_{\sigma}(x_\sigma)} \norm{s(x_\sigma) - \nabla \log q(x_\sigma)}^2 \\
L[s] := \mathbb{E}_{q_0(x_0) q_\sigma(x_\sigma \mid x_0)} \norm{s(x_\sigma) - \nabla \log q(x_\sigma \mid x_0)}^2.
$$

Then $L^0[s] = L[s] + C$, where the constant $C$ does not depend on $s$.

</blockquote>

To prove this result, we expand $L^0[s]$ and $L[s]$ as

$$
L^0[s] = \mathbb{E}_{q_{\sigma}(x_\sigma)} \norm{s(x_\sigma)}^2 \\ + \mathbb{E}_{q_{\sigma}(x_\sigma)} \norm{\nabla \log q(x_\sigma)}^2 \\ + \mathbb{E}_{q_{\sigma}(x_\sigma)} \inner{s(x_\sigma)}{\nabla \log q(x_\sigma)} 
$$

$$
L[s] = \mathbb{E}_{q(x_0) q_{\sigma}(x_\sigma|x_0)} \norm{s(x_\sigma)}^2 \\ + \mathbb{E}_{q(x_0) q_{\sigma}(x_\sigma|x_0)} \norm{\nabla \log q(x_\sigma|x_0)}^2 \\ + \mathbb{E}_{q(x_0) q_{\sigma}(x_\sigma|x_0)} \inner{s(x_\sigma)}{\nabla \log q(x_\sigma)}.
$$

The law of total expectation gives us

$$
\mathbb{E}_{q(x_0) q_{\sigma}(x_\sigma|x_0)} \norm{s(x_\sigma)}^2 = \mathbb{E}_{q_{\sigma}(x_\sigma)} \norm{s(x_\sigma)}^2,
$$

and, should we show that

$$
\mathbb{E}_{q_{\sigma}(x_\sigma)} \inner{s(x_\sigma)}{\nabla \log q(x_\sigma)} = \mathbb{E}_{q(x_0) q_{\sigma}(x_\sigma|x_0)} \inner{s(x_\sigma)}{\nabla \log q(x_\sigma|x_0)},
$$

we find that

$$
L^0[s] - L[s] = \mathbb{E}_{q_{\sigma}(x_\sigma)} \norm{\nabla \log q(x_\sigma)}^2 - \mathbb{E}_{q(x_0) q_{\sigma}(x_\sigma|x_0)} \norm{\nabla \log q(x_\sigma|x_0)}^2,
$$

which does not depend on $s$. Thus, we show the second inequality above, completing the proof. This is done by straightforward calculation, together with the property $f \nabla \log f = \nabla f$:

$$
\mathbb{E}_{q_{\sigma}(x_\sigma)} \inner{s(x_\sigma)}{\nabla \log q(x_\sigma)} = \\
\int \inner{s(x_\sigma)}{\nabla \log q(x_\sigma)} q(x_\sigma) d x_\sigma = \\
\int \inner{s(x_\sigma)}{\nabla q(x_\sigma)} d x_\sigma = \\
\int \inner{s(x_\sigma)}{\nabla \int q(x_\sigma \mid x_0) q(x_0) dx_0} d x_\sigma = \\
\int \int \inner{s(x_\sigma)}{\nabla q(x_\sigma \mid x_0)} q(x_0) dx_\sigma dx_0 = \\
\int \int \inner{s(x_\sigma)}{\nabla \log q(x_\sigma \mid x_0)} q(x_\sigma \mid x_0) q(x_0) dx_\sigma dx_0 = \\
\mathbb{E}_{q(x_0) q_{\sigma}(x_\sigma|x_0)} \inner{s(x_\sigma)}{\nabla \log q(x_\sigma|x_0)}.
$$

<h1> Conclusion </h1>

Now we have all the theory needed to train and sample from a diffusion model. However, we are still missing some heuristic tricks that will make our training much easier, as well as an actual implementation. Just as an example, we will again reparameterize our denoiser $D_\theta(x_\sigma, \sigma)$ to make training much easier. Those are going to be the topics of the second part of this series.

Having said that, notice that, in theory, our problem is complete, the rest is just about improving our training and sampling efficiency.