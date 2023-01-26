---
layout: post
title:  "A tutorial on Large Language Models - Interlude on RLHF"
date:   2023-01-26 12:00:00 -0300
categories: jekyll update
---

# Preamble

In part 1, I said that I would get to ChatGPT. Actually, I lied a bit. ChatGPT exact training was not published, yet, just [a general overview](https://openai.com/blog/chatgpt/). However, OpenAI claims that ChatGPT setup is very similar to the one in the [InstructGPT](https://openai.com/blog/instruction-following/) series, so we will follow the [original InstructGPT paper](https://arxiv.org/pdf/2203.02155.pdf). Yet, to understand that, we need to understand reinforcement learning from human feedback (RLHF), so this is the focus of this part.

# Reinforcement learning - A primer.

To understand, reinforcement learning from human feedback, we must understand reinforcement learning. For a better introduction to what I’m writing here, see the [OpenAI tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html). Still, here is a very concise introduction.

## The building blocks.

Consider an agent acting in some environment through a sequence of steps. The environment starts in some state $$s_0 \in \mathcal{S}$$ with probability $$p(s_0)$$, and, at each step $$t$$, the world is in some state $$s_t$$, from which the agent receives an observation $$o_t \in \mathcal{O}$$, and a reward $$r_t$$, that follows a reward function *that we assume is of the format* $$r_t = r(a_t, o_t)$$. The agent then takes some action $$a_t$$, and the environment then transitions from state $$s_t$$ to some other state $$s_{t+1}$$ with some probability $$p(s_{t+1} \mid s_t, a_t)$$, not necessarily known to the agent. The observation $$o_t$$ also depends on the environment’s state as $$s_t \sim p(o_t \mid s_t)$$. This induces an initial observation probability $$p(o_0) = \int p(o_0 \mid s_0) p(s_0) ds_0$$. If $$o_t = s_t$$, the environment is *fully observable*, and if not, the environment is *partially observable*. Notice that this means that, if we have a prior $$p(s)$$ on the state, we also have $$p(o_{t+1} \mid o_t,a_t)$$ by expansion,

$$
p(o_{t+1} \mid o_t,a_t) = \int p(o_{t+1} \mid s_{t+1})p(s_{t+1} \mid a_t,o_t) ds_{t+1} \\
p(s_{t+1} \mid a_t,o_t) = \int p(s_{t+1} \mid a_t,s_t)p(s_t \mid o_t) s_t \\
p(s_t \mid o_t) = \frac{p(o_t \mid s_t) p(s_t)}{\int p(o_t \mid s_t) p(s_t) ds_t}.
$$

The actions that the agent takes follows some (possibly probabillistic) *policy* $$\pi(a \mid o_t)$$, so that $$a_t \sim \pi(a_t \mid o_t)$$. Sometimes, we will use the the subscript $$\theta$$ as in $$\pi_\theta$$, when we use policies that are modeled by neural networks. Now, let our agent start at $$t=0$$, and act in the environment for $$T$$ time steps until termination. We call this a single *episode*. The probability that the agent follows some *trajectory* $$\tau = \{o_0, a_0, o_1, a_1, \ldots, o_T\}$$ is given by

$$
p_{\pi}(\tau) = p(o_0) \prod_{i=1}^T p(o_{t+1} \mid o_t,a_t)\pi(a_t \mid o_t).
$$

Now, this trajectory will be associated with some cumulative reward $$R(\tau)$$, which may be either the cumulative sum for a *finite horizon* of $$T < \infty$$ time steps,

$$
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t,
$$

for some $$\gamma \in (0, 1]$$ (called the discount rate), or, if $$T=\infty$$ and $$\gamma \leq 1$$, the *infinite horizon* cumulative reward

$$
R(\tau) = \sum_{t=0}^\infty \gamma^t r_t.
$$

Notice that this does *not* mean that the episode runs forever, and, in fact, we may just let $$r_t = 0$$ if the episode has ended (however, if the episode runs forever, we must let $$\gamma < 1$$). We also define the future cumulative rewards starting at some time step $$t$$,

$$
R_t(\tau) = \sum_{s=t}^{T-1} r_t, \quad R_t(\tau) = \sum_{s=t}^{\infty} \gamma^s r_t.
$$

Then, for some policy $$\pi$$, we can consider the expected cumulative reward $$J(\pi)$$ given by

$$
J(\pi) = \mathbb{E}_{\tau \sim p_{\pi}(\tau)}[R(\tau)].
$$

Now, we can finally define the *reinforcement learning objective*: to choose a policy $$\pi(a \mid o)$$ that maximize $$J(\pi)$$. That is, we want to find the *optimal policy*

$$
\pi^*(a \mid o) = \operatorname{argmax}_\pi J(\pi).
$$

In practice, we will use some form of gradient descent to minimize $$-J(\theta) = -J(\pi_\theta)$$ for some parameterized policy $$\pi_\theta$$, using an estimate of $$\nabla J(\theta)$$. However, the next section, which actually gets into optimization algorithm, is optional, since to understand RLHF suffices to say there *are* optimization algorithm, that uses minibatch gradient descent over trajectories.

## Value functions.

Before moving on to actually optimizing $$J(\pi)$$, we define some useful objects we can associate with our policy $$\pi(a \mid o)$$. Namely, we have the *value function* $$V_\pi(o)$$, denoting is the expected cumulative reward of following $$\pi$$ starting from $$o$$, is given by

$$
V_{\pi}(o) = \mathbb{E}_{\tau \sim p_\pi(\tau)}[R(\tau) \mid o_0=o],
$$

and the *action-value function* $$Q_{\pi}(o, a)$$, denoting the expected cumulative reward of following $$\pi$$ after starting from $$o$$ and taking action $$a$$.

$$
Q_{\pi}(o,a) = \mathbb{E}_{\tau \sim p_\pi(\tau)}[R(\tau)  \mid o_0=o, a_0=a].
$$

Of course, we have that $$V_\pi(o) = \mathbb{E}_{a \sim p(a \mid o)}[Q_\pi(o,a)]$$. For infinite time horizons, both $$V_\pi(s)$$ and $$Q_\pi(s, a)$$ both follows the *Bellman equations*

$$
V_\pi(s) = \mathbb{E}_{o', a \sim p(o' \mid o,a)\pi(a \mid o)}[r(a, o) + \gamma V_\pi(o')] \\ \quad \\
Q_\pi(o, a) = \mathbb{E}_{o' \sim p(o' \mid o,a)}[r(a, o) + \gamma \mathbb{E}_{a' \sim p(a' \mid o')}[Q_\pi(o',a')]].
$$

This can be seen by expanding the definition of $$V_\pi(o)$$ and $$Q_\pi(o,a)$$, and it basically says that the (action)-value from an (action)-state is the immediate expected reward plus the discounted future value of the expected future state. In finite time horizons, the situation is more complicated, since we also need information on *when* we are in the episode. However, if we incorporate this information in our observations $$o_t$$, our Bellman equations will be valid for finite horizons. A final object we define is the advantage function

$$
A_\pi(s, a) = Q_\pi(s,a) - V_\pi(s)
$$

which measures how better or worse is taking some action $$a$$ at state $$s$$, and *then* following $$\pi$$, compared to following $$\pi$$ already on $$s$$.

## Solving reinforcement learning - offline.

There are two main classes of reinforcement learning strategies: *online* reinforcement learning and *offline* reinforcement learning. To put it simply, in online RL, the agent updates its policy *as it is playing the episode*. So, online RL is what we are talking about when we talk about reinforcement learning in humans and animals. After all, life consists of a single episode.

Still, in machines (and in ChatGPT), we are able to use the *offline* RL paradigm. Namely, assuming the finite-horizon setting, we let the agent play a few episodes $$\tau_i \sim p_{\pi_\theta}(\tau)$$ sampled from the $$\theta$$-parameterized policy, *observe* the rewards $$\left(r^{(i)}_1, \ldots, r^{(i)}_{T} \right)$$, and then update $$J(\theta)$$ using the stochastic approximation

$$
J(\theta) = \mathbb{E}_{\tau \sim p_{\pi}(\tau)}[R(\tau)] \approx \frac{1}{N} \sum_{i=1}^N R(\tau_i).
$$

Now, from the format show above, it is not clear how we take the gradient $$\nabla J(\theta)$$ here. After all, the only $$\theta$$-dependent parameter are the samples $$\tau_i$$, which cannot directly differentiated. However, a really nice trick let us do exactly that. The derivation can be found [here](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html), as well as more details, but the end result is that

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim p_{\pi_\theta}(\tau)}\left[ \sum_{i=1}^{T} R(\tau) \nabla_\theta \log \pi_\theta(a_t \mid o_t)\right],
$$

yielding the approximation

$$
\nabla J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left[ \sum_{t=1}^{T} R(\tau_i) \nabla_\theta \log \pi_\theta(a^{(i)}_t \mid o^{(i)}_t)\right].
$$

In practice, it can be show that $$R(\tau_i)$$ can be substituted for $$R_t(\tau_i)$$, which itself can be substituted by $$R_t(\tau_i) - b(s_t)$$, where $$b(s_t)$$ is some function dependent on the state only. We can let $$b(s_t)$$ be a neural network approximation $$V_\phi(s_t)$$ of the true value function $$V_{\pi_\theta}(s_t)$$, given by (approximately) minimizing the residual

$$
V_\phi(s_t) = \argmin_\phi \sum_{i=1}^N \sum_{t=1}^T (V_\phi(s^{(i)}_t - R_t(\tau_i))^2
$$

using stochastic gradient descent. All those substitutions make the RL optimization more stable in practice. A commented example of the algorithm in practice, on the CartPole, can be found in [this notebook](https://colab.research.google.com/drive/1lz_FFlWFOexvU_LY80O5qqJGllPE_Ryo?usp=sharing) that I made a while ago, for other reasons.

The offline RL zoo is vast, and in InstructGPT an algorithm called *proximal policy optimization* (PPO) is instead used, whose details can be found [here](https://spinningup.openai.com/en/latest/algorithms/ppo.html). The point, however, is less about what specific RL algorithm is being used, and instead, understand what is being solved. This is fundamental to move to the next step, *reinforcement learning with human feedback*.

# From RL to RLHF.

Now that we have our background in reinforcement learning, we can actually move to reinforcement learning from human feedback. The fundamental insight here is the agent does not *have access to the rewards $$r_t$$ anymore. Instead, he must have a *model* of the reward function, that will come from feedback by humans. In this section, we mainly follow [Christiano et al.](https://arxiv.org/pdf/1706.03741.pdf) in explaining RLHF.

Consider that the agent has a *model of* a reward function $$\hat{r}(a, o)$$, created from feedback by humans. Sometimes we use the subscript $$\phi$$ as in $$\hat{r}_\phi$$ to make it clear that a neural network models the reward function. We will not say that this models the *true* human reward function because that is a much more fuzzy concept. Instead, $$\hat{r}$$ is just a reward function that fits preferences, as we will see below.

Consider two trajectories of the agent $$\tau, \tau'$$, done in two different episodes. We say that a human says that $$\tau \succ \tau'$$ if the human prefers trajectory $$\tau$$ over trajectory $$\tau'$$. We model the probability $$p(\tau \succ \tau' \mid \hat{r})$$ as

$$
p(\tau \succ \tau' \mid \hat{r}) = \sigma(\hat{R}(\tau) - \hat{R}(\tau')) \\
\quad \\
\sigma = \frac{e^x}{e^x + 1}.
$$

Similarly, we assume that $$p(\tau \prec \tau^\prime \mid \hat{r}) = 1 - p(\tau \succ \tau^\prime \mid \hat{r})$$, that is, we do not model indifference between trajectories neither non-comparability. Now, assume that we have a dataset of ordered pairs of trajectories $$\mathcal{D} = \{(\tau_i, \tau_i^\prime)\}_i$$ such that, for each $$i$$, some human judged $$\tau_i \succ \tau_i^\prime$$. Again, if the human has judged indifference or non-comparability between trajectories, we exclude that judgment from our dataset. Then, our loss function for $$\hat{r} = \hat{r}_\phi$$ is given by

$$
l(\phi;\mathcal{D}) = \mathbb{E}_{(\tau, \tau^\prime) \sim \mathcal{D}}[-\log p(\tau \succ \tau' \mid \phi)]
$$

The loss is then minimized using minibatch gradient descent, and the resulting learned reward function $$\hat{r}_\phi$$ is used to adjust the agent policy $$\pi_\theta$$ minimizing $$J(\theta;\phi)$$. So, we have a cyclic process, starting from some policies:

1. Sample some trajectories $$\mathcal{D}^u = \{\tau^u_k\}_k$$ from current policy $$\pi_\theta$$ acting in the environment. Here, the superscript $$u$$ is used to make it clear that those trajectories are unlabeled.
2. Get some humans to use $$\mathcal{D}^u$$ to produce trajectory preference pairs $$\mathcal{D} = \{(\tau_i, \tau_i’)\}_i$$.
3. Learn reward function $$\hat{r}_\phi$$ by minimizing the loss $$l(\phi;\mathcal{D})$$ using minibatch gradient descent.
4. Use the learned reward function $$\hat{r}_\phi$$ to optimize the policy $$\pi_\theta$$ by maximizing $$J(\pi;\hat{r}_\phi)$$ using some reinforcement learning algorithm.
5. Repeat from 1

In the next post, we will use this idea to actually explain ChatGPT training in detail, starting from the base GPT model.
