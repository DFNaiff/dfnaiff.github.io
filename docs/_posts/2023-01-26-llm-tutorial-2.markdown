---
layout: post
title:  "A technical tutorial on Large Language Models - Interlude on Reinforcement Learning"
date:   2023-01-26 12:00:00 -0300
categories:
---

[Part 1]({% link _posts/2023-01-14-llm-tutorial-1.markdown %})

# Update (22/07 (of 2024))

By looking into the text after a long time, while preparing for giving a class in Reinforcement Learning, I realized that the text is... quite subpar, to be honest. I'm giving a major overhaul to it, so until then, consider this document as sort of deprecated (not the theory, just the usefulness).

# Update (23/06)

I turned this into a general interlude for RL and RLHF, and I decided to write the next part of actually applying this to language models.

# Update (24/05)

This text has been revised and edited by ChatGPT 3.5 to improve grammar and overall structure.

# Preamble

In Part 1, I mentioned that we would delve into ChatGPT. However, I must confess that I slightly misled you. The exact training details of ChatGPT have not been published yet. OpenAI has provided only a [general overview](https://openai.com/blog/chatgpt/). Nonetheless, OpenAI claims that the setup of ChatGPT is highly similar to the one used in the [InstructGPT](https://openai.com/blog/instruction-following/) series. Therefore, we will rely on the information presented in the [original InstructGPT paper](https://arxiv.org/pdf/2203.02155.pdf). However, to comprehend InstructGPT, we first need to understand both supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF). Yet, to do that, we need to understand reinforcement learning, which is the point of this interlude.


# Reinforcement learning - A primer.

To comprehend reinforcement learning from human feedback (RLHF), it is essential to grasp the fundamentals of reinforcement learning. For a more detailed introduction, refer to the [OpenAI tutorial](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html). Here, I provide a concise overview.

## The building blocks.

Consider an agent that interacts with an environment through a sequence of steps. We will model this interaction as follows:

- At each time $t$, the environment (which includes the agent itself) is found in a state $s_t \in \mathcal{S}$, where $\mathcal{S}$ is the set of possible states the environment can be found in.
- Crucially, the agent never has access to the environment itself, but instead, at each time $t$, it receives an observation $o_t \in \mathcal{O}$ from the environment, that follows some distribution $\rho(o_t|s_t)$.
- Given an observation $o_t$, the agent decides to take an action $a_t$ following some policy distribution $\pi(a_t|o_t)$.
- The action $a_t$, together with the state $s_t$ of the environment, influences the next state of environment $s_{t+1}$ with probability $p(s_{t+1}|s_t, a_t)$.

Cru


The environment begins in a state $$s_0 \in \mathcal{S}$$ with probability $$p(s_0)$$. At each step $$t$$, the agent receives an observation $$o_t \in \mathcal{O}$$ and a reward $$r_t$$ based on a reward function of the form $$r_t = r(a_t, o_t)$$. The agent takes an action $$a_t$$, causing the environment to transition from state $$s_t$$ to another state $$s_{t+1}$$ with a probability $$p(s_{t+1} \mid s_t, a_t)$$ (not necessarily known to the agent). The observation $$o_t$$ is also dependent on the environment's state, given by $$s_t \sim p(o_t \mid s_t)$$. If $$o_t = s_t$$, the environment is *fully observable*; otherwise, it is *partially observable*. Notably, if we have a prior $$p(s)$$ on the state, we can expand $$p(o_{t+1} \mid o_t,a_t)$$ as follows:

$$
p(o_{t+1} \mid o_t,a_t) = \int p(o_{t+1} \mid s_{t+1})p(s_{t+1} \mid a_t,o_t) ds_{t+1} \\
p(s_{t+1} \mid a_t,o_t) = \int p(s_{t+1} \mid a_t,s_t)p(s_t \mid o_t) ds_t \\
p(s_t \mid o_t) = \frac{p(o_t \mid s_t) p(s_t)}{\int p(o_t \mid s_t) p(s_t) ds_t}.
$$

The agent's actions are guided by a (possibly probabilistic) *policy* $$\pi(a \mid o_t)$$, where $$a_t \sim \pi(a_t \mid o_t)$$. In the context of neural networks, policies are often denoted as $$\pi_\theta$$, incorporating parameters. Let the agent begin at $$t=0$$ and interact with the environment for $$T$$ time steps until termination, constituting a single *episode*. The probability of the agent following a *trajectory* $$\tau = \{o_0, a_0, o_1, a_1, \ldots, o_T\}$$ is given by

$$
p_{\pi}(\tau) = p(o_0) \prod_{i=1}^T p(o_{t+1} \mid o_t,a_t)\pi(a_t \mid o_t).
$$

A trajectory is associated with a cumulative reward $$R(\tau)$$, which can be the cumulative sum for a *finite horizon* of $$T < \infty$$ time steps:

$$
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t,
$$

where $$\gamma \in (0, 1]$$ is the discount rate. Alternatively, for $$T=\infty$$ and $$\gamma \leq 1$$, the *infinite horizon* cumulative reward is given by

$$
R(\tau) = \sum_{t=0}^\infty \gamma^t r_t.
$$

It is important to note that an episode does not necessarily run indefinitely, and if it does, we must ensure $$\gamma < 1$$. We also define future cumulative rewards starting at time step $$t$$ as follows:

$$
R_t(\tau) = \sum_{s=t}^{T-1} r_t, \quad R_t(\tau) = \sum_{s=t}^{\infty} \gamma^s r_t.
$$

Thus, for a policy $$\pi$$, the expected cumulative reward $$J(\pi)$$ is given by

$$
J(\pi) = \mathbb{E}_{\tau \sim p_{\pi}(\tau)}[R(\tau)].
$$

The *reinforcement learning objective* is to find a policy $$\pi(a \mid o)$$ that maximizes $$J(\pi)$$. In practice, we employ some form of gradient descent to minimize $$-J(\theta) = -J(\pi_\theta)$$ for a parameterized policy $$\pi_\theta$$, using an estimate of $$\nabla J(\theta)$$. The next section, which delves into optimization algorithms, is optional as RLHF can be understood without exploring specific optimization algorithms. These algorithms typically employ minibatch gradient descent over trajectories.

## Value functions.

Before delving into optimizing $$J(\pi)$$, let's define some key components associated with the policy $$\pi(a \mid o)$$. First, we have the *value function* $$V_\pi(o)$$, which represents the expected cumulative reward of following policy $$\pi$$ starting from observation $$o$$. It is given by

$$
V_{\pi}(o) = \mathbb{E}_{\tau \sim p_\pi(\tau)}[R(\tau) \mid o_0=o],
$$

where $$R(\tau)$$ is the cumulative reward obtained from trajectory $$\tau$$. Similarly, the *action-value function* $$Q_{\pi}(o, a)$$ represents the expected cumulative reward of following policy $$\pi$$ after starting from observation $$o$$ and taking action $$a$$:

$$
Q_{\pi}(o,a) = \mathbb{E}_{\tau \sim p_\pi(\tau)}[R(\tau)  \mid o_0=o, a_0=a].
$$

We also have the relationship $$V_\pi(o) = \mathbb{E}_{a \sim p(a \mid o)}[Q_\pi(o,a)]$$. For infinite time horizons, both $$V_\pi(o)$$ and $$Q_\pi(o, a)$$ satisfy the *Bellman equations*:

$$
V_\pi(o) = \mathbb{E}_{o', a \sim p(o' \mid o,a)\pi(a \mid o)}[r(a, o) + \gamma V_\pi(o')], \\
Q_\pi(o, a) = \mathbb{E}_{o' \sim p(o' \mid o,a)}[r(a, o) + \gamma \mathbb{E}_{a' \sim p(a' \mid o')}[Q_\pi(o',a')]],
$$

where $$\gamma$$ is the discount factor. These equations capture the notion that the value or action/value from a state or action is the immediate expected reward plus the discounted future value of the expected future state. In finite time horizons, the situation becomes more complex as we need to consider the timing within the episode. However, by incorporating this information in our observations $$o_t$$, the Bellman equations remain valid for finite horizons. Another important concept is the advantage function:

$$
A_\pi(o, a) = Q_\pi(o,a) - V_\pi(o),
$$

which quantifies the advantage of taking action $$a$$ in state $$o$$ followed by policy $$\pi$$ compared to simply following policy $$\pi$$ from state $$o$$.

## Solving reinforcement learning - offline.

There are two main approaches to reinforcement learning: *online* reinforcement learning and *offline* reinforcement learning. In online RL, the agent updates its policy during the episode, similar to how humans and animals learn. However, in machine learning settings like ChatGPT, we typically employ *offline* RL. In offline RL, the agent plays multiple episodes $$\tau_i \sim p_{\pi_\theta}(\tau)$$ using the current policy $$\pi_\theta$$, observes the rewards $$\left(r^{(i)}_1, \ldots, r^{(i)}_{T} \right)$$, and then updates $$J(\theta)$$ using the stochastic approximation:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_{\pi}(\tau)}[R(\tau)] \approx \frac{1}{N} \sum_{i=1}^N R(\tau_i).
$$

To compute the gradient $$\nabla J(\theta)$$, we encounter a challenge because the trajectories $$\tau_i$$ are not directly differentiable with respect to the policy parameters $$\theta$$. However, a clever trick allows us to overcome this issue. The derivation and more details can be found [here](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html), but the key result is:

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim p_{\pi_\theta}(\tau)}\left[ \sum_{t=1}^{T} R(\tau) \nabla_\theta \log \pi_\theta(a_t \mid o_t)\right],
$$

which leads to the following approximation:

$$
\nabla J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left[ \sum_{t=1}^{T} R(\tau_i) \nabla_\theta \log \pi_\theta(a^{(i)}_t \mid o^{(i)}_t)\right].
$$

In practice, we can substitute $$R(\tau_i)$$ with $$R_t(\tau_i)$$ and further replace $$R_t(\tau_i)$$ with $$R_t(\tau_i) - b(s_t)$$, where $$b(s_t)$$ is a function that only depends on the state. We can approximate $$b(s_t)$$ using a neural network $$V_\phi(s_t)$$, which estimates the true value function $$V_{\pi_\theta}(s_t)$$. The estimation is done by minimizing the following residual:

$$
V_\phi(s_t) = \operatorname{argmin}_\phi \sum_{i=1}^N \sum_{t=1}^T (V_\phi(s^{(i)}_t) - R_t(\tau_i))^2
$$

using stochastic gradient descent. These substitutions improve the stability of RL optimization. An annotated example of this algorithm applied to the CartPole environment can be found in [this notebook](https://colab.research.google.com/drive/1lz_FFlWFOexvU_LY80O5qqJGllPE_Ryo?usp=sharing) I created earlier for other purposes.

The field of offline RL encompasses various algorithms, and in the case of InstructGPT, an algorithm called *proximal policy optimization* (PPO) is used. More details about PPO can be found [here](https://spinningup.openai.com/en/latest/algorithms/ppo.html). However, the important point is not the specific RL algorithm being used, but rather understanding the underlying problem being addressed. This understanding will serve as the foundation for the next step: *reinforcement learning from human feedback*.

# From RL to RLHF.

Having gained an understanding of reinforcement learning, we can now explore reinforcement learning from human feedback (RLHF). The key insight of RLHF is that the agent no longer has access to the true reward signal $$r_t$$; instead, it must rely on a *model* of the reward function, which is obtained through feedback from humans. In this section, we will follow the explanation of RLHF provided by [Christiano et al.](https://arxiv.org/pdf/1706.03741.pdf).

Let's consider that the agent has a model of the reward function $$\hat{r}(a, o)$$, which is created based on human feedback. Sometimes, we use the subscript $$\phi$$ (e.g., $$\hat{r}_\phi$$) to indicate that a neural network models the reward function. It's important to note that $$\hat{r}$$ represents a reward function that fits human preferences, rather than the true underlying human reward function, which can be more subjective. 

We define the probability of a human preferring trajectory $$\tau$$ over trajectory $$\tau'$$ as:

$$
p(\tau \succ \tau' \mid \hat{r}) = \sigma(\hat{R}(\tau) - \hat{R}(\tau')),
$$

where $$\sigma$$ is the sigmoid function. Similarly, we assume that $$p(\tau \prec \tau' \mid \hat{r}) = 1 - p(\tau \succ \tau' \mid \hat{r})$$, meaning that we do not model indifference or non-comparability between trajectories. 

Suppose we have a dataset of ordered pairs of trajectories $$\mathcal{D} = \{(\tau_i, \tau_i^\prime)\}_i$$, where each pair $$(\tau_i, \tau_i^\prime)$$ represents a preference judgment made by a human (e.g., $$\tau_i \succ \tau_i^\prime$$). We construct our loss function for $$\hat{r} = \hat{r}_\phi$$ as:

$$
l(\phi;\mathcal{D}) = \mathbb{E}_{(\tau, \tau^\prime) \sim \mathcal{D}}[-\log p(\tau \succ \tau' \mid \phi)].
$$

This loss is minimized using minibatch gradient descent, and the resulting learned reward function $$\hat{r}_\phi$$ is used to update the agent's policy $$\pi_\theta$$ by maximizing $$J(\theta;\phi)$$. This creates a cyclic process involving multiple steps:

1. Sample trajectories $$\mathcal{D}^u = \{\tau^u_k\}_k$$ from the current policy $$\pi_\theta$$.
2. Collect human preferences based on $$\mathcal{D}^u$$ to obtain trajectory preference pairs $$\mathcal{D} = \{(\tau_i, \tau_i’)\}_i$$.
3. Learn the reward function $$\hat{r}_\phi$$ by minimizing the loss $$l(\phi;\mathcal{D})$$ using minibatch gradient descent.
4. Optimize the policy $$\pi_\theta$$ using the learned reward function $$\hat{r}_\phi$$ through a reinforcement learning algorithm that maximizes $$J(\pi;\hat{r}_\phi)$$.
5. Repeat from step 1.

Next, we will use that general framework to explain instruct-based language models, and how they are crafted from base models.