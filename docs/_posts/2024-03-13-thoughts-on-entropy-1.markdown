---
layout: post
title:  "Entropy and the world - Part 1"
date:   2024-03-13 12:00:00 -0300
categories:
---

# Updates

2024-03-15 - Understood a bit more of the theory, in particular the relation with the volume. The text reflects that chance. Also, the first version of this text had the final part incomplete, reflecting that lack of previous understanding.

# Preamble

One thing that has been on my mind, on and off, for a long time, is thermodynamics. It may be that, for a while, thermodynamics just seemed _mysterious_, with a bunch of heuristics that did not make much sense, and lots of talks about engines, cycles and so on. Recently, I delved more deeply into the axiomatic approach given by [Callen's book](https://www.amazon.com/Thermodynamics-Intro-Thermostat-2E-Clo/dp/0471862568), and suddenly classical thermodynamics made sense to me. Sure, entropy was still a mysterious object, but it was a _well-behaved_ mysterious object, following certain laws, just like force or mass. The laws of thermodynamics made sense, and we had a complete theory. I could grasp classical thermodynamics.

Still, there was the classical and statistical thermodynamics correspondence. I was now curious about how this correspondence worked and if I could get by myself some nice insight. Also, there seems to be a deeper topic here. The connection between the microscopic world and the macroscopic world passes through statistical mechanics, it seems. So, in a poorly formed way, part of the mystery of how the world is what it is seems to pass through that connection.


After some research and some thoughts, inspired mainly by [Walter Grandy's book](https://www.amazon.com/Entropy-Evolution-Macroscopic-International-Monographs/dp/0199546177/), and some Wikipedia pages, I could at least form some inner thoughts on that. And, well, entropy did not become _less_ mysterious, but it is now mysterious in a different way. In a sense, classical thermodynamics is just the maximum entropy principle, a mathematical subject, and not only a physical subject _per se_. I got excited, and decided to give my thoughts on the subject, because why not? That is why I have a website, after all.

# Classical thermodynamics

In Callen's, classical thermodynamics is formulated through a set of postulates, for macroscopic systems. They are as follows:

> *Postulate I*. There exist particular states (called equilibrium states) of
systems that, macroscopically, are characterized completely by the
internal energy $U$, and a set of extensive parameters $X_1, \ldots, X_n$ to be later specifically enumerated.

Which of these extensive parameters is to be used depends on the nature of our system. In practice, we'll focus on simple systems, defined as 

> Systems that are macroscopically homogeneous, isotropic, and uncharged,
that are large enough so that surface effects can be neglected, and that are
not acted on by electric, magnetic, or gravitational fields.

In simple systems, we can consider as extensive parameters only the volume $V$ and the number of specific species $N_1, \ldots, N_r$ constituents of our system. Philosophically, the most important thing about this postulate is that there _is_ an internal energy variable $U$ at all, which is not obvious at all. The second and third postulates will introduce our second key player in thermodynamics, entropy.

> *Postulate II*. There exists a function (called the entropy) of the extensive
parameters, defined for all equilibrium states, and having the following
property. The values assumed by the extensive parameters in the absence of a
constraint are those that maximize the entropy over the manifold of con-
strained equilibrium states.

> *Postulate III*. The entropy of a composite system is additive over the
constituent subsystems (whence the entropy of each constituent system 1s a
homogeneous first-order function of the extensive parameters). The entropy is
continuous and differentiable and is a monotonically increasing function of
the internal energy.

Notice that these postulates are not in the game of explaining what entropy _is_, just that there is something called entropy that is maximized, and has the properties described in the third postulate. Therefore, classical thermodynamics is a phenomenological theory, that has no business in interpreting the phenomena it describes, but just describing the actual phenomena.

Focusing on simple systems, we denote the entropy function as

$$
S = \hat{S}(U, V, N_1, \ldots, N_r),
$$

and the second postulate implies that this function is always to be maximized, under certain constraints. In particular, if we assume that the universe evolves by the removal of constraints specified by its initial conditions, this postulate _is_ the second law of thermodynamics. The third postulate implies some properties of the entropy. A fundamental one is that we can invert $\hat{S}$ with relation to $U$, therefore arriving at the energy formulation

$$
U = \hat{U}(S, V, N_1, \ldots, N_r).
$$

By taking the differential of $U$, we arrive at

$$
dU = \frac{\partial{\hat{U}}}{\partial S} dS + \frac{\partial{\hat{U}}}{\partial V} dV + \sum_{j=1}^r \frac{\partial{\hat{U}}}{\partial N_j} dN_j.
$$

Now, we _define_ the temperature $T$, pressure $P$ and chemical potentials $\\{\mu_j\\}$ as

$$
T := \frac{\partial{\hat{U}}}{\partial S} \\
P := -\frac{\partial{\hat{U}}}{\partial V} \\
\mu_j := -\frac{\partial{\hat{U}}}{\partial N_j},
$$

thus arriving at the first law of thermodynamics (where $dQ = T dS$ and $dW = P dV$)

$$
dU = T dS - P dV + \sum_{j=1}^r \mu dN_j
$$


Of course, we must show that our definition of temperature and pressure corresponds to what we know to be temperature and pressure, but that _can_ be shown (just check the first few chapters of _Callen_!). Given these definitions in terms of $\hat{U}$, we can define the same variables in term of the $\hat{S}$ (using some manipulations of partial derivatives) as

$$
\frac{1}{T} = \frac{\partial \hat{S}}{\partial U} \\
\frac{P}{T} = \frac{\partial \hat{S}}{\partial V} \\
\frac{\mu_j}{T} = -\frac{\partial \hat{S}}{\partial N_j}
$$

and write the differential of $S$ as 

$$
dS = \frac{1}{T} dU + \frac{P}{T} dV \sum_{j=1}^r \frac{\mu_j}{T} d N_j.
$$

Another implication of the third postulate, in particular of additivity, is that the entropy function (and the energy function) obeys the _Euler relation_, so that we can write

$$
\hat{S}(U, V, N_1, \ldots, N_r) = \frac{\partial{\hat{S}}}{\partial U} U + \frac{\partial{\hat{S}}}{\partial V} V + \sum_{j=1}^r \frac{\partial{\hat{S}}}{\partial N_j} N_j = \frac{1}{T} U + \frac{P}{T} V - \sum_{r=1}^n \frac{\mu_j}{T} N_j.
$$

There is also a fourth postulate, associated with the third law of thermodynamics, but we don't need to concern ourselves with it in what is to follow. The important thing is that, with just these three postulates, some careful consideration of our systems of interest, and deploying the mathematical tool of _Legendre transformations_, we have a complete and well-defined theory of classical thermodynamics, as shown in Callen's book.

# The maximum information entropy principle.

No, forget about classical thermodynamic entropy. Forget that I've ever written the first part. Pretend that this is a text on probability, and was always thus. We instead consider the following problem. Say we have a random variable $\Theta$ taking values in $\theta_1, \theta_2, \ldots$, with some probability

$$
\pmb{p}^{(\text{true})} = ( p_i^{(\text{true}}); \ p_i = P(\Theta = \theta_i).
$$

Suppose that we do not know \pmb{p}^{(\text{true})}, although we may know some constraints on \pmb{p}^{(\text{true})}. We want to then estimate some probability $\pmb{p} \\{p_i\\}$ such that $\pmb{p}$ is a good estimate for $\pmb{p}^{(\text{true})}$.From a Bayesian epistemological point of view, what even means for $\Theta$ to have a \pmb{p}^{(\text{true})} assigned by nobody is odd, but we will not consider this for now. We bypass this important problem for now and instead just ask

> What is the best way to choose an probability distribution $\pmb{p}$?", possibly under sume contraints on $\pmb{p}$, such as we will treat $\Theta$ as being distributed according to $\pmb{p}$?

The [maximum entropy principle](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy) gives a simple answer: for some positive constant $k_B$, choose the probability distribution $p$ such that $p$ maximizes the _information entropy_

$$
H(\pmb{p}) = -k_B \sum_i p_i \log p_i
$$

under some possible constraints on $\pmb{p}$.

The idea here is that $H(\pmb{p})$ is a measure of the uncertainty of the probability distribution $\pmb{p}$, and you choose the distribution with as much uncertainty as possible. For finite $\theta_i, \ldots, \theta_m$, we can see that $H(\pmb{p})$ is a measure of uncertainty since it is minimized with value $0$ for some distribution $\pmb{p}$ that, for some $\theta_j$, assign probability one to $P(\Theta = \theta_j)$, that is, assign certainty to $\Theta = \theta_j$. Moreover, $H(\pmb{p})$ is maximized if $P(\theta = \theta_i) = 1/m$, that is, the distribution $\pmb{p}$ is as uniform-like, thus uncertain, as possible. Therefore, the argument goes, by maximizing $H(\pmb{p})$ under some constraints, we are choosing the most uninformative distribution as we possibly can.

A more thorough argument for $H(p)$ being the correct measure of uncertainty is found in E. T. Jaynes' [treatise on probability](https://www.amazon.com/Probability-Theory-Science-T-Jaynes/dp/0521592712), showing that the information entropy is (up to the constant $k_B$) the _only_ measure of uncertainty $H(p)$, for an arbitrary discrete probability distribution, that satisfies:

1. For fixed $m$, $H_m(p_1, \ldots, p_m)$ is continuous (small changes in the distribution lead to small changes in the uncertainty), and achieves its maximum at $H(1/m, \ldots, 1/m)$.
2. Letting $h(m) = H_m(1/m, \ldots, 1/m)$, $h(m)$ is monotonically increasing (so, a uniform distribution with more elements is more uncertain than one with fewer elements).
3. $H$ is consistent in the following manner: given an ensemble of $n$ uniformly distributed elements that are divided into $k$ boxes with $b_1, ..., b_k$ elements each, the entropy of the whole ensemble should be equal to the sum of the entropy of the system of boxes and the individual entropies of the boxes, each weighted with the probability of being in that particular box (wording taken from [the Wikipedia page on information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

Since this is true for every $\theta_1, \ldots, \theta_m$ finite, it stands to reason that it should stand for $\theta_1, \theta_2, \ldots$ countably infinite, although we do not have a uniform distribution here, and thus the distribution can only be maximized under some suitable constraint. If we take another shoddy leap, we can even assume that the possible states of $\Theta$ take places in some domain $\Omega$, and by analogy we choose the probability density function $p(\theta)$ for $\Theta$ such that it maximizes

$$
H(p(\theta)) = -\int_\Omega p(\theta) \log p(\theta) d\theta.
$$

A deeper discussion on the maximum entropy principle is postponed since it is part of what I want to understand here by writing these essays.

# The maximum (information) entropy probability distribution.

With the maximum information entropy principle in hand, we will consider the following problem: 

Say that we have a random variable $\Theta$ taking values in the states $\theta_1(\alpha), \theta_2(\alpha), \ldots$, possibly infinite, such that every $\theta_i$ vary continuously in some positive variable $\alpha$, whose value is given to us. We assume $\alpha$ fixed. We will later associate this variable $\alpha$ with a volume in the physical case. Suppose we do not know about the probability distribution $\pmb{p}^{(\text{true})}(\alpha) = p^{(\text{true})}_1(\alpha), p^{(\text{true})}_2(\alpha)$

$$
p^{(\text{true})}_i(\alpha) = P(\Theta = \theta_i(\alpha))
$$

associated with the random variable $\Theta$. However, we have a set of real-value functions $\\{f_j(\theta)\\}_{j=1}^m$ such that we know or have measured the expected value of $f_j(\Theta)$

$$
\bar{f}_j = \mathbb{E}_{\Theta \sim p(\alpha)} f_j(\Theta(\alpha)) = \sum_i f_{ij}(\alpha) \\
f_{ij}(\alpha) := f_j(\theta_i(\alpha))
$$

for every $j=1, \ldots, m$. We want to them make a probability assignment $\pmb{p} \\{p_i\\}_{i=1}$ such that $\pmb{p} \approx \pmb{p}^{(\text{true})}$. That is, we want again to answer the question

> For some $\alpha$, what is the best way to choose an probability distribution $\pmb{p}$?", such as we will treat some random variable $\Theta$, taking values in $\{\theta_i(\alpha)\}$, as being distributed according to $\pmb{p}$, such that $\pmb{p}$ should obey the constraints $\bar{f}\_j = \mathbb{E}\_{\pmb{p}} [f_j(\Theta)]$ associated with our measured values $\bar{f}_j$?.

It is important to notice that, in this maximization problem, $\{\bar{f}_j\}\_j$ are, together with $\alpha$, our fixed variables, since they are the results of our measurements. The principle of maximum entropy dictates that we should choose $\pmb{p}$ such that it is the solution of following the maximization problem:

$$
\pmb{p} = \max_{\pmb{p'}} H(\pmb{p'}) = -k_B \sum_{i} p'_i \log p'_i, \\
\text{s.t.} \quad \sum_{i} p'_i = 1, \\
\text{s.t.} \quad  \sum_{i}^m f_{i}(\alpha) p'_i = \bar{f}_j, \ \forall j=1, \ldots, n.
$$

Crucially, the probability distribution $\pmb{p}$ that maximizes this problem will be a function of $\{\bar{f}_j\}$ and $\alpha$. Here, $k_B$ is just a positive constant, and will not influence our maximization problem. We can use Lagrange multipliers $\lambda_0, \lambda_1, \ldots, \lambda_m$ to arrive at the equivalent system of equations

$$
-k_b \left(\log p_i + (1 + \lambda_0) - \sum_{j=1}^n \lambda_j f(\theta_i;\alpha)  = 0
\right) \ \forall i=1, \ldots, m\\
\sum_i p_i = 1, \\
\sum_i f_{ij}(\alpha) p_i = \bar{f}_j, \ \forall j=1, \ldots, n.
$$

Solving these equations, we find that

$$
p_i := \frac{1}{Z} e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)}; \\
$$

where $Z$ is given by

$$
Z = \sum_i e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)}.
$$

Finally, we find that $\lambda_1, \ldots, \lambda_n$ are found by solving the following system of equations

$$
\sum_i f_{ij}(\alpha) e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)} = \bar{f}_j \sum_i e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)}, \ \forall j=1, \ldots, n.
$$

Of course, solving this system is brutally hard. Still, if we assume the system to be uniquely solvable (we know that it is solvable at all since the Lagrange multipliers exist), we find that

$$
\pmb{p} = \hat{p}(Z, \pmb{\lambda}), \\
Z = \hat{Z}(\pmb{\lambda}, \alpha) = \sum_i e^{-\pmb{\lambda} \cdot \pmb{f}_i}, \\
\pmb{\lambda} = \hat{\lambda}(\pmb{\bar{f}}, \alpha),
$$

where we conveniently define

$$
\pmb{\lambda} := (\lambda_1, \ldots, \lambda_n) \\
\pmb{f}_i := (f_{i1}, \ldots, f_{in}) \\
\pmb{\bar{f}} := (\bar{f}_1, \ldots, \bar{f}_n).
$$

The most important of these functions will be $\hat{Z}$, which we refer to as the partition function. An important property of the partition function is that its derivatives are related to the constraints $\bar{f}_j$, and, by differentiating $\hat{Z}$ in respect to $\lambda_j$,

$$
\frac{\partial \hat{Z}}{\partial \lambda_j} = \sum_i -f_{ij} e^{-\pmb{\lambda} \cdot \pmb{f}_i} = \hat{Z} \bar{f}_j \implies \bar{f}_j = -\frac{\partial \log \hat{Z}}{\partial \lambda_j}.
$$

What about the value $H(p)$ found by the maximization? Defining $S_I$ as this maximum value, by plugging the value of $p$ back in the formula for $H(p)$, we find that,

$$
S_I = \hat{S}_I(\pmb{\bar{f}}, \alpha) := H(\hat{p}(\pmb{\bar{f}}, \alpha)) = k_B \left(\pmb{\lambda} \cdot \pmb{\bar{f}} + \log Z\right).
$$

Now, using the relation between the partial derivatives of $\log \hat{Z}$ and the expected values $\{\bar{f}_j\}$, we find that

$$
\frac{\partial \hat{S}_I}{\partial \bar{f}_j} = k_B \left(\lambda_j + \pmb{\bar{f}} \cdot \frac{\partial \hat{\lambda_j}}{\partial \bar{f}_j} + \frac{\partial \log \hat{Z}}{\partial \lambda_j} \frac{\partial \hat{\lambda_j}}{\partial \bar{f}_j}\right) = k_B \lambda_j \\
\frac{\partial \hat{S}_I}{\partial \alpha} = k_B \left(\pmb{\bar{f}} \cdot \frac{\partial \hat{\lambda_j}}{\partial \alpha} + \frac{\partial \log \hat{Z}}{\partial \lambda_j} \frac{\partial \hat{\lambda_j}}{\partial \bar{f}_j} + \frac{\partial \log \hat{Z}}{\partial \alpha} \right) = k_B \frac{\partial \log \hat{Z}}{\partial \alpha}
$$

Thus we find that

$$
S_I = \hat{S}_I(\pmb{\bar{f}}, \alpha) =  k_B \log \hat{Z} + \sum_{j=1}^m \frac{\partial \hat{S}_I}{\partial \bar{f}_j} \bar{f}_j, \\
d S_I = k_B \frac{\partial \log \hat{Z}}{\partial \alpha} dV + \sum_{j=1}^m \frac{\partial \hat{S}_I}{\partial \bar{f}_j} d \bar{f}_j.
$$

Now, this is almost Euler relation, except the term $k_B \log \hat{Z}$. If we are to make a full comparison with classical thermodynamics, we must handle this term. Before doing that, we use the theory to get a "first law of thermodynamics" for the information entropy. Assume we have only one function of the states $f(\cdot, \alpha)$, therefore only one set $\{f_i\}\_i$ and $\bar{f}$. We find that

$$
\delta \bar{f} = \sum_{i} p_i \delta f_i + \sum_i f_i \delta p_i.
$$

If we define $\delta Q_I := \sum_i f_i \delta p_i$, we find that

$$
\delta \bar{f} - \bar{\delta f} = \delta Q_I.
$$

Moreover, since we have

$$
\delta S_I = -\sum_i \delta (p_i \log p_i) = - \sum_i \delta p_i - \sum_i \log p_i \delta p_i =
\sum_i \lambda f_i \delta p_i + \log Z \delta \sum_i p_i = \sum_i \lambda f_i \delta p_i,
$$

since $\delta \sum_i p_i = \delta 1 = 0$, we find that

$$
\frac{1}{\lambda} d S_I = d Q_I,
$$

and that

$$
\bar{\delta f} = k_B \frac{\log \hat{Z}}{\frac \partial \alpha} dV,
$$

which can alo be directly derived from the properties of $\log Z$ described previously.

Finally, to get to Euler's relation, we will need to _assume_ that

$$
\log \hat{Z} \propto \alpha.
$$

Notice that this assumption may not hold, although the rest of the theory above is valid, we will not get Euler's relation. Yet, assuming the above relation, we will find that,

$$
k \frac{\partial \log \hat{Z}}{\partial \alpha} = \frac{\partial \hat{S}_I}{\partial \alpha} = \text{\cte},
$$

and arrive at the Euler's relation

$$
\hat{S}_I = \frac{\partial \hat{S}_I}{\partial \alpha} \alpha + \sum_{j=1}^m \frac{\partial \hat{S}_I}{\partial \bar{f}_j} \bar{f}_j,
$$

thus finding that the entropy function is extensive, so that

$$
\hat{S}_I(a \pmb{\bar{f}}, a \alpha) = a \hat{S}_I(\pmb{\bar{f}}, \alpha).
$$

That is enough for now. It should be noted that, although we assumed some enumerable set $\theta_1, \theta_2, \ldots$, the same theory also holds if we assume the continuous case of the maximum entropy principle. With that said, we go back to the physical.

# Information entropy and physical entropy

We go back to being a text on thermodynamics. Let us apply the maximum entropy principle to a physical situation. Suppose we are studying some macroscopic system. We know that our system has some volume $V$, and, given $V$, our macroscopic system is associated with some microscopic system $\Theta$, that at any given time takes the value of some microscopic state $\theta_1(V), \theta_2(V), \ldots$. Now, let's assume that "being in state $\theta\_i(V)$" is a random variable $\Theta$. This is reasonable to do because, from a subjective perspective, we cannot know in which state $\Theta$ for a given time. We can think of $\Theta$ as quickly jumping between the states $\theta_1(V), \theta_2(V), \ldots$ with some probability $p^{(true)}$ that we do not have access to.

However, suppose can measure the internal energy $U = \mathbb{E}{u(\Theta)}$, that is, thinking of $\Theta$ as jumping through states, the time-averaged measurement of the energy associated with $\Theta$. Then, we can use the maximum entropy principle. We want to make our best guess about P(\Theta = \theta_i(V)). We are under the constraint that $U = \mathbb{E} u(\Theta)$. By the maximum entropy principle, we find that

$$
p_i = \frac{1}{Z} e^{-\lambda U} \\
S_I = k_B \log Z + \lambda U
d S_I = k_B \frac{\partial Z}{\partial \alpha} dV + \lambda U.
$$

Now, renaming the variables $\lambda$ and $k_B \frac{\partial \log Z}{\partial \alpha}$ as

$$
\lambda = \frac{1}{T} \\
k_B \frac{\partial \log Z}{\partial \alpha} = \frac{P}{T}
$$

and find that

$$
d S_I = \frac{1}{T} dU + \frac{P}{T} dV,
$$

Finally, if we assume that $\log Z \propto V$, we find that

$$
S_I = \frac{1}{T} U + \frac{P}{T} V.
$$

Thus, $S_I$ is an extensive function of the internal energy $U$ and volume $V$, that always takes a maximum under some possible constrainte (since it is the result of a maximiziation). If we assume $\lambda > 0$ for every $U$ and $V$, it is also monotonically increasing function of the internal energy. We arrive at all the axioms of the thermodynamic entropy, so we can equate the thermodynamic entropy $S$ with the information entropy $S_I$.

Assuming that $\lambda > 0$ is the same as assuming that microstates are more unlikely the higher is their energy. As for why $\log Z \propto V$, it can be proven for a variety of systems under very general assumptions, although for some system such that this is untrue, then classical thermodynamics cannot describe this system. Those are not the main problems

# Why does this works at all?

Why this works at all? We arrived at this equation by assuming the maximum information entropy principle. However, the maximum entropy principle is mostly a logical derivation, or some sort of "best practices". There is no principled reason why the true distribution $p^{(true)}$ should _be_ the one we find by the maximum information entropy principle. After all, $p^{(true)}$ is arrived through some physical process on the microstates, and this process should _not care at all for the maximum information entropy principle_. Yet, we go out there, make our measurements, and it seems that it does follows exactly this principle, since we know classical thermodynamics holds.

So, the maximum entropy principle works, but it seem it should not. There is something to be investigated here. This will be the focus of the second part.