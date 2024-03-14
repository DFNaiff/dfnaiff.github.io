---
layout: post
title:  "Entropy and the world - Part 1"
date:   2023-09-24 12:00:00 -0300
categories:
---

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

No, forget about classical thermodynamic entropy. We will talk instead about a different object named "entropy". Forget about classical thermodynamics at all, in what follows.

Instead, consider the following problem: say we have $m$ mutually exclusive propositions $i=1, \ldots, m$. We want to assign a probability distribution $p_i$ for each of those $i$ propositions, such that $p = \{p_1, \ldots, p_m\}$ satisfies some constraint. Formally, we want to choose $p$ from a constrained subset $\mathcal{P}$ of the probability distributions over $\Delta^m = \{1, \ldots, m\}$. How do we do that?

The [maximum entropy principle](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy) gives a simple answer: choose the probability distribution $p$ such that $p$ maximizes the _information entropy_

$$
H(p) = -k_B \sum_{i=1}^m p_i \log p_i,
$$

Where $k_B$ is some positive constant. The idea is that $H(p)$ is a measure of uncertainty, such that $H(p)$ is minimized over $\Delta^m$ for $p_i = 1$ for some $i$ and $p_j = 0$ for $j \neq i$, and $H(p)$ is maximized over $\Delta^m$ for the uniform distribution $p_i = 1/m$. Therefore, the argument goes, by maximizing $H(p)$ under constraint $p \in \mathcal{P}$ we are choosing the most "uniform-like", or uninformative, distribution in $\mathcal{P}$.

A more thorough derivation is found in E. T. Jaynes' [treatise on probability](https://www.amazon.com/Probability-Theory-Science-T-Jaynes/dp/0521592712), showing that the information entropy is (up to the constant $k_B$) the _only_ measure of uncertainty $H(p)$, for an arbitrary discrete probability distribution, that satisfies:

1. For fixed $m$, $H_m(p_1, \ldots, p_m)$ is continuous (small changes in the distribution lead to small changes in the uncertainty), and achieves its maximum at $H(1/m, \ldots, 1/m)$.
2. Letting $h(m) = H_m(1/m, \ldots, 1/m)$, $h(m)$ is monotonically increasing (so, a uniform distribution with more elements is more uncertain than one with fewer elements).
3. $H$ is consistent in the following manner: given an ensemble of $n$ uniformly distributed elements that are divided into $k$ boxes with $b_1, ..., b_k$ elements each, the entropy of the whole ensemble should be equal to the sum of the entropy of the system of boxes and the individual entropies of the boxes, each weighted with the probability of being in that particular box (wording taken from [the Wikipedia page on information entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

Maybe in the future, I'll edit this part and put the proof here, but for now, just take my word for it.[1]

# The maximum (information) entropy probability distribution.

Now, consider a particular case from the above: say that we have a random variable $\Theta$ taking values in the states $\theta_1, \ldots, \theta_m$, that we do not know the probability distribution $p_i = P(\Theta = \theta_i)$. Now, we also have $n$ real-valued functions of the states $\\{f_j(\theta; \alpha)\\}\_{j=1}^m$, possibly parameterized by some $\alpha \in \mathbb{R}$, whose importance will be clear later. Again, we want to make a probability assignment $\pmb{p} = \\{p_i\\}_{i=1}^m$. The information that we have is that we know the expected value of $p$ for each of our $n$ variables, such that

$$
\mathbb{E} f_j(\Theta; \alpha) = \sum_{i=1}^m f_j(\theta_i;\alpha) p_i = \bar{f}_j, \quad \forall j=1, \ldots, n.
$$

It is important to notice that $\{\bar{f}_j\}\_j$ are, together with $\alpha$, the independent variables of our problem since they are the results of our measurements. Moreover, the values $\{ f_j(\theta_i, \alpha)\}\_{i, j}$ for us are functions only of $\alpha$, since both the possible states $\theta_i$ are fixed, as well as the real value functions, given $\alpha$. Therefore, we can avoid confusion by writing

$$
f_{ij}(\alpha) := f_j(\theta_i, \alpha).
$$

The principle of maximum entropy dictates that we should choose $\pmb{p}$ such that it is the solution of following the maximization problem:

$$
\max_{\pmb{p}} H(\pmb{p}) = -k_B \sum_{i=1}^m p_i \log p_i, \\
\text{s.t.} \quad \sum_{i=1}^m p_i = 1, \\
\text{s.t.} \quad  \sum_{i=1}^m f_{ij}(\alpha) p_i = \bar{f}_j, \ \forall j=1, \ldots, n.
$$

Notice that we slipped a positive constant $k_B$ in here, but for now, that is just a positive constant, and it will not change our maximization problem. This is a standard constrained optimization problem in $\mathbb{R}^m$, so we can use Lagrange multipliers to turn into the system of equations

$$
-k_b \left(\log p_i + (1 + \lambda_0) - \sum_{j=1}^n \lambda_j f(\theta_i;\alpha)  = 0
\right) \ \forall i=1, \ldots, m\\
\sum_{i=1}^m p_i = 1, \\
\sum_{i=1}^m f_{ij}(\alpha) p_i = \bar{f}_j, \ \forall j=1, \ldots, n,
$$

therefore finding that

$$
p_i = \frac{1}{Z} e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)}; \\
$$

where $Z$ is given by

$$
Z = \sum_{i=1}^m e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)}.
$$

Finally, we find that $\lambda_1, \ldots, \lambda_n$ are found by solving the following system of equations

$$
\sum_{i=1}^m f_{ij}(\alpha) e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)} = \bar{f}_j \sum_{i=1}^m e^{-\sum_{j=1}^n \lambda_j f_{ij}(\alpha)}, \ \forall j=1, \ldots, n.
$$

Of course, solving this system is brutally hard. Still, the fact that there is such a system (if solvable), and remembering that our independent variables are $\{\bar{f}_j\}\_j$ are, together with $\alpha$, we find that


$$
\pmb{p} = \hat{p}(Z, \pmb{\lambda}), \\
Z = \hat{Z}(\pmb{\lambda}, \alpha) = \sum_{i=1}^m e^{-\pmb{\lambda} \cdot \pmb{f}_i}, \\
\pmb{\lambda} = \hat{\lambda}(\pmb{\bar{f}}, \alpha),
$$

where we conveniently define

$$
\pmb{\lambda} := (\lambda_1, \ldots, \lambda_n) \\
\pmb{f}_i := (f_{i1}, \ldots, f_{in}) \\
\pmb{\bar{f}} := (\bar{f}_1, \ldots, \bar{f}_n).
$$

The most important of these functions will be $\hat{Z}$, which we refer as the partition function. An important property of the partition function is that its derivatives are related to the constraints $\bar{f}_j$, and, by differentiating $\hat{Z}$ in respect to $\lambda_j$,

$$
\frac{\partial \hat{Z}}{\partial \lambda_j} = \sum_{i=1}^m -f_{ij} e^{-\pmb{\lambda} \cdot \pmb{f}_i} = \hat{Z} \bar{f}_j \implies \bar{f}_j = -\frac{\partial \log \hat{Z}}{\partial \lambda_j}.
$$

What about the value $H(p)$ found by maximization? Defining $S_I$ as this maximum value, by plugging the value of $p$ back in the formula for $H(p)$, we find that,

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
S_I = \hat{S}_I(\pmb{\bar{f}}, \alpha) =  k_B \log \hat{Z} + \sum_{j=1}^m \frac{\partial \hat{S}_I}{\partial \bar{f}_j} \bar{f}_j.
$$

Now, this is almost Euler relation, except the term $k_B \log \hat{Z}$. If we are to make a comparison with classical thermodynamics, we must handle this term. If we _assume_ that

$$
\log \hat{Z} \propto \alpha,
$$

we can get Euler's relation

$$
\hat{S}_I = \frac{\partial \hat{S}_I}{\partial \alpha} \alpha + \sum_{j=1}^m \frac{\partial \hat{S}_I}{\partial \bar{f}_j} \bar{f}_j,
$$

and the entropy function is extensive, so that

$$
\hat{S}_I(a \pmb{\bar{f}}, a \alpha) = a \hat{S}_I(\pmb{\bar{f}}, \alpha).
$$

Finally, we can get a "first law" of thermodynamics from the theory. Assume we have only one function of the states $f(\cdot, \alpha)$, therefore only one set $\{f_i\}\_i$ and $\bar{f}$. We find that

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
\frac{1}{\lambda} d S_I = d Q_I.
$$

Good. Now, we are ready to go back to the physical.

# Back to the physical

Now, suppose then that what we are measuring is the internal energy and the volume of a macroscopic system. This macroscopic system is associated with a variety of microscopic states $\theta_1, \ldots, \theta_n$. Now, let's assume that "being in state $\theta\_n$" is a random variable $\Theta$. This is reasonable to do because, from a subjective perspective, we cannot know at any one time $\Theta$. However, for _some reason_, that we'll have to discuss below, we can measure the expected value of some functions of $\Theta$. In particular, we can measure the internal energy $U = \mathbb{E}{u(\Theta)}$ and the amount of species $N_j = \mathbb{E} n_j(\Theta)$, for $j \geq 1$. Finally, $V$ is a known variable in our system of interest, playing the role of $\alpha$.

Well, then we get back to our previous case. We want to make our best guess about $\\{p_i = P(\Theta = \theta_i)\\}$. We have $U = \mathbb{E} u(\Theta) = \sum_i u_i p_i$ and $X_j = \mathbb{E} n_j(\Theta) = \sum_i x_{x,j} p_i$. The rest is obvious. We find that

$$
p_i = \frac{1}{Z} e^{-\lambda_0 U - \sum_{j \geq 1} \lambda_j X_j}.
$$

and that, if we equate the thermodynamic entropy $S$ with the information entropy $H$, we then can equate

$$
\lambda_0 = \frac{1}{T}, \ \lambda_{j} = \frac{-P_j}{T},
$$

and find that

$$
S = \frac{1}{T} U - \sum_{j \geq 1} \frac{P_j}{T} X_j + \log Z.
$$

Now, calculating $\log Z$ is brutally difficult. But, if we assume that $\log Z \propto V$, then we can get back to our original relation

$$
S = \frac{1}{T} U + \frac{P}{T} V + \sum_{j \geq 1} \frac{P_j}{T} X_j.
$$

Crucially, Euler's relation cuts both way, and it implies that entropy is extensive over its constituent systems.

Now, why is it that $\log Z \propto V$, in general? And why does this works actually, in nature? This is a topic for the second part of this series.

