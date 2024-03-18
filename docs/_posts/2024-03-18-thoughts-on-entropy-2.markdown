---
layout: post
title:  "Entropy and the world - Part 2"
date:   2024-04-18 12:00:00 -0300
categories:
---


# Preamble


So, in the previous post, we concluded that, if we assume the maximum entropy principle, we arrive at classical thermodynamics. This is the basis of statistical mechanics, and nothing new was said there. Yet, a question remains: why?

# The Boltzmann argument

First, we consider a particular case, which we are going to explore in depth.

# The quantum mechanical approach: states and measurement.

To explore fully the above argument, we will make use of some basic quantum mechanics. Although this may seem like it makes the problem much harder in principle, we will find that the quantum mechanical framework not only is appropriate because it is true but because it ends up making things much easier.

As a primer, we review the [basic framework](https://en.wikipedia.org/wiki/Mathematical_formulation_of_quantum_mechanics#Postulates_of_quantum_mechanics) of quantum mechanics, for a single isolated physical system. First, we have to define what are the systems we are describing. In quantum mechanics, we have that each isolated system is associated with a separable Hilbert space $\mathbb{H}$ with inner product $\braket{\phi}{\psi}$, or, in a very informal language, to a vector space $\mathbb{C}^D$ with $D$ being possibly _very_ large, that is, infinite. At each time $t$, the physical system is at a state $\ket{\psi} \in \mathbb{H}$. We assume those states to be unit vectors, that is, $\braket{\psi}{\psi} = 1$.

Physical systems are to be measured. In quantum mechanics, we assume that each measurable physical quantity is associated with a Hermitian operator $A$. To avoid using the full apparatus of [spectral theory](https://en.wikipedia.org/wiki/Spectral_theorem), of which I only know the basics, we treat $A$ as we would in the finite-dimensional case (in other words, we assume $A$ to be a [compact Hermitian operator](https://en.wikipedia.org/wiki/Compact_operator_on_Hilbert_space)). Therefore, we can use the spectral theorem to postulate that $A$ can be uniquely decomposed as

$$
A = \sum_i \lambda_i P_i,
$$

where $\lambda_1, \lambda_2, \ldots$ are distinct eigenvalues of $A$, and $P_1, P_2, \ldots$ are projections onto a set of orthogonal subspaces $S_1, S_2, \ldots$ that span $A$. The reason why we are directly talking about projections, instead of eigenvalues, is that the fact that the spectra can be degenerated is crucial in what is to follow.

With the above decomposition, we can describe the measurement of a system as follows. When we make a measurement $A$ on the physical system in state $\ket{\psi}$, two things happens:

1. The result of our measurement is given by one of the eigenvalues $\lambda_1, \lambda_2, \ldots$, with the probability of measuring each eigenvalue $\lambda_i$ being given by $p_i = \bra{\psi} P_i \ket{\psi}$.
2. Having the measurement result $\lambda_i$, our system immediately transitions (collapses) from state $\ket{\psi}$ to the normalized state $\frac{P_i \ket{\psi}}{\bra{\psi} P_i \ket{\psi}}$. That is, our system is projected to the subspace $S_i$ and then normalized.

The above axiom gives rise to the (main problem)[https://en.wikipedia.org/wiki/Measurement_problem] of (interpreting quantum mechanics)[https://en.wikipedia.org/wiki/Interpretations_of_quantum_mechanics], but, for now, we will remain agnostic on what measurement _really_ means, and just accept that this is how the world works.

Importantly, we can calculate the expected value $\mean{A}$ of our measurement $A$ (of the physical system in state $\ket{\psi}$) as

$$
\mean{A} = \sum_i \lambda_i \bra{\psi} P_i \ket{\psi} = \bra{\psi} \sum_i \lambda_i P_i \ket{\psi} = \bra{\psi} A \ket{\psi}.
$$

Also, the measurement axiom makes clear why we want state vectors to be normalized, given that we want our probabilities to sum to one

$$
1 = \sum_i p_i = \sum_i \bra{\psi} P_i \ket{\psi} = \bra{\psi} \sum_i P_i \ket{\psi}.
$$

Also, the projection themselves $P_i$ can be thought of as measurable physical quantities, since they are Hermitian operators. In particular, we can think of $P_i$ as answering the question "Has the system collapsed to the subspace $S_i$?" with the value of 1 if the answer is "yes", and 0 if the answer is no. The expected value here is exactly $\bra{\psi}|P_i|\ket{\psi}$, which measures "how much" was state vector $\bra{\psi}$ overlapping with subspace $S_i$ to begin with.

We will not get right now to the last axiom, which concerns the time evolution of a physical system. This is because, for now, we will consider our system as static in time, bar a measurement. Of course, a later treatment will need to consider this time evolution, which is why [Grandy's book](https://www.amazon.com/Entropy-Evolution-Macroscopic-International-Monographs/dp/0199546177/)] is called "The Entropy and _Time Evolution_ of Macroscopic Systems".

# Mixed systems


