---
layout: post
title:  "Entropy and the world - Part 3: Interlude on QM."
date:   2024-04-10 12:00:00 -0300
categories:
---


# Preamble

Previously, we justified the maximum entropy principle being an actual physical principle for "most physics", when we consider coarse measurements in time and space. However, "most physics" is not enough. After all, in the previous discussion, we had to assume a dynamical law with fluctuations. This is an approximation of the actual world, and, to get deeply into how entropy works in this world here, we will need to interact with the actual physics of the world.

# The quantum mechanical approach: states and measurement.

To continue our exploration of entropy in physical systems, we will have to move to quantum mechanics. Although this may seem like it makes the problem much harder in principle, we will find that the quantum mechanical framework not only is appropriate because it is true but because it ends up making things much easier.

As a primer, we review the [basic framework](https://en.wikipedia.org/wiki/Mathematical_formulation_of_quantum_mechanics#Postulates_of_quantum_mechanics) of quantum mechanics, for a single isolated physical system. First, we have to define what are the systems we are describing. In quantum mechanics, we have that each isolated system is associated with a separable Hilbert space $\mathbb{H}$ with inner product $\braket{\phi}{\psi}$, or, in a very informal language, to a vector space $\mathbb{C}^D$ with $D$ being possibly _very_ large, that is, infinite. At each time $t$, the physical system is at a state $\ket{\psi} \in \mathbb{H}$. We assume those states to be unit vectors, that is, $\braket{\psi}{\psi} = 1$.

Physical systems are to be measured. In quantum mechanics, we assume that each measurable physical quantity is associated with a Hermitian operator $A$. To avoid using the full apparatus of [spectral theory](https://en.wikipedia.org/wiki/Spectral_theorem), of which I only know the basics, we treat $A$ as we would in the finite-dimensional case (in other words, we assume $A$ to be a [compact Hermitian operator](https://en.wikipedia.org/wiki/Compact_operator_on_Hilbert_space)). Therefore, we can use the spectral theorem to postulate that $A$ can be uniquely decomposed as

$$
A = \sum_i \lambda_i P_i,
$$

where $\lambda_1, \lambda_2, \ldots$ are distinct eigenvalues of $A$, and $P_1, P_2, \ldots$ are projections onto a set of orthogonal subspaces spanning $A$. We abuse our notation and also denote these spaces as $P_1, P_2, \ldots$. The reason why we are directly talking about projections, instead of eigenvalues, is that the fact that the spectra can be degenerated is crucial in what is to follow.

With the above decomposition, we can describe the measurement of a system as follows. When we make a measurement $A$ on the physical system in state $\ket{\psi}$, two things happens:

1. The result of our measurement is given by one of the eigenvalues $\lambda_1, \lambda_2, \ldots$, with the probability of measuring each eigenvalue $\lambda_i$ being given by $p_i = \bra{\psi} P_i \ket{\psi}$.
2. Having the measurement result $\lambda_i$, our system immediately transitions (collapses) from state $\ket{\psi}$ to the normalized state $\frac{P_i \ket{\psi}}{\bra{\psi} P_i \ket{\psi}}$. That is, our system is projected to the subspace $S_i$ and then normalized.

The above axiom gives rise to the (main problem)[htteigenbasisps://en.wikipedia.org/wiki/Measurement_problem] of (interpreting quantum mechanics)[https://en.wikipedia.org/wiki/Interpretations_of_quantum_mechanics], but, for now, we will remain agnostic on what measurement _really_ means, and just accept that this is how the world works.

Importantly, we can calculate the expected value $\mean{A}$ of our measurement $A$ (of the physical system in state $\ket{\psi}$) as

$$
\mean{A} = \sum_i \lambda_i \bra{\psi} P_i \ket{\psi} = \bra{\psi} \sum_i \lambda_i P_i \ket{\psi} = \bra{\psi} A \ket{\psi}.
$$

Also, the measurement axiom makes clear why we want state vectors to be normalized, given that we want our probabilities to sum to one

$$
1 = \sum_i p_i = \sum_i \bra{\psi} P_i \ket{\psi} = \bra{\psi} \sum_i P_i \ket{\psi}.
$$

Also, the projection themselves $P_i$ can be thought of as measurable physical quantities, since they are Hermitian operators. In particular, we can think of $P_i$ as answering the question "Has the system collapsed to the subspace $S_i$?" with the value of 1 if the answer is "yes", and 0 if the answer is no. The expected value here is exactly $\bra{\psi} P_i\ket{\psi}$, which measures "how much" was state vector $\ket{\psi}$ overlapping with subspace $S_i$ to begin with.

A particular projection we have is the projection $P_\psi := \ket{\psi} \bra{\psi}$, associated with the quantum state $\ket{\phi}$. To indicate we are talking about a function of the physical state, rather than a measurement, we refer to this projection as the density operator and use the symbol $\rho$ instead of $P_\psi$. Three properties of the density operator, that can be easily proven, are

$$
\trace{\rho} = 1 \\
\bra{\psi} A \ket{\psi} = \trace{\rho A} \\
\frac{P_i \ket{\psi}}{\bra{\psi} A \ket{\psi}} \frac{\bra{\psi} A}{\bra{\psi} A \ket{\psi}} = \frac{A \rho A}{\trace{A \rho A}}.
$$

Thus, we can use the density operator $\rho$ in place of the state vector $\ket{\psi}$ when defining our physical system. An advantage of the density operator is that the state vectors $\ket{\psi}$ and $e^{i \theta} \ket{\psi}$ describe the same physical system, so we do not have a unique representation. However, since we have that $\ket{e^{i \theta} \ket{\psi}} \bra{e^{i \theta} \ket{\psi}} = \ket{\psi} \bra{\psi}$, we do have a unique representation of our physical system state when given by the density matrix $\rho$.


# Mixed systems and the density matrix.

First, consider a pure system as above, in state $\ket{\psi}$. associated with this quantum state, we have the projection operator $P_{\psi}:= \ket{\psi} \bra{\psi}$. To indicate we are talking about a function of the physical state, rather than a measurement, we refer to this projection as the density operator and use the symbol $\rho$ instead of $P_\psi$. Three properties of the density operator, that can be easily proven, are

$$
\trace{\rho} = 1 \\
\bra{\psi} A \ket{\psi} = \trace{\rho A} \\
\frac{P_i \ket{\psi}}{\bra{\psi} A \ket{\psi}} \frac{\bra{\psi} A}{\bra{\psi} A \ket{\psi}} = \frac{A \rho A}{\trace{A \rho A}}.
$$

Thus, we can use the density operator $\rho$ in place of the state vector $\ket{\psi}$ when defining our physical system. An advantage of the density operator is that the state vectors $\ket{\psi}$ and $e^{i \theta} \ket{\psi}$ describe the same physical system, so we do not have a unique representation. However, since we have that $\ket{e^{i \theta} \ket{\psi}} \bra{e^{i \theta} \ket{\psi}} = \ket{\psi} \bra{\psi}$, we do have a unique representation of our physical system state when given by the density matrix $\rho$.

Now, suppose that we do not know in which quantum state our system is in. That is, our system can be in many possible orthogonal states $\{\ket{\psi_j}\}$, with probabilities $\{w\_j\}$, such that $\sum\_j w\_j = 1$. Those are associated with pure density matrices $\{\ket{\psi\_j} \bra{\psi\_j}\}$. Now, when making a measurement $A$ of this system, we need to consider not only the probabilities given by Born's rule but also the ones given by our lack of knowledge. We define then the density matrix $\rho$, encoding the state of our system _given the uncertainty_ as

$$
\rho := \sum_j w_j \rho_j, \quad \rho_j := \ket{\psi_j} \bra{\psi_j}.
$$

The nice thing about the density matrix is that we can easily extend our postulates when written in terms of pure density matrices $\rho_j$. To see this, denote the random variable _our system is in quantum state $\ket{\psi_j}$ as $\Psi_j$, and, for a measurement $A$, denote the random variable _result of measurement $A$_ as $\lambda(A)$. We then can find many similar results. For instance, the expected value of $\lambda(A)$ is given by

$$
\trace{\rho A}
$$

In particular, writing $A = \sum_i \lambda_i P_i$, the probability $P(\lambda(A) = \lambda_i)$ that we receive measurement $\lambda_i$ is given by

$$
P(\lambda(A) = \lambda_i) = \sum_j P(\lambda(A) = \lambda_i \mid \Psi = \psi_j) P(\Psi = \psi_j) = \sum_i \trace{\rho_i P_i} w_i = \trace{\rho P_i}.
$$

Similarly, the expected value $\mean{A}$ of $\lambda(A)$ equals

$$
\mathbb{E} [\lambda(A)] = \trace{\rho A}.
$$

After measurement, $\lambda(A) = \lambda_i$, the quantum state of the system will be in some unknown new collapsed state $P_i \rho_j P_i / \trace{\rho_j P_i}$ with some posterior probability $P(\Psi = \psi_j \mid \lambda(A) = \lambda_i)$ that I _was_ in state $\ket{\psi_j}$ given that I've observed $\lambda_i$. This can also be described by a density matrix $\rho, and using Bayes' rule, we find that the density matrix collapses to 

$$
\frac{P_i \rho P_i}{\trace{\rho P_i}}.
$$

Finally, the matrix $\rho$ also has unit trace

$$
\trace{\rho}.
$$

Moreover, we have the property that

$$
\trace{\rho^2} \leq 1,
$$

with equality only if $\rho = \ket{\psi} \bra{\psi}$, that is, our system is in a pure state.


In what follows, we will want to write the density matrix given a set of orthogonal subspaces, whose projections are given by $\{Q_j\}$. We will often also refer to the subspace themselves as $Q_j$. The probability $w_j$ will now mean "the probability that our system's quantum state is in subspace $Q_j$", and we will write

$$
\rho = \sum w_j Q_j,
$$

where now $w_1, w_2, \ldots$ are _distinct_ positive values satisfying

$$
\trace{\rho} = \sum_j w_j n_j = 1.
$$

This way we are given a unique representation of the density matrix $\rho$ given $\{w_j\}$ and $\{Q_j\}$. If, for each subspace $\{Q_j\}$, of dimension $n_j$, we choose a basis $\{\ket{\psi}\_{jk}\}\_{k=1}^{n_j}$, we can diagonalize $\rho$ as

$$
\rho = \sum_j \sum_{j=1}^{n_j} w_j \ket{\psi}_{jk} \bra{\psi}_{jk} \quad n_j := \operatorname{rank}{Q_j}.
$$

In this case, we are tempted again to interpret $\rho$ as saying that "the system is in state $\ket{\psi}_{jk}$ with probability $w_j$". The problem here is that the orthogonal basis of $Q_j$ is not unique, so in a sense saying the previous sentence is arbitrary. What we can say instead is just "the system's state is in subspace $Q_j$ with probability $w_j n_j$", but we can say nothing about any specific state inside $n_j$.

With these properties, we can consider density matrices as the fundamental objects describing our physical system, instead of quantum states. Following [Nielsen and Chuang](https://www.amazon.com/Quantum-Computation-Information-10th-Anniversary/dp/1107002176), we reformulate the state and measurement postulates in terms of density matrices:

*Postulate 1*: Associated to any isolated physical system is a complex vector space $\mathbb{H}$ with inner product $\braket{\phi}{\psi}$. The system is described by its density operator $\rho$, which is both positive-definite and has unit trace. This implies that the density matrix can be decomposed as

$$
\rho = \sum_j w_j \trace{Q_j},
$$

where $\{Q_j\}$ are projections into a set of orthogonal subspaces, and $w_j > 0$ are distinct values. Notice that the exigence of $\trace{\rho} = 1$ implies that

The system is in a _pure state_ if $\rho = \ket{\psi} \bra{\psi}$, and in a _mixed state_ otherwise. Equivalently, a system is in a pure state if $\trace{\rho^2} = 1$ and in a mixed state if $\trace{\rho^2} < 1$. If a quantum system is in state $\rho_k$ with probability $p_k$, then its density operator is given by

$$
\rho = \sum_j p_k \rho_k.
$$

*Postulate 2*: Quantum measurements are described by Hermitian operators $A$, which can be decomposed as

$$
A = \sum_i \lambda_i P_i,
$$

where $\{P_i\}$ are projections into a set of orthogonal subspaces that span $\mathbb{H}$, and $\{\lambda_i\}$ are the distinct possible measurement results. If the state of the system before measurement, is $\rho$, the probability that we measure value $\lambda_i$ is given by

$$
p(\lambda_i) := \trace{\rho P_i},
$$

and, measuring value $\lambda_i$, the system state collapses to a new state $\rho(P_i)$, given by

$$
\rho(P_i) := \frac{P_i \rho P_i}{\trace{\rho P_i}}.
$$

In particular, this implies that the expected value $\mean{A}$ of the measurement $A$ is given by

$$
\mean{A} = \trace{\rho A}.
$$

# Composite systems and entanglement

Another axiom concerns composite physical systems made of more than one part. In terms of state vectors, the third postulate of quantum mechanics says that, if we we have two distinct physical systems with associated Hilbert spaces $\mathbb{H}_A$ and $\mathbb{H}_B$, then the resulting composite physical system Hilbert space $\mathbb{H}$ is the tensor product

$$
\mathbb{H} = \mathbb{H}_A \otimes \mathbb{H}_B
$$

Moreover, _if_ the the systems are prepared in states $\ket{\psi_A}$ and $\ket{\psi_B}$, then the composite system state vector $\ket{\psi}$ is

$$
\ket{\psi} = \ket{\psi_A} \otimes \ket{\psi_B}.
$$

we formulate the composite system postulate in terms of the density operator, by giving *Postulate 3*: Assuming we have $N$ distinct physical systems, each associated with a Hilbert space $\mathbb{H}_n$, then the resulting composite physical system Hilbert space $\mathbb{H}$ is the tensor product $\mathbb{H}_1 \otimes \ldots \otimes \mathbb{H}_N$. If each system $i$ is prepared in state $\rho_i$, then the joint state $\rho$ of the composite system is given by $\rho_1 \otimes \ldots \otimes \rho_N$.

By choosing an orthogonal basis $\{\ket{a}_i\}$ for $\mathbb{H}_A$ and another orthogonal basis $\{\ket{b}_j\}$ for $\mathbb{H}_B$, we have that $\{\ket{a}_i \otimes \ket{b}_j\}$ is a basis for $\mathbb{H}_A \otimes \mathbb{H}_B$, so we can write $\ket{\psi}$ as

$$
\ket{\psi} = \sum_{i,j} c_{ij} \ket{a_i} \otimes \ket{b}_j.
$$

Similarly, we can write the density operator $\rho = \ket{\psi} \bra{\psi}$ as

$$
\rho = \sum_{i,j,k,l} c_{i,j} c_{k, l}^* \ket{a_i} \bra{a_k} \otimes \ket{b_j} \bra{b_l}.
$$

Crucially, not all state vectors $\ket{\psi} \in \mathbb{H}$ can be written as $\ket{\psi_A} \otimes \ket{\psi_B}$. In this case, we will not be able to consider each of the subsystems as being in some state vector. This gives us another strong motivation for dealing with the density operator formulation of quantum mechanics because if the system is in state $\rho = \ket{\psi}\bra{\psi}$, we _can_ consider each system as in states $\rho_A$ and $\rho_B$, given by

$$
\rho_A = \ptrace{B}{\rho} \\
\rho_B = \ptrace{A}{\rho},
$$

where we define the partial trace as

$$
\ptrace{B}{\rho} = \left(\sum_n I_A \odot \bra{b_n}\right) \rho \left(I_A \odot \ket{b_n}\right) \\
= \sum_n c_{in} c_{kn} \ket{a_i} \bra{a_n}. 
$$

This is because, using some calculations, we can show that, for any measurement $M_A$ made in the first system, the corresponding measurement in $\mathbb{H}_A \otimes \mathbb{H}_B$ is $M_A \otimes I_B$, and that

$$
\mean{M_A} = \trace{\rho \left(M_A \otimes I_B\right)} = \trace{\ptrace{B}{\rho} M_A},
$$

We will have much more to say about entanglement in the future, as I strongly suspect that this will be key to understanding the maximum entropy principle. However, that is a future discussion.

# Time evolution

Finally, the last axiom concerns the time evolution of a physical system. In particular, it postulates that, when not perturbed by some measurement, the state of our system $\ket{\psi_0}$ at time $0$ transitions to state $\ket{\psi_t}$ at time $t$ as

$$
\ket{\psi_t} = U(t) \ket{\psi_0}, 
$$

where the operator $U(t)$ is unitary, with $U(0) = I$. The fact that $U(t)$ is unitary satisfies the requirement that $\braket{\psi}{\psi}$ is always equal to one.

Since the operator is unitary  we can show that

$$
U'(t) = -\frac{i}{\hbar} H,
$$

where $H$ is a Hermitian operator. The operator $H$ is called the Hamiltonian, which, when seen as a measurement, measures the total energy of the system. Therefore, the system will evolve according to the Schrödinger equation
$$
\pderiv{\ket{\psi}}{t} = -\frac{i}{\hbar} H \ket{\psi}.
$$

The time evolution postulate and the Schrödinger equation are equivalent, and we can begin with the Schrödinger equation as our postulate, and derive unitary evolution by solving it.

In terms of density operators, we can formulate *Postulate 4*, which says that, when not perturbed by some measurement, the state of our system $\rho_0$ at time $0$ transitions to state $\rho_t$ at time $t$ as

$$
\ket{\psi_t} = U(t) \rho U(t)
$$

where the operator $U(t)$ is unitary, with $U(0) = I$. The Schrödinger equation for the density matrix can be written as

$$
\pderiv{\rho}{t} = -\frac{i}{\hbar} \left[H, \rho\right] \\
\left[H, \rho\right] = H\rho - \rho H
$$

For now, we will consider our system as static in time, bar a measurement, and we will not consider time evolution until much later in our essays. Instead, in the next part, we will use the above formalism to consider the maximum entropy principle in operator form, and its implications.