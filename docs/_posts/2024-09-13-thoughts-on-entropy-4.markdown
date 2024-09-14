---
layout: post
title:  "Entropy and the world - Part 4."
date:   2024-09-13 12:00:00 -0300
categories:
---

# Preamble

In the previous part, we made a very quick run on quantum mechanics paradigm and density operators. In this part, we will use this to derive a quantum-mechanical approach for entropy and statistical mechanics, at least at equilibrium.

# A note on the Boltzmann constant

As I write more, I am more and more convinced of the view espoused in [A Farewell to Entropy by Arieh Ben-Naim](https://www.amazon.com/Farewell-Entropy-Statistical-Thermodynamics-Information/dp/9812707077), in which entropy should be considered as the information entropy, period. In fact, according to Ben-Naim, we should just be calling entropy "missing information", since it is what the information entropy actually measures. Thus, it follows, entropy should not have any units, and the temperature

$$
T = \evalat{\pderiv{U}{S}}{V, X_1, \ldots}
$$

should be considered, for instance, as energy per unit of information. Although I will keep the convention of calling this quantity the entropy, I will forsake the Boltzmann constant $k_B$. In practice, since $S$ can, as a unitless quantity, get quite large, we would have to plug back in the Boltzmann constant for real life calculations. In this case, we would define the Kelvin unit as $k_B$ amounts of Joule, just as we define the mol unit as $N_A$ units of matter, or kilo-whatever as a thousand units of whatever.

However, I will keep calling this quantity the "entropy", following convention, and use the natural logarithm, instead of the binary logarithm. We can just consider "entropy" as a shorthand for "missing information", if necessary.

# The maximum Von-Neumann entropy principle

A measure of mixedness of a density matrix $\rho$, beyond its square's trace $\trace{\rho^2}$, is given by the _von Neumann entropy_

$$
\hat{S}_I(\rho) := -\trace{\rho \log \rho} = -\mean{\log \rho}.
$$

Of course, the name is not randomly chosen. If we diagonalize $\rho$ as 

$$
\rho = \sum_j w_j Q_j = \sum_j \sum_{k=1}^{n_j} w_j \ket{\psi_j}\bra{\psi_j},
$$

we have that

$$
\hat{S}_I(\rho) = -\sum_j n_j w_j \log w_j = \\
\hat{S}_I(\{w_j n_j\}) + \sum_j w_j n_j \hat{S}_I(\{1/n_j\}) = \\
\hat{S}_I(\{w_j n_j\}) + \sum_j w_j n_j \log n_j.
$$

so we interpret $\hat{S}\_I(\rho)$ as the information entropy associated with the distribution "being in state $\ket{\psi}\_{jk}$ with probability $w\_j$". However, as discussed, we cannot so easily interpret the above distribution in this way. When considering degenerate spectra, the interpretation "the system's state is in subspace $Q_j$ with probability $w_j n_j$" works better, with the caveat that we also have extra entropy terms $\hat{S}_I(\{1/n_j\})$ associated with each subspace itself. In fact, if $\rho = \frac{1}{n} Q$, with $Q$ being a projection into a subspace of dimension $n$, then $\hat{S}_I(\rho) = \log n$.

Well, having an entropy, we can again consider the maximum entropy principle for estimating $\rho$. We will only talk about energy and volumes for now. Assume that our system is described by a Hamiltonian $H[V]$, which is dependent on the volume $V$ of the system. Macroscopically, we have access to the expected value $\mean{H}$ of $H[V]$. We perform the same maximization procedure for the von Neumann entropy:

$$
\operatorname{max}_{\rho'} \hat{S}_I = -\trace{\rho' \log \rho'} \\
s.t. \quad \trace{\rho' H[V]} = \mean{H}, j=1, \ldots, n \\
s.t. \quad \trace{\rho'} = 1.
$$

Using a Lagrange multiplier $\beta$, we arrive at

$$
\rho = \frac{1}{Z} e^{-\beta H[V]}, \\
$$
where $Z$ is the partition function, given as
$$
Z = \hat{Z}(\beta, V) = \trace{e^{\beta H[V]}},
$$

Here, $\beta = \hat{\beta}(\mean{H}, V)$ is given as the solution of

$$
\trace{H e^{-\beta H[V]} } = \mean{H} \trace{e^{-\beta H[V]}}.
$$

# Thermodynamic properties

From this maximization procedure, all properties presented in the first part of this essay follows,  now in operator form.

$$
\mean{H} = -\pderivat{\log \hat{Z}}{\beta}{V} \\
\mean{\pderiv{H[V]}{V}} = -\frac{1}{\beta} \pderivat{\log \hat{Z}}{V}{\beta} \\
S := \hat{S}_I(\rho) = \beta \mean{H} + \log Z \\
\beta = \pderivat{S}{\mean{H}}{V} \\
d S = \beta d \mean{H} - \beta \mean{\pderiv{H[V]}{V}} d V.
$$

In particular, by comparison to classical thermodynamics, we find that $\beta = 1/T$ is the inverse temperature. Notice that we need only to be given the partition function to derive all these quantities. Also, in comparison to classical thermodynamics, we find that the pressure term $P$ equals to $-\mean{\pderiv{H[V]}{V}}$.

Consider the eigendecomposition in eigenstates $\ket{\phi_i(V)}$ of the Hamiltonian $H[V]$.

$$
H(V) = \sum_{i} E_i(V) \ket{\phi_i(V)} \bra{\phi_i(V)}.
$$

We then find that the partition function equals to

$$
\log Z(\beta, V) = \log \sum_i e^{-\beta E_i(V)},
$$

where $i$ refers to each eigenstate. However, the map from eigenstates to energy levels is many-to-one, we can instead enumerate the image of the map $i \to E_i(V)$, with distinct energy levels $E_j$, where now $j$ refers to the enumeration of energy levels. Then, letting $n_j(V)$ be the dimension of the eigenspace associated with eigenvalue $E_j(V)$ (that is, the number of eigenstates $\ket{\phi_i(V)}$ associated with eigenvalue $E_j(V)$), we can rewrite $\log Z$ as

$$
\log Z(\beta, V) = \log \sum_i n_j(V) e^{-\beta E_j(V)}.
$$

The nice thing here is that now we removed the reference to the individual enumeration of eigenstates, which, as discussed before, is relatively arbitrary given inside an eigenspace. We can also calculate the probability that our system will have energy $E_j(V)$ when measured as

$$
p(E_j) = \trace{\rho Q_j} = \frac{1}{Z} n_j e^{-\beta E_j(V)}.
$$

So, for every energy level, we have a balance between the multiplicity $n_j$ of that energy level, and the value of the energy level itself, modulated by the inverse temperature $\beta$.

In fact, at least for free particles, we have that $n_j \propto E_j^N$, where $N$ is the number of particles in the system. Therefore, for $N$ very large, $p(E_k) \approx 1$ for some maximum $E_k$. In this case, the overwhelming probable macrostate is the one associated with $E_k$, with all microstates $\ket{\phi}_i$ in the eigenspace of $E_k$ being equally probable.  In this case, the entropy will be given by

$$
S = \log n_j,
$$

# The first law

The second law is just the statement that, under whatever information the system gives to the world, and constraints that the world places on the system, the information entropy is maximized. What about the first law? By straightforward calculation

$$
d \mean{H} = \trace{H d \rho} + \trace{ \mean{\pderiv{H[V]}{V}} \rho} dV = \trace{H d \rho} - P dV.
$$

On the other hand, we find that (using the fact that $d \trace{\rho} = d 1 = 0$)

$$
\frac{1}{\beta} d S = -\frac{1}{\beta} \trace{d\rho \log \rho} = \trace{H d\rho}. 
$$

Associating back with classical thermodynamics, the heat term $dQ$ refers to the change in energy that arises in the change of the density matrix itself, in contrast to the useful predictable work, and, that $dQ = \frac{1}{\beta} dS$, thus having one more correspondence with classical thermodynamics.

# Free energy and phase transitions

It is easier to calculate the partition function with a given $\beta$ and $V$. For that, we can deploy the Helmholtz free energy

$$
F(\beta, V) = \mean{H} - \frac{1}{\beta} S = -\frac{1}{\beta} \log Z(\beta, V),
$$

and, since it is a Legendre transformation, we can derive all thermodynamic calculations with $F(\beta, V)$. We arrive at one of the goals of statistical thermodynamics: calculating state functions for classical thermodynamics. Moreover, we need only to look at

$$
\log Z(\beta, V) = \log \trace{e^{-\beta H[V]}}
$$

in our analysis.

We can rewrite $\log Z(\beta, V)$ as

$$
-\frac{1}{\beta} \log Z(\beta, V) = -\operatorname{logsumexp}_\beta\left(-F_j(V)\right) \\
F_j(V) := E_j(V) - \frac{1}{\beta} \log n_j,
$$

where the operator $\operatorname{logsumexp}_\beta$ is defined as

$$
\operatorname{logsumexp}_\beta(x_1, x_2, \ldots) := \frac{1}{\beta} \log \sum_i e^{\beta x_i}.
$$

The name $F_j(V)$ is intentional. Remember that $\log n_j$ is the von Neumann entropy associated with the density matrix

$$
\rho_j = \frac{1}{n_j} Q_j,
$$

with $Q_j$ being the projection operator on the eigenspace associated with $E_j$. Therefore, $F_j$ is the von Neumann entropy associated with the information "our system is the eigenspace of $E_j$".


As for operator $-\operatorname{logsumexp}_\beta(-F_1, -F_2, \ldots)$ is a [smooth minimum](https://en.wikipedia.org/wiki/Smooth_maximum), so that

$$
\lim_{\beta \to \infty} -\operatorname{logsumexp}_\beta(-F_1, -F_2, \ldots) = \operatorname{min}(F_1, F_2, \ldots).
$$

So we have a toy explanation for phase transitions. Namely, as $\beta$ changes, the "dominant" $F_i$ changes. Since the free energy determines every thermodynamic property of our system, macroscopically, we see our system changing phases.

# 
Good. There is nothing new here, just regular statistical mechanics, from a quantum point of view. From now on I could derive things like [Bose-Einstein](https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics) or [Fermi-Dirac](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_statistics) statistics, or do many other fun things. But, this in not the point of this series of essay. What I am more interested here is _what has happened_. Here is a story:

1. The system enters in contact with the world, and gives its mean energy $\mean{H}$.
2. The best information the world has of the density matrix is given by maximizing the Von-Neumann entropy of the system, given $\mean{H}$.
3. Therefore, the density matrix of the system is given, and all thermodynamic properties follow.

This is... a strange story to be told? What the hell is the nature of this interaction after all? How is it the world giving a "best guess" results in something physically measurable? In the next part, I will... well, I do not know. But at least the puzzle here is a bit clear. Guess it is time to chat a bit with [Claude](https://claude.ai). See you in the next part.