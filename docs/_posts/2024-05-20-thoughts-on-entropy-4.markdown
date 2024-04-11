---
layout: post
title:  "Entropy and the world - Part 4."
date:   2024-05-28 12:00:00 -0300
categories:
---

# The maximum Von-Neumann entropy principle

A measure of mixedness of a density matrix $\rho$, beyond its square's trace $\trace{\rho^2}$, is given by the _Von Neumann entropy_

$$
\hat{S}_I(\rho) := -k_B \trace{\rho \log \rho} = -k_B \mean{\rho}.
$$

Of course, the name is not randomly chosen. If we diagonalize $\rho$ as 

$$
\rho = \sum_j w_j Q_j = \sum_j \sum_{k=1}^{n_j} w_j \ket{\psi_j}\bra{\psi_j},
$$

we have that

$$
\hat{S}_I(\rho) = -k_B \sum_j n_j w_j \log w_j = \hat{S}_I(\{w_j n_j\}) + \sum_j w_j n_j \hat{S}_I(\{1/n_j\})
$$

so we interpret $\hat{S}\_I(\rho)$ as the information entropy associated with the distribution "being in state $\ket{\psi}\_{jk}$ with probability $w\_j$". However, as discussed, we cannot so easily interpret the above distribution in this way. When considering degenerate spectra, the interpretation "the system's state is in subspace $Q_j$ with probability $w_j n_j$" works better, with the caveat that we also have extra entropy terms $\hat{S}_I(\{1/n_j\})$ associated with each subspace itself.

Well, having an entropy, we can again consider the maximum entropy principle for estimating $\rho$. Assume that we have some Hermitian operators $F_1(\alpha), \ldots, F_n(\alpha)$ acting in our physical system, parameterized by some $\alpha$. For instance, for a system of $N$ particles, $\alpha = V$ can be the volume of our system, and $F(\alpha) = H(V)$ is the $N$ particle Hamiltonian of the system, dependent on the volume $V$. Assume also that we have access to the expected value of these operators $\mean{F_j}$. Then, we find the density matrix $\rho$ that maximizes the entropy according to

$$
\operatorname{max}_{\rho'} \hat{S}_I = -k_B \trace{\rho' \log \rho'} \\
s.t. \quad \trace{\rho' F_j(\alpha)} = \hat{F_j}, j=1, \ldots, n \\
s.t. \quad \trace{\rho'} = 1.
$$

Again using Lagrange multipliers $\lambda_1, \ldots, \lambda_m$, we arrive at

$$
\rho = \frac{1}{Z} e^{-\sum_j \lambda_j F_j} \\
Z = \hat{Z}(\pmb{\lambda}) = \trace{e^{-\sum_{j=1}^n \lambda_j F_j}},
$$

with $\lambda = \hat{\lambda}(\mean{F_j}, \alpha), \alpha$ given as the solution of

$$
\trace{F_j e^{-\sum_i \lambda_i F_i} } = \mean{F_j} \trace{e^{-\sum_i \lambda_i F_i}}, \quad i=1, \ldots, n.
$$

From here all of the properties presented in the first part follow now in operator form

$$
F_i = -\pderiv{\log \hat{Z}}{\lambda_i} \\
S := \hat{S}_I(\rho) = \hat{S}(\mean{F_j}, \alpha) = k_B \left(\sum_j \lambda_j \mean{F_j} + \log Z \right) \\
d S = k_B \pderiv{\log \hat{Z}}{\alpha} d \alpha + \sum_j \pderiv{\hat{S}}{\mean{F_j}} d \mean{F_j} \\
\lambda_j = \pderiv{\hat{S}}{\mean{F_j}}.
$$

However, the interpretation here can be a little more subtle,