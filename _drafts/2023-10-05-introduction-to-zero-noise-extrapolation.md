---
title: "Introduction to zero-noise extrapolation"
subtitle: "Zero-noise extrapolation (ZNE) is an error mitigation technique that
allows us to amplify noise in a quantum circuit then extrapolate from that
what the expectation value would be if there was no (zero) noise.
This post follows a paper from Unitary Fund to explain the technical details
of ZNE and presents some sample code as to its implementation and application."
layout: default
date: 2023-10-05
keywords: error mitigation, zero-noise extrapolation, zne, unitary fund
toc: true
published: false
---

## Introduction
Current quantum computers are very noisy, that is short qubit lifetimes
and imperfect quantum gates. Mathematically, this means that given
the current error correction codes, we can't do error correction on them.

Nonetheless we are able to use them using heuristic algorithms
such as variational quantum eigensolver (VQE), variational quantum simulation (VQS),
and a host of other algorithms that don't rely on error correction.

For instance, in the case of VQE, our goal is to find the ground state
energy of some Hamiltonian of interest. Still, the presence of noise
does affect our ability to accurately compute the ground state energy.

Error mitigation comes to the rescue. Instead of trying to correct
for errors, we try to counter their effects.  
Zero-noise extrapolation (ZNE) is one such technique where we amplify the
noise in the circuit of interest then try to estimate what the
ground state would be if there was no noise.

The goal of this post is to build some intuition about ZNE by via very simple
examples and introduce the tools that allow us to use ZNE professionally.

This post came about as I'm taking part of a graduate-level course
on error mitigation offer by [QWorld](https://qworld.net/qcourse551-1/),
an initiave by the [University of Latvia](https://www.df.lu.lv/en/),
faculty of Computing in collaboration with [Classiq](https://www.classiq.io/)
an Israeli quantum software company.

This post brings nothing new to the table, in fact it is based
on the paper *Digital zero noise extrapolation for quantum error mitigation*
{% cite Giurgica_Tiron_2020 %} from the [Unitary Fund](https://unitary.fund/).

Our goal is to present the aforementioned paper in manner that's
much easier for (quantum) software developers to quickly understand.
That means, where relevant, we do all the calculations and interpretations,
and more important we write some code to validate the technique via concrete
results.

### Prerequisites
I assume that the reader can do simple linear algebra
and can program in Python.

Given that error mitigation is all about compensating for the presence
of errors in the execution of quantum programs, the reader is expected
to have a basic understanding of quantum computation in open systems.
Essentially, what are density matrices and quantum channels.

### Notation
We will make use of the following quantum gates:

- Identity $I$ matrix:
{% katexmm %}
$$
I =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $X$ matrix:
{% katexmm %}
$$
X =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $Y$ matrix:
{% katexmm %}
$$
Y =
\begin{bmatrix}
0 & -i \\
i & 0
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $Z$ matrix:
{% katexmm %}
$$
Z =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$
{% endkatexmm %}

And of matrices that generate the Clifford group:

- Hadamard gate:
{% katexmm %}
$$
H =
\dfrac{1}{\sqrt{2}}\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
$$
{% endkatexmm %}

- Phase gate:
{% katexmm %}
$$
S =
\begin{bmatrix}
1 & 0 \\
0 & i
\end{bmatrix}
$$
{% endkatexmm %}

### Organization of the post
In the tooling section we introduce the tools necessary to
follow along with the tutorial.
The reader is encouraged to install the necessary software tools
and to brush up on the necessary theory.

The section that follows, we formulate ZNE as an estimation problem.
First we justify via a simple example why we can parametrize
the calculation of the expectation value of some observable
with some real number that represents the "amount" of noise
in the system. Then introduce the ZNE estimation problem.

The section on noise amplification shows how to amplify
noise in a quantum circuit so we can later extrapolate
from that amplification what the expectation value would be
in the absence of noise.

How we can apply ZNE differs when we are dealing with incoherent
errors versus when we are dealing with coherent errors.
The section that follows noise amplification shows that
strategies for applying under such different noise models.

Then in the section that follows we elaborate on the different
ways to do estimation procedures.

Last, we use [Mitiq](https://github.com/unitaryfund/mitiq) to do
error mitigation using ZNE using the Classiq platform
and using PennyLane.

## Tooling
It is assumed that the reader can program in Python and can use
Numpy and Scipy.

Of course, since we are doing quantum computation, it is necessary
to know some linear algebra.

### Theoretical tools
On theory side of things, it is important to know:
- How to calculate the expectation of an observable given some density matrix.
- Understanding of the depolarizing channel.

### Software tools
I have a tendency to use [PennyLane](https://pennylane.ai/install/)
to validate some ideas so please go ahead and install it.

While we will write much code about ZNE from "scratch",
the reader should have [Mitiq](https://github.com/unitaryfund/mitiq)
installed since professionally, you will probably be using it
and not something written from scratch.

But the goal of this post is to use Mitiq on the Classiq platform,
please go ahead and register for an account on the
[Classiq platform](https://platform.classiq.io/registration) and download the
[Classiq Python SDK](https://docs.classiq.io/latest/getting-started/python-sdk/).

## Zero-noise extrapolation as an estimation problem
ZNE was introduced in 2016 simultaenously in {% cite Li_2017 %}
and {% cite Temme_2017 %} as technique for "active" error
minimization.

In fact, our justification for computing the expectation
value as a function the strength of noise is based on {% cite Li_2017 %}.

So what does it mean to say that ZNE is an estimation problem?
Let's recall that calculating the expectation value of some
observable is equivalent to computing the average of measured
eigenvalues of the observable of interest:

{% katexmm %}
$$
    \hat{\mathbb{E}}(H) = \hat{\braket{H}} = \sum_{i}^N h_i \, p(h_i)
$$
{% endkatexmm %}

Where $N$ is the number of samples and $h_i$ is
the $i^{th}$ eigenvalue of $H$.  
We are essentially trying to estimate the "true" expectation
value via a limited number of samples by using the sample mean
as an estimator the population mean.

Let $\hat{\braket{H}}$ be the sample average as calculated
by the formula above and $\braket{H}$ be the actual population average,
that is the average in the infinite sample size.  
We can express the sample average as the population average plus/minus some
fluctuation as follows:

{% katexmm %}
$$
    \hat{\braket{H}} = \braket{H} + \hat{\delta} \tag{1}
$$
{% endkatexmm %}

Where $\hat{\delta}$ is a random variable that represents the fact
that the computed expectation value varies about the true expectation value.
*Note that even the true expectation is still be computed*
*in the presence of noise.*

Nonetheless, we already have the necessary ingredient to undersand
ZNE: once we parametrize the computed expectation value with the strength
of the noise we are dealing with, the equation will be quite similar.

### Expectation value as a function of noise
(This subsection is based on subsection VII.A of {% cite Li_2017 %}.)

Let us assume that we are starting with an ideal state
$\ket{\bar{0}} = \ket{00\dots0}$ with the corresponding density
matrix given by $\ket{\bar{0}}\bra{\bar{0}}$.

We then evolve this state using a unitary operator
$U = U_N \cdots U_i \cdots U_1$, where $U_i$ is the $i^{th}$
gate being applied to the initial state $\ket{\bar{0}}\bra{\bar{0}}$.

The operation $U$ is ideal, there are exactly zero errors that
occured after it completed. Let's denote the final state as:

{% katexmm %}
$$
\begin{align}
    \rho^{(0)} &= U(\ket{\bar{0}}\bra{\bar{0}}) \\
    &= U_N \cdots U_i \cdots U_1(\ket{\bar{0}}\bra{\bar{0}})
\end{align}
$$
{% endkatexmm %}

Assume now that operation $U$ is not implemented perfectly
but in fact each individual gate $U_i$ is subject to noise
of the form $\mathcal{\Lambda}_i = (1-\epsilon_i)\mathbb{1} + \epsilon_i \mathcal{E}_i$
where with probability $1-\epsilon_i$ the operation $U_i$ is applied
and with probability $\epsilon_i$ errors $\mathcal{E}_i$ are introduced.

Therefore, under noise, the quantum operation of interest is given by:

{% katexmm %}
$$
    \mathcal{\Lambda}U = \mathcal{\Lambda}_N U_N \cdots \mathcal{\Lambda}_i U_i \cdots \mathcal{\Lambda}_1 U_1
$$
{% endkatexmm %}

The final density matrix $\rho$ under the action of $\mathcal{\Lambda}U$
is given by:

{% katexmm %}
$$
\begin{align}
    \rho &= \mathcal{\Lambda}U(\ket{\bar{0}}\bra{\bar{0}}) \\
    & = \mathcal{\Lambda}_N U_N \cdots \mathcal{\Lambda}_i U_i \cdots \mathcal{\Lambda}_1 U_1(\ket{\bar{0}}\bra{\bar{0}})
\end{align}
$$
{% endkatexmm %}

Given the state $\rho$, the expectation value of $H$ is:

{% katexmm %}
$$
\begin{align}
    \hat{\braket{H}} &= \text{Tr}[H\rho] \\
    &= \text{Tr}[H(\mathcal{\Lambda}_N U_N \cdots \mathcal{\Lambda}_i U_i \cdots \mathcal{\Lambda}_1 U_1(\ket{\bar{0}}\bra{\bar{0}}))]
\end{align}
$$
{% endkatexmm %}

We need to expand $\mathcal{\Lambda} U = \mathcal{\Lambda}_N U_N \cdots \mathcal{\Lambda}_i U_i \cdots \mathcal{\Lambda}_1 U_1$
in order to find a better representation. Given the many terms to multiply
we limit ourselves to expanding two terms and "extrapolate"
the form of the entire expression $\mathcal{\Lambda} U$.

Our goal now is to expand $\mathcal{\Lambda}_2U_2 \mathcal{\Lambda}_1U_1$.

We have:

{% katexmm %}
$$
\begin{align}
    \mathcal{\Lambda}_i &= (1-\epsilon_i)\mathbb{1} + \epsilon_i \mathcal{E}_i \\
    &= \mathbb{1} -\epsilon_i \mathbb{1} + \epsilon_i \mathcal{E}_i
\end{align}
$$
{% endkatexmm %}


It follows then that:

{% katexmm %}
$$
\begin{align}
    \mathcal{\Lambda}_2 U_2 &= U_2 -\epsilon_2 U_2 + \epsilon_2 \mathcal{E}_2 \\
    \mathcal{\Lambda}_1 U_1 &= U_1 -\epsilon_1 U_1 + \epsilon_1 \mathcal{E}_1
\end{align}
$$
{% endkatexmm %}

Consequently:

{% katexmm %}
$$
\begin{align}
    \mathcal{\Lambda}_2 U_2 \mathcal{\Lambda}_1 U_1 &= (U_2 -\epsilon_2 U_2 + \epsilon_2 \mathcal{E}_2) \times (U_1 -\epsilon_1 U_1 + \epsilon_1 \mathcal{E}_1) \\ \\
    &= U_2U_1 \\
    &- \epsilon_1 U_2U_1 - \epsilon_2U_2U_1 + \epsilon_1U_2\mathcal{E}_1U_1 + \epsilon_2\mathcal{E}_2U_2U_1 \\
    &- \epsilon_1\epsilon_2U_2\mathcal{E}_1U_1 + \epsilon_1\epsilon_2U_2U_1 - \epsilon_1\epsilon_2\mathcal{E}_2U_2U_1
    + \epsilon_1\epsilon_2\mathcal{E}_2U_2\mathcal{E}_1U_1 \\ \\
    &= U_2U_1 \\
    &- (\epsilon_1 + \epsilon_2) U_2U_1 \\
    &+\epsilon_1U_2\mathcal{E}_1U_1 + \epsilon_2\mathcal{E}_2U_2U_1 \\
    &+ \epsilon_1\epsilon_2(U_2U_1 - \mathcal{E}_2U_2U_1 - U_2\mathcal{E}_1U_1 + \mathcal{E}_2U_2\mathcal{E}_1U_1)
\end{align}
$$
{% endkatexmm %}

Recalling that $\epsilon_1, \epsilon_2 < 1$
(if $\epsilon_i = 1$ then the system is riddled with noise and unusable),
it follows that $\epsilon_1\epsilon_2 \ll 1$.
We can therefore ignore the higher order terms:

{% katexmm %}
$$
\begin{align}
    \mathcal{\Lambda}_2 U_2 \mathcal{\Lambda}_1 U_1 &= U_2U_1 \\
    &- (\epsilon_1 + \epsilon_2) U_2U_1 \\
    &+\epsilon_1U_2\mathcal{E}_1U_1 + \epsilon_2\mathcal{E}_2U_2U_1 \\
    &+ \mathcal{O}(\epsilon_1\epsilon_2)
\end{align}
$$
{% endkatexmm %}

Extrapolating to the sample size $N$, we have:

{% katexmm %}
$$
\begin{align}
    \mathcal{\Lambda} U &= \mathcal{\Lambda}_N U_N \cdots \mathcal{\Lambda}_i U_i \cdots \mathcal{\Lambda}_1 U_1 \\
    &= U_N \cdots U_i \cdots U_1 \\
    &- (\epsilon_N + \cdots + \epsilon_i + \cdots + \epsilon_1) U_N \cdots U_i \cdots U_1 \\
    &+ \epsilon_N (\mathcal{E}_N U_N \cdots U_i \cdots U_1) + \cdots + \epsilon_i (U_N \cdots \mathcal{E}_i U_i \cdots U_1) 
    + \cdots + \epsilon_1 (U_N \cdots U_i \cdots \mathcal{E}_1 U_1) \\
    &+ \mathcal{O}(\epsilon_1\epsilon_2)
\end{align}
$$
{% endkatexmm %}

In condensed form:

{% katexmm %}
$$
\begin{align}
    \mathcal{\Lambda} U &= U_N \cdots U_i \cdots U_1 \\
    &- (\sum_i^N \epsilon_i) U_N \cdots U_i \cdots U_1 \\
    &+ \sum_i^N \epsilon_i U_N \cdots \mathcal{E}_i U_i \cdots U_1 \\
    &+ \mathcal{O}(\epsilon_1\epsilon_2)
\end{align}
$$
{% endkatexmm %}

Let's rewrite the noise strength $\epsilon_i$ as $\epsilon_i = \lambda \epsilon'_i$.
The expression above becomes:

{% katexmm %}
$$
\begin{align}
    \mathcal{\Lambda} U &= U_N \cdots U_i \cdots U_1 \\
    &- \lambda(\sum_i^N \epsilon'_i) U_N \cdots U_i \cdots U_1 \\
    &+ \lambda \sum_i^N \epsilon'_i U_N \cdots \mathcal{E}_i U_i \cdots U_1 \\
    &+ \mathcal{O}(\lambda^2)
\end{align}
$$
{% endkatexmm %}

We can now complete the calculation of the expectation value:

{% katexmm %}
$$
\begin{align}
    \hat{\braket{H}} &= \text{Tr}[H\rho] \\
    &= \text{Tr}[H(\mathcal{\Lambda}_N U_N \cdots \mathcal{\Lambda}_i U_i \cdots \mathcal{\Lambda}_1 U_1(\ket{\bar{0}}\bra{\bar{0}}))] \\
    &= \text{Tr}[H(U_N \cdots U_i \cdots U_1(\ket{\bar{0}}\bra{\bar{0}}))] \\
    &- \text{Tr}[H(\lambda(\sum_i^N \epsilon'_i) U_N \cdots U_i \cdots U_1(\ket{\bar{0}}\bra{\bar{0}}))] \\
    &+ \text{Tr}[H(\lambda \sum_i^N \epsilon'_i U_N \cdots \mathcal{E}_i U_i \cdots U_1(\ket{\bar{0}}\bra{\bar{0}}))] \\
    &+ \mathcal{O}(\lambda^2)
\end{align}
$$
{% endkatexmm %}

Let us recall that $\rho^{(0)} = U_N \cdots U_i \cdots U_1(\ket{\bar{0}}\bra{\bar{0}})$
and let us define $\rho^{(1)} = \lambda \sum_i^N \epsilon'_i U_N \cdots \mathcal{E}_i U_i \cdots U_1(\ket{\bar{0}}\bra{\bar{0}})$.
The expectation value can be rewritten as:

{% katexmm %}
$$
\begin{align}
    \hat{\braket{H}} &= \text{Tr}[H\rho^{(0)}] \\
    &- \left(\lambda \sum_i^N \epsilon'_i\right) \text{Tr}[H\rho^{(0)}] \\
    &+ \lambda \text{Tr}[H\rho^{(1)}] \\
    &+ \mathcal{O}(\lambda^2)
\end{align}
$$
{% endkatexmm %}

We can now see that $\hat{\braket{H}}$ depends on the factor $\lambda$ that
depends on the strength of the noise in the circuit!

So our final expression of the expectation value is given by:

{% katexmm %}
$$
\begin{align}
    \hat{\braket{H}}(\lambda) &= \left(1 - \lambda \sum_i^N \epsilon'_i \right) \hat{\braket{H}}^{(0)} + \lambda \hat{\braket{H}}^{(1)} + \mathcal{O}(\lambda^2) \tag{2}
\end{align}
$$
{% endkatexmm %}

Where $\hat{\braket{H}}^{(0)} = \text{Tr}[H\rho^{(0)}]$ is the expectation value
when there is no noise in the system and $\hat{\braket{H}}^{(1)} = \text{Tr}[H\rho^{(1)}]$
which takes into account noise $\mathcal{E}_i$ affecting each gate $U_i$.

### Noise-parametrized expectation value as an estimation problem
We now know that it is possible to express the calculation of the expectation
value as a function of a noise parameter $\lambda$.

Our goal is to figure out the expectation if there was no noise in the system.
Starting from equation $(1)$, we can now write the estimation of the expectation
value as:

{% katexmm %}
$$
    \hat{\braket{H}}(\lambda) = \braket{H}(\lambda) + \hat{\delta}
$$
{% endkatexmm %}

Where $\hat{\delta}$ is a random variable.

If $\lambda = 1$ then we are just working with the machine noise,
that is $\hat{\braket{H}} = \text{Tr}[H(\mathcal{\Lambda}U(\ket{\bar{0}}\bra{\bar{0}}))]$.

On the other hand if $\lambda = 0$, then there is no noise in the system
as can be inferred from Equation $(2)$: $\hat{\braket{H}} = \hat{\braket{H}}^{(0)}$.

The main idea behind ZNE is the following:  
*We calculate the expectation value for $\lambda = 1$ then artificially increase*
*the noise in the system by increasing $\lambda$. We then fit a curve about the*
*different points of $\hat{\braket{H}}(\lambda)$ then extrapolate the value*
*of $\hat{\braket{H}}(\lambda)$ at $\lambda = 0$*.

Therefore ZNE is an estimation problem of the following form:

{% katexmm %}
$$
    \hat{\braket{H}}(0) = \braket{H}(0) + \hat{\delta} \tag{3}
$$
{% endkatexmm %}

The issue now is to figure out what curve to use to fit the data points
then extrapolate from there. In other words what will be our estimator function?

Much of time spent using ZNE will be choosing what is the best estimator for a given
problem. For instance {% cite Temme_2017 %} found that under certain assumption
one can use Richardson's extrapolation, a polynomial extrapolation.
On the other hand {% cite Endo_2018 %} found that using an exponential extrapolation
has some advantage accuracy-wise in certain circumstances.
It depends on error models assumed and empirical data.

In our case, we will explore estimation under the assumption of the depolarizing noise
model. This noise model is very simplistic so it is perfect for understanding.

How then do we know that ZNE worked?
We do this by calculating the *mean squared error (MSE)*:

{% katexmm %}
$$
\begin{align}
    MSE(\hat{\braket{H}}(0)) &= \mathbb{E}(\hat{\braket{H}}(0) - \braket{H}(0)) \\
    &= Var(\hat{\braket{H}}(0)) + Bias(\hat{\braket{H}}(0))^2
\end{align}
$$
{% endkatexmm %}

When we work through a simple example with the depolarizing noise model,
we will also compute the expectation value without noise then compute
the MSE and consequently evaluate the success of ZNE.

Note that there is a [bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
involved in the estimation procedure. This issue is expounded upon in
{% cite cai2023quantum %} so we won't delve on it, the interested reader
can (and should) read about in the given reference.

In practice though, what we will want to do is compare the performance
of a particular estimation/extrapolation strategy with the unmitigated
case (or with another estimation strategy).  
In this case, {% cite Giurgica_Tiron_2020 %} propose using
the ratio of absolute errors of the mitigated expectation and
the unmitigated expectation (or a mitigated result using a different estimator).

Let $E_m$ be the mitigated expectation value using some estimation strategy.
Let $E_r$ be the reference expectation value such as the unmitigated
expectation value or an expectation value obtained using a different estimation
strategy.

Then the absolute error for the mitigated expectation value is given by
$R_m = \lvert E_m - E(0) \rvert$ where $E(0)$ is the expectation value
in the noiseless regime.  
Equivalently the absolute error for reference expectation value is given by
$R_r = \lvert E_r - E(0) \rvert$.

The performance of the mitigated result with respect to the reference
result is given by $p = R_r / R_m$.

## Noise amplification
In the previous section, we made a case for ZNE: since we can parametrized
the expectation value as a function of the noise strength, using some extrapolation
strategy, we should be able to estimate the expectation in the noiseless regime.

But this relies on the ability to actually increase the noise in the system by varying
the strength $\lambda$.  
There are two ways we can go about that:
- Stretch pulses that implement our quantum gates so that interaction with the environment
    has time to take hold thereby corrupting our system.
- Add identity (or identity-equivalent) gates into our circuits so that the circuit's
    meaning stays the same but it takes longer to complete thereby also increasing
    the chances of corruption.

The first option is generally not favored because most hardware providers
do not provide pulse-level access, and even if they did most quantum software developers
do not have the skillset necessary to deal with pulses.

The second approach is favored and it is the one expounded upon in our reference paper.
We will therefore deal only with the insertion of identity-equivalent gates.

### Digital noise amplification using unitary folding
Unitary folding is a technique where we take some unitary
in the circuit (could be gate or a layer or entire circuit) and the Hermitian transpose
of that unitary then appending their product to the circuit.

Since the product of a unitary and its Hermitian transpose
is the identity, the original circuit's meaning remains unchanged.

Unitary folding is given by the following replacement rule:

{% katexmm %}
$$
    U \rightarrow U(UU^\dagger)^n \tag{4}
$$
{% endkatexmm %}

Where $n$ is a positive integer representing the number
of times we repeat the identity-equivalent unitary $(UU^\dagger)$.

Where $U$ is either a gate or a layer in the circuit.

{% cite Giurgica_Tiron_2020 %} introduce circuit folding
and gate folding as different ways to do unitary folding.
The principle is the same, only differing with respect
to small technical details.

We will elaborate on circuit folding and leave the reader
to explore gate folding on their own.

Let $U$ be a circuit that can be broken into layers $L$
and it has $d$ such layers.

{% katexmm %}
$$
    U = L_d\dots L_2L_1
$$
{% endkatexmm %}

Where $U$ is the entire circuit and each $L_i$ is either a gate or
a layer in the circuit depending on the circuit representation.

In circuit folding, we fold the entire circuit $n$ times
as in Equation $4$. This gives a scaling by $2n + 1$.

In other words, let $d$ be the current circuit's depth and
$d'$ be the new folded circuit depth. Then the following is true:

{% katexmm %}
$$
    d' = d(2n + 1)
$$
{% endkatexmm %}

It is also possible to have a fine-grained scaling by
appending layers/gates within the circuit as follows:

{% katexmm %}
$$
    U \rightarrow U(UU^\dagger)^n L^\dagger_dL^\dagger_{d-1}\dots L^\dagger_{d-s+1}L_{d-s+1}\dots L_{d-1}L_d \tag{5}
$$
{% endkatexmm %}

With this new circuit folding rule, the folded circuit depth is given by:

{% katexmm %}
$$
    d' = d(2n + 1) + 2s \tag{6}
$$
{% endkatexmm %}

So how does this scaling relate to our desired increase in noise $\lambda$?

What we would like is a relation of the type:

{% katexmm %}
$$
    d' = \lambda d  \tag{7}
$$
{% endkatexmm %}

That is the noise strength $\lambda$ scales proportionally with the depth.

From Equation $(6)$, we have:

{% katexmm %}
$$
    d' = d + 2(dn + s)
$$
{% endkatexmm %}

Let us define $k$ as:

{% katexmm %}
$$
    k = dn + s \tag{8}
$$
{% endkatexmm %}

Consequently we have:

{% katexmm %}
$$
    d' = d + 2k
$$
{% endkatexmm %}

Dividing both sides by $d$ we get:

{% katexmm %}
$$
    \dfrac{d'}{d} = 1 + \dfrac{2k}{d}
$$
{% endkatexmm %}

From Equation $(7)$, we recognize that $\lambda = \frac{d'}{d}$
therefore the equation above results in:

{% katexmm %}
$$
    \lambda = 1 + \dfrac{2k}{d} \tag{9}
$$
{% endkatexmm %}

From equation $(8)$, we see that $k$ depends only on the circuit parameters:
$n$ is the number of circuit folds, $d$ is the number of layers/gates, and
$s$ is the number of additional layers/gates.

It follows that equation $(9)$ relates the strength of the noise $\lambda$
to circuit parameters therefore we should be able to build a circuit
for a particular noise strength of our desire.

### Circuit folding algorithm
Since $d$, $n$, and $s$ are positive integers, if we require $0 \le s < n$
Equation $(8)$ is equivalent to Euclidian division.
We therefore have the following:

{% katexmm %}
$$
\begin{align}
    n &= k / d \\
    s &= k \% d \tag{10}
\end{align}
$$
{% endkatexmm %}

The algorithm practically writes itself at this point:
from Equation $(9)$, for a given noise strength $\lambda$,
calculate $k = \frac{d(\lambda - 1)}{2}$.  
Then we use Equation $(10)$ to find $n$ and $s$ given that $d$
is fixed.

<div class='figure'>
<div class='algorithm' markdown='1'>
**_Prepare:_**  
$\quad \lambda \ge 1$  
$\quad V = U$  
$\quad d = \text{len(}U\text{)}$  

**_Initialize:_**  
$\quad k = \Bigl\lfloor\frac{d(\lambda - 1)}{2}\Bigr\rfloor$  
$\quad n = k / d$  
$\quad s = k \% d$   

**while** $n > 0$**:**  
$\quad U \gets U \circ V^\dagger V$  
$\quad n \gets n - 1$  

$L = V[d-s:d]$  
$U \gets U \circ L^\dagger L$  

**return** $U$
</div>
<div class='caption'>
    <span class='caption-label'>
        Noise amplification by unitary folding algorithm:
    </span>
    from the desired noise strength $\lambda$, calculate
    the new circuit paramters $n$ and $s$ via $k$ then
    generate the circuit.
</div>
</div>

## Zero-noise extrapolation under different noise models
### ZNE in the presence of incoherent errors
### ZNE in the presence of coherent errors

## Estimation procedures
### Non-adaptive estimation
### Adaptive estimation

## Zero-noise extrapolation using Mitiq
### Using Mitiq with PennyLane
### Using Mitiq with Classiq

## Conclusion
