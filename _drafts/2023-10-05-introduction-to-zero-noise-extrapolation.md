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
observable is equivalent to computing the average of measuring
the eigenvalues of the observable of interest:

{% katexmm %}
$$
    \hat{\mathbb{E}}(H) = \sum_{i}^N \lambda_i p(\lambda_i)
$$
{% endkatexmm %}

Where $N$ is the number of samples.  
We are essentially trying to estimate the "true" expectation
value via a limited number of samples by using the sample mean
as an estimator the population mean.

*Moving forward, instead of writing $\hat{\mathbb{E}}(H)$, we will write*
*$\hat{\mathbb{E}}$ where it is understood that we are calculating the*
*expectation value of some given observable.*

Let $\hat{\mathbb{E}}$ be the sample average as calculated
by the formula above and $\mathbb{E}$ be the actual population average,
that is the average in the infinite sample size.

We can express the sample average as the population average with some
fluctuation about the population average as follows:

{% katexmm %}
$$
    \hat{\mathbb{E}}(H) = \mathbb{E} + \hat{\delta}
$$
{% endkatexmm %}

Where $\hat{\delta}$ is a random variable that represents the fact
that the computed expectation value varies about the true expectation value.
Note that even the true expectation would still be computed
in the presence of noise.

Nonetheless, we already have the necessary ingredient to undersand
ZNE: once we parametrize the computed expectation value with the strength
of the noise we are dealing with, the equation will be quite similar.

### Expectation value as a function of noise
### Noise-parametrized expectation value as an estimation problem

## Noise amplification
### Digital noise amplification
### Digital noise amplification algorithm

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
