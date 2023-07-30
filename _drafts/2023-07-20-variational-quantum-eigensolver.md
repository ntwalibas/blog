---
title: "[Tutorial] Variational Quantum Eigensolver"
subtitle: "We explore the main components that make the variational
quantum eigensolver work: the ansatz, the optimizer, and the observable
of interest. Our main objective is to look at how the different components
can be tailored to solve practical problems."
layout: default
date: 2023-07-20
keywords: vqe, ansatz, optimizer, gradient descent, spsa
toc: true
published: false
---

## Introduction
The variational quantum eigensolver (VQE) is a hybrid quantum-classical
algorithm that performs the hard computation on a quantum computer then
uses a classical computer to process the measurement result from
the quantum computer.

The idea is quite simple: the premise of quantum computers is that the Hilbert
space is so huge that we can't reasonably explore it efficiently on a classical
computer (under certain conditions - see
[clifford circuits](https://en.wikipedia.org/wiki/Clifford_gates)) so we use
a quantum computer to do the exploration. <br>
But once we have the result from the quantum computer, we use a classical
optimizer to drive us towards the state of interest, which in our case
is the ground state.

The sections that follow are all about making sense of the two paragraphs above
and seeing VQE in action.

### Prerequisites
It is assumed that the reader has some basic knowledge of quantum computation.
That is the reader knows what a circuit is, and what gates and qubits are.
In the tooling section, [theoretical tools](#theoretical-tools) subsection
we list the necessary theoretical tools needed to make fruitful use of this
tutorial.

### Notation
We will work mostly with Pauli matrices.
(The identity matrix is not a Pauli matrix but we need to list it here.)
We will adopt the following notation for those matrices.

- Identity $I$ matrix:
{% katexmm %}
$$
\sigma_i = I =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $X$ matrix:
{% katexmm %}
$$
\sigma_x = X =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $Y$ matrix:
{% katexmm %}
$$
\sigma_y = Y =
\begin{bmatrix}
0 & -i \\
i & 0
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $Z$ matrix:
{% katexmm %}
$$
\sigma_z = Z =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$
{% endkatexmm %}

<div class='figure figure-alert figure-warning' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        Different roles of Pauli matrices
    </div>
    It is important to remain cognizant of the fact that Pauli
    matrices can either be gates or observables.
    And the way they are handled in code depends on whether one
    is dealing with a gate or an observable.<br>
    For instance, as a gate the Pauli matrix $\sigma_x$ will appear
    exactly as it is in both code and equations.
    But as an observable, it will appear as the matrix above
    in equations but will be translated to a Hadamard gate in code.
</div>
</div>

### Organization of the tutorial
In the next section we introduce the tools (theoretical and practical)
needed to understand and implement VQE.

In the basic theory section, we justify why VQE works and how it works.
We will provide the physical justification of the algorithm and
explore two simple examples so that we have something to anchor us
for the remainder of the tutorial.

In the ansatz design section we learn about the different ways ansatze
are designed. We will explore some of the challenges that occur with
particular choices of different designs.

In the optimizer selection section we study two optimizers and
how they behave.

In the observable reduction section we touch upon the fact that
we can combine commuting obervables into one so that we can
speed up calculations.

And last in the practical considerations section we think about the limitations
of VQE and think about the implication of those limitations and how they affect
the current quantum computing landscape.

## Tooling
### Theoretical tools
The reader is expected to know the following mathematics:
1. Finding the eigenvalues and eigenvectors of an operator.

The reader is also expected to know the following physics:
1. The measurement postulate, specifically projective measurements.
We will provide a review of the relevant mathematics nonetheless.

### Software tools
We will use Pennylane from Xanadu to run code we write.
It is an easy to use library and has an excellent documentation
which includes demos, tutorials and reference documentation.

The installation instructions can be found at
[https://pennylane.ai/install/](https://pennylane.ai/install/).

*Note: we will use the noiseless state vector simulator so as not worry*
*our heads with the complications that come with having a noisy device.*
*Maybe a future post will explore the behavior of VQE when noise*
*is taken into account.*

## Basic theory
In physics, if we know the Hamiltonian of a system
then we know the dynamics of the system.
This can be easily seen by looking at Schrodinger's equation.
Therefore, we will first need to have our problem encoded in
a Hamiltonian.

We won't spend our time trying to devise Hamiltonians ---
this is very hard work --- but we will assume that they are given to us.
Our goal will simply be finding the ground state energy (zero-point energy)
of the given Hamiltonian.

We care about the ground state energy because in nature the most
stable state of a physical system will be its ground state.
It is true that sometimes we care about excited states of
a system but in this tutorial we won't worry about that.

We begin by justifying why VQE works. Then justify we why a classical
optimizer is necessary to find the ground state.
Having grounded ourselves -- pun intended -- we tie all the different parts
required to make VQE work in some sort of template and offer a little more
explanation via a diagram.
Last, we code a couple of examples to see if simulations match theoretical
predictions.

<!-- ### The variational method
We begin by recalling the measurement postulate of quantum mechanics,
specifically the projective measurement postulate.<br>
The postulate applies to any observable but we will specialize
it to Hamiltonians alone. -->

### The measurement postulate
Let $H$ be an observable representing the total energy of the
system. By the spectral theorem $H$ has spectral decomposition:

{% katexmm %}
$$
H = \sum_{i} \lambda_i P_i \tag{1}
$$
{% endkatexmm %}

Where $\lambda_i$ is an eigenvalue and $P_i$ is a projector
onto the eigenspace of $H$ with corresponding eigenvalue $\lambda_i$.
That simply means that $P_{i}^{2} = P_i$
and $P_i = \ket{\lambda_i} \bra{\lambda_i}$
where $\\{ \ket{\lambda_i} \\}$ is the eigenspace of $H$
with each $\ket{\lambda_i}$ associated to eigenvalue $\lambda_i$.

We can therefore write equation $(1)$ as:

{% katexmm %}
$$
H = \sum_{i} \lambda_i \ket{\lambda_i} \bra{\lambda_i} \tag{1'}
$$
{% endkatexmm %}

Upon measurement (*before* the measurement but not after)
the **probability** of measuring the eigenvalue $\lambda_i$
given some state $\ket{\psi}$ is given by:

{% katexmm %}
$$
p(\lambda_i) = \bra{\psi} P_i \ket{\psi}
= \braket{\psi\\|\lambda_i} \braket{\lambda_i\\|\psi}
= \braket{\psi\\|\lambda_i} \braket{\psi\\|\lambda_i}^*
= \lvert\braket{\psi\\|\lambda_i}\rvert^{2} \tag{2}
$$
{% endkatexmm %}

The final state *after* measurement is given by:

{% katexmm %}
$$
\ket{\psi} \mapsto \frac{P_i \ket{\psi}}{\sqrt{p(\lambda_i)}}
$$
{% endkatexmm %}

We will care only about post-measurement states for the purpose of calculating
probabilities but not much else beside that.

<div class='figure figure-alert figure-success' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        Measuring observables and eigenvalues
    </div>
    It is important to realize that when we are asked to measure an observable,
    we are being asked to find the probabilities of measuring its eigenvalues
    given some state.<br>
    In practice though we will see the eigenvectors appearing with a certain
    frequency. So we will calculated the probability from those frequencies
    and associated the probability of each eigenvector to the corresponding
    eigenvalue.
</div>
</div>

<div class='figure figure-alert figure-info' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        Different terminology for measurements
    </div>
    It is common to hear/read that a measurement was performed
    in a certain basis.<br>
    Let us recall that the eigenvectors of a Hermitian operator
    form a <i>complete</i> basis. This means that we can express
    any state in the corresponding space as a linear combination
    of the eigenvectors of that Hermitian operator.<br>
    For example the $\sigma_z$ observable, being a Hermitian operator,
    has eigenvectors $\ket{0} = \begin{bmatrix}0\\1\end{bmatrix}$ and
    $\ket{1} = \begin{bmatrix}1\\0\end{bmatrix}$.<br>
    Consequently every one qubit state can be written as
    $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$.
    States written using the eigenvectors of $\sigma_z$ are said to be
    written in the <i>standard basis</i> or simply in the $\sigma_z$ basis.<br>
    (It is called standard basis because it maps to classical binary,
    which is the standard for classical computing.)<br><br>
    Similarly the $\sigma_x$ observable has eigenvectors
    $\ket{+} = \dfrac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$ and
    $\ket{-} = \dfrac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$.<br>
    Therefore every one qubit state can also be written as
    $\ket{\psi} = c_0 \ket{+} + c_1 \ket{-}$.
    States written using the eigenvectors of $\sigma_x$ are said to be
    written in the <i>Hadamard basis</i> or the simply in the $\sigma_x$ basis.<br>
    (It is called the Hadamard basis because the eigenvectors of $\sigma_x$ are
    obtained by applying the Hadamard gate to eigenvectors of $\sigma_z$.)<br><br>
    <b>However, we will find it convenient most of the time to work
    in the standard basis.</b><br><br>
    So performing a measurement in the standard basis is equivalent
    to using projectors $P_0=\ket{0}\bra{0}$ and $P_1=\ket{1}\bra{1}$.<br>
    Equivalently, performing a measurement in the Hadamard basis is equivalent
    to using the projectors $P_+=\ket{+}\bra{+}$ and $P_-=\ket{-}\bra{-}$
</div>
</div>

#### Pauli matrices and the identity form a complete basis
The Pauli matrices with the identity matrix
form a complete basis for all observables on a single qubit.
The tensor products of Pauli matrices along with the identity
matrix form a complete basis for observables on multiple
qubits.

What that means is that every observables we can think
about can be expressed as a linear combination of Pauli
matrices and the identity matrix.<br>
**Therefore we only need to learn how to measure Pauli matrices.**

#### Measurement of $\sigma_z$ with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$
We begin by find the eigenvalues and eigenvectors of $\sigma_z$
then use equation $(2)$ to calculate the probabilities.

* **Eigenvalues**:<br>
{% katexmm %}
$$
\begin{align}
    \det\begin{vmatrix} \sigma_z - \lambda \sigma_i \end{vmatrix} &= 0 \\
    \implies
    \det\begin{vmatrix}
        \begin{bmatrix}
        1-\lambda & 0 \\
        0 & -1-\lambda
        \end{bmatrix}
    \end{vmatrix} &= 0 \\
    \implies (1-\lambda)(-1-\lambda) &= 0 \\
    \implies \lambda &= \pm 1
\end{align}
$$
{% endkatexmm %}

The eigenvalues of $\sigma_z$ are $\lambda_0 = +1$ and $\lambda_1 = -1$.

* **Eigenvectors**:<br>
    * *Eigenvector corresponding to eigenvalue $\lambda_0 = +1$*
        {% katexmm %}
        $$
        \begin{align}
            \sigma_z \ket{\lambda_+} &= +1 \ket{\lambda_+} \\
            \implies
            \begin{bmatrix}
            1 & 0 \\
            0 & -1
            \end{bmatrix}
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix}
            &=
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix} \\
            \implies
            \begin{bmatrix}
            c_0 \\
            -c_1
            \end{bmatrix}
            &=
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix} \\
            \implies
            \begin{cases}
            c_0 &= c_0 \\
            -c_1 &= c_1
            \end{cases}
        \end{align}
        $$
        {% endkatexmm %}

        In the last step, $c_0 = c_0$ tells us nothing useful.
        But $-c_1 = c_1$ tells us that $c_1 = 0$. It follows then that
        $\ket{\lambda_+} = \begin{bmatrix} c_0 \\ 0 \end{bmatrix}$.

        Using the normalization condition,
        we find that $\braket{\lambda_+|\lambda_+}=1$ implies $|c_0|^2=1$ therefore
        $c_0 = 1$.

        Thus $\ket{\lambda_+} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.
        This eigenvector is also written as $\ket{0} = \ket{\lambda_+}$.

    * *Eigenvector corresponding to eigenvalue $\lambda_1 = -1$*<br>
        Repeating the same calculations as above:
        {% katexmm %}
        $$
        \begin{align}
            \sigma_z \ket{\lambda_-} &= -1 \ket{\lambda_-} \\
            \implies
            \begin{bmatrix}
            1 & 0 \\
            0 & -1
            \end{bmatrix}
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix}
            &=
            \begin{bmatrix}
            -c_0 \\
            -c_1
            \end{bmatrix} \\
            \implies
            \begin{bmatrix}
            c_0 \\
            -c_1
            \end{bmatrix}
            &=
            \begin{bmatrix}
            -c_0 \\
            -c_1
            \end{bmatrix} \\
            \implies
            \begin{cases}
            c_0 &= -c_0 \\
            c_1 &= c_1
            \end{cases}
        \end{align}
        $$
        {% endkatexmm %}

        Following the same reasoning that we used to calculate $\ket{\lambda_+}$,
        we find that $c_0 = 0$ and $c_1=1$.

        Thus $\ket{\lambda_-} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.
        This eigenvector is also written as $\ket{1} = \ket{\lambda_-}$.

* **Measurement with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$**<br>
We calculate only the probability of obtaining eigenvalue $+1$ given the state
$\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$. The calculation for eigenvalue
$-1$ is similar and left as an exercise.

{% katexmm %}
$$
\begin{align}
    p(+1) &= \lvert\braket{\psi\\|0}\rvert^{2} \\
    &= \lvert (c_0 \ket{0} + c_1 \ket{1})\ket{0}\rvert^{2} \\
    &= \lvert c_0 \braket{0\\|0} + c_1 \braket{1\\|0} \rvert^{2} \\
    &= \lvert c_0 \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

* **Quantum circuit for performing the measurement**<br>
The measurement of $+1$ corresponds to the use of the projector
$P_+ = \ket{0}\bra{0}$. Therefore we have:

{% katexmm %}
$$
\begin{align}
    p(+1) &= \bra{\psi}P_+\ket{\psi} \\
    &= \braket{\psi\\|0}\braket{0\\|\psi} \\
    &= \braket{\psi\\|I\\|0}\braket{0\\|I\\|\psi} \\
    &= \lvert \braket{0\\|I\\|\psi} \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

Concomitantly, the measurement of $-1$ corresponds to the use of the projector
$P_- = \ket{1}\bra{1}$ leading to:

{% katexmm %}
$$
\begin{align}
    p(-1) &= \bra{\psi}P_-\ket{\psi} \\
    &= \braket{\psi\\|1}\braket{1\\|\psi} \\
    &= \braket{\psi\\|I\\|1}\braket{1\\|I\\|\psi} \\
    &= \lvert \braket{1\\|I\\|\psi} \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

We conclude then that the circuit to perform a measurement
of the $\sigma_z$ observable given a state $\psi$ is as follows:

<div class='figure'>
    <img src='/assets/images/vqe/z-measurement.png'
         style='width: 30%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of the $\sigma_z$ observable:</span>
        since the identity $I$ acts on the state $\ket{\psi}$ we need not do
        anything, we just measure directly.
    </div>
</div>

* **Code for performing the measurement**<br>
We prepare the state $\ket{\psi} = RY(\dfrac{\pi}{2})\ket{0}$ and measure
the $\sigma_z$ observable with respect to that state.
$RY$ is a rotation about the $Y$ axis.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml
from pennylane import numpy as np

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 10000
)

@qml.qnode(dev)
def circuit(y: float):
    # Prepare the state against which to measure
    qml.RY(y, wires = 0)
    # Get the frequencies for each eigenvalue of the Z observable
    return qml.counts(qml.PauliZ(0))

if __name__ == "__main__":
    results = circuit(np.pi / 2)
    print(results)
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>Measurent of the $\sigma_z$ observable:</span>
    we prepare the state $\ket{\psi} = RY(\dfrac{\pi}{2})$
    as a generic example, any will do.
</div>
</div>

Note that if we prepared eigenvectors of $\sigma_z$
we will obtain eigenvalues with $100\%$ probability.
That is if we prepare the $\ket{0}$ state, we will obtain
eigenvalue $+1$ with $100\%$ probability.<br>
Equivalently, if we prepare $\ket{1}$, we will obtain
eigenvalue $-1$ with $100\%$ probability.

#### Measurement of $\sigma_y$ with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$
As with the $\sigma_z$ observable, we find the eigenvalues
and eigenvectors.

* **Eigenvalues**:<br>
{% katexmm %}
$$
\begin{align}
    \det\begin{vmatrix} \sigma_y - \lambda \sigma_i \end{vmatrix} &= 0 \\
    \implies
    \det\begin{vmatrix}
        \begin{bmatrix}
        -\lambda & -i \\
        i & -\lambda
        \end{bmatrix}
    \end{vmatrix} &= 0 \\
    \implies \lambda^{2}-1 &= 0 \\
    \implies \lambda &= \pm 1
\end{align}
$$
{% endkatexmm %}

The eigenvalues of $\sigma_y$ are $\lambda_+ = +1$ and $\lambda_- = -1$.

* **Eigenvectors**:<br>
    * *Eigenvector corresponding to eigenvalue $\lambda_+ = +1$*
        {% katexmm %}
        $$
        \begin{align}
            \sigma_z \ket{\lambda_+} &= +1 \ket{\lambda_+} \\
            \implies
            \begin{bmatrix}
            0 & -i \\
            i & 0
            \end{bmatrix}
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix}
            &=
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix} \\
            \implies
            \begin{bmatrix}
            -i c_1 \\
            i c_0
            \end{bmatrix}
            &=
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix} \\
            \implies
            \begin{cases}
            -i c_1 &= c_0 \\
            i c_0 &= c_1
            \end{cases}
        \end{align}
        $$
        {% endkatexmm %}

        Using $c_1 = i c_0$, we transform $\ket{\lambda_+}$ as follows:
        $\ket{\lambda_+} = \begin{bmatrix}c_0 & i c_0\end{bmatrix}^\intercal$

        Using the normalization condition,
        we find that $\braket{\lambda_+|\lambda_+}=1$ implies $2|c_0|^2=1$
        from which it follows that $c_0 = \dfrac{1}{\sqrt{2}}$.
        Consequently $c_1 = \dfrac{i}{\sqrt{2}}$.

        Thus $\ket{\lambda_+} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ i \end{bmatrix}$.
        This eigenvector is also written as $\ket{+i} = \ket{\lambda_+}$.
        Expressed in the $\sigma_z$ basis, $\ket{+i} = \dfrac{1}{\sqrt{2}}(\ket{0} + \ket{1})$.
    
    * *Eigenvector corresponding to eigenvalue $\lambda_- = -1$*
        {% katexmm %}
        $$
        \begin{align}
            \sigma_z \ket{\lambda_-} &= -1 \ket{\lambda_-} \\
            \implies
            \begin{bmatrix}
            0 & -i \\
            i & 0
            \end{bmatrix}
            \begin{bmatrix}
            c_0 \\
            c_1
            \end{bmatrix}
            &=
            \begin{bmatrix}
            -c_0 \\
            -c_1
            \end{bmatrix} \\
            \implies
            \begin{bmatrix}
            -i c_1 \\
            i c_0
            \end{bmatrix}
            &=
            \begin{bmatrix}
            -c_0 \\
            -c_1
            \end{bmatrix} \\
            \implies
            \begin{cases}
            i c_1 &= c_0 \\
            i c_0 &= -c_1
            \end{cases}
        \end{align}
        $$
        {% endkatexmm %}

        Using $c_0 = i c_1$, we transform $\ket{\lambda_-}$ as follows:
        $\ket{\lambda_-} = \begin{bmatrix}ic_1 & c_1\end{bmatrix}^\intercal$

        Using the normalization condition,
        we find that $\braket{\lambda_-|\lambda_-}=1$ implies $2|c_1|^2=1$
        from which it follows that $c_1 = \dfrac{1}{\sqrt{2}}$.
        Consequently $c_1 = -\dfrac{i}{\sqrt{2}}$.

        Thus $\ket{\lambda_-} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ i \end{bmatrix}$.
        This eigenvector is also written as $\ket{-i} = \ket{\lambda_-}$.
        Expressed in the $\sigma_z$ basis, $\ket{-i} = \dfrac{1}{\sqrt{2}}(\ket{0} - \ket{1})$.

* **Measurement with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$**<br>
We calculate only the probability of obtaining eigenvalue $+1$ given the state
$\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$. The calculation for eigenvalue
$-1$ is similar and left as an exercise.

{% katexmm %}
$$
\begin{align}
    p(+1) &= \lvert\braket{\psi\\|+i}\rvert^{2} \\
    &= \lvert (c_0 \ket{0} + c_1 \ket{1})\ket{+i}\rvert^{2} \\
    &= \lvert c_0 \braket{0\\|+i} + c_1 \braket{1\\|+i} \rvert^{2} \\
    &= \left\lvert c_0 \left(\bra{0} \left(\dfrac{1}{\sqrt{2}} (\ket{0} + i\ket{1})\right)\right)
    + c_1 \left(\bra{1} \left(\dfrac{1}{\sqrt{2}} (\ket{0} - i\ket{1})\right)\right) \right\rvert^{2} \\
    &= \left\lvert \dfrac{c_0}{\sqrt{2}}\left( \braket{0\\|0} \right)
    + \dfrac{-ic_1}{\sqrt{2}}\left( \braket{1\\|1} \right) \right\rvert^{2} \\
    &= \left\lvert \dfrac{c_0}{\sqrt{2}} - \dfrac{ic_1}{\sqrt{2}} \right\rvert^{2} \\
    &= \dfrac{1}{2} \lvert c_0 - ic_1 \rvert^{2} \\
    &= \dfrac{1}{2} \left(\sqrt{c_{0}^{2} + c_{1}^{2}}\right)^{2} \\
    &= \dfrac{c_{0}^{2} + c_{1}^{2}}{2}
\end{align}
$$
{% endkatexmm %}

* **Quantum circuit for performing the measurement**<br>
The measurement of $+1$ corresponds to the use of the projector
$P_+ = \ket{+i}\bra{+i}$. Therefore we have:

{% katexmm %}
$$
\begin{align}
    p(+1) &= \bra{\psi}P_+\ket{\psi} \\
    &= \braket{\psi\\|+i}\braket{+i\\|\psi} \\
    &= \braket{\psi\\|SH\\|0}\braket{0\\|HS^\dagger\\|\psi} \\
    &= \lvert \braket{0\\|HS^\dagger\\|\psi} \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

Similarly, the measurement of $-1$ corresponds to the use of the projector
$P_- = \ket{-i}\bra{-i}$ leading to:

{% katexmm %}
$$
\begin{align}
    p(+1) &= \bra{\psi}P_-\ket{\psi} \\
    &= \braket{\psi\\|-i}\braket{-i\\|\psi} \\
    &= \braket{\psi\\|SH\\|1}\braket{1\\|HS^\dagger\\|\psi} \\
    &= \lvert \braket{1\\|HS^\dagger\\|\psi} \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

We conclude then that the circuit to perform a measurement
of the $\sigma_y$ observable given a state $\psi$ is as follows:

<div class='figure'>
    <img src='/assets/images/vqe/y-measurement.png'
         style='width: 30%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of the $\sigma_y$ observable:</span>
        we need to perform a basis change from the $\sigma_z$ basis to the $\sigma_y$
        basis using $HS^\dagger$ then perform a standard measurement in the
        $\sigma_z$ basis. We will get eigenvectors in the $\sigma_z$ basis
        but the probabilities will correspond to measurements of the
        eigenvectors of $\sigma_y$.
    </div>
</div>

* **Code for performing the measurement**<br>
We prepare the state $\ket{\psi} = H\ket{0}$ and measure
the $\sigma_y$ observable with respect to that state.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml
from pennylane import numpy as np

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 10000
)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires = 0)
    return qml.counts(qml.PauliY(0))


if __name__ == "__main__":
    results = circuit()
    print(results)
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>Measurement of the $\sigma_y$ observable:</span>
    we prepare the state $\ket{\psi} = H\ket{0}$. We note that we obtain
    eigenvalue $+1$ with appromixately $0.5$ probablity and same
    for eigenvalue $-1$. It is easy to verify that this corresponds
    to theoretical predictions.
</div>
</div>

Note that if we prepared eigenvectors of $\sigma_y$
we will obtain eigenvalues with $100\%$ probability.
That is if we prepare the $\ket{+i} = SH\ket{0}$ state,
we will obtain eigenvalue $+1$ with $100\%$ probability.<br>
Equivalently, if we prepare $\ket{-i} = SH\ket{1}$,
we will obtain eigenvalue $-1$ with $100\%$ probability.

#### Example 2: measurement of $\sigma_x \otimes \sigma_i + \sigma_i \otimes \sigma_z$

### Expectation values
From equation $(2)$ we note that measurements in quantum mechanics are
inherently probabilistic in nature. That means we need to make multiple
measurements in order to make coherent conclusions.

The fact that measurements are probablistic also means that we can
calculate quantities such as moments of the probability distribution we get.
Of particular interest we want to know the expectation value of the observale
of interest.<br>
In more words, given a state $\ket{\psi}$ and an observable $H$,
we will generally want to know the expectation value ($i.e.$ average)
of the eigenvalues ($i.e.$ energies since we have a Hamiltonian)
with respect to that state.

In other words, we can't really measure in one measuremement the energy
of the system but we can only try to find the average.
If the state against which we are making measurements happens
to be an eigenvector of the Hamiltonian then after a sufficient
number of measurements we will approach the true eigenvalue
(that is the true energy of the system) with respect to that state.

So how do we find the average energy of a Hamiltonian given a state?
From basic probability theory, the average is simply the sum of
the measured energies times the probability of obtaining that specific energy.

{% katexmm %}
$$
\begin{align}
    \mathbb{E}(H) & = \sum_{i} \lambda_i p(\lambda_i) \\
    & = \sum_{i} \lambda_i \bra{\psi} P_i \ket{\psi} \\
    & = \bra{\psi} \left( \sum_{i} \lambda_i P_i \right) \ket{\psi} \\
    & = \bra{\psi} H \ket{\psi}
\end{align}
$$
{% endkatexmm %}

As a notational convenience, we will adopt:

{% katexmm %}
$$
\braket{H} = \mathbb{E}(H) = \bra{\psi} H \ket{\psi} \tag{3}
$$
{% endkatexmm %}

Using equation $(1')$ for the expansion in the Hamiltonian
eigenbasis, we obtain:

{% katexmm %}
$$
\begin{align}
    \braket{H} & = \bra{\psi} H \ket{\psi} \\
    & = \bra{\psi} \left( \sum_{i} \lambda_i \ket{\lambda_i}
    \bra{\lambda_i} \right) \ket{\psi} \\
    & = \sum_{i} \lambda_i \braket{\psi \\|\lambda_i}
    \braket{\lambda_i\\|\psi} \\
    & = \sum_{i} \lambda_i \braket{\psi\\|\lambda_i}
    \braket{\psi\\|\lambda_i}^* \\
    & = \sum_{i} \lambda_i \lvert\braket{\psi\\|\lambda_i}\rvert^{2}
\end{align}
$$
{% endkatexmm %}

Therefore if we are given a state $\ket{\psi}$ and the eigendecomposition
of $H$, we can find the expectation value using:

{% katexmm %}
$$
\braket{H} = \sum_{i} \lambda_i \lvert\braket{\psi\\|\lambda_i}\rvert^{2}
= \sum_{i} \lambda_i p(\lambda_i) \tag{3'}
$$
{% endkatexmm %}

**Example 1: expectation value of $\sigma_x$ with respect to $\ket{\psi}$**

**Example 2: expectation value of**
**$\sigma_x \otimes \sigma_i + \sigma_i \otimes \sigma_z$**
**with respect to $\ket{\psi}$**

### The variational method
From basic quantum mechanics we know that every system has a lowest
non-negative energy, call it $\lambda_0$.
We are generally interested in finding that energy because it corresponds
to the most stable state of the system.

The variational method allows to find an approximation of that energy.
The idea is very simple: since $\lambda_0 \le \lambda_i, \forall i$,
we have the following:

{% katexmm %}
$$
\begin{align}
    \braket{H} & = \sum_{i} \lambda_i \lvert\braket{\psi\\|\lambda_i}\rvert^{2} \\
    & \ge \sum_{i} \lambda_0 \lvert\braket{\psi\\|\lambda_i}\rvert^{2} \\
    & = \lambda_0 \sum_{i} \lvert\braket{\psi\\|\lambda_i}\rvert^{2} \\
    & = \lambda_0 \sum_{i} p(\lambda_i) \\
    & = \lambda_0
\end{align}
$$
{% endkatexmm %}

Where $\sum_{i} p(\lambda_i) = 1$ because probabilities must sum
to $1$ for normalized states.

It follows then that:

{% katexmm %}
$$
\braket{H} \ge \lambda_0 \tag{4}
$$
{% endkatexmm %}

Equation $(4)$ is the *essence* of the variational method.
It tells us that we can always try to find some state
that approximates the ground state.
Our goal therefore is to keep constructing such a state and
measure until we can't find any state with lower energy
because we can never find a state with lower energy
than the ground state energy.

### The variational algorithm
How then do we find the state $\ket{\psi}$ that approximates
the ground state $\ket{\lambda_0}$? The trick is to parametrize
$\ket{\psi}$ and then vary those parameters until a particular
sequence of parameters leads to a state that appromiximates
the ground state.

In other words we consider a state $\ket{\psi(\vec{\theta})}$
where $\vec{\theta} = [\theta_{n-1},\theta_{n-2}, \cdots, \theta_0]$
are the parameters that we will vary until we appromixate the ground
state. Whence the *variational* aspect of the algorithm.

#### Problem statement
Let us formaly state the variational problem. This is quite easy:
find a sequence of parameters $\vec{\theta}$ such that
$\mathcal{C}(\vec{\theta}) =
\bra{\psi(\vec{\theta})}H\ket{\psi(\vec{\theta})} \ge \lambda_0$.
The function $\mathcal{C}(\vec{\theta})$ is usually called the
*cost function* or the *objective function* and our goal is to minimize it.

Mathematically:

{% katexmm %}
$$
\underset{\vec{\theta}}{min} \: \mathcal{C}(\vec{\theta}) =
\underset{\vec{\theta}}{min} \:
\bra{\psi(\vec{\theta})}H\ket{\psi(\vec{\theta})} \ge \lambda_0
$$
{% endkatexmm %}

That is our goal is to minimize $\mathcal{C}(\vec{\theta})$
by manipulating $\vec{\theta}$.

#### Problem solution
In general we will start with some arbitrary instance
$\ket{\psi} = \ket{\psi(\vec{\theta})}$ where we fix $\vec{\theta}$
to some values (usually selected randomly).
Then we will use an *optimizer* to find new parameters $\vec{\theta}^*$
such at $\mathcal{C}(\vec{\theta}^\*) < \mathcal{C}(\vec{\theta})$.

The process will repeat itself until
$\mathcal{C}(\vec{\theta}^\*) \approx \lambda_0$ at which point
the optimizer stops and reports the result.

<div class='figure figure-alert figure-warning' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        The optimizer and local minima
    </div>
    The description above is a generalization because sometimes
    the optimizer can get stuck in a local minimum of
    $\mathcal{C}(\vec{\theta})$ and won't find a value
    close to $\lambda_0$ but we will worry about that later.
    Our hope though is that we end up with a value close to $\lambda_0$.
</div>
</div>

Each state $\ket{\psi(\vec{\theta})}$ is called an *ansatz*.
We will use circuits that have arbitrary rotations about some axis
to construct those ansatze. The circuits used to prepare arbitrary
ansatze are called *parametrized quantum circuits* ($a.k.a.$ PQCs).

### Examples
Let us work through a couple of examples where we try to find
their ground state energies. We have calculated those energies
before, now we use VQE to find the same.

**Example 1: ground state energy of $\sigma_x$**

**Example 2: ground state energy of**
**$\sigma_x \otimes \sigma_i + \sigma_i \otimes \sigma_z$**

## Ansatz design

## Optimizer selection

## Observable reduction

## Practical considerations

## Conclusion
