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
\sigma^{(i)} = I =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $X$ matrix:
{% katexmm %}
$$
\sigma^{(x)} = X =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $Y$ matrix:
{% katexmm %}
$$
\sigma^{(y)} = Y =
\begin{bmatrix}
0 & -i \\
i & 0
\end{bmatrix}
$$
{% endkatexmm %}

- Pauli $Z$ matrix:
{% katexmm %}
$$
\sigma^{(z)} = Z =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$
{% endkatexmm %}

<div class='figure figure-alert figure-danger' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        Different roles of Pauli matrices
    </div>
    It is important to remain cognizant of the fact that Pauli
    matrices can either be gates or observables.
    And the way they are handled in code depends on whether one
    is dealing with a gate or an observable.<br>
    For instance, as a gate the Pauli matrix $\sigma^{(x)}$ will appear
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

In the ansatz design section we learn about the different ways ansätze
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
2. Basic operations on matrices.
3. Kronecker product, especially using the mixed-product property.

The reader is also expected to know the following basic quantum theory:
1. Quantum states as rays in Hilbert space.
2. Quantum gates that evolve quantum states.

A review of the measurement postulate will be provided
so it is not a prerequisite.

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
    For example the $\sigma^{(z)}$ observable, being a Hermitian operator,
    has eigenvectors $\ket{0} = \begin{bmatrix}0\\1\end{bmatrix}$ and
    $\ket{1} = \begin{bmatrix}1\\0\end{bmatrix}$.<br>
    Consequently every one qubit state can be written as
    $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$.
    States written using the eigenvectors of $\sigma^{(z)}$ are said to be
    written in the <i>standard basis</i> or simply in the $\sigma^{(z)}$ basis.<br>
    (It is called standard basis because it maps to classical binary,
    which is the standard for classical computing.)<br><br>
    Similarly the $\sigma^{(x)}$ observable has eigenvectors
    $\ket{+} = \dfrac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$ and
    $\ket{-} = \dfrac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$.<br>
    Therefore every one qubit state can also be written as
    $\ket{\psi} = c_0 \ket{+} + c_1 \ket{-}$.
    States written using the eigenvectors of $\sigma^{(x)}$ are said to be
    written in the <i>Hadamard basis</i> or the simply in the $\sigma^{(x)}$ basis.<br>
    (It is called the Hadamard basis because the eigenvectors of $\sigma^{(x)}$ are
    obtained by applying the Hadamard gate to eigenvectors of $\sigma^{(z)}$.)<br><br>
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

#### Measurement of $\sigma^{(z)}$ with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$
The eigenvalues and eigenvectors of $\sigma^{(z)}$ are calculated
in the [derivations section](#eigenvalues-and-eigenvectors-of-sigmaz) and are found to be:

1. Eigenvalue $+1$ with eigenvector $\ket{0} = \begin{bmatrix} 1 \\ 0\end{bmatrix}$
2. Eigenvalue $-1$ with eigenvector $\ket{1} = \begin{bmatrix} 0 \\ 1\end{bmatrix}$

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
of the $\sigma^{(z)}$ observable given a state $\psi$ is as follows:

<div class='figure'>
    <img src='/assets/images/vqe/z-measurement.png'
         style='width: 20%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of the $\sigma^{(z)}$ observable:</span>
        since the identity $I$ acts on the state $\ket{\psi}$ we need not do
        anything, we just measure directly.
    </div>
</div>

* **Code for performing the measurement**<br>
We prepare the state $\ket{\psi} = RY(^\pi/_2)\ket{0}$ and measure
the $\sigma^{(z)}$ observable with respect to that state.
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
    <span class='caption-label'>Measurent of the $\sigma^{(z)}$ observable:</span>
    we prepare the state $\ket{\psi} = RY(^\pi/_2)\ket{0}$
    as a generic example, any state would do.
</div>
</div>

Note that if we prepared eigenvectors of $\sigma^{(z)}$
we will obtain eigenvalues with $100\%$ probability.
That is if we prepare the $\ket{0}$ state, we will obtain
eigenvalue $+1$ with $100\%$ probability.<br>
Equivalently, if we prepare $\ket{1}$, we will obtain
eigenvalue $-1$ with $100\%$ probability.

#### Measurement of $\sigma^{(y)}$ with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$
The eigenvalues and eigenvectors of $\sigma^{(y)}$ are calculated
in the [derivations section](#eigenvalues-and-eigenvectors-of-sigmay) and are found to be:

1. Eigenvalue $+1$ with eigenvector $\ket{+i} = \dfrac{1}{\sqrt{2}}(\ket{0} + i\ket{1})$
2. Eigenvalue $-1$ with eigenvector $\ket{-i} = \dfrac{1}{\sqrt{2}}(\ket{0} - i\ket{1})$

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
    + c_1 \left(\bra{1} \left(\dfrac{1}{\sqrt{2}} (\ket{0} + i\ket{1})\right)\right) \right\rvert^{2} \\
    &= \left\lvert \dfrac{c_0}{\sqrt{2}}\left( \braket{0\\|0} \right)
    + \dfrac{ic_1}{\sqrt{2}}\left( \braket{1\\|1} \right) \right\rvert^{2} \\
    &= \left\lvert \dfrac{c_0}{\sqrt{2}} + \dfrac{ic_1}{\sqrt{2}} \right\rvert^{2} \\
    &= \dfrac{1}{2} \lvert c_0 + ic_1 \rvert^{2} \\
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
    p(-1) &= \bra{\psi}P_-\ket{\psi} \\
    &= \braket{\psi\\|-i}\braket{-i\\|\psi} \\
    &= \braket{\psi\\|SH\\|1}\braket{1\\|HS^\dagger\\|\psi} \\
    &= \lvert \braket{1\\|HS^\dagger\\|\psi} \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

We conclude then that the circuit to perform a measurement
of the $\sigma^{(y)}$ observable given a state $\psi$ is as follows:

<div class='figure'>
    <img src='/assets/images/vqe/y-measurement.png'
         style='width: 30%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of the $\sigma^{(y)}$ observable:</span>
        we need to perform a basis change from the $\sigma^{(z)}$ basis to the $\sigma^{(y)}$
        basis using $HS^\dagger$ then perform a standard measurement in the
        $\sigma^{(z)}$ basis. We will get eigenvectors in the $\sigma^{(z)}$ basis
        but the probabilities will correspond to measurements of the
        eigenvalues of $\sigma^{(y)}$.
    </div>
</div>

* **Code for performing the measurement**<br>
We prepare the state $\ket{\psi} = H\ket{0}$ and measure
the $\sigma^{(y)}$ observable with respect to that state.

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
    """Measurement of the Y observable
    using facilities provided by Pennylane.
    """
    # Prepare the state
    qml.Hadamard(wires = 0)

    # Perform the measurement
    return qml.counts(qml.PauliY(0))

@qml.qnode(dev)
def custom_circuit():
    """Custom circuit to measure the Y observable.
    We need to perform a change of basis then
    do a measurement in the standard basis
    as by the circuit above.
    """
    # Prepare the state
    qml.Hadamard(wires = 0)
    
    # Perform a change of basis
    qml.adjoint(qml.S(wires = 0))
    qml.Hadamard(wires = 0)

    # Measure in standard basis
    return qml.counts(qml.PauliZ(0))

if __name__ == "__main__":
    results = circuit()
    print(results)
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>Measurement of the $\sigma^{(y)}$ observable:</span>
    we prepare the state $\ket{\psi} = H\ket{0}$. We note that we obtain
    eigenvalue $+1$ with appromixately $0.5$ probablity and same
    for eigenvalue $-1$. It is easy to verify that this corresponds
    to theoretical predictions.<br>
    Also, note that <code>custom_circuit</code> and <code>circuit</code>
    provide the same results. In the first case, we use the formula derived.
    In the second case, we let Pennylane do it for us.
</div>
</div>

Note that if we prepared eigenvectors of $\sigma^{(y)}$
we will obtain eigenvalues with $100\%$ probability.
That is if we prepare the $\ket{+i} = SH\ket{0}$ state,
we will obtain eigenvalue $+1$ with $100\%$ probability.<br>
Equivalently, if we prepare $\ket{-i} = SH\ket{1}$,
we will obtain eigenvalue $-1$ with $100\%$ probability.

#### Measurement of $\sigma^{(x)}$ with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$
The eigenvalues and eigenvectors of $\sigma^{(x)}$ are calculated
in the [derivations section](#eigenvalues-and-eigenvectors-of-sigmax) and are found to be:

1. Eigenvalue $+1$ with eigenvector $\ket{+} = \dfrac{1}{\sqrt{2}}(\ket{0} + \ket{1})$
2. Eigenvalue $-1$ with eigenvector $\ket{-} = \dfrac{1}{\sqrt{2}}(\ket{0} - \ket{1})$

* **Measurement with respect to $\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$**<br>
We calculate only the probability of obtaining eigenvalue $+1$ given the state
$\ket{\psi} = c_0 \ket{0} + c_1 \ket{1}$. The calculation for eigenvalue
$-1$ is similar and left as an exercise.

{% katexmm %}
$$
\begin{align}
    p(+1) &= \lvert\braket{\psi\\|+}\rvert^{2} \\
    &= \lvert (c_0 \ket{0} + c_1 \ket{1})\ket{+}\rvert^{2} \\
    &= \lvert c_0 \braket{0\\|+} + c_1 \braket{1\\|+} \rvert^{2} \\
    &= \left\lvert c_0 \left(\bra{0} \left(\dfrac{1}{\sqrt{2}} (\ket{0} + \ket{1})\right)\right)
    + c_1 \left(\bra{1} \left(\dfrac{1}{\sqrt{2}} (\ket{0} - \ket{1})\right)\right) \right\rvert^{2} \\
    &= \left\lvert \dfrac{c_0}{\sqrt{2}}\left( \braket{0\\|0} \right)
    - \dfrac{c_1}{\sqrt{2}}\left( \braket{1\\|1} \right) \right\rvert^{2} \\
    &= \left\lvert \dfrac{c_0}{\sqrt{2}} - \dfrac{c_1}{\sqrt{2}} \right\rvert^{2} \\
    &= \dfrac{1}{2} \lvert c_0 - c_1 \rvert^{2} \\
    &= \dfrac{1}{2} \left(\sqrt{c_{0}^{2} + c_{1}^{2}}\right)^{2} \\
    &= \dfrac{c_{0}^{2} + c_{1}^{2}}{2}
\end{align}
$$
{% endkatexmm %}

* **Quantum circuit for performing the measurement**<br>
The measurement of $+1$ corresponds to the use of the projector
$P_+ = \ket{+}\bra{+}$. Therefore we have:

{% katexmm %}
$$
\begin{align}
    p(+1) &= \bra{\psi}P_+\ket{\psi} \\
    &= \braket{\psi\\|+}\braket{+\\|\psi} \\
    &= \braket{\psi\\|H\\|0}\braket{0\\|H\\|\psi} \\
    &= \lvert \braket{0\\|H\\|\psi} \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

Similarly, the measurement of $-1$ corresponds to the use of the projector
$P_- = \ket{-}\bra{-}$ leading to:

{% katexmm %}
$$
\begin{align}
    p(-1) &= \bra{\psi}P_-\ket{\psi} \\
    &= \braket{\psi\\|-}\braket{-\\|\psi} \\
    &= \braket{\psi\\|H\\|1}\braket{1\\|H\\|\psi} \\
    &= \lvert \braket{1\\|H\\|\psi} \rvert^{2} \\
\end{align}
$$
{% endkatexmm %}

We conclude then that the circuit to perform a measurement
of the $\sigma^{(x)}$ observable given a state $\psi$ is as follows:

<div class='figure'>
    <img src='/assets/images/vqe/x-measurement.png'
         style='width: 30%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of the $\sigma^{(x)}$ observable:</span>
        we need to perform a basis change from the $\sigma^{(x)}$ basis to the $\sigma^{(x)}$
        basis using $H$ then perform a standard measurement in the
        $\sigma^{(z)}$ basis. We will get eigenvectors in the $\sigma^{(z)}$ basis
        but the probabilities will correspond to measurements of the
        eigenvalues of $\sigma^{(x)}$.
    </div>
</div>

* **Code for performing the measurement**<br>
We prepare the state $\ket{\psi} = X\ket{0}$ and measure
the $\sigma^{(x)}$ observable with respect to that state.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 10000
)

@qml.qnode(dev)
def circuit():
    """Measurement of the Y observable
    using facilities provided by Pennylane.
    """
    # Prepare the state
    qml.PauliX(wires = 0)

    # Perform the measurement
    return qml.counts(qml.PauliX(0))

@qml.qnode(dev)
def custom_circuit():
    """Custom circuit to measure the X observable.
    We need to perform a change of basis then
    do a measurement in the standard basis
    as by the circuit above.
    """
    # Prepare the state
    qml.PauliX(wires = 0)
    
    # Perform a change of basis
    qml.Hadamard(wires = 0)

    # Measure in standard basis
    return qml.counts(qml.PauliZ(0))

if __name__ == "__main__":
    results = circuit()
    print(results)
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>Measurement of the $\sigma^{(x)}$ observable:</span>
    we prepare the state $\ket{\psi} = \sigma^{(x)} \ket{0}$. We note that we obtain
    eigenvalue $+1$ with appromixately $0.5$ probablity and same
    for eigenvalue $-1$. It is easy to verify that this corresponds
    to theoretical predictions.<br>
    Also, note that <code>circuit</code> and <code>custom_circuit</code>
    provide the same results. In the first case, we use facilities provided by
    PennyLane. In the second case, we use the derived circuit.
</div>
</div>

Note that if we prepared eigenvectors of $\sigma^{(x)}$
we will obtain eigenvalues with $100\%$ probability.
That is if we prepare the $\ket{+} = H\ket{0}$ state,
we will obtain eigenvalue $+1$ with $100\%$ probability.<br>
Equivalently, if we prepare $\ket{-} = H\ket{1}$,
we will obtain eigenvalue $-1$ with $100\%$ probability.

#### Multi-qubits measurement
In order to perform measurements on multiple qubits,
we only need to perform a change of basis on each qubit
individually as dictated by the form of the Hamiltonian.

Let us justify this: we will only consider the case of two qubits
though the procedure can be proven for an arbitrary number of qubits.

Consider a generic 2-qubits Hamiltonian of the form $H = \sigma^{(m)} \otimes \sigma^{(n)}$,
where $n, m \in \\{i, x, y, z\\}$.
We would like to know how to find the probabilities corresponding
to the eigenvalues of $H$.

First, we find the projectors:

{% katexmm %}
$$
\begin{align}
    H &= \sum_i \lambda^{(m)}_i P^{(m)}_i \otimes \sum_j \lambda^{(n)}_j P^{(n)}_j \\
    &= \sum_{i,j} \lambda^{(m)}_i \cdot \lambda^{(n)}_j \left(P^{(m)}_i \otimes P^{(n)}_j\right) \\
    &= \sum_r \lambda_r P_r
\end{align}
$$
{% endkatexmm %}

Where we set $\lambda_r = \lambda^{(m)}_i \cdot \lambda^{(n)}_j$
and $P_r = P^{(m)}_i \otimes P^{(n)}_j$.

In general, if $H = \bigotimes_{k} \sigma_{k}^{(l)}$ with $l \in \\{i, x, y, z\\}$
then we have:

{% katexmm %}
$$
H = \sum_{r=0}^{2^k-1} \left(\prod_{k} \lambda_r^{(k)} \bigotimes_{k} P^{(k)}_r\right)
$$
{% endkatexmm %}

From the equation above, we conclude that the projectors of $H$
are $\bigotimes_{k} P^{(k)}_r$.

Then, we find the probability of measuring an arbitrary eigenvalue $\lambda_r$.
Again, we only make the derivation for the 2-qubits case and make a general
statement for the multiple-qubits case:

{% katexmm %}
$$
\begin{align}
    p(\lambda_r) &= \bra{\psi} P_r \ket{\psi} \\
    &= \bra{\psi} (P_r^{(m)} \otimes P_r^{(n)}) \ket{\psi} \\
    &= \bra{\psi} \left(\ket{m}\bra{m} \otimes \ket{n}\bra{n}\right) \ket{\psi} &P_r^{(\star)} = \ket{\star}\bra{\star} \\
    &= \bra{\psi} \left((\overbrace{G_m\ket{0_m}}^{A}\overbrace{\bra{0_m}G_m^\dagger}^{B}) \otimes
    (\overbrace{G_n\ket{0_n}}^{C}\overbrace{\bra{0_n}G_n^\dagger}^{D}) \right) \ket{\psi} &\ket{\star}=G_\star\ket{0_\star} \\
    &= \bra{\psi} \left( (\overbrace{G_m}^{A'}\overbrace{\ket{0_m}}^{B'} \otimes \overbrace{G_n}^{C'}\overbrace{\ket{0_n}}^{D'})
    (\overbrace{\bra{0_m}}^{A'}\overbrace{G_m^\dagger}^{B'} \otimes \overbrace{\bra{0_n}}^{C'}\overbrace{G_n^\dagger}^{D'}) \right) \ket{\psi}
    &(AB)\otimes(CD)=(A\otimes C)(B\otimes D) \\
    &= \Big( \bra{\psi} (G_m \otimes G_n) (\ket{0_m} \otimes \ket{0_n})\Big)\Big((\bra{0_m} \otimes \bra{0_n}) (G_m^\dagger \otimes G_n^\dagger) \ket{\psi} \Big)
    &(A'B')\otimes(C'D')=(A'\otimes C')(B'\otimes D') \\
    &= \lvert \bra{0_m0_n} G_m^\dagger \otimes G_n^\dagger \ket{\psi} \rvert^2
\end{align}
$$
{% endkatexmm %}

The choice $\ket{\star} = G_\star \ket{0_ \star}$ is arbitrary.
It could have been $\ket{\star}=G_\star\ket{1_\star}$ and the result would
still be similar. In fact, the reader is encouraged to do that calculation.

The main point is that given a 2-qubits state $\ket{\psi}$
we just need to apply the gate $G_m^\dagger$ to the first qubit
and the gate $G_n^\dagger$ to the second
qubit then measure in the standard basis.

In general, given a multi-qubits Hamiltonian with spectral decomposition
$H = \sum_{r=0}^{2^k-1} \left(\prod_{k} \lambda_r^{(k)} \bigotimes_{k} P^{(k)}_r\right)$
where $P^{(k)} = G_k\ket{0_k}\bra{0_k}G_k^\dagger$, the probability of measuring
eigenvalue $\lambda_r = \prod_k \lambda_r^{(k)}$ is given by:

{% katexmm %}
$$
p(\lambda_r) = \Bigg\lvert \bra{\star}^{\otimes k} \Big(\bigotimes_k G_k^{\dagger}\Big) \ket{\psi} \Bigg\rvert^2
$$
{% endkatexmm %}

Where $\bra{\star} = \bra{0}$ or $\bra{\star} = \bra{1}$.

We therefore conclude that the circuit to perform a measurement
of the $H = \bigotimes_{k} \sigma_{k}^{(l)}$ observable
given a state $\ket{\psi}$ is given by the circuit that follows:

<div class='figure'>
    <img src='/assets/images/vqe/generalized-measurements.png'
         style='width: 30%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of the
        $H= \bigotimes_{k} \sigma_{k}^{(l)}$ observable:</span>
        even though only three wires/qubits are shown, the state
        $\ket{\psi}^{\otimes k}$ is built from $k$ qubits,
        whence the dashed lines.
    </div>
</div>

* **Example 1: measurement of $H = \sigma^{(x)} \otimes \sigma^{(z)}$**<br>
As a first example, we will measure $H = \sigma^{(x)} \otimes \sigma^{(z)}$
with respect to one of its ground states.

So let's start by finding the eigenvalues and eigenvectors of $H$.
As the number of qubits increases, it gets difficult to do the calculations
manually. That's why we will use Numpy to do the calculation for us.

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np
import numpy.linalg as la

if __name__ == "__main__":
    H = np.matrix([
        [0,  0,  1,  0],
        [0,  0,  0, -1],
        [1,  0,  0,  0],
        [0, -1,  0,  0]
    ])

    eigvals, eivecs = la.eig(H)
    print(eigvals)
    print(eigvecs)
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Eigenvalues and eigenvectors of $H = \sigma^{(x)} \otimes \sigma^{(z)}$:
    </span>
    we find that $H$ has a degenerate ground state energy,
    that is there are two states with the same ground state energy $-1$.
</div>
</div>

We therefore see that $H$ has the following eigenvalues and eigenvectors:

1. Eigenvalue $-1$ has eigenvectors:
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & -1 & 0\end{bmatrix}^\intercal$
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & 1 & 0 & 1\end{bmatrix}^\intercal$

2. Eigenvalue $+1$ has eigenvectors:
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & 1 & 0\end{bmatrix}^\intercal$
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & -1 & 0 & 1\end{bmatrix}^\intercal$

Therefore, should we prepare the state
$\ket{\psi} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & -1 & 0\end{bmatrix}^\intercal = \dfrac{1}{\sqrt{2}}(\ket{00}-\ket{10})$,
we should expect to measure eigenvalue $-1$ with probability $1$.
The circuit that prepares that state and performs the measurement
of $H$ against that state follows:

<div class='figure'>
    <img src='/assets/images/vqe/xz-groundstate.png'
         style='width: 45%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of $H = \sigma^{(x)} \otimes \sigma^{(z)}$
        against the state $\ket{\psi} = \dfrac{1}{\sqrt{2}}(\ket{00}-\ket{10})$:</span>
        the gates before the zigzag lines correspond to the state preparation.
        The gates afterwards correspond to the change of basis before
        performing the measurement in the standard basis.<br>
        <i>A simplification would result in cancellation of two $H$ gates
        acting on the first qubits. They are left for the purpose of
        clarity and completeness.</i>
    </div>
</div>

The code that follows implements the circuit above.
The regular `circuit` function shows the implementation
using PennyLane facilities.
The `custom_circuit` does the implementation as per the
figure above.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 10000
)

@qml.qnode(dev)
def circuit():
    """Measurement of the XZ observable
    using facilities provided by Pennylane.
    """
    # Prepare the state
    qml.PauliZ(wires = 0)
    qml.Hadamard(wires = 0)

    # Perform the measurement
    # The @ operator calculates the tensor product
    return qml.counts(qml.PauliX(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def custom_circuit():
    """
    """
    # Prepare the state
    qml.PauliZ(wires = 0)
    qml.Hadamard(wires = 0)
    
    # Perform a change of basis
    qml.Hadamard(wires = 0)

    # Measure in standard basis
    return qml.counts(qml.PauliZ(0) @ qml.PauliZ(1))

if __name__ == "__main__":
    print(circuit())
    print(custom_circuit())
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>Measurement of $H$:</span>
    Both <code>circuit</code> and <code>custom_circuit</code>
    should yield eigenvalue $-1$ with probability $1$.
</div>
</div>

<div class='figure figure-alert' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        Exercise
    </div>
    The reader is encouraged to find the circuits that prepare the remaining
    $3$ eigenvectors and verify that the corresponding eigenvalues
    are computed with the predicted probability of $100\%$.
</div>
</div>

* **Example 2: measurement of $H = \sigma^{(z)} \otimes \sigma^{(i)}$**<br>
For our second example, we will measure $H = \sigma^{(z)} \otimes \sigma^{(i)}$
with respect to one of its ground states.

The eigenvalues and eigenvectors are calculated as before and are found to be:

1. Eigenvalue $-1$ has eigenvectors:
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & 1 & 0 & 0\end{bmatrix}^\intercal$
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & 0 & 0 & 1\end{bmatrix}^\intercal$

2. Eigenvalue $+1$ has eigenvectors:
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & 0 & 0\end{bmatrix}^\intercal$
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & 0 & 1 & 0\end{bmatrix}^\intercal$

We will prepare $\ket{\psi} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & 0 & 0 & 1\end{bmatrix}^\intercal = \ket{11}$
and measure $H$ against that state.

The circuit that prepares $\ket{\psi}$ and measures $H$ against that
state is in the figure that follows:

<div class='figure'>
    <img src='/assets/images/vqe/zi-groundstate.png'
         style='width: 30%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Measurement of $H = \sigma^{(z)} \otimes \sigma^{(i)}$
        against the state $\ket{\psi} = \ket{11}$:</span>
        we need only measure the first qubit. This is equivalent
        to measuring both qubits in the standard basis.
    </div>
</div>

The code that follows implements the figure above.
Notice how we don't tensor $\sigma^{(z)}$ with $\sigma^{(i)}$
in the code. We just perform a measurement on the first qubit.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 10000
)

@qml.qnode(dev)
def circuit():
    """Measurement of the ZI observable
    using facilities provided by Pennylane.
    """
    # Prepare the state
    qml.PauliX(wires = 0)
    qml.PauliX(wires = 1)

    # Perform the measurement
    return qml.counts(qml.PauliZ(0))

if __name__ == "__main__":
    print(circuit())
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>Measurement of $H$:</span>
    we only need to measure the first qubit when handed a Hamiltonian
    where there is a tensor with the identity.
</div>
</div>

<div class='figure figure-alert' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        Exercise
    </div>
    The reader is encouraged to find the circuits that prepare the remaining
    $3$ eigenvectors and verify that the corresponding eigenvalues
    are computed with the predicted probability of $100\%$.
</div>
</div>

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

If the observable is of the form $H = \sum_i h_i H_i$ then the expectation value of $H$
is easily verified to be given by:

{% katexmm %}
$$
\braket{H} = \sum_i h_i \braket{H_i} \tag{4}
$$
{% endkatexmm %}

#### Example 1: expectation value of $H = \sigma^{(x)}$
We quickly confirm that if we prepare the ground state
of $\sigma^{(x)}$, the expectation value will correspond
to the ground state energy of $\sigma^{(x)}$ since
we will obtain the ground state energy $-1$ with probability $1$

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 100000
)

@qml.qnode(dev)
def expval():
    qml.PauliX(wires = 0)
    qml.Hadamard(wires = 0)
    return qml.expval(qml.PauliX(0))

if __name__ == "__main__":
    print(expval()) # should print -1
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>Measurement of $H = \sigma^{(x)}$:</span>
    the circuit prepare the ground state of $H = \sigma^{(x)}$
    therefore the expectation value will be the ground state
    energy.
</div>
</div>

#### Example 2: expectation value of $H = \dfrac{1}{\sqrt{2}}\left(\sigma^{(x)}+\sigma^{(z)}\right)$
A quick calculation shows that $H$ has the following eigendecomposition:

1. Eigenvalue $-1$ with eigenvector $\dfrac{1}{\sqrt{4+2\sqrt{2}}} \begin{bmatrix} 1-\sqrt{2} \\ 1\end{bmatrix}$
2. Eigenvalue $+1$ with eigenvector $\dfrac{1}{\sqrt{4+2\sqrt{2}}} \begin{bmatrix} 1+\sqrt{2} \\ 1\end{bmatrix}$

Unlike the previous examples, it is not clear how to prepare the ground state by inspection.
So we can't readily generate a circuit and compute the expectation value that
would result in the ground state energy.

<div class='figure figure-alert figure-info' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        The ground state is generally unkown
    </div>
    If we knew the ground state, we would not need VQE because
    computing the ground state energy would simply amount to
    preparing the ground state and measuring the expectation
    value of the Hamiltonian with respect to the prepared
    state.
</div>
</div>

So we will prepare some generic state and measure the expectation value
with respect to that state. The result is not important, it is how
we achieve that result that's important.

*Note: PennyLane doesn't exactly make it possible to iterate*
*through the counts we get upon measurement so it is not easy*
*for us to manually compute the expectation value according*
*to equation $(\href{#mjx-eqn:3'}{3'})$.*
*We will directly use their provided function for computing*
*the expectation value and do a simple sanity check.*

In the code that follows, we compute the expectation value
according to equation $(\href{#mjx-eqn:4}{4})$ then ask PennyLane do the same
calculation for us and compare the result.

The sanity check depends on the fact that PennyLane
already has our observable $H$ as the $Hadamard$ observable.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml
from pennylane import numpy as np

# We fix the seed to make results reproducible
np.random.seed(1)

dev = qml.device(
    "default.qubit",
    wires = 1,
    # We request the exact expectation value by not setting shots
    shots = None
)

@qml.qnode(dev)
def x_expval(y):
    qml.RY(y, wires = 0)
    return qml.expval(qml.PauliX(0))

@qml.qnode(dev)
def z_expval(y):
    qml.RY(y, wires = 0)
    return qml.expval(qml.PauliZ(0))

def h_expval(y):
    return (1/np.sqrt(2)) * (x_expval(y) + z_expval(y))

@qml.qnode(dev)
def hadamard_expval(y):
    qml.RY(y, wires = 0)
    return qml.expval(qml.Hadamard(0))

if __name__ == "__main__":
    custom_expval = h_expval(np.pi)
    builtin_expval = hadamard_expval(np.pi)
    print(custom_expval)
    print(builtin_expval)
    print(custom_expval == builtin_expval) # should print True
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Expectation value of $H = \dfrac{1}{\sqrt{2}}\left(\sigma^{(x)}+\sigma^{(z)}\right)$:
    </span>
    we see a confirmation of equation $(\href{#mjx-eqn:4}{4})$ since both
    <code>custom_expval</code> and <code>builtin_expval</code>
    contain the same value.
</div>
</div>

#### Example 3: expectation value of $H = \sigma^{(x)} \otimes \sigma^{(z)} + \sigma^{(i)} \otimes \sigma^{(z)}$
It is easily and quickly verified that $H$ has the following eigendecomposition:

1. Eigenvalue $-2$ has eigenvector $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & 1 & 0 & 1\end{bmatrix}^\intercal$
2. Eigenvalue $0$ has two eigenvectors:
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & -1 & 0\end{bmatrix}^\intercal$
    - $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 0 & -1 & 0 & 1\end{bmatrix}^\intercal$
3. Eigenvalue $2$ has eigenvector $\dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 & 0 & 1 & 0\end{bmatrix}^\intercal$

Therefore, if we compute the expectation value with respect to the
state $\ket{\psi} = \dfrac{1}{\sqrt{2}} \left( \ket{01} + \ket{11} \right)$
we should get the eigenvalue $-2$.
The code below confirms that.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml

dev = qml.device(
    "default.qubit",
    wires = 2,
    shots = 100000
)

@qml.qnode(dev)
def xz_expval():
    qml.Hadamard(wires = 0)
    qml.PauliX(wires = 1)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def zi_expval():
    qml.Hadamard(wires = 0)
    qml.PauliX(wires = 1)
    return qml.expval(qml.PauliZ(1))

def h_expval():
    return xz_expval() + zi_expval()

if __name__ == "__main__":
    print(h_expval()) # should print -2
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Expectation value of $H = \sigma^{(x)} \otimes \sigma^{(z)} + \sigma^{(i)} \otimes \sigma^{(z)}$:
    </span>
    since we prepared the ground state $\ket{\psi} = \ket{+}\ket{1}$
    the expectation value yields the ground state energy $-2$.
</div>
</div>

### The variational method
From basic quantum mechanics we know that every system has a lowest
energy, call it $\lambda_0$.
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
\braket{H} \ge \lambda_0 \tag{5}
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
\bra{\psi(\vec{\theta})}H\ket{\psi(\vec{\theta})} \approx \lambda_0$.
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

That is our goal is to find parameters $\vec{\theta}$ that minimize
$\mathcal{C}(\vec{\theta})$.

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

Each state $\ket{\psi(\vec{\theta})}$ where $\vec{\theta}$ is fixed is called an *ansatz*.
We will use circuits that have arbitrary rotations about some axis
to construct those ansätze. The circuits used to prepare arbitrary
ansätze are called *parametrized quantum circuits* ($a.k.a.$ PQCs).

#### The algorithm
While there will be slight variations in implementations,
the flow of VQE is quite the same across all implementations.
We present that flow as an algorithm:

<div class='figure'>
<div class='algorithm' markdown='1'>
**Prepare:**  
$\quad cost(\vec{\theta}) = \bra{\vec{\theta}}H\ket{\vec{\theta}}$  
$\quad optimizer = Optimizer()$  

**Initialize:**  
$\quad maxiter > 0$  
$\quad iter = 0$  
$\quad \vec{\theta} = rand()$  
$\quad energy = cost(\vec{\theta})$  

**while** $iter < maxiter$**:**  
$\qquad \vec{\theta}, energy \gets optimizer(cost, \vec{\theta})$  
$\qquad iter \gets iter + 1$  

**return** $energy$
</div>
<div class='caption'>
    <span class='caption-label'>
        VQE algorithm:
    </span>
    we prepare the cost function as a circuit that prepares
    the state $\ket{\psi(\vec{\theta})}$, initialize
    the parameters $\vec{\theta}$ to some random values
    and let the optimizer take it from there for a maximum
    number of iterations after which we report the
    computed energy.
</div>
</div>

### Examples
Let us work through a couple of examples where we try to find
their ground state energies. We have calculated those energies
before, now we use VQE to find the same.

* **Ansatz design:**<br>
For both examples, we rely on arbitrary state preparation circuits.
This means that starting from fiduciary states $\ket{0}$ and $\ket{00}$,
we create circuits that allow us to generate arbitrary single-qubit and
two-qubits states.  
The PQC for single-qubit systems is derived in [single-qubit state preparation](#single-qubit-state-preparation).  
And a PQC for two-qubits systems is derived in [two-qubits state preparation](#two-qubits-state-preparation).  

* **Optimizer selection:**<br>
We chose the SPSA optimizer because it works out of the box
without requiring additional knowledge beyond what we have already learned
thus far. When we look at gradient descent, we will seee we require
the ability to find the gradient of the cost function and we haven't learned how.


#### Example 1: ground state energy of $H = \dfrac{1}{\sqrt{2}}\left(\sigma^{(x)}+\sigma^{(z)}\right)$
We already know from [calculating the expectation value of $H$](#example-2-expectation-value-of-h--dfrac1sqrt2leftsigmaxsigmazright)
that it has ground state energy $-1$. We just couldn't manually construct
the ground state itself.

The code that follows implements VQE as per the algorithm above and it does
find the ground state energy. We plot the optimization steps in the figure
that follows the code so we can see the optimizer in action.

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt

dev = qml.device(
    "default.qubit",
    wires = 1,
    shots = 100000
)

@qml.qnode(dev)
def hadamard_cost(theta):
    qml.RY(theta[1], wires = 0)
    qml.PhaseShift(theta[0], wires = 0)
    return qml.expval(qml.Hadamard(0))

def vqe(cost, theta, maxiter):
    optimizer = qml.SPSAOptimizer(maxiter = maxiter)
    energy = cost(theta)
    history = [energy]

    for iter in range(maxiter):
        theta, energy = optimizer.step_and_cost(
            cost,
            theta
        )

        # Print the optimizer progress every 10 steps
        if iter % 10 == 0:
            print(f"Step = {iter},  Energy = {history[-1]:.8f}")
        
        # Save the full energy optimization history
        history.append(energy)
    
    return energy, history

if __name__ == "__main__":
    # Initialize theta from the normal distribution with mean 0 and spread np.pi
    # The last argument is set to 2
    # because we need to pass 2 parameters to the cost function
    init_theta = np.random.normal(0, np.pi, 2)
    
    # We try 151 iterations
    maxiter = 151

    # Run VQE
    energy, history = vqe(hadamard_cost, init_theta, maxiter)

    # Print the final energy
    print(energy)

    # Plot the optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(range(maxiter + 1), history, "go", ls = "dashed", label = "Energy")
    plt.xlabel("Optimization step", fontsize=13)
    plt.ylabel("Energy", fontsize=13)
    plt.show()
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Ground state energy of $H = \dfrac{1}{\sqrt{2}}\left(\sigma^{(x)}+\sigma^{(z)}\right)$:
    </span>
    while we may not get exactly $-1$, we will get comfortably close to it.
</div>
</div>

We can see how the optimizer gets closer to the ground state energy
even though the initial energy estimation is not too far away from the true
energy:

<div class='figure' markdown='1'>
{% highlight text %}
Step = 0,  Energy = -0.84014000
Step = 10,  Energy = -0.92522000
Step = 20,  Energy = -0.96302000
Step = 30,  Energy = -0.97982000
Step = 40,  Energy = -0.98670000
Step = 50,  Energy = -0.99214000
Step = 60,  Energy = -0.99522000
Step = 70,  Energy = -0.99638000
Step = 80,  Energy = -0.99744000
Step = 90,  Energy = -0.99860000
Step = 100,  Energy = -0.99874000
Step = 110,  Energy = -0.99902000
Step = 120,  Energy = -0.99922000
Step = 130,  Energy = -0.99936000
Step = 140,  Energy = -0.99960000
Step = 150,  Energy = -0.99970000
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Optimization evolution every 10 steps:
    </span>
    sometimes we will start close to the true ground state energy,
    other times not. But we clearly see the optimizer converging
    towards the true ground state energy.
</div>
</div>

The plot generated by the code above should help drive
home the point of how VQE works:

<div class='figure'>
    <img src='/assets/images/vqe/h-vqe.png'
         style='width: 80%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>VQE optimization landscape for
        $H = \dfrac{1}{\sqrt{2}}\left(\sigma^{(x)}+\sigma^{(z)}\right)$:</span>
        we can clearly see the optimizer approaching the true ground state energy
        of our Hamiltonian $H$.
    </div>
</div>

#### Example 2: ground state energy of $H = \sigma^{(x)} \otimes \sigma^{(z)} + \sigma^{(i)} \otimes \sigma^{(z)}$
The procedure is pretty much the same as with the Hadamard
observable, except we will make use of equation $(\href{#mjx-eqn:4}{4})$
since we will calculate the ground state energies of
$\sigma^{(x)} \otimes \sigma^{(z)}$ and $\sigma^{(i)} \otimes \sigma^{(z)}$
separately.

Without further ado, here is the code:

<div class='figure' markdown='1'>
{% highlight python %}
import pennylane as qml
import pennylane.numpy as np

dev = qml.device(
    "default.qubit",
    wires = 2,
    shots = 100000
)

def ansatz(params):
    qml.RY(params[0], wires = 1)
    qml.PhaseShift(params[3], wires = 1)
    qml.CRY(params[1], wires = [1, 0])
    qml.ControlledPhaseShift(params[5] - params[3], wires = [0, 1])
    qml.CRY(params[2], wires = [0, 1])
    qml.PauliX(wires = 1)
    qml.ControlledPhaseShift(params[4] - params[5], wires = [1, 0])
    qml.PauliX(wires = 1)

@qml.qnode(dev)
def xz_cost(params):
    ansatz(params)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

@qml.qnode(dev)
def iz_cost(params):
    ansatz(params)
    return qml.expval(qml.PauliZ(1))

def vqe(cost, params, maxiter):
    optimizer = qml.SPSAOptimizer(maxiter = maxiter)
    energy = cost(params)
    history = [energy]

    for iter in range(maxiter):
        params, energy = optimizer.step_and_cost(
            cost,
            params
        )

        # Print the optimizer progress every 40 steps
        if iter % 40 == 0:
            print(f"Step = {iter},  Energy = {history[-1]:.8f}")
        
        # Save the full energy optimization history
        history.append(energy)
    
    return energy, history

if __name__ == "__main__":
    # Initialize params from the normal distribution with mean 0 and variance np.pi
    init_params = np.random.normal(0, np.pi, 6)
    
    # We try 401 iterations
    maxiter = 401

    # Run VQE
    print("Optimizer progress for the XZ observable:")
    xz_energy, _ = vqe(xz_cost, init_params, maxiter)
    print("\nOptimizer progress for the IZ observable:")
    iz_energy, _ = vqe(iz_cost, init_params, maxiter)

    # Print the final energy
    energy = xz_energy + iz_energy
    print("\nFinal energy:", energy)
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Ground state energy of $H = \sigma^{(x)} \otimes \sigma^{(z)} + \sigma^{(i)} \otimes \sigma^{(z)}$:
    </span>
    we should expect to get an energy close to $-2$.
</div>
</div>

And here is a sample run on my machine:

<div class='figure' markdown='1'>
{% highlight text %}
Optimizer progress for the XZ observable:
Step = 0,  Energy = 0.81258000
Step = 40,  Energy = 0.38758000
Step = 80,  Energy = 0.04470000
Step = 120,  Energy = -0.10830000
Step = 160,  Energy = -0.25992000
Step = 200,  Energy = -0.51630000
Step = 240,  Energy = -0.72498000
Step = 280,  Energy = -0.80484000
Step = 320,  Energy = -0.88252000
Step = 360,  Energy = -0.90248000
Step = 400,  Energy = -0.93032000

Optimizer progress for the IZ observable:
Step = 0,  Energy = -0.63556000
Step = 40,  Energy = -0.93758000
Step = 80,  Energy = -0.97082000
Step = 120,  Energy = -0.97892000
Step = 160,  Energy = -0.98604000
Step = 200,  Energy = -0.98908000
Step = 240,  Energy = -0.99122000
Step = 280,  Energy = -0.99288000
Step = 320,  Energy = -0.99420000
Step = 360,  Energy = -0.99448000
Step = 400,  Energy = -0.99518000

Final energy: -1.92782
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Optimization evolution every 40 steps and final energy:
    </span>
    while we didn't get exactly $-2$, we got pretty close to it.
</div>
</div>

### Next steps
The reader who just wanted to get the basics and play a little
should free to stop here.

For the reader that wants to delve a
little deeper, the sections that follow elaborate on ways
ansätze are designed, what choices of optimizers we have,
and what is meant by combining observables that are commuting.

Even the reader who just wanted to get the basics is encouraged
to read the final section on practical considerations
so they understand the limitations of VQE, especially
the *measurement problem*.

## Ansatz design

## Optimizer selection

## Observable reduction

## Practical considerations

## Derivations

### Eigenvalues and eigenvectors of $\sigma^{(z)}$
#### Eigenvalues
{% katexmm %}
$$
\begin{align}
    \det\begin{vmatrix} \sigma^{(z)} - \lambda \sigma^{(i)} \end{vmatrix} &= 0 \\
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

The eigenvalues of $\sigma^{(z)}$ are $\lambda_0 = +1$ and $\lambda_1 = -1$.

#### Eigenvectors
* *Eigenvector corresponding to eigenvalue $\lambda_0 = +1$*
    {% katexmm %}
    $$
    \begin{align}
        \sigma^{(z)} \ket{\lambda_+} &= +1 \ket{\lambda_+} \\
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
        \sigma^{(z)} \ket{\lambda_-} &= -1 \ket{\lambda_-} \\
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

### Eigenvalues and eigenvectors of $\sigma^{(y)}$
#### Eigenvalues
{% katexmm %}
$$
\begin{align}
    \det\begin{vmatrix} \sigma^{(y)} - \lambda \sigma^{(i)} \end{vmatrix} &= 0 \\
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

The eigenvalues of $\sigma^{(y)}$ are $\lambda_+ = +1$ and $\lambda_- = -1$.

#### Eigenvectors
* *Eigenvector corresponding to eigenvalue $\lambda_+ = +1$*
    {% katexmm %}
    $$
    \begin{align}
        \sigma^{(y)} \ket{\lambda_+} &= +1 \ket{\lambda_+} \\
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
    $\ket{\lambda_+} = \begin{bmatrix}c_0 \\ i c_0\end{bmatrix}$

    Using the normalization condition,
    we find that $\braket{\lambda_+|\lambda_+}=1$ implies $2|c_0|^2=1$
    from which it follows that $c_0 = \dfrac{1}{\sqrt{2}}$.
    Consequently $c_1 = \dfrac{i}{\sqrt{2}}$.

    Thus $\ket{\lambda_+} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ i \end{bmatrix}$.
    This eigenvector is also written as $\ket{+i} = \ket{\lambda_+}$.
    Expressed in the $\sigma^{(z)}$ basis, $\ket{+i} = \dfrac{1}{\sqrt{2}}(\ket{0} + i\ket{1})$.

* *Eigenvector corresponding to eigenvalue $\lambda_- = -1$*
    {% katexmm %}
    $$
    \begin{align}
        \sigma^{(y)} \ket{\lambda_-} &= -1 \ket{\lambda_-} \\
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
    $\ket{\lambda_-} = \begin{bmatrix}ic_1 \\ c_1\end{bmatrix}$

    Using the normalization condition,
    we find that $\braket{\lambda_-|\lambda_-}=1$ implies $2|c_1|^2=1$
    from which it follows that $c_1 = \dfrac{1}{\sqrt{2}}$.
    Consequently $c_1 = -\dfrac{i}{\sqrt{2}}$.

    Thus $\ket{\lambda_-} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -i \end{bmatrix}$.
    This eigenvector is also written as $\ket{-i} = \ket{\lambda_-}$.
    Expressed in the $\sigma^{(z)}$ basis, $\ket{-i} = \dfrac{1}{\sqrt{2}}(\ket{0} - i\ket{1})$.

### Eigenvalues and eigenvectors of $\sigma^{(x)}$
#### Eigenvalues
{% katexmm %}
$$
\begin{align}
    \det\begin{vmatrix} \sigma^{(x)} - \lambda \sigma^{(i)} \end{vmatrix} &= 0 \\
    \implies
    \det\begin{vmatrix}
        \begin{bmatrix}
        -\lambda & 1 \\
        1 & -\lambda
        \end{bmatrix}
    \end{vmatrix} &= 0 \\
    \implies \lambda^{2}-1 &= 0 \\
    \implies \lambda &= \pm 1
\end{align}
$$
{% endkatexmm %}

The eigenvalues of $\sigma^{(x)}$ are $\lambda_+ = +1$ and $\lambda_- = -1$.

#### Eigenvectors
* *Eigenvector corresponding to eigenvalue $\lambda_+ = +1$*
    {% katexmm %}
    $$
    \begin{align}
        \sigma^{(x)} \ket{\lambda_+} &= +1 \ket{\lambda_+} \\
        \implies
        \begin{bmatrix}
        0 & 1 \\
        1 & 0
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
        c_1 \\
        c_0
        \end{bmatrix}
        &=
        \begin{bmatrix}
        c_0 \\
        c_1
        \end{bmatrix} \\
        \implies
        \begin{cases}
        c_1 &= c_0 \\
        c_0 &= c_1
        \end{cases}
    \end{align}
    $$
    {% endkatexmm %}

    Using $c_1 = c_0$, we transform $\ket{\lambda_+}$ as follows:
    $\ket{\lambda_+} = \begin{bmatrix}c_0 \\ c_0\end{bmatrix}$

    Using the normalization condition,
    we find that $\braket{\lambda_+|\lambda_+}=1$ implies $2|c_0|^2=1$
    from which it follows that $c_0 = \dfrac{1}{\sqrt{2}}$.
    Consequently $c_1 = \dfrac{1}{\sqrt{2}}$.

    Thus $\ket{\lambda_+} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.
    This eigenvector is also written as $\ket{+} = \ket{\lambda_+}$.
    Expressed in the $\sigma^{(z)}$ basis, $\ket{+} = \dfrac{1}{\sqrt{2}}(\ket{0} + \ket{1})$.

* *Eigenvector corresponding to eigenvalue $\lambda_- = -1$*
    {% katexmm %}
    $$
    \begin{align}
        \sigma^{(x)} \ket{\lambda_-} &= -1 \ket{\lambda_-} \\
        \implies
        \begin{bmatrix}
        0 & 1 \\
        1 & 0
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
        c_1 \\
        c_0
        \end{bmatrix}
        &=
        \begin{bmatrix}
        -c_0 \\
        -c_1
        \end{bmatrix} \\
        \implies
        \begin{cases}
        c_1 &= -c_0 \\
        c_0 &= -c_1
        \end{cases}
    \end{align}
    $$
    {% endkatexmm %}

    Using $c_0 = -c_1$, we transform $\ket{\lambda_-}$ as follows:
    $\ket{\lambda_-} = \begin{bmatrix}c_0 \\ -c_0\end{bmatrix}$

    Using the normalization condition,
    we find that $\braket{\lambda_-|\lambda_-}=1$ implies $2|c_0|^2=1$
    from which it follows that $c_0 = \dfrac{1}{\sqrt{2}}$.
    Consequently $c_1 = -\dfrac{1}{\sqrt{2}}$.

    Thus $\ket{\lambda_-} = \dfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1\end{bmatrix}$.
    This eigenvector is also written as $\ket{-} = \ket{\lambda_-}$.
    Expressed in the $\sigma^{(z)}$ basis, $\ket{-} = \dfrac{1}{\sqrt{2}}(\ket{0} - \ket{1})$.

### Parametrized quantum circuits via state preparation
We can use state preparation circuits as a starting point
for the design of parametrized quantum circuits.

We do derivations for single-qubit and two-qubits states
though the procedure can be extented to multiple qubits.

*Note: except for the single-qubit case, any circuit*
*of more than one qubit obtained by the procedure below*
*will be inefficient depth-wise and CNOT count wise.*

#### Single-qubit state preparation
A single qubit has the trigonometric parametrization:

{% katexmm %}
$$
\begin{align}
\ket{\psi} = \cos\dfrac{\theta}{2} \ket{0} + {\rm e}^{i\phi} \sin\dfrac{\theta}{2} \ket{1}
\end{align}
$$
{% endkatexmm %}

Our task then is to design a circuit that would prepare such a state
starting from the state $\ket{0}$.

We begin by noting that
$RY(\theta)\ket{0} = \cos\dfrac{\theta}{2} \ket{0} + \sin\dfrac{\theta}{2} \ket{1}$.
We also know that application of the phase shift gate confers
a phase to a qubit in the $\ket{1}$ but does nothing to the $\ket{0}$ state.

So we it follows that
$P(\phi)RY(\theta)\ket{0} = \cos\dfrac{\theta}{2} \ket{0} + {\rm e}^{i\phi} \sin\dfrac{\theta}{2} \ket{1}$.

And thus we have our circuit:

<div class='figure'>
    <img src='/assets/images/vqe/single-qubit-state-preparation.png'
         style='width: 35%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Preparation of an arbitrary single qubit state:</span>
        we apply a rotation about Y then a phase shift gate.
    </div>
</div>

#### Two-qubits state preparation
A two-qubits state has the trigonometric parametrization:

{% katexmm %}
$$
\begin{align}
\ket{\psi}  &= \cos\dfrac{\theta_1}{2} \ket{00} \\
            &+ {\rm e}^{i\phi_1} \sin\dfrac{\theta_1}{2} \cos\dfrac{\theta_2}{2} \ket{01} \\
            &+ {\rm e}^{i\phi_2} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \cos\dfrac{\theta_3}{2}  \ket{10} \\
            &+ {\rm e}^{i\phi_3} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \sin\dfrac{\theta_3}{2}  \ket{11} \\
\end{align}
$$
{% endkatexmm %}

We design the circuit by following the exact same steps as for the single qubit case,
starting from the $\ket{00}$ state.

1. Apply $RY(\theta_1)$ to qubit $1$:
    {% katexmm %}
    $$
    \begin{align}
    \ket{\psi_1} &= RY_{1}(\theta_1) \ket{00} \\
                &= \cos\dfrac{\theta_1}{2} \ket{00} + \sin\dfrac{\theta_1}{2} \ket{01}
    \end{align}
    $$
    {% endkatexmm %}

2. Apply $P(\phi_1)$ to qubit $1$:
    {% katexmm %}
    $$
    \begin{align}
    \ket{\psi_2} &= P_{1}(\phi_1) \ket{\psi_1} \\
                &= \cos\dfrac{\theta_1}{2} \ket{00} + {\rm e}^{i\phi_1} \sin\dfrac{\theta_1}{2} \ket{01}
    \end{align}
    $$
    {% endkatexmm %}

3. Apply controlled-$RY(\phi_{2})$ to qubit $0$ if qubit $1$ is set:
    {% katexmm %}
    $$
    \begin{align}
    \ket{\psi_3} &= CRY^{1}_{1\to 0}(\theta_2) \ket{\psi_2} \\
                &= \cos\dfrac{\theta_1}{2} \ket{00} \\
                &+ \underbrace{ {\rm e}^{i\phi_1} }_{\text{wrong phase}} \sin\dfrac{\theta_1}{2} \cos\dfrac{\theta_2}{2} \ket{01} \\
                &+ {\rm e}^{i\phi_1} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \ket{11}
    \end{align}
    $$
    {% endkatexmm %}
    Comparing with the original state $\ket{\psi}$ we are trying to construct,
    it clear that the term ${\rm e}^{i\phi_1} \sin\dfrac{\theta_1}{2} \cos\dfrac{\theta_2}{2} \ket{11}$
    in $\ket{\psi_3}$ has the wrong phase; it should be ${\rm e}^{i\phi_3}$ and not ${\rm e}^{i\phi_1}$.
    We do the correction in the next step.

4. Apply controlled-$P(\phi_3-\phi_1)$ to qubit $1$ if qubit $0$ is set:
    {% katexmm %}
    $$
    \begin{align}
    \ket{\psi_4} &= CP^{1}_{0\to 1}(\phi_3 - \phi_1) \ket{\psi_3} \\
                &= \cos\dfrac{\theta_1}{2} \ket{00} \\
                &+ {\rm e}^{i\phi_1} \sin\dfrac{\theta_1}{2} \cos\dfrac{\theta_2}{2} \ket{01} \\
                &+ {\rm e}^{i\phi_3} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \ket{11}
    \end{align}
    $$
    {% endkatexmm %}

4. Apply controlled-$RY(\theta_3)$ to qubit $1$ if qubit $0$ is set:
    {% katexmm %}
    $$
    \begin{align}
    \ket{\psi_5} &= CRY^{1}_{0\to 1}(\theta_3) \ket{\psi_4} \\
                &= \cos\dfrac{\theta_1}{2} \ket{00} \\
                &+ {\rm e}^{i\phi_1} \sin\dfrac{\theta_1}{2} \cos\dfrac{\theta_2}{2} \ket{01} \\
                &+ \underbrace{ {\rm e}^{i\phi_3} }_{\text{wrong phase}} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \cos\dfrac{\theta_3}{2} \ket{10} \\
                &+ {\rm e}^{i\phi_3} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \sin\dfrac{\theta_3}{2} \ket{11} \\
    \end{align}
    $$
    {% endkatexmm %}
    Again, we see that there is a term with the wrong phase,
    ${\rm e}^{i\phi_3} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \cos\dfrac{\theta_3}{2} \ket{10}$.
    It should be ${\rm e}^{i\phi_2}$ and not ${\rm e}^{i\phi_3}$. We correct the phase in the next step.

4. Apply controlled-$P(\phi_2 - \phi_3)$ to qubit $0$ if qubit $1$ is **not** set:
    {% katexmm %}
    $$
    \begin{align}
    \ket{\psi_6} &= CP^{0}_{0\to 1}(\phi_2 - \phi_2) \ket{\psi_5} \\
                &= \cos\dfrac{\theta_1}{2} \ket{00} \\
                &+ {\rm e}^{i\phi_1} \sin\dfrac{\theta_1}{2} \cos\dfrac{\theta_2}{2} \ket{01} \\
                &+ {\rm e}^{i\phi_2} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \cos\dfrac{\theta_3}{2} \ket{10} \\
                &+ {\rm e}^{i\phi_3} \sin\dfrac{\theta_1}{2} \sin\dfrac{\theta_2}{2} \sin\dfrac{\theta_3}{2} \ket{11}
    \end{align}
    $$
    {% endkatexmm %}
    Notice that the term $\cos\dfrac{\theta_1}{2} \ket{00}$ doesn't acquire any phase.
    This is because, remember, the target qubit is in the $\ket{0}$ state.

And we have recovered the original state $\ket{\psi}$.
The circuit corresponding to the gates sequences is presented below:

<div class='figure'>
    <img src='/assets/images/vqe/two-qubits-state-preparation.png'
         style='width: 100%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Preparation of an arbitrary two-qubits state:</span>
        we notice that we have 4 controlled operations which will be quite expensive.
    </div>
</div>

The last negated control (change target if control **is not** set)
can be replaced by a positive control (change target if control **is** set)
by sandwiching the control between two $X$ gates. This replacement yields
the following circuit:

<div class='figure'>
    <img src='/assets/images/vqe/two-qubits-state-preparation-final.png'
         style='width: 100%; height: auto; display: block; margin: 0 auto'/>
    <div class='caption'>
        <span class='caption-label'>Preparation of an arbitrary two-qubits state:</span>
        we change the negative control to a positive control so the circuit is easy
        to work with in PennyLane.
    </div>
</div>

<div class='figure figure-alert figure-warning' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        The derived parametrized quantum circuit is inefficient
    </div>
    Notice that we require 4 controlled rotations that will amount
    to 8 controlled CNOTs and 8 single-qubit rotations.
    <b>This is wildly inefficient.</b><br>
    This derivation is shown for completeness sake,
    there are better PQCs!
</div>
</div>

## Conclusion
