---
title: "Unitary designs: faster averages over unitaries"
subtitle: "A common task in life is calculating the average
of some quantities. The average is the most common and intuitive statistic every
human learns and uses. Unitary designs allow us to compute averages over
unitaries faster than doing uniform sampling. In this post, we build some intuition
about unitary designs and how they relate to integrals over the unitary group."
layout: default
date: 2023-08-25
keywords: unitary designs, integration over the unitary group
toc: true
published: false
---

## Introduction
Consider the following problem: given a quantum gate, what is the probability
that under noise it fails to evolve a given state the expected way?

Our first step will be to define the quality of implementation of a quantum gate.
This is done using the notion of *fidelity* which measures how close the implemented
quantum gate approximates the ideal quantum gate.

The method used to determine fidelity can either be quantum process tomography (QPT)
or quantum gate set tomography (GST).

The problem with both QPT and GST is that they are expensive to carry out.
So an alternative method is to find a large random number of states
and see how the implemented gate performs on each of those states compared
to the performance of the ideal gate.
We then take the *average* of fidelities over all those states.

But that can also be quite expensive because it will require a large number
of states if we want the calculated average to be meaningful.  
And that's where *designs* enter the picture: by using a small well-chosen
representative set of states we can calculate the average fidelity
without using a large number of samples.

It is our goal in this post to build some intuition about designs,
in particular unitary designs, by working through small examples.
For the sake of this goal, we won't dig deep into the mathematics
and focus more on the basic principles via illustrative examples
and code.

### Prerequisites
It is assumed the reader can do basic integral calculations
and can program in Python.

The reader is also assumed to know basics of quantum computation,
up to a basic ability to describe quantum channels in terms
of Kraus operators though this figure only once.

### Notation
Quantum states can be written in a variety of ways but for computational
purposes, we will adopt the trigonometric form since it will be convenient in
computing integrals:

{% katexmm %}
$$
\begin{align}
\ket{\psi} = \cos\frac{\theta}{2} \ket{0} + {\rm e}^{i\phi} \sin\frac{\theta}{2} \ket{1}
\end{align}
$$
{% endkatexmm %}

The equivalent vector form can also be handy:

{% katexmm %}
$$
\begin{align}
\ket{\psi} =
\begin{bmatrix}
\cos\frac{\theta}{2} \\
{\rm e}^{i\phi} \sin\frac{\theta}{2}
\end{bmatrix}
\end{align}
$$
{% endkatexmm %}

We will make extensive use of Pauli matrices:

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
In the next section, we introduce all the tools we need to successfully
make use of this post.

The section that follows on spherical designs: before we touch upon
quantum designs, it is fruitful acquaint ourselves with classical designs
so that we can appreciate the power of designs.  

The section of the Haar measure is introduced before moving to
quantum designs because integration over states or unitaries
requires that we take special care of defining how such integrations are
to be carried out.

We follow with states designs as a prelude to unitary designs
so as to strengthen our intuition about what integration over the Haar
measure and quantum designs are about.

Then we explore unitary designs and work through a couple of examples
where the power of unitary designs come to rescue us from computing
complicated integrals and more importantly from performing expensive computations.

We then talk about what we didn't touch upon and consequently
what's worth studying more assuming we have to appreciate
the power of designs.

Last we give some concluding remarks.

## Tooling
Passing familiarity with Python and Numpy is always assumed on the software side
and linear algebra on the theory side.

### Theoretical tools
On the mathematics side of things, the following will be useful to know:
- Computing simple single, double and triple trigonometric integrals.
- Performing matrix and vector operations such as addition and multiplication.

On the quantum computing side of things we will need to know:
- What quantum states and gates are.
- Basic ideas about quantum channels,
  but nothing fancy besides dealing with Kraus operators.

### Software tools
We will use PennyLane from Xanadu to run simulations.

The installation instructions can be found at
[https://pennylane.ai/install/](https://pennylane.ai/install/).

## Spherical designs
Say we want to find the average of some function over a sphere.
You might go: wait, why would I want to do that?
In general, the problem we are working with will dictate whether
we need to use a sphere, a line, a cylinder, etc.
In the case of quantum computing, for instance, an ideal qubit
can be described geometrically as points on the surface of a sphere,
the Bloch sphere.
So learning to work with spheres is a good starting point.

We will start with a refresher though: computing averages
over a line (an interval to be exact).
Then we will move to spheres. Note that a circle is also defined
as a sphere, a $2$ dimensional one.
The sphere we are used to is a $3$ dimensional one.
And there are spheres in dimensions more than $3$,
we just won't bother with them.

### Average of a function over a line
To get started, as a segue to the fun stuff,
let's compute the average of a simple function over an interval.

This subsection won't introduce designs but will show
a general method of computing averages of functions by using random
samples drawn from an interval.

Our mission is to find the average of
$f(x) = -x^2+4$ over the interval $[-1, 1]$.

The average of an arbitrary function $f(x)$ over an interval
$[a,b]$ is computed using the formula:

{% katexmm %}
$$
f_{avg} = \dfrac{1}{b-a} \int_{a}^{b} f(x)\,dx
$$
{% endkatexmm %}

Therefore in our case, the average is computed as follows:

{% katexmm %}
$$
\begin{align}
    f_{avg} &= \dfrac{1}{2} \int_{-1}^{1} (4-x^2) \, dx \\
    &= \dfrac{1}{2} \left( 4 \int_{-1}^{1} \,dx - \int_{-1}^{1} x^2\,dx \right) \\
    &= \dfrac{1}{2} \left( 4 \times \left[x\right]_{-1}^{1} - \left[\dfrac{x^3}{3}\right]_{-1}^{1} \right) \\
    &= \dfrac{1}{2} \left( (4 \times 2) - \dfrac{2}{3} \right) \\
    &= \dfrac{1}{2} \times \dfrac{22}{3} \\
    &= \dfrac{11}{3} \\
    &\approx 3.667 \\
\end{align}
$$
{% endkatexmm %}

Let us now compute that average using a computer.
The procedure is very simple: uniformly take $N$ samples from the interval $[a, b]$,
calculate $f(x)$ for each sample $x$ then divide by the total number of samples.
Essentially the same way a normal average is calculated:

{% katexmm %}
$$
f_{avg} = \dfrac{1}{N} \sum_{i=0}^{N-1} f(x_i)
$$
{% endkatexmm %}

Coding the above formula in Python, we obtain a similar result to the
analytical result above:

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np

def average(f, a, b):
    """Compute the average of a function `f` in the interval [a,b]"""
    if b - a == 0:
        raise ValueError(f"Cannot compute the average in the interval [{a},{b}].")

    vectorized_f = np.vectorize(f)
    samples = np.random.uniform(-1, 1, 100000)
    return np.sum(vectorized_f(samples)) / len(samples)

if __name__ == "__main__":
    f = lambda x: 4 - x**2
    print(average(f, -1, 1))
{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Average of $f(x) = 4 - x^2$ in the interval $[-1,1]$:
    </span>
    even though <code>np.random.uniform</code>
    actually samples from $[a,b)$ meaning the interval is open to the right,
    the final result will not be affected.
</div>
</div>

In the code above, we used $100000$ samples. The lower the number of samples
we use, the farther we get away from the true average. The more samples we use,
the longer it takes to obtain the result we seek.

This observation is very important because it forms the justification of using
designs as we will quickly see in the following subsections.

### Average of a function over a circle
#### Average of a function over the unit 1-sphere
#### Circular designs
### Average of a function over a sphere
#### Average of a function over the unit 2-sphere
#### Spherical designs

## The Haar measure

## State designs

## Unitary designs

## Next steps

## Conclusion

## Derivations
