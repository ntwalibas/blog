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

#### Average of a function over the line
Our mission is to find the average of
$f(x) = -x^2+4$ over the interval $[-1, 1]$.

The average of an arbitrary function $f(x)$ over an interval
$[a,b]$ is computed using the formula:

{% katexmm %}
$$
\bar{f} = \dfrac{1}{b-a} \int_{a}^{b} f(x)\,dx
$$
{% endkatexmm %}

Therefore in our case, the average is computed as follows:

{% katexmm %}
$$
\begin{align}
    \bar{f} &= \dfrac{1}{2} \int_{-1}^{1} (4-x^2) \, dx \\
    &= \dfrac{1}{2} \left( 4 \int_{-1}^{1} \,dx - \int_{-1}^{1} x^2\,dx \right) \\
    &= \dfrac{1}{2} \left( 4 \times \left[x\right]_{-1}^{1} - \left[\dfrac{x^3}{3}\right]_{-1}^{1} \right) \\
    &= \dfrac{1}{2} \left( (4 \times 2) - \dfrac{2}{3} \right) \\
    &= \dfrac{1}{2} \times \dfrac{22}{3} \\
    &= \dfrac{11}{3} \\
    &\approx 3.667 \\
\end{align}
$$
{% endkatexmm %}

#### Average of a function over a line using a computer
Let us now compute that average using a computer.
The procedure is very simple: uniformly take $N$ samples from the interval $[a, b]$,
calculate $f(x)$ for each sample $x$ then divide by the total number of samples.
Essentially the same way a normal average is calculated:

{% katexmm %}
$$
\bar{f} = \dfrac{1}{N} \sum_{i=0}^{N-1} f(x_{i}^{line})
$$
{% endkatexmm %}

Where $x_{i}^{line}$ means that each point $x_i$ is sampled
uniformly from some line.

Coding the above formula in Python, we obtain a similar result to the
analytical result above:

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np

def monte_carlo_average(f, a, b, sample_size):
    """Compute the average of a function `f` in the interval [a,b)
    The interval is right-open because of the implementation of
    `np.random.uniform` but it doesn't affect our final result
    because we just need to get arbitrarily close to  `b`.
    """
    if b - a == 0:
        raise ValueError(f"Cannot compute the average in the interval [{a},{b}].")

    samples = np.random.uniform(-1, 1, sample_size)
    total = 0
    for sample in samples:
        total += sample
    return total / len(samples)

if __name__ == "__main__":
    f = lambda x: 4 - x**2
    print(monte_carlo_average(f, -1, 1, 100_000))

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
As mentioned before, circles are also spheres but we treat
them separately so that our intuition is built in a smooth way.

Integration over the circle exhibits all the necessary
aspects that will allow us to introduce designs
but at the same time it is easy to work with, especially
finding analytical solutions.

This is important because as we move to spheres and beyond,
it gets more complicated to find analytical solutions.

#### Area of a circle
To calculate the average, we first need to calculate the area
of a circle. More importantly, we get to look at the Jacobian
determinant which will allow us to understand the concept
of measure.

Let $z = g(x, y)$ be a continuous function over a region $R$
in the $xy$-plane describing a surface.
The surface area enclosed by $z$ is given by:

{% katexmm %}
$$
S = \iint_R g(x, y)\,dA
$$
{% endkatexmm %}

The equation above can be read as taking the sum of all the little
areas $dA$. In cartesian coordinates $dA = \,dx\,dy$.

In the case of a circle, the equation is $r^2 = x^2 + y^2$.
We can try to find the area in cartesian coordinates by expressing
$y$ in terms of $x$ with:

{% katexmm %}
$$
\begin{align}
    y &= \pm \sqrt{r^2-x^2} &|y| < r
\end{align}
$$
{% endkatexmm %}

We can then calculate the integral:

{% katexmm %}
$$
S = \int_{-r}^{r} \int_{-\sqrt{r^2-x^2}}^{\sqrt{r^2-x^2}}\,dy\,dx
$$
{% endkatexmm %}

And get $S = \pi r^2$.

A much more elegant method is to move to polar coordinates from the beginning.
This involves a change of variables so let's talk about that.

Let $x = r(v,w)$ and $y = s(v,w)$. Let us assume that the first derivatives
of $r$ and $s$ exist and are continuous.
Let $J(v,w) = (x,y)$ be a transformation that maps a region $S$ to the region $R$.

Integration by a change of variables is given by:

{% katexmm %}
$$
\iint_R g(x, y)\,dA = \iint_S g(r(v,w), s(v,w)) \left\lvert \dfrac{\partial(x,y)}{\partial(v,w)} \right\rvert \,dv\,dw
$$
{% endkatexmm %}

The transformation $J = \dfrac{\partial(x,y)}{\partial(v,w)}$ is called the Jacobian.
It is a linear map that allows us to do the transformation.
The quantity $\det J = \left\lvert \dfrac{\partial(x,y)}{\partial(v,w)} \right\rvert$
is called the Jacobian determinant. It quantifies how much the linear transformation
affects the area in space.  
If you changed a square in cartesian coordinates to polar coordinates,
the area *will* change. The Jacobian determinant **measures** that change.

This is very important to grasp because not only we will see
the same for spheres but also because even when we move
to integration over unitary matrices, we will have to take into account
the integration measure in order to properly account for change in area/volume.

We won't go over the detailed calculations of the Jacobian since this
is can be found in most books on calculus.
The change of variable is given by $x = r\cos\theta$ and $y = r\sin\theta$.
The Jacobian is found to be $J(r,\theta) = r$, and since $r \ge 0$ we have
$\det J = r$. It follows that $dA = rdrd\theta$.  
The new limits of integration are $0 \le r \le R$ for some fixed radius $R$
and $0 \le \theta \le 2\pi$.

Consequently the surface area of the circle is:

{% katexmm %}
$$
\begin{align}
    S &= \int_{0}^{2\pi} \int_{0}^{R} r\,dr\,d\theta \\
    &= \dfrac{R^2}{2} \int_{0}^{2\pi} \,d\theta \\
    &= \pi R^2
\end{align}
$$
{% endkatexmm %}

<div class='figure figure-alert figure-success' style='margin-top: 10px'>
<div class='caption'>
    <div class='caption-label'>
        Goal of calculating the surface area using polar coordinates
    </div>
    The primary goal of performing this calculation was not to find the area
    of the circle, it was to understand the meaning and importance of the Jacobian (determinant).
    When we move to sphere, the Jacobian will be different. And similarly,
    when we need to perform integrations in higher dimensions, the measure
    that quantifies the change in volume will also be different.<br><br>
    <b>If there is any single lesson to be learned, it is to pay attention
    to the measure before integrating.</b><br><br>
    Sure, we will need to know the surface area of the circle to find the average
    of a function over it but this could have been easily found in any geometry/calculus book.
</div>
</div>

#### Average of a function over a circle
In general, the average of a function $f(x,y)$ over a circle
of unknown radius is given by:

{% katexmm %}
$$
\bar{f} = \dfrac{1}{\pi R^2} \int_{0}^{2\pi} \int_{0}^R f(\theta, r) r\,dr\,d\theta
$$
{% endkatexmm %}

Notice that $r$ is the unknown radius of an arbitrary circle while $R$
is potentially fixed radius once it is known.
What that means is that once we have calculated the average of $f$,
it will depend on $R$.

Let us proceed a find the average of $f(x,y) = x^2$.

First, we move to polar coordinates and find $f(\theta, r)$.
The circle we're integrating over has equation $x^2 + y^2 = R^2$
for some unknown radius $R$ to be fixed when we need it.
Since $f$ depends only on $x$, and the change to polar coordinates
implies $x = R\cos\theta$.
It follows then that $f(\theta, r) = R^2\cos^2\theta$.

Now, we can calculate the average using the formula above:

{% katexmm %}
$$
\begin{align}
    \bar{f} &= \dfrac{1}{\pi R^2} \int_{0}^{2\pi} \int_{0}^R f(\theta, r) r\,dr\,d\theta \\
    &= \dfrac{1}{\pi \cancel{R^2}} \int_{0}^{2\pi} \int_{0}^R \cancel{R^2} cos^2\theta \,r\,dr\,d\theta \\
    &= \dfrac{1}{\pi} \int_{0}^{2\pi} cos^2\theta \int_{0}^R r\,dr\,d\theta \\
    &= \dfrac{R^2}{2\pi} \int_{0}^{2\pi} cos^2\theta \,d\theta \\
    &= \dfrac{R^2}{2\pi} \left[\dfrac{\theta}{2} + \dfrac{1}{4}\sin(2\theta)\right]_{0}^{2\pi} \\
    &= \dfrac{R^2}{2\pi} \times \pi \\
    &= \dfrac{R^2}{2}
\end{align}
$$
{% endkatexmm %}

If we are integrating over the unit circle ($R = 1$) then the average will be
$\bar{f} = \dfrac{1}{2}$. If $R = 2$, then $\bar{f} = \dfrac{2^2}{2} = 2$.

#### Average of a function over the unit circle
Now assume that we know $R$ from the beginning. Then there is no need
to integrate over the radius, it is a constant from the get-go.
The function $f(\theta,r)$ no longer depends on $r$ so we denote it by
$f(\theta;R)$.  
This is exactly the case when we are averaging over the unit circle since
we know that $R = 1$.

In that case we can save ourselves the pain of performing a double integral
using the formula:

{% katexmm %}
$$
\begin{align}
    \bar{f} = \dfrac{1}{2\pi} \int_{0}^{2\pi} f(\theta; R) \,d\theta
\end{align}
$$
{% endkatexmm %}

For a sanity check, let's repeat the calculation above; to find the average of
$f(x,y) = x^2$ over the unit circle. Since $R=1$, it follows that
$f(\theta; R) = R^2 \cos^2\theta = \cos^2\theta$.

{% katexmm %}
$$
\begin{align}
    \bar{f} &= \dfrac{1}{2\pi} \int_{0}^{2\pi} f(\theta; R) \,d\theta \\
    &= \dfrac{1}{2\pi} \int_{0}^{2\pi} \cos^2\theta \,d\theta \\
    &= \dfrac{1}{2\pi} \left[\dfrac{\theta}{2} + \dfrac{1}{4}\sin(2\theta)\right]_{0}^{2\pi} \\
    &= \dfrac{1}{2\pi} \times \pi \\
    &= \dfrac{1}{2}
\end{align}
$$
{% endkatexmm %}

And this matches the result obtained by not fixing $R$ and providing it later.

The advantage of this latter method is that most of the time we already
know the radius and for problems we will care about $R=1$ --
whence the *unit* circle.

This is not at all particularly interesting or practically useful,
but an important stepping stone that will lead us towards our goal of
finding these averages using designs.

#### Average of a function over the unit circle using a computer
Just as with the line, we need to sample uniformly from the unit circle.
The formula itself doesn't change:

{% katexmm %}
$$
\bar{f} = \dfrac{1}{N} \sum_{i=0}^{N-1} f(x_{i}^{circle}, \, y_{i}^{circle})
$$
{% endkatexmm %}

Where $x_{i}^{circle}$ ($y_{i}^{circle}$) means that each $x_i$ ($y_i$)
is sampled uniformly from the circle.

To sample uniformly from the circle, we use the circle trigonometric
parametrization {% cite Ozols_2009 %}:

{% katexmm %}
$$
\mathbb{S}^1 = \left\{ \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix} \Bigg| 0 \le \theta < 2\pi \right\}
$$
{% endkatexmm %}

Where $\mathbb{S}^1$ is the 1-sphere (circle).  
Uniform sampling simply involves uniformly sampling from the interval $[0, 2\pi)$.

With that said the code is straightforward. So let's see if for $f(x,y)=x^2$
we do get the average $\bar{f}=\dfrac{1}{2}=0.5$.

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np

def monte_carlo_average(f, sample_size):
    """Compute the average of a function `f` over the unit circle.
    """
    def sample_from_circle(size):
        x_samples = np.random.uniform(0, 2 * np.pi, size)
        y_samples = np.random.uniform(0, 2 * np.pi, size)
        return zip(np.cos(x_samples), np.sin(y_samples))

    samples = sample_from_circle(sample_size)
    total = 0
    for sample in samples:
        total += f(*sample)
    return total / sample_size

if __name__ == "__main__":
    f = lambda x, y: x**2
    print(monte_carlo_average(f, 100_000))

{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Average of $f(x, y) = x^2$ over the unit circle:
    </span>
    as we increase the number of samples, we get closer to
    the true value of $0.5$.
</div>
</div>

We are now in a position to introduce designs, and of course
these are going to be circular designs.  
Understanding circular designs is going to be enough to appreciate
the value they bring to the table!

#### Circular designs
Using a variant of [Monte Carlo methods](https://en.wikipedia.org/wiki/Monte_Carlo_method),
we've been able to calculate the average of some function $f(x,y)$ over the circle.
This method requires drawing random (uniform) samples then computing the average
the usual way.

With circular designs, under certain conditions, we can find that average by using
a set of carefully chosen points depending on the degree of the polynomial
that makes the function we wish to average.

We start by defining what a circular design is:

> **Definition:**
> let $f_t: \mathcal{S}(\mathbb{R}^2) \rightarrow \mathbb{R}$ be a polynomial in $2$ variables
> homogeneous in degree at most $t$.
> A set $X = \{ x \vert x \in \mathcal{S}(\mathbb{R}^2) \}$ is a circular
> $t$-design if:
> {% katexmm %}
> $$
> \dfrac{1}{\lvert X \rvert} \sum_{x \in X} f_t(x) = \dfrac{1}{2\pi} \int_{0}^{2\pi} f(\theta; R) \,d\theta
> $$
> {% endkatexmm %}
> holds for all possible $f_t$.
> Moreover a circular $t$-design is also a circular $s$-design for all $0 < s < t$.

Let's first understand the definition before finding what our circular $t$-designs are:
1. We require that the function $f_t(x,y)$ be a polynomial.
2. The polynomial $f_t$ must be homogeneous in degree at most $t$.
    This means the sum of all the degrees of the terms in the polynomial is the same and it is $t$.
3. The set $X$ is finite otherwise calculating $\dfrac{1}{\lvert X \rvert}$ is not possible.
4. Given $s < t$, wherever an $s$-design works, so will a $t$-design.
5. **To accurately compute the average of polynomial of degree $t$, we must use a $t$-design.**

Here are a few examples to help understand the definition:
- $f(x,y) = x^2$ is a homgeneous polynomial of degree $2$.
- $f(x, y) = x^2 + y$ is *not* a homogeneous polynomial.
- $f(x, y) = x^2 + xy$ is also a homogeneous polynomial of degree $2$.
- $f(x, y) = x^8 + x^5\,y^3 + x\,y^7$ is a homogeneous polynomial of degree $8$.

So then, what are the circular designs we will work with?
It has been shown that polygons are circular designs {% cite 10.1016/j.ejc.2008.11.007 %},
specifically the $(t+1)$-gon is a circular $t$-design.

As examples we have that the [digon](https://en.wikipedia.org/wiki/Digon)
is a $1$-design, the [triangle](https://en.wikipedia.org/wiki/Triangle)
is a $2$-design and so on.

So we have a polynomial of degree $5$, we should use a $6$-gon (hegaxon) as
our $5$-design.

What then are the points we will need to evaluate the polynomial
$f(x,y)$ at in order to compute the average we seek?
Those are simply the points corresponding to the vertex corners of the polygon.

Without further ado, let's write some code that computes
the average of $f(x,y) = x^8 + x^5\,y^3 + x\,y^7$ using circular t-designs.

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np
import numpy.linalg as la

def circular_design_average(f, t):
    """Computes the average of a function `f` using circular
    `t`-designs, specifically polygons.
    """
    def polygon(number_of_sides):
        """
        Given the number of sides, this function returns
        the coordinates of the vertex corners of the polygon with
        the given number of sides.
        """
        internal_angle = 2 * np.pi / number_of_sides
        coordinates = []
        
        for point in range(number_of_sides):
            new_angle = point * internal_angle
            coordinates.append([np.cos(new_angle), np.sin(new_angle)])
        
        return coordinates

    # Calculate the mean of the function evaluated at
    # the vertex corners of the polygon
    return np.mean([f(*point) for point in polygon(t + 1)])

if __name__ == "__main__":
    f = lambda x, y: x**8 + (x**5)*(y**3) + x*(y**7)
    for vertex_count in range(2, 15):
        print(f"Design = {vertex_count + 1}\t\
                average = {circular_design_average(f, vertex_count)}")

{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Average of $f(x, y) = x^8 + x^5\,y^3 + x\,y^7$
        over the unit circle using circular $t$-designs:
    </span>
    using $100000$ samples in the Monte Carlo method will yield $\bar{f}=0.2740250458146402$.
    Using a $9$-design will yield $\bar{f}=0.27343749999999994$.
    They are pretty close, to the second decimal!
</div>
</div>

It is worth looking at the output of the code above and make some comments:

<div class='figure' markdown='1'>
{% highlight text %}
Design = 3      average = 0.33593749999999994
Design = 4      average = 0.5
Design = 5      average = 0.2734375
Design = 6      average = 0.33593749999999994
Design = 7      average = 0.2734375
Design = 8      average = 0.28125
Design = 9      average = 0.27343749999999994
Design = 10     average = 0.27343749999999994
Design = 11     average = 0.27343750000000006
Design = 12     average = 0.27343749999999994
Design = 13     average = 0.27343749999999994
Design = 14     average = 0.2734375
Design = 15     average = 0.2734374999999999

{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Average of $f(x, y) = x^8 + x^5\,y^3 + x\,y^7$ for different $t$-designs:
    </span>
    we note that the value becomes stable after using the $9$-gon (nonagon).
</div>
</div>

So what have we learned from the result above?
1. If we tried computing the average analytically (via integration),
    we would have ended up needing to evaluate integrals with sines
    and cosines of degree at least $3$, not fun!
2. Using Monte Carlo, we need a large number of samples in order to approximate
    the average of the function.
    Using a circular $9$-design, we required only $9$ evaluation of the function.
    Moreover, because Monte Carlo relies on random sampling, we will not
    get the same exact result with each run.
3. While we may get the correct result using an $s$-design for $s \le 8$
    (e.g.: $s = 5$), we can't rely on that. Notice that for $s = 6$ right after
    we got the wrong result.  
    But notice how the result remains pretty consistent for $t \ge 9$.
    And that makes sense: we are just using more points so accuracy should either
    increase or stay the same.

And that's pretty much the essence of using designs: approximate
function averaging over some set.  
In the sections that follow, we wil simply introduce new measures and
new designs to tackle different problems.

### Average of a function over a sphere
#### Average of a function over the unit 2-sphere
#### Spherical designs

## The Haar measure

## State designs

## Unitary designs

## Next steps

## Conclusion

## Derivations
