---
title: "Quantum designs: faster averaging over quantum states and unitaries"
subtitle: "A common task in life is calculating the average
of some quantities. The average is the most common and intuitive statistic every
human learns and uses. Unitary designs allow us to compute averages over
unitaries faster than doing Monte Carlo integration. In this post, we build some intuition
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
quantum designs, it is fruitful to acquaint ourselves with classical designs
so that we can appreciate the power of designs.

We follow with states designs as a prelude to unitary designs
so as to strengthen our intuition about what integration over the Haar
measure and quantum designs are about.

Then we explore unitary designs and work through a couple of examples
where the power of unitary designs come to rescue us from computing
complicated integrals and more importantly from performing expensive computations.

We then talk about what we didn't touch upon and consequently
what's worth learning more about assuming we have come to appreciate
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

    return np.mean(np.random.uniform(a, b, sample_size))

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
them separately so that our intuition is built in a smooth way,
starting with something very simple.

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
\mathbb{S}^1 = \left\{ \left. \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix} \right|\,  0 \le \theta < 2\pi \right\}
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
        theta = np.random.uniform(0, 2 * np.pi, size)
        return zip(
            np.cos(theta),
            np.sin(theta)
        )

    return np.mean([f(*sample) for sample in sample_from_circle(sample_size)])

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
> A set $X = \\{ x \vert x \in \mathcal{S}(\mathbb{R}^2) \\}$ is a circular
> $t$-design if:
> {% katexmm %}
> $$
> \dfrac{1}{\lvert X \rvert} \sum_{x \in X} f_t(x) = \dfrac{1}{\pi R^2} \int_{0}^{2\pi} \int_{0}^R f_t(\theta, r) r\,dr\,d\theta
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
5. **To accurately compute the average of a polynomial of degree $t$, we must use a $t$-design.**

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
is a $2$-design, the [square](https://en.wikipedia.org/wiki/Square) a $3$-design,
and so on.

Thus if we have a polynomial of degree $5$, we should use a $6$-gon (hexagon) as
our $5$-design.

What then are the points we will need to evaluate the polynomial
$f(x,y)$ at in order to compute the average we seek?
Those are simply the points corresponding to the vertex corners of the polygon.

Without further ado, let's write some code that computes
the average of $f(x,y) = x^8 + x^5\,y^3 + x\,y^7$ using circular t-designs.

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np

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
3. While we may get the correct result using a $t$-design $t < 9$
    (e.g.: $t = 5$), we can't rely on that. Notice that for $t = 6$, we got
    the wrong result even though the result was correct for $t = 5$.  
    But notice how the result remains pretty consistent for $t \ge 9$.
    And that makes sense: we are just using more points so accuracy should either
    increase or stay the same.

And that's pretty much the essence of using designs: *approximate*
*function averaging over some set using as few function evaluations as possible.*  
In the sections that follow, we wil simply introduce new sets over which to measure
and new designs to tackle different problems.

### Average of a function over a sphere
Averaging over a $2$-sphere is pretty much the same as averaging over
the circle. So we won't go through the exact same motions and just
state the results and perform some simple calculations.
Then we introduce the designs that work for the $2$-sphere.

Moving forward, we will refer to the $2$-sphere simply as sphere
even if mathematically a sphere generally refer to the $n$-sphere.

#### Volume of a sphere
The move from cartesian coordinates to spherical coordinates is given by:

{% katexmm %}
$$
\begin{align}
    x &= r\sin\phi\cos\theta \\
    y &= r\sin\phi\sin\theta \\
    z &= r\cos\phi
\end{align}
$$
{% endkatexmm %}

The region of integration is given by:

{% katexmm %}
$$
\begin{align}
    0 &\le r \le R \\
    0 &\le \phi \le \pi \\
    0 &\le \theta \le 2\pi
\end{align}
$$
{% endkatexmm %}

The Jacobian determinant of a sphere is given by:

{% katexmm %}
$$
\det J = r^2\sin\theta
$$
{% endkatexmm %}

The volume of a sphere is therefore calculated as:

{% katexmm %}
$$
\begin{align}
    V &= \int_0^{2\pi} \int_0^{\pi} \int_0^{R} r^2\sin\phi \,dr\,d\phi\,d\theta \\
    &= \dfrac{R^3}{3} \int_0^{2\pi} \int_0^{\pi} \sin\phi \,d\phi\,d\theta \\
    &= \dfrac{2}{3} R^3 \int_0^{2\pi} \,d\theta \\
    &= \dfrac{4}{3} \pi R^3
\end{align}
$$
{% endkatexmm %}

Again, we can appreciated the importance of using the correct measure
obtained from the Jacobian determinant allowing us to compute
the correct volume of a sphere of radius $R$.

#### Average of a function over a sphere
Averaging over a sphere is very similar to averaging over a circle.
The only change will be the use of a different measure obtained
from the Jacobian determinant. Because of that, we will use the volume
of the sphere, not its surface area in the general formula.

In general, the average of a function $f(x, y, z)$ over a sphere
of unknown radius is given by:

{% katexmm %}
$$
\bar{f} = \dfrac{3}{4\pi R^3} \int_0^{2\pi} \int_0^{\pi} \int_0^{R} f(r,\phi,\theta) \,r^2\sin\phi \,dr\,d\phi\,d\theta
$$
{% endkatexmm %}

As an example, let us find the average of $f(x,y,z)=x^4$ over a sphere
of unknown radius $R$.  
Moving from cartesian coordinates to spherical coordinates, $f$ is now given by
$f(r,\phi,\theta)=R^4\sin^4\phi\cos^4\theta$.

Evaluating the integral above, we get:

{% katexmm %}
$$
\begin{align}
    \bar{f} &= \dfrac{3}{4\pi \cancel{R^3}} \int_0^{2\pi} \int_0^{\pi} \int_0^{R} R^{\cancel{4}}\sin^4\phi\cos^4\theta \,r^2\sin\phi \,dr\,d\phi\,d\theta \\
    &= \dfrac{3}{4\pi R} \int_0^{2\pi} \int_0^{\pi} \int_0^{R} r^2\sin^5\phi\cos^4\theta \,dr\,d\phi\,d\theta \\
    &= \dfrac{3}{4\pi \cancel{R}} \times \dfrac{R^{\cancel{3}}}{3} \int_0^{2\pi} \int_0^{\pi} \sin^5\phi\cos^4\theta \,d\phi\,d\theta \\
    &= \dfrac{R^2}{4\pi} \times \dfrac{16}{15} \int_0^{2\pi} cos^4 \,d\theta \\
    &= \dfrac{4R^2}{15\pi} \times \dfrac{3\pi}{4} \\
    &= \dfrac{R^2}{5}
\end{align}
$$
{% endkatexmm %}

#### Average of a function over the unit sphere
Same as with the circle, we can make a simplification of the averaging
formula if the radius is a known constant.
The function to average will therefore have a dependence
only on $\phi$ and $\theta$ and is given as $f(\phi,\theta;R)$.

The average of a sphere of known radius is therefore given by:

{% katexmm %}
$$
\bar{f} = \dfrac{1}{4\pi} \int_0^{2\pi} \int_0^{\pi}f(\phi,\theta;R) \,\sin\phi \,d\phi\,d\theta
$$
{% endkatexmm %}

It is straightforward to verify that the average of $f(x,y,z) = x^4$
over the unit sphere is given by $\bar{f}=\dfrac{1^2}{5}=\dfrac{1}{5}=0.2$:

{% katexmm %}
$$
\begin{align}
    \bar{f} &= \dfrac{1}{4\pi} \int_0^{2\pi} \int_0^{\pi} \sin^4\phi\cos^4\theta \sin\phi \,d\phi\,d\theta \\
    &= \dfrac{1}{4\pi} \times \dfrac{16}{15} \int_0^{2\pi} \cos^4\theta\,d\theta \\
    &= \dfrac{1}{4\pi} \times \dfrac{16}{15} \times \dfrac{3\pi}{4} \\
    &= \dfrac{1}{5}
\end{align}
$$
{% endkatexmm %}

Again, nothing fancy, just mundane integration.
The only thing that has changed thus far is that dimensions increased
and the Jacobian determinant has accounted for this change.

<div class='figure figure-alert figure-info' style='margin-top: 10px'>
<div class='caption'>
    Recall that the Bloch sphere is a unit sphere.
    So when we will need to integrate over single-qubit states,
    the formula above will be quite useful so it is important to remember it.
</div>
</div>

#### Average of a function over the unit sphere using a computer
As before, we do Monte Carlo integration but this time we need to sample
over the sphere. The uniform sampling procedure we will use is based on
that in subsection $2.2.1$ in {% cite Ozols_2009 %}:

{% katexmm %}
$$
\mathbb{S}^2 = \left\{ \left. \begin{bmatrix}x \\ y \\ z\end{bmatrix} = \begin{bmatrix} \sin\phi\cos\theta \\ \sin\phi\sin\theta \\ \cos\phi \end{bmatrix} \right|\, 0 \le \phi \le \pi; 0 \le \theta < 2\pi \right\}
$$
{% endkatexmm %}

Where $\mathbb{S}^2$ is the sphere. To sample uniformly from a sphere,
we sample $\theta$ uniformly in the interval $[0,2\pi]$
and we sample $p \in [-1,1]$ uniformly then calculate $\phi$ from $p$
using $\phi = \arccos p$.

The formula stays the same as with the circle, we just now sample uniformly
from the sphere:

{% katexmm %}
$$
\bar{f} = \dfrac{1}{N} \sum_{i=0}^{N-1} f(x_{i}^{sphere}, \, y_{i}^{sphere}, \, z_{i}^{sphere})
$$
{% endkatexmm %}

Where $x_{i}^{sphere}$ ($y_{i}^{sphere}$, $z_{i}^{sphere}$)
means that each $x_i$ ($y_i$, $z_i$) is sampled uniformly from the sphere.

Here follows the code that calculates the average of $f(x,y,z) = x^4$
using the procedure just described above:

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np

def monte_carlo_average(f, sample_size):
    """Compute the average of a function `f` over the unit sphere
    """
    def sample_from_sphere(size):
        theta = np.random.uniform(0, 2 * np.pi, size)
        p = np.random.uniform(-1, 1, size)
        phi = np.arccos(p)
        return zip(
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        )

    return np.mean([f(*sample) for sample in sample_from_sphere(sample_size)])

if __name__ == "__main__":
    f = lambda x, y, z: x**4
    print(monte_carlo_average(f, 100_000))

{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Average of $f(x, y, z) = x^4$ over the unit sphere:
    </span>
    as we increase the number of samples, we get closer to
    the true value of $0.2$.
</div>
</div>

Now that we have validated that Monte Carlo integration helps
us compute the average, let's do the same using the far better
tool of designs.

#### Spherical designs
Computing the average over a sphere using designs is pretty much the same
as computing averages over a circle.
The crucial difference is how we choose the points: unsuprisingly,
instead of polygons, we use polyhedra.

Let us define spherical designs, the same way we defined circular designs:

> **Definition:**
> let $f_t: \mathcal{S}(\mathbb{R}^3) \rightarrow \mathbb{R}$ be a polynomial in $3$ variables
> homogeneous in degree at most $t$.
> A set $X = \\{ x \vert x \in \mathcal{S}(\mathbb{R}^3) \\}$ is a spherical
> $t$-design if:
> {% katexmm %}
> $$
> \dfrac{1}{\lvert X \rvert} \sum_{x \in X} f_t(x) = \dfrac{3}{4\pi R^3} \int_0^{2\pi} \int_0^{\pi} \int_0^{R} f(r,\phi,\theta) \,r^2\sin\phi \,dr\,d\phi\,d\theta
> $$
> {% endkatexmm %}
> holds for all possible $f_t$.
> Moreover a spherical $t$-design is also a spherical $s$-design for all $0 < s < t$.

The exact same comments we made about circular designs apply.

The following polyhedra correspond to the respective spherical $t$-designs
{% cite 10.1016/j.ejc.2008.11.007 %}. We limit ourselves to platonic solids:
- The regular [tetrahedron](https://en.wikipedia.org/wiki/Tetrahedron) is a $2$-design.
- The [cube](https://en.wikipedia.org/wiki/Cube) is a $3$-design.
- The regular [octahedron](https://en.wikipedia.org/wiki/Octahedron) is a $3$-design.
- The regular [icosahedron](https://en.wikipedia.org/wiki/Regular_icosahedron) is a $5$-design.
- The regular [dodecahedron](https://en.wikipedia.org/wiki/Regular_dodecahedron) is a $5$-design.

For practice, let's compute the average of two functions
using different spherical designs. Given there are duplicate designs
of same strength, we will choose representative $3$- and $5$-designs.

We will compute the average of the following functions:
- The function $f(x,y,z) = x^2 + xz + yz + z^2$ has analytically computed average $\bar{f} = \dfrac{2}{3} \simeq 0.6666667$.
- The function $f(x,y,z) = x^4$ has analytically computed average $\bar{f} = \dfrac{1}{5} = 0.2$.

The first function requires at least a $2$-design while the second requires at least
a $4$-design.

<div class='figure' markdown='1'>
{% highlight python %}
import numpy as np
import numpy.linalg as la

def tetrahedron():
    """The tetrahedron as a spherical 2-design."""
    coordinates = np.array([
        [ 1,  0, -1/np.sqrt(2)],
        [-1,  0, -1/np.sqrt(2)],
        [ 0,  1,  1/np.sqrt(2)],
        [ 0, -1,  1/np.sqrt(2)],
    ])
    # Normalize so the tetrahedron fits into the unit sphere
    return np.array(
        [point / la.norm(point) for point in coordinates]
    )

def cube():
    """The cube as the representative spherical 3-design."""
    coordinates = np.array([
        [ 1,  1,  1], [-1, -1, -1],
        [-1, -1,  1], [ 1,  1, -1],
        [ 1, -1, -1], [-1,  1,  1],
        [-1,  1, -1], [ 1, -1,  1],
    ])
    # Normalize so the cube fits into the unit sphere
    return np.array(
        [point / la.norm(point) for point in coordinates]
    )

def icosahedron():
    """The icosahedron as the representative spherical 5-design."""
    g = (1 + np.sqrt(5)) / 2
    coordinates = np.array([
        [ 0,  1,  g], [ 0, -1,  g],
        [ 0,  1, -g], [ 0, -1, -g],

        [ 1,  g,  0], [-1,  g,  0],
        [ 1, -g,  0], [-1, -g,  0],

        [ g,  0,  1], [ g,  0, -1],
        [-g,  0,  1], [-g,  0, -1],
    ])
    # Normalize so the icosahedron fits into the unit sphere
    return np.array(
        [point / la.norm(point) for point in coordinates]
    )

def spherical_design_average(f, points):
    """Computes the average of a function `f` using the spherical
    t-design provided as `points`, specifically polyhedra vertex corners.
    """
    return np.mean([f(*point) for point in points])

if __name__ == "__main__":
    f1 = lambda x, y, z: x**2 + x*z + y*z + z**2
    print("We expect all designs to provide the average:")
    print(f"f1 average using 2-design: {spherical_design_average(f1, tetrahedron())}")
    print(f"f1 average using 3-design: {spherical_design_average(f1, cube())}")
    print(f"f1 average using 5-design: {spherical_design_average(f1, icosahedron())}")

    f2 = lambda x, y, z: x**4
    print("\nWe expect the average to work only for the icosahedron:")
    print(f"f2 average using 2-design: {spherical_design_average(f2, tetrahedron())}")
    print(f"f2 average using 3-design: {spherical_design_average(f2, cube())}")
    print(f"f2 average using 5-design: {spherical_design_average(f2, icosahedron())}")

{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Averages of $f(x, y, z) = x^2 + xz + yz + z^2$ and $f(x, y, z) = x^4$
        over the unit sphere using spherical $t$-designs:
    </span>
    as before, we note that for polynomials of degree $t$,
    any $s$-design with $s < t$ yields unreliable averages.
</div>
</div>

The output of my run is:

<div class='figure' markdown='1'>
{% highlight text %}
We expect all designs to provide the average:
f1 average using 2-design: 0.6666666666666667
f1 average using 3-design: 0.6666666666666667
f1 average using 5-design: 0.6666666666666666

We expect the average to work only for the icosahedron:
f2 average using 2-design: 0.22222222222222235
f2 average using 3-design: 0.11111111111111117
f2 average using 5-design: 0.20000000000000004

{% endhighlight %}
<div class='caption'>
    <span class='caption-label'>
        Averages of $f(x, y, z) = x^2 + xz + yz + z^2$ and $f(x, y, z) = x^4$ for different $t$-designs:
    </span>
    as expected, for <code>f1</code> we get the correct value for all $t$-designs.
    And as expected, for <code>f2</code> only the icosahedron works.
</div>
</div>

From the result above, we have (numerically) confirmed that polyhedra are spherical $t$-designs.

### Average of a function over $n$-spheres
It is possible to define averages over $n$-spheres but we won't bother doing
so here since we don't use them at all.

## State designs
Consider now the Bloch sphere and say we want to average some
function over it. This is where state designs come into play.

If we are dealing with two-qubits states, we will have to average
over a 7-sphere and that's just way too much work so we will not bother
about such states.

Moreover, quantum states are created by evolving some fudiciary state
(such as the $\ket{00}$) towards the desired state.
It will prove much more convenient to deal with two-qubits states
via unitary designs than state designs.

### Functions of a quantum state and their average
The first thing we want to do is understand (at least loosely)
what functions of quantum states are. It is not terribly complicated
even if seeing them the first time can pose some problems
wrapping our heads around. It gets easier once one
sees a bunch of them and remembers that quantum states are just
vectors.  
We will introduce one function that we will use as running example
for the rest of this section and that was introduced in {% cite horodecki1999general %}.

After we have introduced the function in question, we will give it
an operational meaning as given in {% cite horodecki1999general %}
when averaging over all states.

#### Function of a quantum state
A function of a quantum state is pretty much like any other function
but its domain will be over quantum states.
The range can be quantum states, matrices, complex number, etc.

For example, here is a function we will work with in the following
subsection:

{% katexmm %}
$$
\begin{align}
f : \mathbb{C}^n &\rightarrow \mathbb{R} \\
\ket{\psi} &\mapsto \bra{\psi}\mathcal{E}(\ket{\psi}\bra{\psi})\ket{\psi}
\end{align}
$$
{% endkatexmm %}

Where $\mathcal{E}$ is a channel.

Of course, functions we work will need to have some meaning
that allows us to interpret the output of their "execution".

In the case of the function above, let's assume that the channel
$\mathcal{E}$ is a unitary channel. Then it follows that:

{% katexmm %}
$$
\mathcal{E}(\ket{\psi}\bra{\psi}) = \mathcal{E} \ket{\psi}\bra{\psi} \mathcal{E}^\dagger
$$
{% endkatexmm %}

Consequently $f$ can be rewritten as:

{% katexmm %}
$$
\begin{align}
f : \mathbb{C}^n &\rightarrow \mathbb{R} \\
\ket{\psi} &\mapsto \bra{\psi}\mathcal{E} \ket{\psi}\bra{\psi} \mathcal{E}^\dagger\ket{\psi} \\
&\mapsto \lvert \bra{\psi}\mathcal{E} \ket{\psi} \rvert^2
\end{align}
$$
{% endkatexmm %}

This form is easily amenable to interpretation: $f$ measures
how close the states $\ket{\psi}$ and $\mathcal{E} \ket{\psi}$ are.
This is something we can compute with the [SWAP test](https://en.wikipedia.org/wiki/Swap_test).

What's truly important to know about this function is to note
that since $\mathcal{E}$ is fixed but $\ket{\psi}$ varies, the function $f$
actually tells us something about $\mathcal{E}$: it tells how close
states created by $\mathcal{E}$ are close to the input state $\ket{\psi}$.

Hence $f$ calculates the so-called *fidelity* of the channel $\mathcal{E}$:
that is how well the channel $\mathcal{E}$ preserves its input state.
If $\ket{\psi}$ remained unchanged then $f = 1$, otherwise $f < 1$.

#### Average of a function of quantum states
Given our running example using the function $f$,
it is reasonable to ask how to choose the representative
inputs for $f$. Of course, we randomly sample uniformly
the input states then average over all of them.

The average is given by:

{% katexmm %}
$$
\begin{align}
\bar{f} &= \dfrac{1}{Vol(\mathcal{S}(\mathbb{C}^n))} \int_{\mathcal{S}(\mathbb{C}^n)} f(\ket{\psi}) d_{\mu}\ket{\psi} \\
&= \dfrac{1}{Vol(\mathcal{S}(\mathbb{C}^n))} \int_{\mathcal{S}(\mathbb{C}^n)} \bra{\psi}\mathcal{E}(\ket{\psi}\bra{\psi})\ket{\psi} d_{\mu}\ket{\psi}
\end{align}
$$
{% endkatexmm %}

Where $\mathcal{S}(\mathbb{C}^n)$ correspond to "points" from the complex $n$-sphere
and $d_{\mu}\ket{\psi}$ is the appropriate measure.

To understand what the former sentence means, let's place it in the context
of computing the integral above over the Block sphere:
- $\mathcal{S}(\mathbb{C}^n)$ correspond to "points" from the complex $n$-sphere:
    for single qubits, the sphere in the Bloch sphere. So the "points" are going to be
    single-qubit states taken from the Bloch sphere such as $\ket{0}$,
    $\ket{+}$, $\ket{+i}$, etc.
- $d_{\mu}\ket{\psi}$ is the appropriate measure:
    for single-qubit states "living" on the Bloch sphere,
    the measure is exactly as calculated for regular spheres.
    (For two-qubit states, we are dealing with a $7$-sphere {% cite Mosseri_2001 %}.
    That's just too much to bother about so we will not elaborate on them.)

And of course, we always need to make sure to normalize
our integral by dividing the result by the volume of the sphere
$Vol(\mathcal{S}(\mathbb{C}^n))$.
In the case of the Bloch sphere, it is exactly the same as before
when we learned about regular spheres.  
*It is normal to see the normalization factor omitted but it is always assumed*
*present. A generous author will let you know that the measure is normalized*
*to remind you of that normalization factor.*

Now back to the integral above: since we are averaging over all states,
$\bar{f}$ is called the *average fidelity* of the channel $\mathcal{E}$.
{% cite horodecki1999general %} give it the following operational
interpretation which coincides with our preliminary interpretation
of $f$: $\bar{f}$ is the probability that the output state
$\mathcal{E}(\ket{\psi}\bra{\psi})$ passes the test of being the input
state $\ket{\psi}$, averaged over all input states.  
(Note that the **Horodecki**s call $\bar{f}$ *fidelity* but in
current literature it has the more appropriate name of *average fidelity*
and we will stick to this current nomenclature {% cite Nielsen_2002 %}.)

Since we are averaging over all states, $\bar{f}$ depends on the channel
$\mathcal{E}$ and not on the states:

{% katexmm %}
$$
\begin{align}
\bar{f}: \, &\mathbb{C}^{n \times n} \rightarrow \mathbb{R} \\
&\mathcal{E} \mapsto \dfrac{1}{Vol(\mathcal{S}(\mathbb{C}^n))} \int_{\mathcal{S}(\mathbb{C}^n)} \bra{\psi}\mathcal{E}(\ket{\psi}\bra{\psi})\ket{\psi} d_{\mu}\ket{\psi}
\end{align}
$$
{% endkatexmm %}

Based on our previous interpretation of $f$, $\bar{f}$ quantifies how
well $\mathcal{E}$ preserves quantum states: if it equal to $1$
then $\mathcal{E}$ perfectly preserves quantum states. This will be
the case if $\mathcal{E} = \mathbb{1}_n$, that is the identity matrix.
And if it is $0$, then the channel doesn't preserve quantum states.

### Average of a function over the Bloch sphere: analytic solution

### Average of a function over the Bloch sphere: Monte Carlo integration

### Average of a function over the Bloch sphere: state designs

## Unitary designs

## Next steps

## Conclusion

## Derivations
