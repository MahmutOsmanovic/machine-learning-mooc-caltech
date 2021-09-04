# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 10:58:21 2021

@author: Mahmut Osmanovic
"""

# import libraries
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.style as style
style.use('bmh')

"""
https://github.com/homefish/edX_Learning_From_Data_2017/blob/master/homework_9_final/hw9_p1_2_3_4_5_6.ipynb
• Nonlinear transforms
1. The polynomial transform of order Q = 10 applied to X of dimension d = 2 
results in a Z space of what dimensionality (not counting the constant 
coordinate x0 = 1 or z0 = 1)?
[a] 12
[b] 20
[c] 35
[d] 100
[e] None of the above
"""

"""
Let's have a look at an example from Homework 5, Problem 3. We had a polynomial
transform of order $Q = 4$ and dimension $d=2$:

$\mathbf{x} \mapsto  \mathbf{z} = (1,   x_1,       x_2,     x_1^2,   x_1 x_2, 
  x^2_2, x_1^3, x^2_1 x_2, x_1 x_2^2,     x_2^3,
  x_1^4, x_1^3 x_2, x_1^2 x_2^2, x_1 x_2^3, x_2^4)
  $

The vector $\mathbf{z}$ has $15$ components which we can list as follows:

constant: $1$
order $1$: $ x_1,       x_2$
order $2$: $x_1^2,   x_1 x_2, x^2_2$
order $3$: $x_1^3, x^2_1 x_2, x_1 x_2^2,     x_2^3$
order $4$: $x_1^4, x_1^3 x_2, x_1^2 x_2^2, x_1 x_2^3, x_2^4$
Let's count the monomials in our example:

constant: $1$, number of monomials $ = 1$
order $1$: $ x_1,       x_2$, number of monomials $ = 2$
order $2$: $x_1^2,   x_1 x_2, x^2_2$, number of monomials $ = 3$
order $3$: $x_1^3, x^2_1 x_2, x_1 x_2^2,     x_2^3$, number of monomials $ = 4$
order $4$: $x_1^4, x_1^3 x_2, x_1^2 x_2^2, x_1 x_2^3, x_2^4$, number of 
monomials $ = 5$. You can see that there are $k+1$ monomials of order $k$, 
because the monomials have the form:
$x_1^k x_2^0, x_1^{k-1} x_2^1, x_1^{k-2} x_2^2, ...,
x_1^{1} x_2^{k-1}, x_1^{0} x_2^k,$ with a total of $k+1$ monomials,
e.g. for $k = 2$ we have the $k+1 = 3$ monomials $x_1^2,   
x_1 x_2, x^2_2$. Each monomial corresponds to an entry in the {z} vector, so 
the total number of monomials gives us the length of the vector $\mathbf{z}$ 
which is equal to the dimensionality of the $\cal{Z}$ space.

In total we have $1 + 2 + 3 + 4 + 5 = 15$ monomials, so the dimensionality 
should be $15$. And indeed, we can confirm this by counting the number of 
components of $\mathbf{z}$.

In general, if the highest order is $Q$, we have 
$1 + 2 + ... + (Q+1) = \frac{(Q+1)(Q+2)}{2}$ monomials, so the dimensionality 
of $\cal{Z}$ is $\frac{(Q+1)(Q+2)}{2}$. In our example we have $Q = 4$ which 
yields the dimensionality $\frac{(Q+1)(Q+2)}{2} = \frac{(4+1)(4+2)}{2} = 15$.

The problem asks us for the dimensionality of $\cal{Z}$ if the highest order is 
$Q = 10$. Using this value we get the dimensionality 
$\frac{(Q+1)(Q+2)}{2} = \frac{(10+1)(10+2)}{2} = 66$. 
Ignoring the constant coordinate $z_0 = 1$ the answer is $65$, which is not 
among the given choices, so the correct answer is 1[e].
"""
