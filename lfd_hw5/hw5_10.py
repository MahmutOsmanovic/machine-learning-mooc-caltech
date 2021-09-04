# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 04:09:06 2021

@author: Mahmu
"""

"""
The Perceptron Learning Algorithm can be implemented as SGD using which
of the following error functions en(w)? Ignore the points w at which en(w) is
not twice differentiable.
[a] en(w) = e
−ynw|xn
[b] e_n(w) = e**(−y_nw**(T)xn)
[c] e_n(w) = (yn − w|xn)**2
[d] e_n(w) = ln(1 + e−ynw|xn )
[e] e_n(w) = − min(0, ynw|xn)

"""

# ANSWER

"""
https://github.com/homefish/edX_Learning_From_Data_2017/blob/master/homework_5/hw5_p10_PLA_as_SGD.ipynb

In this problem we are asked to choose the error function $e(\mathbf{w})$ in SGD such that the PLA is implemented.

Recall that the weight vector $\mathbf{w}$ in PLA is updated via

$\mathbf{w} \leftarrow \mathbf{w} + y \mathbf{x}$

where $(\mathbf{x}, \mathbf{y})$ is a misclassified point (see also slide 13 of Lecture 1).

Let's choose $e(\mathbf{w}) = -y \mathbf{w}^T \mathbf{x}$ for SGD. The partial derivatives are

$\frac{\partial e(\mathbf{w})}{\partial w_k} 
= \frac{\partial }{\partial w_k} (-y \mathbf{w}^T \mathbf{x})
= -y x_k $

with $k \in \{0,1,2 \}$ .

So the gradient is:

$\nabla e = [\frac{\partial e(\mathbf{w})}{\partial w_0},\frac{\partial e(\mathbf{w})}{\partial w_1}, \frac{\partial e(\mathbf{w})}{\partial w_2}] 
          = [-y x_0, -y x_1, -y x_2]
          = -y [x_0, x_1, x_2]
          = -y \mathbf{x}$

In SGD we update $\mathbf{w}$ via

$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla e(\mathbf{w})$

(see slide 3 and 4 of lecture 10 and slide 23 of lecture 9)

If we now insert the error function $e(\mathbf{w})$ we chose, then we get

$\mathbf{w} \leftarrow \mathbf{w} - \eta (-y \mathbf{x})$

$\mathbf{w} \leftarrow \mathbf{w} + \eta (y \mathbf{x})$

With $\eta = 1$ we get

$\mathbf{w} \leftarrow \mathbf{w} + y \mathbf{x}$

However, we also have to accomodate for the fact that we only pick misclassified points in PLA. This can be done by choosing

$e(\mathbf{w}) = - \min(0, y \mathbf{w}^T \mathbf{x})$

because for correctly classified points the expression $y \mathbf{w}^T \mathbf{x}$ is positive, and in that case

$e(\mathbf{w}) = - \min(0, y \mathbf{w}^T \mathbf{x}) = 0$

(see slide 12 of lecture 1)

So for correctly classified points the gradient is

$\nabla e(\mathbf{w}) = \nabla 0 = \mathbf{0}$

which means that correctly classified points do not contribute to a change of $\mathbf{w}$ during the update.

So the correct answer to this problem is 10[e] $e(\mathbf{w}) = - \min(0, y \mathbf{w}^T \mathbf{x})$ .
"""

