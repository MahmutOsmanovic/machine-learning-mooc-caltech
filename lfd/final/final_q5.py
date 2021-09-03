# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 21:22:33 2021

@author: Mahmut Osmanovic
"""

"""
https://github.com/homefish/edX_Learning_From_Data_2017/blob/master/homework_9_final/hw9_p1_2_3_4_5_6.ipynb
"""

"""
If $\mathbf{w}_{\text{lin}}$ is the linear regression solution, and additionally satisfies the constraint $\mathbf{w}_{\text{lin}}^T \Gamma^T \Gamma \mathbf{w}_{\text{lin}} \leq C$, then $\mathbf{w}_{\text{lin}}$ is a solution to

minimize $\frac{1}{N} \sum_{i=1}^{N} (\mathbf{w}^T \mathbf{x} - y_n)$ subject to $\mathbf{w}^T \Gamma^T \Gamma \mathbf{w} \leq C$

If $\mathbf{w}_{\text{reg}}$ is the solution to the regularized linear regression problem described above, then we can write $\mathbf{w}_{\text{reg}} = \mathbf{w}_{\text{lin}}$. Thus, the correct answer is 5[a].

The least squares linear regression solution satisfies the constraint. 
Thus the regularization has no effect, and w_reg = w_lin.

"""