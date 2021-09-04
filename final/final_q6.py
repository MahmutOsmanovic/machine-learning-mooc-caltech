# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 21:25:38 2021

@author: Mahmut Osmanovic
"""

"""
A soft order constraint $\sum_{i=0}^{Q} w_q^2 \leq C$ leads to the augmented error

$E_{\text{aug}}(\mathbf{w}) = E_{\text{in}}(\mathbf{w}) + \frac{\lambda}{N} \mathbf{w}^T \mathbf{w}$

(see lecture 12, slides 8, 9 and 10)

Thus, answer 6[b] is the correct answer.

https://github.com/homefish/edX_Learning_From_Data_2017/blob/master/homework_9_final/hw9_p1_2_3_4_5_6.ipynb
https://home.work.caltech.edu/slides/slides12.pdf

Augmented error = Errorrate minus as large of a subset as possible from the
overfitting.
"""