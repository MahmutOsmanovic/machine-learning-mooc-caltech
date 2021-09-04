# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:45:28 2021

@author: Mahmut Osmanovic
"""

"""
BEST ANSWER:
Problem 3
Option 3[d] states that we can determine if there is overfitting by comparing the values of the difference $(E_{out} - E_{in}$) only without knowing $E_{in}$ and $E_{out}$.

However, by definition overfitting occurs if $E_{in}$ decreases while $E_{out}$ increases. This means we need to know the values of $E_{in}$ and $E_{out}$ to determine if there is overfitting, and it is not sufficient to know the difference $(E_{out} - E_{in}$) only.

Thus, the correct answer to Problem 3 is 3[d].

"""

""" 
MY ANSWER:
    
    A.  Yes, since different hypothesis have different complexity they should 
    gather ever so slightly different in sample results.
    B. Same reasons as for A, applies to the out of sample error aswell.
    C. Follows from A and B.
    D. False. The difference "d", d = E_out-E_in might, let us say increase
    although the E_out error stays the same, just the E_in decreases.
    For overfitting to occur, the E_out ought to increase as E_in decreases.
    Thus D is false, meaning, you can NOT determine whether or not there
    is overfitting solely by know d without knowing E_out and E_in.
    E. err(h1) = c1, err(h2) = c2, both error rates are gathered equivalently
    using the most optimal method. You can only assert that model h2 is 
    overfitting when you know the E_in and E_out of both h1 and h2. Since at
    that point you can compare them. Given that E_in(h1) =/= E_out(h2) and
    E_out(h1) << E_out(h2) you can conclude that h2 is overfitting. Meaning,
    you know whether or not a model is overfitting when you can compare its 
    error rates with another. The case of early stopping is related to finding
    the optimal error rate given model h_i, it doesn't tell us to what extend
    model h_i is overfitting the data. The best weights might still be
    overfitting the data relative to another model. One can optimize all models
    to gain their best weights, w.r.t that model. What you want is to 
    discriminate between models, and whether you overfitting relative to
    another model. The only way to know an answer which is in nature relative
    is to compare it to another model.

"""