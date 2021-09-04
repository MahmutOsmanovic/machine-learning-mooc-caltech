# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:25:42 2021

@author: Mahmu
"""

"Problem 7"
"The notation $H(Q,C,Q_0)$ means that we are setting all coefficients $w_q$ with $q \geq Q_0$ to the value $C$"

"In 7[c] we are setting coefficients to zero, e.g. $H(10,0,3)$ sets all coefficients greater or equal to $3$ to zero."
"This means $H(10,0,3)$ consists of all second order polynomials. Similarly, $H(10,0,4)$ consists of all third order polynomials." 
"Hence, the intersection consists of all second order polynomials, and we have"

"$H(10,0,3) \cap H(10,0,4) = H_{2}$"

"So the correct answer is 7[c]."

"Note:"

"By setting coefficients to zero we are constraining the available hypotheses. You can think of this as regularization"
"if you recall that regularization means constraining the solutions to within a circle (see lecture 12, slide 9)."

"You can also think of the Legendre polynomials as basis vectors who are orthogonal to each other with respect to some" 
"inner product, and who form a complete set meaning that you can represent any function in a certain space as a linear" 
"combination of Legendre polynomials. By setting coefficients to zero you are reducing the span."
print("ANS: 7[c]")