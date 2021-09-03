# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:52:41 2021

@author: Mahmut Osmanovic
"""

"""
• Cross Validation
In the next two problems, we will experiment with 10-fold cross validation for the
polynomial kernel. Because Ecv is a random variable that depends on the random
partition of the data, we will try 100 runs with different partitions and base our
answer on how many runs lead to a particular choice.
4
7. Consider the 1 versus 5 classifier with Q = 2. We use Ecv to select C ∈
{0.0001, 0.001, 0.01, 0.1, 1}. If there is a tie in Ecv, select the smaller C. 
[Within the 100 random runs, which of the following statements is correct?]
[a] C = 0.0001 is selected most often.
[b] C = 0.001 is selected most often.
[c] C = 0.01 is selected most often.
[d] C = 0.1 is selected most often.
[e] C = 1 is selected most often.
8. Again, consider the 1 versus 5 classifier with Q = 2. [For the winning selection
in the previous problem, the average value of Ecv over the 100 runs is closest to]
(a) 0.001
(b) 0.003
(c) 0.005
(d) 0.007
(e) 0.009
"""

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.style as style
style.use('bmh')

df_train = pd.read_csv('D:\\AI\\lfd\\lfd_hw8\\features_train.txt', names =
           ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)
df_test = pd.read_csv('D:\\AI\\lfd\\lfd_hw8\\features_test.txt', names =
           ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)

ones = df_train[df_train['digit'] == 1].assign(y = np.ones(df_train[df_train['digit'] == 1].shape[0]))
fives = df_train[df_train['digit'] == 5].assign(y = -np.ones(df_train[df_train['digit'] == 5].shape[0]))
ones_test = df_test[df_test['digit'] == 1].assign(y = np.ones(df_test[df_test['digit'] == 1].shape[0]))
fives_test = df_test[df_test['digit'] == 5].assign(y = -np.ones(df_test[df_test['digit'] == 5].shape[0]))

df_1_vs_5 = ones.append(fives, ignore_index=True)
df_test_1_vs_5 = ones_test.append(fives_test, ignore_index=True)

X_train_1_vs_5 = np.c_[df_1_vs_5['intensity'], df_1_vs_5['symmetry']]
y_train_1_vs_5 = np.array(df_1_vs_5['y'])
X_test_1_vs_5 = np.c_[df_test_1_vs_5['intensity'], df_test_1_vs_5['symmetry']]
y_test_1_vs_5 = np.array(df_test_1_vs_5['y'])

# glue X_train_1_vs_5 and y_train_1_vs_5
print("Five first rows of glued_X_y_train_1_vs_5:")
glued_X_y_train_1_vs_5 = np.c_[X_train_1_vs_5, y_train_1_vs_5]
print(glued_X_y_train_1_vs_5[:5, :])
print(glued_X_y_train_1_vs_5.shape, end='\n\n')

# shuffle the data
print("Five first rows after shuffling glued_X_y_train_1_vs_5:")
np.random.shuffle(glued_X_y_train_1_vs_5)
print(glued_X_y_train_1_vs_5[:5, :])
print(glued_X_y_train_1_vs_5.shape, end='\n\n')

# partition the indices into 10 parts
k_fold = 10
indices = np.arange(glued_X_y_train_1_vs_5.shape[0])
partition_indices = np.array_split(indices, k_fold)
#print(*partition_indices, sep='\n\n')

# for each chunk we store minimum and maximum index
print("This list contains the ranges of each indices chunk.")
partition_ranges = [(min(p), max(p)) for p in partition_indices]
print(partition_ranges, end='\n\n')

RUNS = 100
C_values = [10**k for k in [-4, -3, -2, -1, 0]]
E_cv_avg_values = []

#-------------------------------------------------

RUNS = 100
C_values = [10**k for k in [-4, -3, -2, -1, 0]]
winners = []

#-------------------------------------------------

    
# do 100 runs, for each run determine which C yields the lowest cross validation error
for run in range(RUNS):

    # glue X_train_1_vs_5 and y_train_1_vs_5
    glued_X_y_train_1_vs_5 = np.c_[X_train_1_vs_5, y_train_1_vs_5]

    # shuffle the data
    np.random.shuffle(glued_X_y_train_1_vs_5)

    # get 10-fold partitions
    XX = glued_X_y_train_1_vs_5
    k_fold = 10
    indices = np.arange(XX.shape[0])
    partition_indices = np.array_split(indices, k_fold)
    partition_ranges = [(min(p), max(p)) for p in partition_indices]


    min_C = None
    min_E_cv = 2**64
    
    for C in C_values:
        
        
        # Start cross validating
        e_values = []    # errors on validation sets  
        
        for min_p, max_p in partition_ranges:
            # all data except the part from min_p to max_p
            D_train = np.r_[ XX[ : min_p], XX[max_p + 1 : ] ]
            D_val = XX[min_p : max_p+1]

            X_train = D_train[:, :2]
            y_train = D_train[:, 2]
            X_val = D_val[:, :2]
            y_val = D_val[:, 2]

            clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1)
            clf_1_vs_5.fit(X_train,  y_train)

            # cross validation error e_n
            y_predict_val_1_vs_5 = clf_1_vs_5.predict(X_val)
            N_val = X_val.shape[0]
            e_n = sum(y_val != y_predict_val_1_vs_5) / N_val

            e_values.append(e_n)

        # see slide 17, lecture 13
        E_cv = np.mean(e_values)
        
        if E_cv < min_E_cv:
            min_E_cv = E_cv
            min_C = C
            
    winners.append(min_C)
    
    #---------------

d_count = {C:0 for C in C_values}

for item in winners:
    d_count[item] += 1

for item, count in d_count.items():
    print("C = {} won \t{} times".format(item, count))

y_values = d_count.values()



fig, ax = plt.subplots(1, 1)    
ax.vlines(C_values, 0, y_values, colors='b', lw=5, alpha=0.5)

    
plt.semilogx(C_values, y_values, 'ro')
plt.title('Distribution of winners for 100 runs')
plt.xlabel('$C$')
plt.ylabel('count')
plt.show()

"""
Result for Problem 7
The classifier with $C = 0.001$ hast the most wins over 100 runs. This is interesting because $C = 0.01$ yields the lowest average value of $E_{cv}$.
"""

"""
Addendum for Problem 8
We again plot the average cross validation error as a function of $C$, however this time we change the order of the loops to ensure that the different $C$ values 'see' the same shuffled data set.

In contrast, previously we generated 100 new shuffled data sets for each $C$.
"""

# I want to switch order of loops from Problem 7, because previously each of the C values 
# gets a different permutation of the the data set.

RUNS = 100
C_values = [10**k for k in [-4, -3, -2, -1, 0]]
E_cv_avg_values = []   # for each C_value store the average E_cv
E_cv_values = [[] for _ in range(5)]    # collect E_cv of each run and use np.mean => expected E_cv

#-------------------------------------------------

    
# for each C value do 100 runs
for run in range(RUNS):

    # glue X_train_1_vs_5 and y_train_1_vs_5
    glued_X_y_train_1_vs_5 = np.c_[X_train_1_vs_5, y_train_1_vs_5]

    # shuffle the data
    np.random.shuffle(glued_X_y_train_1_vs_5)

    # get 10-fold partitions
    XX = glued_X_y_train_1_vs_5
    k_fold = 10
    indices = np.arange(XX.shape[0])
    partition_indices = np.array_split(indices, k_fold)
    partition_ranges = [(min(p), max(p)) for p in partition_indices]

    
    for index, C in enumerate(C_values):

        e_values = []    # errors on validation sets        
        
        # Start cross validating
        for min_p, max_p in partition_ranges:
            # all data except the part from min_p to max_p
            D_train = np.r_[ XX[ : min_p], XX[max_p + 1 : ] ]
            D_val = XX[min_p : max_p+1]

            X_train = D_train[:, :2]
            y_train = D_train[:, 2]
            X_val = D_val[:, :2]
            y_val = D_val[:, 2]

            clf_1_vs_5 = svm.SVC(C, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1)
            clf_1_vs_5.fit(X_train,  y_train)

            # cross validation error e_n
            y_predict_val_1_vs_5 = clf_1_vs_5.predict(X_val)
            N_val = X_val.shape[0]
            e_n = sum(y_val != y_predict_val_1_vs_5) / N_val

            e_values.append(e_n)

        # see slide 17, lecture 13
        E_cv = np.mean(e_values)
        E_cv_values[index].append(E_cv)
        #print(E_cv_values)        
        #print("index = {}, E_cv = {}".format(index, E_cv))



for i in range(5):
    E_cv_avg_values.append(np.mean(E_cv_values[i]))

#---------------

for C, E_cv_avg in zip(C_values, E_cv_avg_values):
    print("C = {} => \tE_cv_avg = {}".format(C, E_cv_avg))

plt.semilogx(C_values, E_cv_avg_values, 'bo-')
plt.xlabel("$C$")
plt.ylabel("average $E_{cv}$ over 100 runs")
plt.title("Cross validation error")
plt.show()

"""
Result:
The graph looks similar to the one we created before where we shuffled the data independently for each $C$. Here, we use the same shuffled data for each $C$.
"""

"""
Explanation for why the winning classifier doesn't yield the lowest average $E_{in}$ ?
I want to explore a little why the winning classifier with $C=0.001$ does not yield the lowest average $E_{in}$. The lowest average $E_{in}$ is achieved for $C=0.01$
"""

# E_cv_for_each_C is a list of arrays, where each array contains the E_cv values
# of each C over 100 runs
E_cv_for_each_C = np.array(E_cv_values).T
df_whole = pd.DataFrame(E_cv_for_each_C, columns = ['C=0.0001', 'C=0.001', 'C=0.01', 'C=0.1', 'C=1'])
print(df_whole.head(5))
print('\n\n')
print(df_whole.describe())


print('----------------------------')

NUMBER_OF_BINS = 100

hist_C_01 = plt.hist(df_whole['C=0.01'], edgecolor = 'black', color = 'orange', bins = NUMBER_OF_BINS)
plt.title('Distribution of E_cv for C=0.01')
#plt.xscale('log') 
plt.show()
print(df_whole['C=0.01'].describe())

print('----------------------------')

hist_C_001 = plt.hist(df_whole['C=0.001'], edgecolor = 'black', color = 'c', bins = NUMBER_OF_BINS)
plt.title('Distribution of E_cv for C=0.001')
#plt.xscale('log') 
plt.show()


print(df_whole['C=0.001'].describe())

print('----------------------------')


plt.hist(df_whole['C=0.001'], edgecolor = 'black', bins = NUMBER_OF_BINS, color = 'c', alpha = 0.5, label='C=0.001')
plt.hist(df_whole['C=0.01'], edgecolor = 'black', bins = NUMBER_OF_BINS, color = 'orange', alpha = 0.5, label='C=0.01')
#plt.xscale('log') 
plt.show()
plt.legend()
print(df_whole['C=0.001'].describe())

"""
Conclusion
Well, I don't have any conclusions. I can't see from the data why the classifier with the lowest average $E_{cv}$ is not the winner.

Let's try to dig a little deeper by looking at every single value of $E_cv$ over the 100 runs.
"""

# distribution of E_cv values for C = 0.01
E_cv_values_C_0_01 = E_cv_values[1]
dict_C_0_01 = {}

for E_cv in E_cv_values_C_0_01:
    if not E_cv in dict_C_0_01:
        dict_C_0_01[E_cv] = 0
    dict_C_0_01[E_cv] += 1
    
sorted_list_E_cv_C_0_01 = sorted(dict_C_0_01.items(), key=lambda t: t[0])
print("For C = 0.01:")
print("(E_cv, occurence):\n")
print(*sorted_list_E_cv_C_0_01[:10], sep='\n')

# distribution of E_cv values for C = 0.001
E_cv_values_C_0_001 = E_cv_values[2]
dict_C_0_001 = {}

for E_cv in E_cv_values_C_0_001:
    if not E_cv in dict_C_0_001:
        dict_C_0_001[E_cv] = 0
    dict_C_0_001[E_cv] += 1
    
sorted_list_E_cv_C_0_001 = sorted(dict_C_0_001.items(), key=lambda t: t[0])
print("For C = 0.001:")
print("(E_cv, occurence):\n")
print(*sorted_list_E_cv_C_0_001[:10], sep='\n')

x_values_C_0_01 = [tup[0] for tup in sorted_list_E_cv_C_0_01]
y_values_C_0_01 = [tup[1] for tup in sorted_list_E_cv_C_0_01]
plt.plot(x_values_C_0_01, y_values_C_0_01, 'ro-', label='C=0.01')

x_values_C_0_001 = [tup[0] for tup in sorted_list_E_cv_C_0_001]
y_values_C_0_001 = [tup[1] for tup in sorted_list_E_cv_C_0_001]
plt.plot(x_values_C_0_001, y_values_C_0_001, 'bo-', label='C=0.001')


plt.title("Comparing distribution of E_cv")
plt.legend()
plt.show()

"""
Unfortunately, I still can't make any conclusions.

Update: KFold from scikit-learn to split indices
I just learned that there is a class called KFold to split indices into almost equally sized chunks.
"""

# testing k_fold
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

import numpy as np
from sklearn.model_selection import KFold
vv = np.arange(10) 
X = np.c_[vv, vv*vv, vv+3]
y = np.arange(10)

print("Training set X:")
print(X)

print("\nLabels y:")
print(y)
print("\n")

kf = KFold(n_splits=3)
kf.get_n_splits(X)
print(kf, end='\n\n')  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "VAL:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X[train_index, :], end='\n\n')
    
# interesting, I could have used this!
#print(X[np.array([0,2,4]),1:3])