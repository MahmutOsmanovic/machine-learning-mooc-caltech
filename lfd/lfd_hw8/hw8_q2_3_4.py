# import libraries
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.style as style
style.use('bmh')
"""
# Read data and visualize some distributions
train = pd.read_csv("D:\\AI\\lfd\\lfd_hw8\\features_train.txt", names = ['digit', 'intensity', 'symmetry'], sep='\s+', dtype=np.float64)

print(train.head(5))

print("\nShape of training data: ", train.shape)

plt.hist(train['digit'], color='tab:blue', edgecolor = 'black')
plt.title('Distributions of Digits')
plt.show()

plt.hist(train['intensity'], color='tab:orange', edgecolor='black')
plt.title('Distribution of Intensity')
plt.show()

plt.hist(train['symmetry'], color='tab:pink', edgecolor='black')
plt.title('Distribution of Symmetry')
plt.show()"""

# Prepare data
# Examine the training data and re-label data for x-vs-all

# Reading in data with np.loadtxt instead of Pandas
# Read with Pandas when you want to make tables, etc.
# Read with Numpy when you want to numerically manipulate the data.
TRAIN_DATA = np.loadtxt("D:\\AI\\lfd\\lfd_hw8\\features_train.txt")
print(TRAIN_DATA[:5,:])
print(TRAIN_DATA.dtype)

# training data ignoring the 0-th column which contains the digit labels
X_TRAIN = TRAIN_DATA[:,1:]

# 0-th column contains digit labels
DIGIT_LABELS = TRAIN_DATA[:,0]

# number of training points
N_TRAIN = X_TRAIN.shape[0]

# function get_labels (one vs. all)
# This function creates new labels 'y' according to the problem statement
def get_labels(x, digit_labels):
    """
    - Takes integer 'x' (digit 0-9)
    - Takes labels
    - Returns new labels
      where the label is 1 for digit == x, and -1 otherwise.
    """
    y = []
    for d in digit_labels:
        if d == x:
            y.append(1)
        else:
            y.append(-1)
    return np.array(y)

# Setting up the classifier
# For SVM we will use https://scikit-learn.org/stable/modules/svm.html, which internally uses libsvm
# classifier clf with the parameters as stated by the homework problem
clf_poly = svm.SVC(C = 0.01, kernel = 'poly', degree = 2, coef0 = 1, gamma = 1)

# Compute in-sample error E_in for x-vs-all
def get_E_in_x_vs_all(x, DIGIT_LABELS, X_TRAIN):
    '''
    - Takes integer x
    - Takes vector DIGIT_LABELS containing true digit labels
    - Takes matrix X_TRAIN with features intensity and symmetry
    - Returns in-sample error E_in for binary classifier with label
      y = 1 if digit == x, otherwise y = -1
    '''
    y_x_vs_all = get_labels(x, DIGIT_LABELS)
    N_TRAIN = X_TRAIN.shape[0]

    # fit and predict
    clf_poly.fit(X_TRAIN, y_x_vs_all)
    y_predict_x_vs_all = clf_poly.predict(X_TRAIN)

    # calculate E_in
    E_in_x_vs_all = sum(y_predict_x_vs_all != y_x_vs_all) / N_TRAIN
    return E_in_x_vs_all


# Compute the digit among [0,2,4,6,8] with the highest E_in
max_error = -1
max_digit = None
selected_classifier_problem_2 = None
error_values = []

for digit in [0,2,4,6,8]:
    current_E_in = get_E_in_x_vs_all(digit, DIGIT_LABELS, X_TRAIN)
    error_values.append(current_E_in)
    print("E_in_{}_vs_all = {}".format(digit, current_E_in))
    if current_E_in > max_error:
        max_error = current_E_in
        max_digit = digit
        
print("\nMaximum error {} is achieved for digit {}".format(max_error, max_digit))

plt.plot([0,2,4,6,8], error_values, 'bo-')
plt.title('Problem 2 - plot for digits in [0,2,4,6,8]')
plt.xlabel('digit in digit-vs-all classifier')
plt.ylabel('in-sample error $E_{in}$')
plt.show()

# plot with vertical bars

fig, ax = plt.subplots(1, 1)    
ax.vlines([0,2,4,6,8], 0, error_values, colors='b', lw=5, alpha=0.5)
plt.plot([0,2,4,6,8], error_values, 'bo')
plt.title('Problem 2 - plot for digits in [0,2,4,6,8]')
plt.xlabel('digit in digit-vs-all classifier')
plt.ylabel('in-sample error $E_{in}$')
plt.show()

# Let's determine the number of support vectors for the classifier 0-vs-all because we will need it for Problem 4.
y_0_vs_all = get_labels(0, DIGIT_LABELS)
selected_classifier_problem_2 = clf_poly.fit(X_TRAIN, y_0_vs_all)
num_support_vectors_problem_2 = sum(selected_classifier_problem_2.n_support_)
print("number of support vectors of classifier chosen in problem 2: ", num_support_vectors_problem_2)

d_values = []
num_support_vectors_values = []

for d in range(0, 10):
    clf_poly.fit(X_TRAIN,  get_labels(d, DIGIT_LABELS))
    num_support_vectors = sum(clf_poly.n_support_)
    print("d={} vs all yields {} support vectors".format(d, num_support_vectors))
    
    d_values.append(d)
    num_support_vectors_values.append(num_support_vectors)
    
plt.plot(d_values, num_support_vectors_values, 'ro-')
plt.xlabel('digit for classifier')
plt.ylabel('number of support vectors')
plt.title('Plot for all digits')
plt.show()

d_values = []
E_in_values = []

for d in range(0,10):
    y_d_vs_all_label = get_labels(d, DIGIT_LABELS)
    clf_poly.fit(X_TRAIN, y_d_vs_all_label)
    y_predict = clf_poly.predict(X_TRAIN)
    current_E_in = np.sum(y_predict != y_d_vs_all_label) / X_TRAIN.shape[0]
    print("E_in_{}_vs_all = {}".format(d, current_E_in))
    
    d_values.append(d)
    E_in_values.append(current_E_in)


plt.plot(d_values, E_in_values, 'go-')
plt.xlabel('digit for classifier')
plt.ylabel('In-sample error $E_{in}$')
plt.title('Plot for all digits')
plt.show()

# Q: 3

min_error = 2**64
min_digit = None
error_values = []

for digit in [1,3,5,7,9]:
    current_E_in = get_E_in_x_vs_all(digit, DIGIT_LABELS, X_TRAIN) 
    error_values.append(current_E_in)
    
    print("E_in_{}_vs_all = {}".format(digit, current_E_in))
    if current_E_in < min_error:
        min_error = current_E_in
        min_digit = digit
    
print("\nMinimum error {} is achieved for digit {}".format(min_error, min_digit))

plt.plot([1,3,5,7,9], error_values, 'bo-')
plt.title('Problem 3 - plot for digits in [1,3,5,7,9]')
plt.xlabel('digit')
plt.ylabel('classifier for digit-vs-all')
plt.show()

y_1_vs_all = get_labels(1, DIGIT_LABELS)
selected_classifier_problem_3 = clf_poly.fit(X_TRAIN, y_1_vs_all)
num_support_vectors_problem_3 = sum(selected_classifier_problem_3.n_support_)
print("The number of support vectors of classifier chosen in problem 3: ", num_support_vectors_problem_3)
print("The number of support vectors of classifier chosen in problem 2: ", max(num_support_vectors_values))

# ANS: 1 vs all

# Q: 4
diff_support_vectors_p1_p2 = abs(num_support_vectors_problem_3 - num_support_vectors_problem_2)
print("The number of support vectors from problem 2 and 3 differ by ", diff_support_vectors_p1_p2)

min_distance_4 = 2**64
min_choice_4 = None

choices_4 = [600, 1200, 1800, 2400, 3000]

for choice in choices_4:
    current_distance = abs(diff_support_vectors_p1_p2 - choice)
    if current_distance < min_distance_4:
        min_distance_4 = current_distance
        min_choice_4 = choice
        
print("The closest choice is: ", min_choice_4)

# ANS: 1800