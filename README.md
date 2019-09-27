
# Image Recognition with PCA - Lab

## Introduction

In this lab, you'll explore the classic MNIST dataset of handwritten digits. While not as large as the previous dataset on facial image recognition, it still provides a 64 dimensional dataset that is ripe for feature reduction.

## Objectives

You will be able to:
* Train a baseline classifier using sci-kit learn
* Use grid search to optimize the hyperparameters of a classifier
* Perform dimensionality reduction using PCA
* Calculate the time savings and performance gains of layering in PCA as a preprocessing step in machine learning pipelines

## Load the Data

To start, load the dataset using `sklearn.datasets.load_digits`.


```python
#Your code here
```


```python
# __SOLUTION__ 
#Your code here

from sklearn.datasets import load_digits
data = load_digits()
print(data.data.shape, data.target.shape)
```

    (1797, 64) (1797,)


## Preview the Dataset

Now that the dataset is loaded, display the images of the first 20 pictures.


```python
#Your code here
```


```python
# __SOLUTION__ 
#Your code here

import matplotlib.pyplot as plt
%matplotlib inline

fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(10,10))
for n in range(20):
    i = n //5
    j = n%5
    ax = axes[i][j]
    ax.imshow(data.images[n], cmap=plt.cm.gray)
plt.title('First 20 Images From the MNIST Dataset');
```


![png](index_files/index_6_0.png)


## Baseline Model

Now it's time to fit an initial baseline model to compare against. Fit a support vector machine to the dataset using `sklearn.sv.SVC()`. Be sure to perform a train test split, record the training time and print the training and testing accuracy of the model.


```python
#Your code here
```


```python
# __SOLUTION__ 
from sklearn import svm
from sklearn.model_selection import train_test_split
```


```python
# __SOLUTION__ 
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=22)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (1347, 64) (450, 64) (1347,) (450,)



```python
# __SOLUTION__ 
clf = svm.SVC(gamma='auto')
%timeit clf.fit(X_train, y_train)
```

    252 ms ± 14.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
# __SOLUTION__
# Naive Baseline
```


```python
# __SOLUTION__ 
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print('Training Accuracy: {}\tTesting Accuracy: {}'.format(train_acc, test_acc))
```

    Training Accuracy: 1.0	Testing Accuracy: 0.58


### Grid Search Baseline

Refine the initial model slightly by using a grid search to tune the hyperparameters. The two most important parameters to adjust are "C" and "gamma". Once again, be sure to record the training time as well as the train and test accuracy.


```python
#Your code here
# ⏰ Your code may take several minutes to run
```


```python
# __SOLUTION__ 
import numpy as np
from sklearn.model_selection import GridSearchCV

clf = svm.SVC()
param_grid = {"C" : np.linspace(.1, 10, num=11),
             "gamma" : np.linspace(10**-3, 5, num=11)}
grid_search = GridSearchCV(clf, param_grid, cv=5)
%timeit grid_search.fit(X_train, y_train)
```

    1min 22s ± 127 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
# __SOLUTION__ 
grid_search.best_params_
```




    {'C': 2.08, 'gamma': 0.001}




```python
# __SOLUTION__ 
train_acc = grid_search.best_estimator_.score(X_train, y_train)
test_acc = grid_search.best_estimator_.score(X_test, y_test)
print('Training Accuracy: {}\tTesting Accuracy: {}'.format(train_acc, test_acc))
```

    Training Accuracy: 1.0	Testing Accuracy: 0.9911111111111112


## Compressing with PCA

Now that you've fit a baseline classifier, it's time to explore the impacts of using PCA as a preprocessing technique. To start, perform PCA on X_train. (Be sure to only fit PCA to X_train; you don't want to leak any information from the test set.) Also, don't reduce the number of features quite yet. You'll determine the number of features needed to account for 95% of the overall variance momentarily.


```python
#Your code here
```


```python
# __SOLUTION__ 
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_style('darkgrid')
```


```python
# __SOLUTION__ 
pca = PCA()
X_pca = pca.fit_transform(X_train)
```

## Plot the Explained Variance versus Number of Features

In order to determine the number of features you wish to reduce the dataset to, it is sensible to plot the overall variance accounted for by the first n principle components. Create a graph of the variance explained versus the number of principle components.


```python
#Your code here
```


```python
# __SOLUTION__ 
plt.plot(range(1,65), pca.explained_variance_ratio_.cumsum())
```




    [<matplotlib.lines.Line2D at 0x1a19ddb7f0>]




![png](index_files/index_25_1.png)


## Determine the Number of Features to Capture 95% of the Datasets Variance

Great! Now determine the number of features needed to capture 95% of the dataset's overall variance.


```python
#Your code here
```


```python
# __SOLUTION__ 
total_explained_variance = pca.explained_variance_ratio_.cumsum()
n_over_95 = len(total_explained_variance[total_explained_variance >= .95])
n_to_reach_95 = X.shape[1] - n_over_95 + 1
print("Number features: {}\tTotal Variance Explained: {}".format(n_to_reach_95, total_explained_variance[n_to_reach_95-1]))
```

    Number features: 29	Total Variance Explained: 0.9549611953216074


## Subset the Dataset to these Principle Components which Capture 95%+ of the Overall Variance

Use your knowledge to reproject the dataset into a lower dimensional space using PCA. 


```python
#Your code here
```


```python
# __SOLUTION__ 
pca = PCA(n_components=n_to_reach_95)
X_pca_train = pca.fit_transform(X_train)
pca.explained_variance_ratio_.cumsum()[-1]
```




    0.954954231338425



## Refit a Model on the Compressed Dataset

Now, refit a classification model to the compressed dataset. Be sure to time the required training time, as well as the test and training accuracy.


```python
#Your code here
```


```python
# __SOLUTION__ 
X_pca_test = pca.transform(X_test)
clf = svm.SVC(gamma='auto')
%timeit clf.fit(X_pca_train, y_train)
```

    137 ms ± 958 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
# __SOLUTION__ 
train_pca_acc = clf.score(X_pca_train, y_train)
test_pca_acc = clf.score(X_pca_test, y_test)
print('Training Accuracy: {}\tTesting Accuracy: {}'.format(train_pca_acc, test_pca_acc))
```

    Training Accuracy: 1.0	Testing Accuracy: 0.14888888888888888


### Grid Search

Finally, use grid search to find optimal hyperparameters for the classifier on the reduced dataset. Be sure to record the time required to fit the model, the optimal hyperparameters and the test and train accuracy of the resulting model.


```python
#Your code here
# ⏰ Your code may take several minutes to run
```


```python
# __SOLUTION__ 
clf = svm.SVC()
param_grid = {"C" : np.linspace(.1, 10, num=11),
             "gamma" : np.linspace(10**-3, 5, num=11)}
grid_search = GridSearchCV(clf, param_grid, cv=5)
%timeit grid_search.fit(X_pca_train, y_train)
```

    51.8 s ± 917 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
# __SOLUTION__ 
grid_search.best_params_
```




    {'C': 2.08, 'gamma': 0.001}




```python
# __SOLUTION__ 
train_acc = grid_search.best_estimator_.score(X_pca_train, y_train)
test_acc = grid_search.best_estimator_.score(X_pca_test, y_test)
print('Training Accuracy: {}\tTesting Accuracy: {}'.format(train_acc, test_acc))
```

    Training Accuracy: 0.9992576095025983	Testing Accuracy: 0.9933333333333333


## Summary

Well done! In this lab, you employed PCA to reduce a high dimensional dataset. With this, you observed the potential cost benefits required to train a model and performance gains of the model itself.
