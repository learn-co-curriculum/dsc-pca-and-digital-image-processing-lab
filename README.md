# Image Recognition with PCA - Lab

## Introduction

In this lab, you'll explore the classic MNIST dataset of handwritten digits. While not as large as the previous dataset on facial image recognition, it still provides a 64-dimensional dataset that is ripe for feature reduction.

## Objectives

In this lab you will: 

- Use PCA to discover the principal components with images 
- Use the principal components of  a dataset as features in a machine learning model 
- Calculate the time savings and performance gains of layering in PCA as a preprocessing step in machine learning pipelines 

## Load the data

Load the `load_digits` dataset from the `datasets` module of scikit-learn. 


```python
# Load the dataset

data = None
print(data.data.shape, data.target.shape)
```

## Preview the dataset

Now that the dataset is loaded, display the first 20 images.


```python
# Display the first 20 images 

```

## Baseline model

Now it's time to fit an initial baseline model. 

- Split the data into training and test sets. Set `random_state=22` 
- Fit a support vector machine to the dataset. Set `gamma='auto'` 
- Record the training time 
- Print the training and test accucary of the model 


```python
# Split the data



X = None
y = None
X_train, X_test, y_train, y_test = None
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```


```python
# Fit a naive model 
clf = None

```


```python
# Training and test accuracy
train_acc = None
test_acc = None
print('Training Accuracy: {}\nTesting Accuracy: {}'.format(train_acc, test_acc))
```

### Grid search baseline

Refine the initial model by performing a grid search to tune the hyperparameters. The two most important parameters to adjust are `'C'` and `'gamma'`. Once again, be sure to record the training time as well as the training and test accuracy.


```python
# Your code here
# ⏰ Your code may take several minutes to run
```


```python
# Print the best parameters 

```


```python
# Print the training and test accuracy 
train_acc = None
test_acc = None
print('Training Accuracy: {}\tTesting Accuracy: {}'.format(train_acc, test_acc))
```

## Compressing with PCA

Now that you've fit a baseline classifier, it's time to explore the impacts of using PCA as a preprocessing technique. To start, perform PCA on `X_train`. (Be sure to only fit PCA to `X_train`; you don't want to leak any information from the test set.) Also, don't reduce the number of features quite yet. You'll determine the number of features needed to account for 95% of the overall variance momentarily.


```python
# Your code here
```

## Plot the explained variance versus the number of features

In order to determine the number of features you wish to reduce the dataset to, it is sensible to plot the overall variance accounted for by the first $n$ principal components. Create a graph of the variance explained versus the number of principal components.


```python
# Your code here
```

## Determine the number of features to capture 95% of the variance

Great! Now determine the number of features needed to capture 95% of the dataset's overall variance.


```python
# Your code here
```

## Subset the dataset to these principal components which capture 95% of the overall variance

Use your knowledge to reproject the dataset into a lower-dimensional space using PCA. 


```python
# Your code here
```

## Refit a model on the compressed dataset

Now, refit a classification model to the compressed dataset. Be sure to time the required training time, as well as the test and training accuracy.


```python
# Your code here
```

### Grid search

Finally, use grid search to find optimal hyperparameters for the classifier on the reduced dataset. Be sure to record the time required to fit the model, the optimal hyperparameters and the test and train accuracy of the resulting model.


```python
# Your code here
# ⏰ Your code may take several minutes to run
```


```python
# Print the best parameters 

```


```python
# Print the training and test accuracy 
train_acc = None
test_acc = None
print('Training Accuracy: {}\tTesting Accuracy: {}'.format(train_acc, test_acc))
```

## Summary

Well done! In this lab, you employed PCA to reduce a high dimensional dataset. With this, you observed the potential cost benefits required to train a model and performance gains of the model itself.
