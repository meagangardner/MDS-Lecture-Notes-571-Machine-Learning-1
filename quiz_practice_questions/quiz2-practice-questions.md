DSCI 571 - Quiz 2 Practice questions
================

_Note that these are just sample questions for you to get an idea about what to expect in the quiz. They are neither meant to cover all the topics we have covered in the last four lectures nor meant to be indicative of the number of questions in the actual quiz._

### Question 1: Preprocessing

We introduced SimpleImputer to deal with missing values. Earlier in the course, we introduced pipelines and discussed how they help avoid violations of the Golden Rule. And yet, we don't always need pipelines to avoid violating the Golden Rule. Consider the code

```imp = SimpleImputer(strategy='?????')
X_train_imputed = imp.fit_transform(X_train)
cross_val_score(classifier, X_train_imputed, y_train)
```

Above I left out the strategy hyperparameter. Take a look at the SimpleImputer documentation; it seems there are 4 possible values of strategy, "mean", "median", "most_frequent", and "constant". For which of these values of the strategy would the above code NOT violate the Golden Rule? Briefly justify your answer. Maximum 3 sentences.

**Solution**
"constant". Because in this case, the imputation is done separately for each row. With the other 3 strategies, the imputed value depends on all rows in the training set.

### Question 2: `DummyClassifier`

In hw4, you implemented DummyClassifier from scratch. This involved implementing four methods: `fit`, `predict`, `predict_proba`, and `score`. If you were asked to implement a different classifier like `DecisionTreeClassifier` or `LogisticRegression`, would any of these 4 functions be the same as for DummyClassifier? Briefly explain. Max 3 sentences.

**Solution**
`score` would be the same. `fit` and `predict_proba` would definitely be different. `predict` might be the same if it happened to be implemented as a threshold on `predict_proba` - it depends on the implementation (either answer is OK).

### Question 3: State whether True or False
Suppose you are building a spam classification using a large corpus with a bag-of-words model. Setting a very small value for the `max_features` hyperparameter of `CountVectorizer` may lead to underfitting. 

**Solution:** True because small value for `max_features` means we are ignoring most of the words in our representation, which is likely to result in underfitting. 

### Question 4: 
What goes wrong in naive Bayes when Laplace smoothing is not used? 

**Solution:**
If any of the conditional probabilities is zero because the feature value does not occur with the target in the training data, the probability estimate of the class would be zero. This happens because naive Bayes naively multiplies all the feature likelihoods together. 

### Question 5: 
Give one advantage of using `RandomizedSearchCV` over `GridSearchCV`. 

**Solution:**
Some possible answers
- `RandomizedSearchCV` is faster compared to `GridSearchCV` because we can control the number of iterations. 
- `RandomizedSearchCV` is a better choice when some parameters are more important than others; adding parameters that do not influence the performance does not affect efficiency.
- By passing probability distributions to `RandomizedSearchCV`, we can explore different parts of the search space. 

### Question 6: 
In Lecture 6 we covered the idea of overfitting on the validation set. We showed that the validation score might not lead you to the hyperparameters that actually give the best test score. In that case, if validation scores are not a faithful representation of test scores, why not just use the test set directly to tune your hyperparameters? 

**Solution**
That would violate the Golden Rule and would lead to overfitting on the test set itself. And then we'd have no unseen test set left. The purpose of the test set is to provide an unbiased evaluation of model performance on data it hasn't seen during training or hyperparameter tuning. If we use the test set for tuning, we make it part of the training process, which means our final test score won't accurately reflect the model's performance on truly unseen data. 

### Question 7: 

Below are several scenarios. For each, recommend one of the above models to your client and briefly justify your choice. (More than one answer may be suitable in each case.)
- SVM with RBF kernel
- Logistic Regression
  
#### 7.1
Your client wants to predict whether a customer will commit fraud. They have two requirements: (1) the model should provide not only predictions but also the level of certainty in these predictions, and it should be fast and scalable; (2) they want insight into the most important features influencing the predictions.

**Solution:**
Logistic Regression, as it provides reasonable probability estimates that can indicate the level of certainty in predictions. Additionally, the model coefficients can help identify the most important features influencing the prediction.

#### 7.2
You are working with sparse data and suspect a non-linear relationship. Achieving high accuracy is a priority, while speed is less crucial.

**Solution:**
SVM with RBF kernel, as it can effectively model non-linear relationships and often achieves strong performance in terms of evaluation scores.

### Question 8
Assuming that all the appropriate libraries are imported, consider the code below. 

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
vec = CountVectorizer()
X_train_enc = vec.fit_transform(X_train)
lr = LogisticRegression(max_iter=2000)

param_dist = {
    "C": loguniform(1e-3, 1e3)
}

random_search = RandomizedSearchCV(
    lr,
    param_distributions=param_dist,
    n_iter=50,
    n_jobs=-1,
    return_train_score=True
)

random_search.fit(X_train_enc, y_train)
```

#### 8.1 
Do you think that the code is violating the golden rule? Briefly explain.  

**Solution:**
Yes, we will be breaking the golden rule because `RandomizedSearchCV` carries out cross-validation under the hood. We are passing already preprocessed data when we call `fit` of random search. But this means that the validation portion from each split was already used in the calls for fit of the `CountVectorizer`. This is information leakage during cross-validation inside the random search.  



#### 8.2 
If you want to also optimize `max_features` of `CountVectorizer` how would you modify the code above? 

**Solution:**

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
pipe_lr = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=2000))
pipe_lr.fit(X_train, y_train)
vocab = pipe_lr.named_steps['countvectorizer'].get_feature_names_out()

param_dist = {
    "logisticregression__C": loguniform(1e-3, 1e3),
    "countvectorizer__max_features": randint(100, len(vocab))
}
random_search = RandomizedSearchCV(
    pipe_lr,
    param_distributions=param_dist,
    n_iter=50,
    n_jobs=-1,
    return_train_score=True
)

random_search.fit(X_train, y_train)
```