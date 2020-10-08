
# Logistic Regression

## Learning goals

1. Compare predicting a continuous outcome to predicting a class
2. Compare linear to logistic regression as classification models
3. Describe how logistic regression works under the hood
4. Learn how to interpret a trained logistic model's coefficients
5. Familiarize ourselves with Maximum Likelihood Estimation
6. Explore the C (inverse regularization) paramater and hyperparameter tune
7. Describe the assumptions of logistic regression


```python
from src.student_caller import three_random_students, one_random_student
from src.student_list import student_first_names
```

# Why logistic regression as the 1st of our classifiers?

- It is widely used.

- Logistic regression takes a concept we are familiar with, a linear equation, and translates it into a form fit for predicting a class.  

- It generally can't compete with the best supervised learning algorithms, but it is **simple, fast, and interpretable**.  

- As we will see in mod 4, it will also serve as a segue into our lessons on **neural nets**.


# 1. Compare predicting a continuous outcome to predicting a class

Thus far, we have worked to predict continuous target variables using linear regression. 

  - Continous target variables:
        - Sales price of a home
        - MPG of a car
        - A country's life expectancy rate
        
We will now transition into another category of prediction: classification. Instead of continous target variables, we will be predicting whether records from are data are labeled as a particular class.  Whereas the output for the linear regression model can be any number, the output of our classification algorithms can only be a value designated by a set of discrete outcomes.

  - Categorical target variables:
        - Whether an employee will stay at a company or leave (churn)
        - Whether a tumor is cancerous or benign
        - Whether a flower is a rose, a dandelion, a tulip, or a daffodil
        - Whether a voter is Republican, Democrat, or Independent



Let's navigate to the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table), and browse there classification datasets.  Which one's catch your eye?

![discuss](https://media.giphy.com/media/l0MYIAUWRmVVzfHag/giphy.gif)


```python
three_random_students(student_first_names)
```

### We are still dealing with **labeled data**.

![labels](https://media.giphy.com/media/26Ff5evMweBsENWqk/giphy.gif)


This is still supervised learning. 

But now, instead of the label being a continuous value, such as house price, the label is the category.  This can be either binary or multiclass.  But we still need the **labels** to train our models.



# 2. Compare linear to logistic regression as classification models


The goal of logistic regression, and any classification problem, is to build a model which accurately separates the classes based on independent variables.  

We are already familiar with how linear regression finds a best-fit "line".  It uses the **MSE cost function** to minimize the difference between true and predicted values.  

A natural thought would be to use that "line" to descriminate between classes: Everything with an output greater than a certain point is classified as a 1, everything below is classified as a 0.

Logistic regression does just this, but with some fancy tricks. 

The logistic classifer is **parametric, discriminitive** function.  The best fit parameters ($\beta$)s creates a decision boundary which allows us to discriminate between the classes.

![decision_boundary](https://www.researchgate.net/publication/325813999/figure/fig5/AS:638669773893635@1529282148432/Classification-decision-boundary-using-logistic-regression-The-blue-area-corresponds-to.png)


```python
#img source https://www.researchgate.net/publication/325813999/figure/fig5/AS:638669773893635@1529282148432/Classification-decision-boundary-using-logistic-regression-The-blue-area-corresponds-to.png
```

# Breast Cancer Dataset

Logistic regression "is widely used in biostatistical applications where binary responses (two classes) occur quite frequently. For example, patients survive
or die, have heart disease or not, or a condition is present or absent."   [Elements of Statistical Learning, Ch. 4, p. 119](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf)


[data_source](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

![breast_cancer_cells](https://storage.googleapis.com/kaggle-datasets-images/180/384/3da2510581f9d3b902307ff8d06fe327/dataset-card.jpg)


```python
# Breast Cancer identification dataset
import pandas as pd

df = pd.read_csv('data/breast_cancer.csv', index_col= 0)

df.drop(columns= ['Unnamed: 32'], inplace = True)
df.head()
```

We have 30 predictor columns, and 1 target column.  Our target column, however, is not in a form suitable for classification.  

Assumption: **Binary logistic** regression requires the dependent variable to be binary.

We will define malignant as our positive "1" class, and Benign as our "0" class.


```python
df['Target'] = df.diagnosis.map(lambda x: 1 if x == 'M' else 0)
```


```python
df.Target.value_counts()
```

We have a fairly **balanced dataset**.  The logistic regression model will likely be able to pick up on the signal of the minority class.  If it were heavily imbalanced, our model might predict only the majority class, and we would have to use resampling techniques to pick up on the signal.

To begin exploring how logistic regression works, we will fit a linear regression model, and using techniques we know, make predictions.

# Individual Exercise (turn off camera, take 2 minutes)

- Use 'area_mean' as the independent variable and 'Target' as dependent variable and apply a linear regression model to this dataset.



```python
# Your code here
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

y_hat = None # Continuous prediction here
```


```python
#__SOLUTION__

lr = LinearRegression()
lr.fit(df[['area_mean']], df[['Target']])

y_hat = lr.predict(df[['area_mean']])
```


```python
# Plot the resulting regression line.
fig, ax = plt.subplots(figsize=(7,7))
sns.scatterplot(df.area_mean, df.Target, hue=df.Target)
sns.scatterplot(df.area_mean, list(y_hat),color='black')
```



- According to the linear regression model, what would be your prediction if area_mean = 350?

- What about if 'area_mean' is 5?

- What about 2000?


```python
import numpy as np
```


```python
#__SOLUTION__
lr.predict(np.array([[5],[350], [2000]]))
```

Those predictions are not within the bounds of our target's sample space. In fact, linear regression could produce predictions from **-$\infty$ to $\infty$**  


In order to fix that, we can set a threshold which determines a 0 or 1 value.
Let's set the threshhold at .5.


```python
lr_prd = (y_hat > .5).astype(int)
df['lr_yhat'] = lr_prd
```


```python
# Now, let's plot again


fig, ax = plt.subplots(figsize=(7,7))
sns.scatterplot(df.area_mean, df.Target, hue=df.Target)
sns.scatterplot(df.area_mean, list(y_hat), hue=df.lr_yhat)
ax.hlines(.5, xmin=0, xmax=2500)
ax.get_legend().remove();
```

Let's look at how many predictions linear regression got wrong.


```python
print(f"Linear Regression missed {(df.Target != df.lr_yhat).sum()} predictions")
```

The **confusion matrix** will be an important visualization in classification. It will allow us to see the distribution of prediction results. 


```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_lin = confusion_matrix(df.Target, df.lr_yhat)
cm_lin
```


```python
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cm_lin, annot=True, ax=ax, square=True, cmap="coolwarm", fmt='g', xticklabels=['B', 'M'],yticklabels=['B', 'M'] )
plt.rcParams.update({'font.size': 30})
ax.set_xlabel('Predicted', fontdict={'size': 15})
ax.set_ylabel('True', fontdict={'size': 15})
ax.set_title('Linear Regression\n Tumor Diagnostic Predictions', fontdict={'size': 15} )
plt.show()
```

Volunteer from below to interpret the above CM as type I/II error


```python
three_random_students(student_first_names)
```

# Now Let's Try Logistic Regression


```python
# Note: Same module as Linear Regression
from sklearn.linear_model import LogisticRegression

# Same process: Instantiate instance of the algorithm
log_reg = LogisticRegression(penalty='none', solver='lbfgs')

# Fit to data
log_reg.fit(df[['area_mean']], df[['Target']])

# Predict 
yhat_log = log_reg.predict(df[['area_mean']])

# predict_proba: This is new!
yhat_log_proba = log_reg.predict_proba(df[['area_mean']])

df['yhat_log_proba'] = yhat_log_proba[:,1]
df['yhat_log'] = yhat_log


```


```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,7))
sns.scatterplot(df.area_mean, df.Target, hue=df.Target)
sns.scatterplot(df.area_mean, df.yhat_log_proba, hue=df.yhat_log)
ax.hlines(.5, xmin=0, xmax=2500)
ax.get_legend().remove()
ax.set_title('Logistic Regression Predictions')
plt.style.use(['default']);
```

Look at that nice S-shape that fits our data so much more naturally.

__Your Turn__

Use the trained logistic regression and make predictions for 

- area_mean = 350
- area_mean = 5
- area_mean = 2000


```python
# Your code here
```


```python
#__SOLUTION__
log_reg.predict([[5], [350],[2000]])
```

Logistic regression's predict function automatically converts the predicted probabilities to categorical predctions.
To return the probabilities, use the predict_proba method.


```python
log_reg.predict_log_proba([[5], [350],[2000]])
```


```python
# What does the above output mean?
one_random_student(student_first_names)
```

How did our logistic regression model compare with our linear regression?


```python
print(f"Logistic Regression missed {(df.Target != df.yhat_log).sum()} predictions")
```


```python
# Let's look at how many differences there were between our two predictions
(df.lr_yhat != df.yhat_log).sum()
```


```python
lr_logr_diffs = df[(df.lr_yhat != df.yhat_log) & (df.lr_yhat == df.Target)]
print(lr_logr_diffs.shape[0])
```


```python
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,10))

# Linear Regression Confusion Matrix
sns.heatmap(cm_lin, annot=True, ax=ax1,cbar=False, square=True, 
            cmap="coolwarm", fmt='g', xticklabels=['B', 'M'],yticklabels=['B', 'M'] )
plt.rcParams.update({'font.size': 30})
ax1.set_xlabel('Predicted', fontdict={'size': 15})
ax1.set_ylabel('True', fontdict={'size': 15})
ax1.set_title('Linear Regression', fontdict={'size': 15} )


# Logistic Regression Confusion Matrix
cm_log = confusion_matrix(df.Target, df.yhat_log)
sns.heatmap(cm_log, annot=True, ax=ax2,  square=True, 
            cbar=False, cmap="coolwarm", fmt='g',
            xticklabels=['B', 'M'],yticklabels=['', ''] )

ax2.set_xlabel('Predicted', fontdict={'size': 15})

ax2.set_title('Logistic Regression', fontdict={'size': 15} )
plt.show()
plt.tight_layout();
```

# Threshold

By default, the predict() method applies a threshold of .05 to our prediction probabilities.


```python
# We see that converting all values of the column associated with the probability of a class 1 (Malignant)
# is equal to the output of the predict() method
(log_reg.predict_proba(df[['area_mean']])[:,1] > .5).astype(int) == log_reg.predict(df[['area_mean']])
```

However, we may want to be more conservative in our estimate. With medical diagnostics in particular, certain errors are more important to catch.

Which errors have particularly negative consequences in the scenario above?


```python
one_random_student(student_first_names)
```


```python
#__SOLUTION__
'''type II errors, false negatives, are particularly dangerous.  
A false negative means a sample was predicted to be benign, when in fact it was malignant'''
```

To err on the side of caution, we can force our model to predict more conservitavely.  

By lowering the threshold from .5, our model will predict more positive values, thereby decreasing our false negatives.  Consequently, our false positive rate will go up.


```python
# Code here
yhat_lower_thresh = None
```


```python
#__SOLUTION__
yhat_lower_thresh = (log_reg.predict_proba(df[['area_mean']])[:,1] > .4).astype(int)

```


```python
cm_lower_thresh = confusion_matrix(df.Target, yhat_lower_thresh)
```


```python
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,10))

# Linear Regression Confusion Matrix
sns.heatmap(cm_log, annot=True, ax=ax1,cbar=False, square=True, 
            cmap="coolwarm", fmt='g', xticklabels=['B', 'M'],yticklabels=['B', 'M'] )
plt.rcParams.update({'font.size': 30})
ax1.set_xlabel('Predicted', fontdict={'size': 15})
ax1.set_ylabel('True', fontdict={'size': 15})
ax1.set_title('Threshold .5', fontdict={'size': 15} )


# Logistic Regression Confusion Matrix
cm_log = confusion_matrix(df.Target, df.yhat_log)
sns.heatmap(cm_lower_thresh, annot=True, ax=ax2,  square=True, 
            cbar=False, cmap="coolwarm", fmt='g',
            xticklabels=['B', 'M'],yticklabels=['', ''] )

ax2.set_xlabel('Predicted', fontdict={'size': 15})

ax2.set_title('Threshold .4', fontdict={'size': 15} )
plt.show()
plt.tight_layout();
```

# 3. Logistic Regression Under the Hood

As we have seen above, linear regression outputs a value that can range anywhere from $-\infty$ to $\infty$.  

Logistic regression attempts to convert those linear outputs to a range of probabilities, i.e. a value between 0 and 1.

To make this conversion, we use the sigmoid function.

![sigmoid](https://media.giphy.com/media/GtKtQ9Gb064uY/giphy.gif)


<img src='https://cdn-images-1.medium.com/max/1600/1*RqXFpiNGwdiKBWyLJc_E7g.png' />

As ‘Z’ goes to infinity, Y(predicted) will inch closer to 1, and as ‘Z’ goes to negative infinity, Y(predicted) will inch closer to 0.

Using the sigmoid function above, if X = 1, the estimated probability would around .7. This tells that there is 70% chance that this observation would fall in the positive class.



```python
# Let's code out the sigmoid:

```


```python
#__SOLUTION__
def sigmoid(x):
    """
    params: input from linear equation
    returns: probability between 0 and 1
    """
    
    return 1/(1+np.e**(-x))


```


```python
# Outputs of sigmoid function
print(sigmoid(1))
print(sigmoid(10))
print(sigmoid(-10))
```

If we substitute the product of our linear equation for x in the function above, and rephrase the objective of logistic regression as computing the probability of a class (assume positive class 1) given a set of $\beta$ parameters, our formula becomes:

$$\Large P(Class = 1|X = x) =  \frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}} $$



Some arithmetic (see appendix) allows us to see what the linear equation represents in our logistic regression:
<br><br>
    $\ln{\dfrac{ P(G = 1|X = x) }{P(G = 0|X =x)}} = \beta_0 + \beta_1*X_1 + \beta_2*X_2...\beta_n*X_n$
    

Our linear function calculates the log of the probability we predict 1, divided by the probability of predicting 0.  In other words, the linear equation is calculating the **log of the odds** that we predict a class of 1.
    

## Probability and odds

$$probability = \frac {one\ outcome} {all\ outcomes}$$

$$odds = \frac {one\ outcome} {all\ other\ outcomes}$$

Examples:

- Dice roll of 1: probability = 1/6, odds = 1/5
- Even dice roll: probability = 3/6, odds = 3/3 = 1
- Dice roll less than 5: probability = 4/6, odds = 4/2 = 2

$$odds = \frac {probability} {1 - probability}$$

$$probability = \frac {odds} {1 + odds}$$

Odds can assume any positive value, from **0 to $\infty$**

The log of the odds can take an value from **-$\infty$ to $\infty$**, which allows us to map it to the output of the linear equation.


```python
# Dice roll less than 5: probability = 4/6, odds = 4/2 = 2
np.log(2)
```


```python
# Dice roll >=  5: probability = 2/6, odds = 2/4 = .5
np.log(.5)
```

Let's check our understanding with the probability predictions of our logistic regression predict_proba output


```python
proba_sample_1 = log_reg.predict_proba(df[['area_mean']])[0,1]
proba_sample_1
```


```python
odds_sample_1 = proba_sample_1/(1-proba_sample_1)
odds_sample_1
```


```python
# Log odds Sample_1
np.log(odds_sample_1)

```


```python
# What about sample 43

proba_sample_4 = log_reg.predict_proba(df[['area_mean']])[3,1]
odds_sample_4 = proba_sample_4/(1-proba_sample_4)
np.log(odds_sample_4)
```

# Help me out
Reproduce the above log_odds prediction using the coef_ and intercept_ attributes of our fitted log_reg model.


```python
three_random_students(student_first_names)
```


```python
# Code here
```


```python
#__SOLUTION__
log_reg.coef_[0] * df.area_mean.iloc[3] + log_reg.intercept_
```

Now, apply the sigmoid function above to convert the log-odds back to a probability.


```python
# Your code here
```


```python
#__SOLUTION__
sigmoid(np.log(odds_sample_4))
```

## 4. Interpreting Logistic Regression Coefficients

Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

<img src='img/betas.png' width=700/>


```python
log_reg.coef_
```

**Interpretation:** A 1 unit increase in 'area size' is associated with a .0118 unit increase in the log-odds of a malignant result.

**Bottom line:** Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# 5. Maximum Likelihood Estimation

Instead of OLS, we will use Maximimum Likelihood Estimation to calculate our $\beta$s. 

You could use the cost function we used for linear regression, mean-squared error.  However, with a binary outcome, the MSE is **non-convex**.  Gradient descent risks missing the global minimum in favor of a local minimum.

Instead of optimizing the coefficients based on mean squared error, logistic regression looks to maximize the likelihood of seeing the probabilities given the true class using the following likelihood function.

$$ \Large negative\ loglikelihood = 
-\frac{1}{n} \sum\limits_{i=1}^N y_i\log{p_i} + (1-y_i)\log(1-p_i) $$

The p variable represents the probabilities of class 1 calculated for each sample, and y represents the true value of the sample.   

Take a moment to think through how the above negative loglikelihood function rewards coefficients which yield high probabilities of a class matched to the true value.

![log_cost](img/cost_curve_log.png)

When looking at the above plots of the cost function, we see that as our hypothesis gets closer predicting the correct value, the slope gets much smaller. The effect is "the cost function penalizes confident and wrong predictions more than it rewards confident and right predictions. The corollary is increasing prediction accuracy (closer to 0 or 1) has diminishing returns on reducing cost due to the logistic nature of our cost function." [source](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#id4)



Unlike linear regression and its normal equation, there is no closed form solution to minimize the derivative. That is why you may see that non-convergence error.  

See [here](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) for more detail on MLE


We have covered how this works for **binary classification problems** (two response classes). But what about **multi-class classification problems** (more than two response classes)?

- Most common solution for classification models is **"one-vs-all"** (also known as **"one-vs-rest"**): decompose the problem into multiple binary classification problems
- **Multinomial logistic regression** can solve this as a single problem

# 6. Hyperparameter Tuning the C Variable

We have discussed 'L1' (lasso)  and 'L2' (ridge) regularization.  If you looked at the docstring of Sklearn's Logistic Regression function, you may have noticed that we can specify different types of regularization when fitting the model via the `penalty` parameter.

We can also specificy the strength of the regularization via the `C` parameter. `C` is the inverse regularization strength.  So, a low `C` means high regularization strength.

Let's run through our train test split process, and tune our C parameter.   


```python
df = pd.read_csv('data/breast_cancer.csv', index_col= 0)

df.drop(columns= ['Unnamed: 32'], inplace = True)
df['Target'] = df.diagnosis.map(lambda x: 1 if x == 'M' else 0)
df.drop('diagnosis', axis=1, inplace=True)

y = df.Target
X = df.drop("Target", axis=1)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=42)
```

Scaling is important when implementing regularization, since it penalizes the magnitude of the coefficients.

To correctly implement scaling, we scale only on the training data.

# Pair Annotation

With a partner, put annotations after the empty # comments in the KFold implementation below.


```python
from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler

c_recall = {}

#
kf = KFold(n_splits=4)

#
for c in np.linspace(1,1000,100):
    
    mean_recall = []
    
    #
    for train_ind, val_ind in kf.split(X_train, y_train):
        
        #
        X_tt, y_tt = X_train.iloc[train_ind], y_train.iloc[train_ind]
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
        
        ss = StandardScaler()
        
        #
        X_tt = ss.fit_transform(X_tt)
        X_val = ss.transform(X_val)
        
        #
        log_reg = LogisticRegression(C=c, solver='lbfgs', max_iter=400)
        
        log_reg.fit(X_tt, y_tt)
        
        mean_recall.append(log_reg.score(X_val, y_val))
    
    # 
    c_recall[c] = np.mean(mean_recall)


```


```python
#__SOLUTION__

from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler

c_recall = {}

# Instantiate a Kfolds instance which we will use to split training into 4 parts
kf = KFold(n_splits=4)

# Cycle through a series of 100 numbers equally spaced beteen 1 and 1000
# To be used as the C parameter in the logistic regression models fit below
for c in np.linspace(1,1000,100):
    
    mean_recall = []
    
    # Loop through the KFold splits (total of 4 loops), each time reserving a different
    # quarter of the training data as the validation data
    for train_ind, val_ind in kf.split(X_train, y_train):
        
        X_tt, y_tt = X_train.iloc[train_ind], y_train.iloc[train_ind]
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
        
        ss = StandardScaler()
        
        
        # Fit the scaler to the training data for each loop.
        # Then transform the validation set.
        # This will prevent data leakage.
        
        X_tt = ss.fit_transform(X_tt)
        X_val = ss.transform(X_val)
        
        # Fit the logistic regression with the c candidate value
        # linked to the current step in the loop.
        log_reg = LogisticRegression(C=c, solver='lbfgs', max_iter=400)
        
        log_reg.fit(X_tt, y_tt)
        
        mean_recall.append(log_reg.score(X_val, y_val))
    
    # Calculate the mean recall score of all validation sets (4 in total)
    # related to the c value of the previous loop
    # and add to the dictionary outside of the loop to store the progress.
    c_recall[c] = np.mean(mean_recall)

```


```python
sorted(c_recall.items(), key=lambda kv: kv[1], reverse=True)[:10]
```


```python
from sklearn.model_selection import KFold 
from sklearn.preprocessing import StandardScaler

c_recall = {}

kf = KFold(n_splits=4)

# Iterate with another set of C parameters with a range zeroed in on greater strength
for c in np.linspace(1,100,100):
    mean_recall = []
    for train_ind, val_ind in kf.split(X_train, y_train):

        X_tt, y_tt = X_train.iloc[train_ind], y_train.iloc[train_ind]
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]
        
        ss = StandardScaler()
        X_tt = ss.fit_transform(X_tt)
        X_val = ss.transform(X_val)
        
        log_reg = LogisticRegression(C=c, solver='lbfgs', max_iter=400)
        log_reg.fit(X_tt, y_tt)
        mean_recall.append(log_reg.score(X_val, y_val))
        
    c_recall[c] = np.mean(mean_recall)

```


```python
sorted(c_recall.items(), key=lambda kv: kv[1], reverse=True)[:10]
```


Now that we have selected a C hyperparameter that performs well, fit to the entire training set.


```python

ss = StandardScaler()

X_train_sc = ss.fit_transform(X_train)
log_reg = LogisticRegression(C=2, solver='lbfgs', max_iter=5000)
log_reg.fit(X_train_sc, y_train)

y_hat = log_reg.predict(X_train_sc)
```


```python
fig, ax = plt.subplots()
cm = confusion_matrix(y_train, y_hat)
sns.heatmap(cm, ax=ax, annot=True,  square=True, 
            cbar=False, cmap="coolwarm", fmt='g',
            xticklabels=['B', 'M'],yticklabels=['B', 'M'] )

ax.set_xlabel('Predicted', fontdict={'size': 15})
ax.set_ylabel('True', fontdict={'size': 15})
ax.set_title('Logistic Regression: C2', fontdict={'size': 15})
```

We can adjust the threshold to catch more false positives.


```python

y_hat = (log_reg.predict_proba(X_train_sc)[:,1] >.4).astype(int)

```


```python
fig, ax = plt.subplots()
cm = confusion_matrix(y_train, y_hat)
sns.heatmap(cm, ax=ax, annot=True,  square=True, 
            cbar=False, cmap="coolwarm", fmt='g',
            xticklabels=['B', 'M'],yticklabels=['B', 'M'] )

ax.set_xlabel('Predicted', fontdict={'size': 15})
ax.set_ylabel('True', fontdict={'size': 15})
ax.set_title('Logistic Regression: Threshold .4', fontdict={'size': 15})
```

# Now apply to the test set


```python
X_test_sc = ss.transform(X_test)
```


```python
y_hat = (log_reg.predict_proba(X_test_sc)[:,1] >.4).astype(int)
```


```python
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_hat)
sns.heatmap(cm, ax=ax, annot=True,  square=True, 
            cbar=False, cmap="coolwarm", fmt='g',
            xticklabels=['B', 'M'],yticklabels=['B', 'M'] )

ax.set_xlabel('Predicted', fontdict={'size': 15})
ax.set_ylabel('True', fontdict={'size': 15});
ax.set_title('Holdout Test Results', fontdict={'size': 15})
```

With our logistic regression coefficients, we can inspect which features our model thinks are most important for the different classifications.


```python
beta_values = {name:coef for name, coef in zip(list(X_test.columns), list(log_reg.coef_[0]))}
col = [item[0] for item in sorted(beta_values.items(), key= lambda kv: kv[1], reverse=True)]
beta = [item[1] for item in sorted(beta_values.items(), key= lambda kv: kv[1], reverse=True)]

```


```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,7))

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

plt.rc('font', **font)

bars = ax.barh(col,beta, color='g')
  
for bar in range(1,8):
    bars[-bar].set_color('red')

ax.set_title('Relative Importance of Features');
```

# 8. Assumptions of Logistic Regression

Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, and homoscedasticity.

First, logistic regression does not require a linear relationship between the dependent and independent variables.  Second, the error terms (residuals) do not need to be normally distributed.  Third, homoscedasticity is not required.  

**The following assumptions still apply:**

1.  Binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.

2. Logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data.

3. Logistic regression requires there to be little or no multicollinearity among the independent variables.  This means that the independent variables should not be too highly correlated with each other.

4. Logistic regression assumes linearity of independent variables and log odds.  although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.

5. Logistic regression typically requires a large sample size.  A general guideline is that you need at minimum of 10 cases with the least frequent outcome for each independent variable in your model. For example, if you have 5 independent variables and the expected probability of your least frequent outcome is .10, then you would need a minimum sample size of 500 (10*5 / .10).

# Appendix: Converting sigmoid to log-odds.

If we substitute the product of our linear equation for x in the function above, and rephrase the objective of logistic regression as computing the probability of a class (assume positive class 1) given a set of $\beta$ parameters, our formula becomes:

$$\Large P(Class = 1|X = x) =  \frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}} $$



Now, with some arithmetic:

You can show that, by multiplying both numerator and denominator by $e^{(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}$


$$ \Large P(G = 1|X = x) = \displaystyle \frac{e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}{1+e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}$$

As a result, you can compute:

$$ \Large P(G = 0|X =x) = 1- \displaystyle \frac{e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}{1+e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}= \displaystyle \frac{1}{1+e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}$$


Further:

$$ \Large \dfrac{ P(G = 1|X = x) }{P(G = 0|X =x)} = e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n} $$

This expression can be interpreted as the *odds in favor of class 1*.  

## Probability and odds

$$probability = \frac {one\ outcome} {all\ outcomes}$$

$$odds = \frac {one\ outcome} {all\ other\ outcomes}$$

Examples:

- Dice roll of 1: probability = 1/6, odds = 1/5
- Even dice roll: probability = 3/6, odds = 3/3 = 1
- Dice roll less than 5: probability = 4/6, odds = 4/2 = 2

$$odds = \frac {probability} {1 - probability}$$

$$probability = \frac {odds} {1 + odds}$$

This expression can be interpreted as the *odds in favor of class 1*.  

$$ \Large \dfrac{ P(G = 1|X = x) }{P(G = 0|X =x)} = e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n} $$



Finally, taking the log of both sides leads to:
<br><br>
    $\ln{\dfrac{ P(G = 1|X = x) }{P(G = 0|X =x)}} = \beta_0 + \beta_1*X_1 + \beta_2*X_2...\beta_n*X_n$
    
Here me can see why we call it logisitic regression.

Our linear function calculates the log of the probability we predict 1, divided by the probability of predicting 0.  In other words, the linear equation is calculating the **log of the odds** that we predict a class of 1.
    
