
# Logistic Regression

## Learning goals

1. Compare predicting a continuous outcome to predicting a class
2. Compare linear to logistic regression as classification models
3. Describe how logistic regression works under the hood
4. Learn how to interpret a trained logistic model's coefficients
5. Familiarize ourselves with Maximum Likelihood Estimation
6. Explore the C (inverse regularization) paramater and hyperparameter tune
7. Learn how to adjust the threshold of a logistic model
8. Describe the assumptions of linear regression

# Why logistic 1st of our classifiers?

Approximately 70% of problems in Data Science are classification problems. There are lots of classification algorithms that are available, but the logistics regression is common and is a useful regression method for solving the binary classification problem.


Logistic regression takes a concept we are familiar with, a linear equation, and translates it into a form fit for predicting a class.  

It generally can't compete with the best supervised learning algorithms, but it is **simple, fast, and interpretable**.  

As we will see in mod 4, it will also serve as a segue into our lessons on **neural nets**.

# 1. Compare predicting a continuous outcome to predicting a class

Thus far, we have worked to predict continuous target variables using linear regression. 

  - Continous target variables:
        - Sales price of a home
        - MPG of a car
        - A Country's Life Expectancy Rate
        
We will now transition into another category of prediction: classification. Instead of continous target variables, we will be predicting whether records from are data are labeled as a particular class.  Whereas the output for the linear regression model can be any number, the output of our classification algorithms can only be a value designated by a set of discrete outcomes.

  - Categorical target variables:
        - Whether an employee will stay at a company or leave (churn)
        - Whether a tumor is cancerous or benign
        - Whether a flower is a rose, a dandelion, a tulip, or a daffodil
        - Whether a voter is Republican, Democrat, or Independent



Let's navigate to the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table), and browse there classification datasets.  Which one's catch your eye?

![discuss](https://media.giphy.com/media/l0MYIAUWRmVVzfHag/giphy.gif)

### We are still dealing with **labeled data**.

![labels](https://media.giphy.com/media/26Ff5evMweBsENWqk/giphy.gif)


This is still supervised learning. 

But now, instead of the label being a continuous value, such as house price, the label is the category.  This can be either binary or multiclass.  But we still need the **labels** to train our models.



# 2. Compare linear to logistic regression as classification models


The goal of logistic regression, and any classification problem, is to build a model which accurately separates the classes based on independent variables.  

We are already familiar with how linear regression finds a best-fit "line".  It uses the **MSE cost function** to minimize the difference between true and predicted values.  

A natural thought would be to use that "line" to descriminate between classes: Everything with an output greater than a certain point is classified as a 1, everything below is classified as a 0.

In this way, logistic classifer is **parametric, discriminitive** function.  The best fit parameters ($\beta$)s creates a decision boundary which allows us to discriminate between the classes.

![decision_boundary](https://www.researchgate.net/publication/325813999/figure/fig5/AS:638669773893635@1529282148432/Classification-decision-boundary-using-logistic-regression-The-blue-area-corresponds-to.png)

# Breast Cancer Dataset

Logistic regression "is widely used in biostatistical applications where binary responses (two classes) occur quite frequently. For example, patients survive
or die, have heart disease or not, or a condition is present or absent."   [Elements of Statistical Learning, Ch. 4, p. 119](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf)


[data_source](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

We have a fairly **balanced dataset**.  The logistic regression model will likely be able to pick up on the signal of the minority class.  If it were heavily imbalanced, our model might predict only the majority class, and we would have to use resampling techniques to pick up on the signal.

To begin exploring how logistic regression works, we will fit a linear regression model, and using techniques we know, make predictions.

# Individual Exercise (turn off camera, take 2 minutes)

- Use 'area_mean' as the independent variable and 'Target' as dependent variable and apply a linear regression model to this dataset.



```python

lr = LinearRegression()
lr.fit(df[['area_mean']], df[['Target']])

y_hat = lr.predict(df[['area_mean']])
```



- According to the linear regression model, that would be your prediction if area_mean = 350?

- What about if 'area_mean' is 5?

- What about 2000?


```python
lr.predict(np.array([[5],[350], [2000]]))
```




    array([[-0.26100989],
           [ 0.07533936],
           [ 1.68396621]])



Those predictions are not within the bounds of our target's sample space. In fact, linear regression could produce predictions from **-$\infty$ to $\infty$**  


In order to fix that, we can set a threshold which determines a 0 or 1 value.
Let's set the threshhold at .5.

Let's look at how many predictions linear regression got wrong.

# Now Let's Try Logistic Regression

__Your Turn__

Use the trained logistic regression and make predictions for 

- area_mean = 350
- area_mean = 5
- area_mean = 2000


```python
log_reg.predict([[5], [350],[2000]])
```




    array([0, 0, 1])



Logistic regression's predict function automatically converts the predicted probabilities to categorical predctions.
To return the probabilities, use the predict_proba method.

How did our logistic regression model compare with our linear regression?

# Threshold

By default, the predict() method applies a threshold of .05 to our prediction probabilities.

However, we may want to be more conservative in our estimate. With medical diagnostics in particular, certain errors are more important to catch.

Which errors have particularly negative consequences in the scenario above?


```python
'''type II errors, false negatives, are particularly dangerous.  
A false negative means a sample was predicted to be benign, when in fact it was malignant'''
```




    'type II errors, false negatives, are particularly dangerous.  \nA false negative means a sample was predicted to be benign, when in fact it was malignant'



To err on the side of caution, we can force our model to predict more conservitavely.  

By lowering the threshold from .5, our model will predict more positive values, thereby decreasing our false negatives.  Consequently, our false positive rate will go up.


```python
yhat_lower_thresh = (log_reg.predict_proba(df[['area_mean']])[:,1] > .4).astype(int)

```

# 3. Logistic Regression Under the Hood

As we have seen above, linear regression outputs a value that can range anywhere from $-\infty$ to $\infty$.  

Logistic regression attempts to convert those linear outputs to a range of probabilities, i.e. a value between 0 and 1.

To make this conversion, we use the sigmoid function.

![sigmoid](https://media.giphy.com/media/GtKtQ9Gb064uY/giphy.gif)


<img src='https://cdn-images-1.medium.com/max/1600/1*RqXFpiNGwdiKBWyLJc_E7g.png' />

As ‘Z’ goes to infinity, Y(predicted) will inch closer to 1, and as ‘Z’ goes to negative infinity, Y(predicted) will inch closer to 0.

Using the sigmoid function above, if X = 1, the estimated probability would be 0.8. This tells that there is 80% chance that this observation would fall in the positive class.



```python
def sigmoid(x):
    """
    params: input from linear equation
    returns: probability between 0 and 1
    """
    
    return 1/(1+np.e**(-x))


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

Let's check our understanding with the probability predictions of our logistic regression predict_proba output

# Help me out
Reproduce the above log_odds prediction using the coef_ and intercept_ attributes of our fitted log_reg model.


```python
log_reg.coef_[0] * df.area_mean.iloc[3] + log_reg.intercept_
```




    array([-3.43032516])



## 4. Interpreting Logistic Regression Coefficients

Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

<img src='img/logistic_betas.png' width=700/>

**Interpretation:** A 1 unit increase in 'area size' is associated with a .0118 unit increase in the log-odds of a malignant result.

**Bottom line:** Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# 5. Maximum Likelihood Estimation

Instead of OLS, we will use Maximimum Likelihood Estimation to calculate our $\beta$s. 

The **cost function**, i.e. the distance from the truth of all data points, is is calculated using the probabilities of the Bernouli distribution.



Instead of optimizing the coefficients based on mean squared error, logistic regression looks to maximize the likelihood of seeing the probabilities given the class.

Likelihood functions model the goodness of fit of our hypothesis.  

In otherwords, they describe how likely a particular set of $\beta$\s are given the true values of our sample.

Because we are dealing with a binary outcome, our likelihood equation comes from the Bernouli distribution:

$$ \Large Likelihood=\prod\limits_{i=0}^N p_i^{y_i}(1-p_i)^{1-y_i}$$

The p variable represents the probabilities of class 1 calculated for each sample, and y represents the true value of the sample.  Take a moment to think through how the above Likelihood function rewards coefficients which yield high probabilities of a class matched to the true value.

Because of issues of [computational complexity](https://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution), we take the log of this cost function. And since we generally want to minimize the derivative to find an optimal solution, we take the negative of log-likelihood as our cost function.


$$ \Large negative\ loglikelihood = \sum\limits_{i=1}^N - y_i\log{p_i} - (1-y_i)\log(1-p_i) $$

Our algorithms calculate the derivitave of the cost function to find the $\beta$ values which maximize the likelihood (minimize negative loglikelihood) that they represent the true process which generated the data, given the prior distribution of our target.

Unlike linear regression and its normal equation, there is no closed form solution to minimize the derivative. That is why you may see that non-convergence error.  

See [here](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) for more detail on MLE


We have covered how this works for **binary classification problems** (two response classes). But what about **multi-class classification problems** (more than two response classes)?

- Most common solution for classification models is **"one-vs-all"** (also known as **"one-vs-rest"**): decompose the problem into multiple binary classification problems
- **Multinomial logistic regression** can solve this as a single problem

# 6. Hyperparameter Tuning the C Variable

We have discussed 'L1' (lasso)  and 'L2' (ridge) regularization.  If you looked at the docstring of Sklearn's Logistic Regression function, you may have noticed that we can specify different types of regularization when fitting the model via the `penalty` parameter.

We can also specificy the strength of the regularization via the `C` parameter. `C` is the inverse regularization strength.  So, a low `C` means high regularization strength.

Let's run through our train test split process, and tune our C parameter.   

Scaling is important when implementing regularization, since it penalizes the magnitude of the coefficients.

To correctly implement scaling, we scale only on the training data.

For the purpose of comparison, Let's now fit without any hyperparamter tuning.

That did really well.  But let's check out recall, since false negatives are very important to protect against in medical diagnostic tests.


Even though hyperparameter tuning will have dimishing returns in this scenario, let's see if we can do any better.

Let's cycle through C values, and pick the best value based on accuracy.

Our model was very confident about the negative prediction for our one prediction error.


# 8. Assumptions of Logistic Regression

Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, and homoscedasticity.

First, logistic regression does not require a linear relationship between the dependent and independent variables.  Second, the error terms (residuals) do not need to be normally distributed.  Third, homoscedasticity is not required.  

**The following assumptions still apply:**

1.  Binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.

2. Logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data.

3. Logistic regression requires there to be little or no multicollinearity among the independent variables.  This means that the independent variables should not be too highly correlated with each other.

4. Logistic regression assumes linearity of independent variables and log odds.  although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.

5. Logistic regression typically requires a large sample size.  A general guideline is that you need at minimum of 10 cases with the least frequent outcome for each independent variable in your model. For example, if you have 5 independent variables and the expected probability of your least frequent outcome is .10, then you would need a minimum sample size of 500 (10*5 / .10).

# Appendix: Kfolds

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
    
