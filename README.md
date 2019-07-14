### Project Overview

 The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales. We have train (8523) and test (5681) data set, train data set has both input and output variable(s).



### Learnings from the project

 After completing this project, you will have the better understanding of how to build a regularized regression model. In this project, you will apply the following concepts.

- Train-test split

- Correlation between the features

- Linear Regression

- Polynomial Regressor

- Lasso Regressor

- Ridge Regressor

- R squared Evaluation Metrics


### Additional pointers

 Questions to be solved are as follows, 

- What will happen to R-Square score if you increase the no. of predictors in your model.Use all features except Item_Outlet_Sales','Item_Identifier for prediction and implement a linear regression model?

- What will happen if we remove some features('Item_Outlet_Sales','Item_Identifier', 'Item_Visibility', 'Outlet_Years') from the previous model. Also, how would associate the change in model accuracy with addition or negation of a feature in the model ?

- Heteroskedacity should be avoided at all costs in our model. How will you detect it?

- How can we have a look at the model coefficients or weights of different features of various features?

- Our model tends to become overfit if it learns too much from the data, to avoid this we need to generalize the model so that it is flexible enough to work on new data points. Regularization will help us to achieve this. Use Ridge regression on the data and find out the results.

- Use Lasso model on the data and find out the difference between Lasso and Ridge regression.

- What if we created a bunch of train/test splits, calculated the testing accuracy for each, and averaged the results together?(take a look at the concept of cross validation)


