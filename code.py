import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn import model_selection
import matplotlib.pyplot as plt

## Load the data
df = pd.read_csv(path)

## Split the data and preprocess
print(df.source.unique())
train_df, test_df = df[df['source'] == 'train'], df[df['source'] == 'test']

## Baseline regression model
X = train_df[['Item_Weight', 'Item_MRP', 'Item_Visibility']]
Y = train_df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

# Intiating baseline model

reg = linear_model.LinearRegression(normalize=True)
reg.fit(X_train, y_train)
print('Regression Coefficient is',reg.coef_)

# Predicting on the sample subset
y_pred = reg.predict(X_test)

# Calculating error

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error is',mse) 

# R-Square
r2 = r2_score(y_test, y_pred)
print('R Squared Score is',r2) 

# =======================================================

## Effect on R-square if you increase the number of predictors
# Let's try out to set up a baseline model with just two explanatory variables
X1 = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source'])

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y, test_size=0.2, random_state = 42)

# Fitting baseline model 
reg.fit(X1_train, y1_train) 

# Predicting on the sample subset
y1_pred = reg.predict(X1_test)

# Calculating error

# Mean Squared Error
mse1 = mean_squared_error(y1_test, y1_pred)
print('Mean Square Error is',mse1) 

# R2 Square
r2_1 = r2_score(y1_test, y1_pred)
print('R2 Square Score is',r2_1)

# =======================================================

## Effect of decreasing feature from the previous model

X2 = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source', 'Item_Visibility', 'Outlet_Years'])

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y, test_size=0.2, random_state = 42)

# Fitting baseline model 
reg.fit(X2_train, y2_train)

# Predicting on the sample subset 
y2_pred = reg.predict(X2_test)

# Calculating error

# Mean Squared Error
mse2 = mean_squared_error(y2_test, y2_pred)
print('Mean Squared Error is',mse2)

# R2 Square
r2_2 = r2_score(y2_test, y2_pred)
print('R2 Square Score is',r2_2)

# =======================================================

# Implementing adjusted r square 
def adj_r2_score(model,y,y_pred):
    from sklearn import metrics
    adj = 1 - float(len(y)-1)/(len(y)-len(model.coef_)-1)*(1 - metrics.r2_score(y,y_pred))
    return adj


# Comparing r square and adjusted r square across three models
adj_score_model1 = adj_r2_score(reg, y_test, y_pred)
adj_score_model2 = adj_r2_score(reg, y1_test, y1_pred)
adj_score_model3 = adj_r2_score(reg, y2_test, y2_pred)


print('R square {} and adjusted R square {} of model 1 '.format(r2_score(y_test, y_pred), adj_score_model1))
print('R square {} and adjusted R square {} of model 2 '.format(r2_score(y1_test, y1_pred), adj_score_model2))
print('R square {} and adjusted R square {} of model 3 '.format(r2_score(y2_test, y2_pred), adj_score_model3))

# =======================================================

## Detecting hetroskedacity
plt.scatter(y_pred, (y_pred - y_test))

plt.hlines(y=0, xmin= -1000, xmax=5000)

plt.title('Residual plot')

## Model coefficients

coef = pd.DataFrame(X2.columns, reg.coef_).reset_index()

plt.figure(figsize=(10,10))
coef.plot(kind='bar', title='Modal Coefficients')

# =======================================================

## Ridge regression
clf = linear_model.Ridge(alpha=1.0)
X_ridge = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source', 'Item_Visibility', 'Outlet_Years'])
X_ridge_train, X_ridge_test, y_train, y_test = train_test_split(X_ridge, Y, test_size=0.2, random_state = 42)

# Fitting baseline model 
clf.fit(X_ridge_train, y_train)

# Predicting on the sample subset 
y_pred = clf.predict(X_ridge_test)

# Calculating error

# Mean Squared Error
mse_ridge = mean_squared_error(y_test, y_pred)
print('Ridge Mean Squared Error is',mse_ridge)

# R2 Square
r2_ridge = r2_score(y_test, y_pred)
print('Ridge R Squared Score is',r2_ridge)

# =======================================================

## Lasso regression
clf = linear_model.Lasso(alpha=1e-6)
X_lasso = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source', 'Item_Visibility', 'Outlet_Years'])
X_lasso_train, X_lasso_test, y_train, y_test = train_test_split(X_lasso, Y, test_size=0.2, random_state = 42)

# Fitting baseline model 
clf.fit(X_lasso_train, y_train)

# Predicting on the sample subset 
y_pred = clf.predict(X_lasso_test)

# Calculating error

# Mean Squared Error
mse_lasso = mean_squared_error(y_test, y_pred)
print('Lasso Mean Squared Error is',mse_lasso) 

# R2 Square
r2_lasso = r2_score(y_test, y_pred)
print('Lasso R Squared Score is',r2_lasso)

# Difference between Lasso and Ridge regression
diff = r2_lasso - r2_ridge
print('The difference between Lasso and Ridge regression is'.format(diff))

## Cross validation
kf = model_selection.KFold(n_splits=3) 
res = model_selection.cross_val_score(clf, X, Y, cv=kf) 
print('Result is',res) 

