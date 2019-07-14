# --------------
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


reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
reg.fit(X_train, y_train)

print(reg.coef_)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(mse) 
print(r2)

## Effect on R-square if you increase the number of predictors
X1 = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source'])
X1_train, X1_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state = 42)
reg.fit(X1_train, y_train)

y_pred = reg.predict(X1_test)
mse1 = mean_squared_error(y_test, y_pred)
r2_1 = r2_score(y_test, y_pred)
print(mse1) 
print(r2_1)


## Effect of decreasing feature from the previous model

X2 = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source', 'Item_Visibility', 'Outlet_Years'])
X2_train, X2_test, y_train, y_test = train_test_split(X2, Y, test_size=0.2, random_state = 42)
reg.fit(X2_train, y_train)

y_pred = reg.predict(X2_test)
mse2 = mean_squared_error(y_test, y_pred)
r2_2 = r2_score(y_test, y_pred)
print(mse2) 
print(r2_2)

## Detecting hetroskedacity
plt.scatter(y_pred, (y_pred - y_test))

## Model coefficients

print(pd.DataFrame(X2.columns, reg.coef_))

## Ridge regression
clf = linear_model.Ridge(alpha=1.0)
X_ridge = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source', 'Item_Visibility', 'Outlet_Years'])
X_ridge_train, X_ridge_test, y_train, y_test = train_test_split(X_ridge, Y, test_size=0.2, random_state = 42)
clf.fit(X_ridge_train, y_train)

y_pred = clf.predict(X_ridge_test)
mse_ridge = mean_squared_error(y_test, y_pred)
r2_ridge = r2_score(y_test, y_pred)
print(mse_ridge) 
print(r2_ridge)

## Lasso regression
clf = linear_model.Lasso(alpha=1e-6)
X_lasso = train_df.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'source', 'Item_Visibility', 'Outlet_Years'])
X_lasso_train, X_lasso_test, y_train, y_test = train_test_split(X_lasso, Y, test_size=0.2, random_state = 42)
clf.fit(X_lasso_train, y_train)

y_pred = clf.predict(X_lasso_test)
mse_lasso = mean_squared_error(y_test, y_pred)
r2_lasso = r2_score(y_test, y_pred)
print(mse_lasso) 
print(r2_lasso)

# Difference between Lasso and Ridge regression
diff = r2_lasso - r2_ridge
print('The difference between Lasso and Ridge regression is'.format(diff))

## Cross validation
kf = model_selection.KFold(n_splits=3) 
res = model_selection.cross_val_score(clf, X, Y, cv=kf) 
print(res) 






