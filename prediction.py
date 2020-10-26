import pandas as pd

from classifiers.linear_regression import BayesianLinearRegression
from classifiers.util import mean_squared_error, root_mean_squared_error, max_error

# reading the train, test, answers
train = pd.read_csv("./dataset/train_processed.csv")
test = pd.read_csv("./dataset/test_processed.csv")
y_test = pd.read_csv("./dataset/submission.csv")["Item_Outlet_Sales"]

X_train = train.drop(["Item_Outlet_Sales"], axis=1).values
y_train = train["Item_Outlet_Sales"].values
X_test = test.drop(["Item_Outlet_Sales"], axis=1).values

# training
model = BayesianLinearRegression()
model.fit_map(X_train, y_train)
y_predictions = model.predict(X_test)

print("Linear Regression Model :: ")
print(f"\t Mean squared error :: {mean_squared_error(y_test, y_predictions)}")
print(f"\t Root mean squared error :: {root_mean_squared_error(y_test, y_predictions)}")
print(f"\t Max residual error :: {max_error(y_test, y_predictions)}")

outputDataFrame = pd.DataFrame({"Item_Outlet_Sales": y_test, "Item_Outlet_Sales_Pred": y_predictions})
outputDataFrame.to_csv("./dataset/output_predicted.csv", index=False)
