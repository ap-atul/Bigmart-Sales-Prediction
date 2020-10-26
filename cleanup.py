import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# reading both the data sets
trainDf = pd.read_csv("dataset/train.csv")
testDf = pd.read_csv("dataset/test.csv")

# checking lengths
trainingSetIndex = len(trainDf)
print(len(trainDf), len(testDf))

# data type check
print(trainDf.dtypes)

# null check
# OP: Item_Weight, Outlet_Size
print(trainDf.isnull().sum())

# combining both
combination = trainDf.append(testDf)
print(len(combination))

# dropping identifier columns
combination = combination.drop(["Item_Identifier", "Outlet_Identifier"], axis=1)

# replacing the null in the ItemWeight
combination = combination.fillna(combination.median())

# replacing nominal values
combination["Item_Fat_Content"] = combination["Item_Fat_Content"].replace({"LF": 0, "reg": 1})
combination["Item_Fat_Content"] = combination["Item_Fat_Content"].replace({"Low Fat": 0, "Regular": 1})
combination["Item_Fat_Content"] = combination["Item_Fat_Content"].replace({"low fat": 0, "Regular": 1})
print(combination["Item_Fat_Content"].unique())

perishable = ["Breads", "Breakfast", "Dairy", "Fruits and Vegetables", "Meat", "Seafood"]
non_perishable = ["Baking Goods", "Canned", "Frozen Foods", "Hard Drinks", "Health and Hygiene", "Household",
                  "Soft Drinks", "Snack Foods", "Starchy Foods", "Others"]
combination["Item_Type"] = combination["Item_Type"].replace(to_replace=perishable, value="perishable")
combination["Item_Type"] = combination["Item_Type"].replace(to_replace=non_perishable, value="non_perishable")
combination["Item_Type"] = combination["Item_Type"].replace({"perishable": 0, "non_perishable": 1})
print(combination["Item_Type"].unique())

combination["Outlet_Size"] = combination["Outlet_Size"].replace({"Small": 0,
                                                                 "High": 1,
                                                                 "Medium": 2,
                                                                 np.nan: 3})
print(combination["Outlet_Size"].unique())

combination["Outlet_Location_Type"] = combination["Outlet_Location_Type"].replace({"Tier 3": 0,
                                                                                   "Tier 2": 1,
                                                                                   "Tier 1": 2})
print(combination["Outlet_Location_Type"].unique())

combination["Outlet_Type"] = combination["Outlet_Type"].replace({"Grocery Store": 0,
                                                                 "Supermarket Type1": 1,
                                                                 "Supermarket Type2": 2,
                                                                 "Supermarket Type3": 3})
print(combination["Outlet_Type"].unique())

# splitting again the cleaned data sets
trainDfClean = combination[:trainingSetIndex]
testDfClean = combination[trainingSetIndex:]

# saving the sets
trainDfClean.to_csv("./dataset/train_processed.csv", index=False)
testDfClean.to_csv("./dataset/test_processed.csv", index=False)

# plotting the hist
trainDfClean.hist()
testDfClean.hist()
plt.show()
