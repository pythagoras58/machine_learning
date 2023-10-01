import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv("data/student-mat.csv", sep=";")

# print(data.head())

data = data[["G1","G2","G3","studytime","failures","absences"]]

print(data.head())

# let predict represent the label
predict = "G3"

X = np.array(data.drop([predict], 1)) # drop the dataframe without the label :: G3

Y = np.array(data[predict])

# split the data into 4 arrays  
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1) 



