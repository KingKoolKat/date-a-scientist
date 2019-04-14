import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
df = pd.read_csv("profiles.csv")

def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = []
  
  for value in lst:
    normalized_num = (value - minimum) / (maximum - minimum)
    normalized.append(normalized_num)
  
  return normalized

education_mapping = {"working on college/university": 3, "working on space camp":0, "graduated from masters program":2, "graduated from college/university": 3, "working on two-year college":2, "graduated from high school":1, "working on masters program":4, "graduated from space camp":0, "college/university":3, "dropped out of space camp":0, "graduated from ph.d program" :5, "graduated from law school":4, "working on ph.d program":5, "two-year college":2, "graduated from two-year college" :2, "working on med school":5, "dropped out of college/university": 3, "space camp":0, "graduated from med school" :5, "dropped out of high school" :1, "working on high school" :1, "masters program" :4, "dropped out of ph.d program" :4, "dropped out of two-year college" :2, "dropped out of med school" :4, "high school" :1, "working on law school" :4, "law school" :4, "dropped out of masters program" :3, "ph.d program":5, "dropped out of law school" :3, "med school":5}
job_mapping = {"other": 1, "student:": 2, "computer / hardware / software": 3, "science / tech / engineering": 4, "artistic / musical / writer": 5, "sales / marketing / biz dev": 6, "medicine / health": 7, "education / academia": 8, "entertainment / media": 9, "executive / management": 10, "banking / financial / real estate": 11, "law / legal services": 12, "hospitality / travel": 13, "construction / craftsmanship": 14, "clerical / administrative": 15, "political / government": 16, "rather not say": 17, "transportation": 18, "unemployed": 19, "retired": 20, "military": 21}
pets_mapping = {"likes dogs and likes cats": 0, "likes dogs": 1, "likes dogs and has cats": 0, "has dogs": 1, "has dogs and likes cats": 0, "likes dogs and dislikes cats": 1, "has dogs and has cats": 0, "has cats": -1, "likes cats": -1, "has dogs and dislikes cats": 1, "dislikes dogs and likes cats": -1, "dislikes dogs and dislikes cats":0, "dislikes cats": 1, "dislikes dogs and has cats" :-1, "dislikes dogs": -1}

df["education_code"] = df['education'].map(education_mapping)
df["job_code"] = df['job'].map(job_mapping)
df["pets_code"] = df['pets'].map(pets_mapping)

df = df.dropna(subset = ["education_code"])
df = df.dropna(subset = ["income"])
df = df.dropna(subset = ["job_code"])
df = df.dropna(subset = ["pets_code"])

x = df[["education_code", "job_code", "income"]]
y = df[["pets_code"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_predict = mlr.predict(x_test)
plt.scatter(y_predict, y_test)
plt.xlabel("predicted")
plt.ylabel("actual (-1 = cat, 1 = dog)")
plt.show()
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
plt.scatter(y_predict, y_test)
plt.xlabel("predicted")
plt.ylabel("actual (-1 = cat, 1 = dog)")
plt.show()
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))
x2 = df[["education_code", "income"]]
x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size = 0.8, test_size = 0.2, random_state=6)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
plt.scatter(y_predict, y_test)
plt.xlabel("predicted")
plt.ylabel("actual (-1 = cat, 1 = dog)")
plt.show()
print(classifier.score(x_train, y_train))
print(classifier.score(x_test, y_test))
x = df[["education_code", "job_code", "income"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)
classifier = SVC(kernel = "rbf", gamma = 0.5, C = 2)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
plt.scatter(y_predict, y_test)
plt.xlabel("predicted")
plt.ylabel("actual (-1 = cat, 1 = dog)")
plt.show()
print(classifier.score(x_train, y_train))
print(classifier.score(x_test, y_test))

