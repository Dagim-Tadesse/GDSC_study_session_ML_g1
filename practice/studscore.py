import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([45, 50, 60, 68, 75])

model = LinearRegression()
model.fit(x, y)


pred = model.predict([[6]])
print("predicted value of study: ", pred[0])

plt.scatter(x, y, color="blue")
plt.xlabel("study time")
plt.ylabel("result")
plt.title("linear regression example")
plt.plot(x, model.predict(x), color="red")
plt.show()
