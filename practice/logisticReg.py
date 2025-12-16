import numpy as np
# import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def cost_function(y, y_pred):
    m = len(y)
    cost = - (1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost


X = np.array([0.2, 0.4, 0.6])
y = np.array([0, 1, 1])

theta = 0.5
z = X * theta
y_pred = sigmoid(z)

print("Predicted possiblity: ", y_pred)
print("cost function value: ", cost_function(y, y_pred))

# plt.scatter(X, y, color="blue")
# plt.xlabel("study time")
# plt.ylabel("result")
# plt.title("linear regression example")
# plt.plot(X, y_pred, color="red")
# plt.show()
