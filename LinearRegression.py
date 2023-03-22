import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Create 100 random points 0<X<2, y=3*x+4 +random noise
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)  #

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

print(X)
print(np.c_[X, np.ones((m, 1))])

X_b = np.c_[X, np.ones((m, 1))]  # add x1 = 1 to each instance
w_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(w_best)

A = np.linalg.inv(X_b.T.dot(X_b))
print(A)

X_new = np.array([[0], [2]])
X_new_b = np.c_[X_new, np.ones((2, 1))]  # add x1 = 1 to each instance
y_predict = X_new_b.dot(w_best)
print(y_predict)

plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()

print(np.linalg.pinv(X_b).dot(y))

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))
