import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
d = 2

# Create 100 random points 0<X<2, y = -2X2 + 3X + 4 + random noise
m = 100
X = 2 * np.random.rand(m, 1)
y = -2 * X * X + 3 * X + 4 + np.random.randn(m, 1)
print('X:', *X)

c = [X, np.ones((m, 1))]
c.insert(0, c[0] * X)
print(*np.c_[c])
X_b = np.c_[X * X, X, np.ones((m, 1))]  # add x1 = 1 to each instance
print('X_b:', *X_b)
A_inv = np.linalg.inv(X_b.T.dot(X_b))
print('A Inverse:', *A_inv)
b = X_b.T.dot(y)
print('b:', *b)
w_best = A_inv.dot(b)
print('Best w:', *w_best)

print('w (using psuedo-inv):', *np.linalg.pinv(X_b).dot(y))

fig1 = plt.figure()
x_curve = np.linspace(0, 2, 100)
plt.plot(x_curve, [np.polyval(w_best, i) for i in x_curve], "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.xlim([0, 2])
plt.show()
