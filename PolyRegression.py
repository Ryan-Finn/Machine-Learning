import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Create 100 random points 0<X<2, y = -2X^2 + 3X + 4 + random noise
m = 100
X = 2 * np.random.rand(m, 1)
Y = -2 * X**2 + 3 * X + 4 + np.random.randn(m, 1)
Y_avg = np.mean(Y)
SST = sum([(y - Y_avg)**2 for y in Y])[0]

x_curve = np.linspace(-1, 3, 200)
c = [np.ones((m, 1))]
for d in range(1, 10):
    c.insert(0, c[0] * X)
    X_b = np.concatenate(c, axis=1)
    A_inv = np.linalg.inv(X_b.T.dot(X_b))
    b = X_b.T.dot(Y)
    w_best = A_inv.dot(b)
    # print('w (using psuedo-inv):', *np.linalg.pinv(X_b).dot(y))
    plt.plot(x_curve, [np.polyval(w_best, i) for i in x_curve], "-", linewidth=2, label=f"Degree {d}")
    R2 = 1 - sum([(y - np.polyval(w_best, x))**2 for x, y in zip(X, Y)])[0] / SST
    print('Degree = ', d, '|', 'R2 =', R2)

plt.plot(X, Y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.xlim([-1, 3])
plt.ylim([1.5 * min(Y), 1.5 * max(Y)])
plt.show()
