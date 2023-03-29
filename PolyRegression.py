import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# Create 100 random points 0<X<2, y = -2X^2 + 3X + 4 + random noise
m = 100
X = np.linspace(0, 2, m).reshape(m, 1)  # 2 * np.random.rand(m, 1)
Y = -2 * X**2 + 3 * X + 4 + np.random.rand(m, 1)
Y_avg = np.mean(Y)
SST = sum([(y - Y_avg)**2 for y in Y])[0]

x_curve = np.linspace(-1, 3, 200)
c = [np.ones((m, 1))]
for d in range(1, 11):
    c.insert(0, c[0] * X)
    if d != 1 and d % 2 != 0:
        continue

    X_b = np.concatenate(c, axis=1)
    w_best = np.linalg.pinv(X_b).dot(Y)
    # A_inv = np.linalg.inv(X_b.T.dot(X_b))
    # b = X_b.T.dot(Y)
    # w_best = A_inv.dot(b)

    R2 = 1 - sum([(y - np.polyval(w_best, x))**2 for x, y in zip(X, Y)])[0] / SST
    R2 = np.floor(R2 * 1000) / 1000

    w = np.floor(w_best * 100) / 100
    enum = reversed(list(enumerate(w[1:])))
    func = ' + '.join("{}x^{}".format(coeff[0], exp + 1) for exp, coeff in enum) + ' + ' + str(w[0][0])

    plt.plot(x_curve, [np.polyval(w_best, i) for i in x_curve], "-", linewidth=2, label=f"{func} | R2 = {R2}")
    # print(func)

plt.plot(X, Y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.xlim([-1, 3])
diff = max(Y)[0] - min(Y)[0]
plt.ylim([min(Y)[0] - 0.5 * diff, max(Y)[0] + 0.5 * diff])
plt.show()
