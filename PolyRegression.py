import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
poly_order = 2  # Order of the random data points
m = 100  # Number of random data points

# Inits
X = np.linspace(0, 2, m).reshape(m, 1)  # 2 * np.random.rand(m, 1)

coeffs = 10 * np.random.rand(poly_order + 1) - 5
coeffs = np.floor(coeffs * 100) / 100
Y = [np.polyval(coeffs, x) for x in X] + np.random.rand(m, 1)

Y_avg = np.mean(Y)
SST = sum([(y - Y_avg) ** 2 for y in Y])[0]

x_curve = np.linspace(-1, 3, 200)
diff = max(Y)[0] - min(Y)[0]


# Plot the random data points
def plotData():
    plt.plot([0, 0], [min(Y)[0] - 0.5 * diff, max(Y)[0] + 0.5 * diff], 'k--', linewidth=1)

    enum = enumerate(coeffs[:-2])
    func = ' + '.join("${}x^{{{}}}$".format(coeff, len(coeffs) - exp - 1) for exp, coeff in enum)
    app = f'${coeffs[-2]}x$ + ${coeffs[-1]}$ + Noise[0, 1]'
    if func == '':
        func = app
    else:
        func += ' + ' + app

    plt.plot(X, Y, "b.", label=f"{func}")

    plt.xlabel("$x$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=10)
    plt.xlim([-1, 3])
    plt.ylim([min(Y)[0] - 0.5 * diff, max(Y)[0] + 0.5 * diff])
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.show()


# Main stuff
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

    R2 = 1 - sum([(y - np.polyval(w_best, x)) ** 2 for x, y in zip(X, Y)])[0] / SST
    R2 = np.floor(R2 * 1000) / 1000

    w = np.floor(w_best * 100) / 100
    enum = enumerate(w[:-2])
    func = ' + '.join("${}x^{{{}}}$".format(coeff[0], len(w_best) - exp - 1) for exp, coeff in enum)
    app = f'${w[-2][0]}x$ + ${w[-1][0]}$'
    if func == '':
        func = app
    else:
        func += ' + ' + app

    plt.plot(x_curve, [np.polyval(w_best, i) for i in x_curve], "-", linewidth=2, label=f"{func} | $R^2$ = {R2}")

plotData()
