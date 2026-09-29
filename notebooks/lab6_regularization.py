import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# 1. Simulate/Load Data
np.random.seed(42)

X = np.sort(6 * np.random.rand(100, 1) + 4)

y = np.sin(X).ravel() + np.random.normal(
    0, 0.2, X.shape[0]
)


# 2. Split dataset into 80% Training and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 3. Transform features to a Higher-Order Polynomial
poly_degree = 15

poly = PolynomialFeatures(
    degree=poly_degree
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 4. Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)


# 5. Define a range of tuning parameters
lambdas = np.logspace(-4, 4, 200)

train_errors = []
test_errors = []


# 6. Loop over each lambda value
for lam in lambdas:

    ridge = Ridge(alpha=lam)

    ridge.fit(
        X_train_scaled,
        y_train
    )

    # Predict on training data
    y_train_pred = ridge.predict(
        X_train_scaled
    )

    # Predict on testing data
    y_test_pred = ridge.predict(
        X_test_scaled
    )

    # Calculate Mean Squared Error
    train_errors.append(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    test_errors.append(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )


# 7. Plot the Error Curves
plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    train_errors,
    label='Training Error'
)

plt.plot(
    lambdas,
    test_errors,
    label='Testing Error',
    linestyle='--'
)

plt.xscale('log')

plt.xlabel('Regularization Parameter (lambda / alpha)')
plt.ylabel('Mean Squared Error')

plt.title(
    'Regularization Path: Ridge Regression Overfitting Control (Degree 15)'
)

plt.legend()

plt.grid(True, which="both", linestyle="--")

plt.show()