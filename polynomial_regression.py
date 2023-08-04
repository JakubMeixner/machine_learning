import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data from files
with open('outputX.txt', 'r') as f:
    X = np.array(ast.literal_eval(f.read()))

with open('outputy.txt', 'r') as f:
    y = np.array(ast.literal_eval(f.read()))

# Preprocess the data to handle missing and invalid values
X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
y = np.nan_to_num(y, nan=0, posinf=1e10, neginf=-1e10)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Polynomial Regression function
def polynomial_regression(x_train, y_train, x_test, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_train_poly = polynomial_features.fit_transform(x_train)
    x_test_poly = polynomial_features.transform(x_test)

    reg_model = LinearRegression()
    reg_model.fit(x_train_poly, y_train)

    y_pred = reg_model.predict(x_test_poly)
    return y_pred
# Set the polynomial degree (a hyperparameter that controls the complexity of the model)
polynomial_degree = 5

# Predict the target variable for the test set using Polynomial Regression
y_pred = polynomial_regression(X_train[:, 0].reshape(-1, 1), y_train, X_test[:, 0].reshape(-1, 1), polynomial_degree)

# Replace NaN, infinity, and very large values in y_pred with a constant value
y_pred[np.isnan(y_pred)] = 0
y_pred[np.isinf(y_pred)] = 0
y_pred[y_pred > 1e10] = 0
y_pred[y_pred < -1e10] = 0

# Calculate the Mean Squared Error (MSE) and Mean Absolute Error (MAE) as evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Scatter Plot of Number of Co Atoms vs. Magnetocrystalline Anisotropy Energy
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label='Data')
plt.xlabel('Number of Co Atoms')
plt.ylabel(' Anisotropy Energy')
plt.title('Scatter Plot of Number of Co Atoms vs. Magnetocrystalline Anisotropy Energy')

# Polynomial Regression line
X_poly_regression = np.linspace(min(X[:, 0]), max(X[:, 0]), 100).reshape(-1, 1)
y_poly_regression = polynomial_regression(X_train[:, 0].reshape(-1, 1), y_train, X_poly_regression, polynomial_degree)
plt.plot(X_poly_regression, y_poly_regression, color='blue', label='Polynomial Regression')

plt.legend()
plt.savefig('co_atoms_vs_magnetocrystalline_anisotropy.png')
plt.close()

# Plot the predicted values against the true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)

# Add dotted lines representing the MAE above and below the regression line
for true_val, pred_val in zip(y_test, y_pred):
    plt.plot([true_val, pred_val], [true_val + mae, pred_val + mae], 'r:', alpha=0.5)
    plt.plot([true_val, pred_val], [true_val - mae, pred_val - mae], 'r:', alpha=0.5)
plt.xlabel('True  Anisotropy Energy')
plt.ylabel('Predicted Magnetocrystalline Anisotropy Energy')
plt.title('Polynomial Regression: True vs. Predicted with MAE Bounds')
plt.tight_layout()
plt.savefig('polynomial_regression_true_vs_predicted.png')
plt.close()
# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='dashed', lw=2)
plt.xlabel('Predicted Magnetocrystalline Anisotropy Energy')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot_polynomial_regression.png')
plt.close()
