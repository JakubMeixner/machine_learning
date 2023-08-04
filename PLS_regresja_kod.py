import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load your data here and replace X and y with your input features and target variable
# X should be a 2D array-like object where rows are samples and columns are features
# y should be a 1D array-like object representing the target variable

# Example:
# X = np.array([[feature1_sample1, feature2_sample1, ...],
#               [feature1_sample2, feature2_sample2, ...],
#               ...
#               [feature1_sampleN, feature2_sampleN, ...]])

# y = np.array([target_sample1, target_sample2, ..., target_sampleN])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize PLS regression with the desired number of components (latent variables)
n_components = 3
pls = PLSRegression(n_components=n_components)

# Fit the PLS model to the training data
pls.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = pls.predict(X_test)

# Calculate the Mean Squared Error (MSE) and Mean Absolute Error (MAE) as evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Scatter Plot of Number of Co Atoms vs. Magnetocrystalline Anisotropy Energy
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5, label='Data')
plt.xlabel('Number of Co Atoms')
plt.ylabel('Magnetocrystalline Anisotropy Energy')
plt.title('Scatter Plot of Number of Co Atoms vs. Magnetocrystalline Anisotropy Energy')
plt.tight_layout()  # Automatically adjust subplots to avoid label cutoff

# Fit a linear regression model to the data
reg_model = LinearRegression()
reg_model.fit(X[:, 0].reshape(-1, 1), y)
y_reg = reg_model.predict(X[:, 0].reshape(-1, 1))
plt.plot(X[:, 0], y_reg, color='red', label='Linear Regression')

# PLS regression line
X_pls = np.linspace(min(X[:, 0]), max(X[:, 0]), 100).reshape(-1, 1)
X_pls_with_features = np.hstack((X_pls, np.mean(X_train[:, 1])*np.ones((100, 1)), np.mean(X_train[:, 2])*np.ones((100, 1))))
y_pls = pls.predict(X_pls_with_features)
plt.plot(X_pls, y_pls, color='blue', label='PLS Regression')

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

plt.xlabel('True Magnetocrystalline Anisotropy Energy')
plt.ylabel('Predicted Magnetocrystalline Anisotropy Energy')
plt.title('PLS Regression: True vs. Predicted with MAE Bounds')
plt.tight_layout()  # Automatically adjust subplots to avoid label cutoff
plt.savefig('pls_regression_true_vs_predicted.png')
plt.close()

# Residual Plot
residuals = y_test - y_pred.flatten()
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='dashed', lw=2)
plt.xlabel('Predicted Magnetocrystalline Anisotropy Energy')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()  # Automatically adjust subplots to avoid label cutoff
plt.savefig('residual_plot.png')
plt.close()





























