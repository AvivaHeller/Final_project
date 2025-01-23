from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Select features (brainwaves) and target (attention or meditation)
features = data[brainwave_columns]
target_attention = data['attention']
target_meditation = data['meditation']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_attention, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"R² score for Attention: {r2:.4f}")

# Display model coefficients
coefficients = pd.DataFrame({'Brainwave': brainwave_columns, 'Coefficient': model.coef_})
print(coefficients)
