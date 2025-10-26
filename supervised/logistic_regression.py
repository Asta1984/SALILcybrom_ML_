from sklearn.datasets import load_digits

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Preprocessing step - Scale features (IMPORTANT for Logistic Regression)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000,  # Increase iterations
    solver='lbfgs',  # Good for multiclass
    multi_class='multinomial'  # Better for multiclass
)
model.fit(X_train_scaled, y_train)

# Save both model 
import joblib
joblib.dump(model, "Logistic_model_number_det.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved")

# Model prediction
pred = model.predict(X_test_scaled)

# Model measurement metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

