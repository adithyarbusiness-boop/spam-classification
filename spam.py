import pandas as pd

data = pd.read_csv("data/spambase.data", header=None)
print(data.head())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Dataset shape:", data.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print(y.value_counts())
print("Missing values:", data.isnull().sum().sum())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Scaling (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create KNN model (K = 5)
knn = KNeighborsClassifier(n_neighbors=5)

# Train model
knn.fit(X_train_scaled, y_train)

# Predict using test data
y_pred_knn = knn.predict(X_test_scaled)

# Evaluate results
print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\nKNN Classification Report:\n")
print(classification_report(y_test, y_pred_knn))

print("\nKNN Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_knn))
for k in [3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    print(f"K = {k}, Accuracy = {acc}")

from sklearn.naive_bayes import GaussianNB

# Naive Bayes model
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Predictions
y_pred_nb = nb.predict(X_test_scaled)

# Results
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

print("\nNaive Bayes Classification Report:\n")
print(classification_report(y_test, y_pred_nb))

print("\nNaive Bayes Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_nb))
















    
