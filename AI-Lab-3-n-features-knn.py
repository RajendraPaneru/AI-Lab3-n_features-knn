import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Step 1: Generate random N-feature dataset
# -------------------------------
def generate_n_feature_data(n_samples=100, n_features=5):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=3, n_redundant=0, n_classes=2, random_state=42)
    return X, y

# -------------------------------
# Step 2: Get user input for N-feature query
# -------------------------------
def get_user_input(n_features):
    print(f"Enter {n_features} feature values separated by space:")
    try:
        values = list(map(float, input().strip().split()))
        if len(values) != n_features:
            print(f"❌ You must enter exactly {n_features} values.")
            exit()
        return np.array(values).reshape(1, -1)
    except ValueError:
        print("❌ Invalid input. Please enter only numbers.")
        exit()

# -------------------------------
# Step 3: Run the KNN algorithm
# -------------------------------
def run_knn(X, y, query_point, k):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    query_scaled = scaler.transform(query_point)

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_scaled, y)

    prediction = knn_model.predict(query_scaled)
    print(f"\n✅ Predicted Class for input {query_point.flatten().tolist()} is: {prediction[0]}")

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    n_features = int(input("Enter the number of features (N): "))
    k = int(input("Enter value of K (number of neighbors): "))

    X, y = generate_n_feature_data(n_samples=100, n_features=n_features)
    query_point = get_user_input(n_features)

    run_knn(X, y, query_point, k)
