import streamlit as st
import pandas as pd
import joblib

# Load the trained KMeans model
kmeans_model = joblib.load("random_forest_model.pkl")

# cluster label
cluster_names = {
    0: "Big Spenders",
    1: "Budget Conscious",
    2: "Average Spenders",
    3: "Savers",
    4: "Aspirational Spenders"
}

# Function to predict cluster for given input
def predict_cluster(age, annual_income, spending_score):
    # Create a DataFrame with input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual Income': [annual_income],
        'Spending Score': [spending_score]
    })
    # Predict cluster label
    cluster_label = kmeans_model.predict(input_data)
    return cluster_label[0]

# Streamlit app
def main():
    st.title("Customer Segmentation App")

    # Input fields for Age, Annual Income, and Spending Score
    age = st.number_input("Enter Age:", min_value=0, max_value=120, step=1)
    annual_income = st.number_input("Enter Annual Income:", min_value=0, step=1)
    spending_score = st.number_input("Enter Spending Score:", min_value=0, max_value=100, step=1)

    # Predict cluster label for input
    if st.button("Predict"):
        cluster_label = predict_cluster(age, annual_income, spending_score)
        cluster_label_name = cluster_names[cluster_label]
        st.write(f"Predicted Cluster Label: {cluster_label_name}")

if __name__ == "__main__":
    main()