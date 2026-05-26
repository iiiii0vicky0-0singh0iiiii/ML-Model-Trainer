# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ML Model Trainer", layout="wide")

st.title("Machine Learning Model Trainer")
st.write("Upload CSV file, select model, and train instantly.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Select Target Column")
    target_column = st.selectbox("Target Column", df.columns)

    # Encode categorical columns
    le_dict = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    X = df.drop(columns=[target_column])
    y = df[target_column]

    st.subheader("Select Machine Learning Model")

    model_name = st.selectbox(
        "Choose Model",
        [
            "KNN",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    if st.button("Train Model"):

        if model_name == "KNN":
            k = st.slider("Select K Value", 1, 15, 5)
            model = KNeighborsClassifier(n_neighbors=k)

        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()

        elif model_name == "Random Forest":
            model = RandomForestClassifier()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Model Trained Successfully!")

        st.metric("Accuracy", f"{accuracy:.2f}")

        st.subheader("Predictions")
        result_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        st.dataframe(result_df.head(20))
