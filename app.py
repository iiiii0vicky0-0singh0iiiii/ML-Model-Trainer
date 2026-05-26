import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AutoML Studio",
    layout="wide"
)

st.title("AutoML Studio")
st.write("Machine Learning Training Dashboard")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # ---------------- LOAD DATA ----------------

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Dataset Shape:", df.shape)

        # ---------------- CLEAN DATA ----------------

        # Remove empty columns
        df = df.dropna(axis=1, how='all')

        # Fill missing values
        df = df.fillna(0)

        # ---------------- ENCODE CATEGORICAL DATA ----------------

        label_encoders = {}

        for col in df.columns:

            if df[col].dtype == 'object':

                le = LabelEncoder()

                df[col] = le.fit_transform(
                    df[col].astype(str)
                )

                label_encoders[col] = le

        # ---------------- CONVERT TO NUMERIC ----------------

        for col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors='coerce'
            )

        # Replace NaN values
        df = df.fillna(0)

        # Keep only numeric columns
        df = df.select_dtypes(
            include=[np.number]
        )

        # ---------------- CHECK DATASET ----------------

        if len(df.columns) < 2:

            st.error(
                "Dataset must contain at least 2 valid columns."
            )

            st.stop()

        # ---------------- PROCESSED DATA ----------------

        st.subheader("Processed Dataset")
        st.dataframe(df.head())

        # ---------------- COLUMN INFO ----------------

        st.subheader("Column Information")

        for col in df.columns:

            st.write(
                f"{col} : {df[col].nunique()} unique values"
            )

        # ---------------- VISUALIZATION ----------------

        st.subheader("Dataset Statistics")

        st.write(df.describe())

        # ---------------- BAR GRAPH ----------------

        st.subheader("Column Distribution")

        visual_col = st.selectbox(
            "Select Column",
            df.columns
        )

        fig1, ax1 = plt.subplots()

        df[visual_col].value_counts().plot(
            kind='bar',
            ax=ax1
        )

        ax1.set_title(
            f"Distribution of {visual_col}"
        )

        st.pyplot(fig1)

        # ---------------- HEATMAP ----------------

        st.subheader("Correlation Heatmap")

        corr = df.corr()

        fig2, ax2 = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm',
            ax=ax2
        )

        st.pyplot(fig2)

        # ---------------- HISTOGRAM ----------------

        st.subheader("Feature Histogram")

        hist_col = st.selectbox(
            "Select Feature",
            df.columns,
            key="hist"
        )

        fig3, ax3 = plt.subplots()

        ax3.hist(
            df[hist_col],
            bins=20
        )

        ax3.set_title(hist_col)

        st.pyplot(fig3)

        # ---------------- TARGET DETECTION ----------------

        possible_targets = []

        for col in df.columns:

            unique_values = df[col].nunique()

            # Good classification target
            if 2 <= unique_values <= 20:

                possible_targets.append(col)

        # No valid target
        if len(possible_targets) == 0:

            st.error(
                "No valid target column found."
            )

            st.write(
                "Target column should contain "
                "2 to 20 unique classes."
            )

            st.stop()

        # ---------------- TARGET SELECTION ----------------

        st.subheader("Select Target Column")

        target_column = st.selectbox(
            "Target Column",
            possible_targets
        )

        st.success(
            f"Suggested Target Columns: {possible_targets}"
        )

        # ---------------- FEATURES ----------------

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ---------------- FEATURE SCALING ----------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # ---------------- SMART SPLIT ----------------

        if len(df) < 50:
            test_size = 0.1
        else:
            test_size = 0.2

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42
        )

        # ---------------- MODEL SELECTION ----------------

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

        # ---------------- TRAIN BUTTON ----------------

        if st.button("Train Model"):

            # ---------------- MODEL CONFIG ----------------

            if model_name == "KNN":

                model = KNeighborsClassifier()

                params = {
                    'n_neighbors': [3, 5, 7]
                }

            elif model_name == "Logistic Regression":

                model = LogisticRegression(
                    max_iter=3000,
                    solver='lbfgs'
                )

                params = {
                    'C': [0.1, 1, 10]
                }

            elif model_name == "Decision Tree":

                model = DecisionTreeClassifier()

                params = {
                    'max_depth': [3, 5, 10]
                }

            elif model_name == "Random Forest":

                model = RandomForestClassifier()

                params = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10]
                }

            # ---------------- SAFE CV ----------------

            min_class_count = y.value_counts().min()

            cv_value = min(2, min_class_count)

            # ---------------- SMALL DATASET ----------------

            if cv_value < 2:

                st.warning(
                    "Dataset too small for GridSearchCV."
                )

                model.fit(X_train, y_train)

                best_model = model

            else:

                # ---------------- GRID SEARCH ----------------

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=cv_value,
                    scoring='accuracy'
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                st.subheader("Best Parameters")

                st.write(grid.best_params_)

            # ---------------- PREDICTION ----------------

            y_pred = best_model.predict(X_test)

            # ---------------- ACCURACY ----------------

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            st.success(
                "Model Trained Successfully"
            )

            st.metric(
                "Accuracy",
                f"{accuracy:.2f}"
            )

            # ---------------- CLASSIFICATION REPORT ----------------

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df)

            # ---------------- CONFUSION MATRIX ----------------

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            cm_df = pd.DataFrame(cm)

            st.dataframe(cm_df)

            # ---------------- PREDICTIONS ----------------

            st.subheader("Prediction Results")

            result_df = pd.DataFrame({

                "Actual": y_test.values,
                "Predicted": y_pred

            })

            st.dataframe(result_df.head(20))

            # ---------------- DOWNLOAD CSV ----------------

            csv = result_df.to_csv(
                index=False
            ).encode('utf-8')

            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(f"Error: {e}")

