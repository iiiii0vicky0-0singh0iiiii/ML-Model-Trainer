
import streamlit as st
import pandas as pd
import numpy as np

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
st.write("Train and Test Machine Learning Models")

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

        # ---------------- ENCODE CATEGORICAL ----------------

        label_encoders = {}

        for col in df.columns:

            try:

                if df[col].dtype == 'object':

                    le = LabelEncoder()

                    df[col] = le.fit_transform(
                        df[col].astype(str)
                    )

                    label_encoders[col] = le

            except:
                pass

        # ---------------- FORCE NUMERIC ----------------

        for col in df.columns:

            try:

                df[col] = pd.to_numeric(
                    df[col],
                    errors='coerce'
                )

            except:
                pass

        # Replace NaN values
        df = df.fillna(0)

        # Keep only numeric columns
        df = df.select_dtypes(
            include=[np.number]
        )

        # ---------------- DATA CHECK ----------------

        if len(df.columns) < 2:

            st.error(
                "Dataset must contain at least 2 columns."
            )

            st.stop()

        st.subheader("Processed Dataset")
        st.dataframe(df.head())

        # ---------------- TARGET COLUMN ----------------

        target_column = st.selectbox(
            "Select Target Column",
            df.columns
        )

        # ---------------- FEATURES ----------------

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ---------------- TARGET VALIDATION ----------------

        if y.nunique() < 2:

            st.error(
                "Target column must contain at least 2 classes."
            )

            st.stop()

        # ---------------- FEATURE SCALING ----------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # ---------------- TEST SIZE ----------------

        test_size = st.slider(
            "Select Test Size",
            0.1,
            0.5,
            0.2
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42
        )

        # ---------------- MODEL SELECTION ----------------

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

            # ---------------- MODEL + PARAMETERS ----------------

            if model_name == "KNN":

                model = KNeighborsClassifier()

                params = {
                    'n_neighbors': [3, 5, 7]
                }

            elif model_name == "Logistic Regression":

                model = LogisticRegression(
                    max_iter=2000,
                    solver='liblinear'
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

            # ---------------- SAFE CV VALUE ----------------

            min_class_count = y.value_counts().min()

            cv_value = min(2, min_class_count)

            if cv_value < 2:

                st.warning(
                    "Dataset too small for tuning. "
                    "Training without GridSearchCV."
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

            st.success("Model Trained Successfully")

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

