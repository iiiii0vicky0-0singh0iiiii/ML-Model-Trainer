
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.feature_selection import (
    VarianceThreshold
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
st.write("Advanced Machine Learning Dashboard")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # ---------------- LOAD DATA ----------------

        df = pd.read_csv(uploaded_file)

        # Keep original copy
        original_df = df.copy()

        st.subheader("Dataset Preview")
        st.dataframe(original_df.head())

        st.write("Dataset Shape:", df.shape)

        # ---------------- CLEAN DATA ----------------

        # Remove empty columns
        df = df.dropna(axis=1, how='all')

        # Fill missing values
        df = df.fillna(0)

        # ---------------- TARGET COLUMN ----------------

        st.subheader("Select Target Column")

        target_column = st.selectbox(
            "Target Column",
            original_df.columns
        )

        st.success(
            f"Selected Target: {target_column}"
        )

        # ---------------- TARGET LABELS ----------------

        target_labels = original_df[target_column]

        st.write(
            "Unique Classes:",
            target_labels.unique()
        )

        # ---------------- FEATURE DATAFRAME ----------------

        feature_df = df.drop(columns=[target_column])

        # ---------------- ENCODE FEATURES ONLY ----------------

        for col in feature_df.columns:

            if feature_df[col].dtype == 'object':

                le = LabelEncoder()

                feature_df[col] = le.fit_transform(
                    feature_df[col].astype(str)
                )

        # ---------------- ENCODE TARGET ----------------

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            target_labels.astype(str)
        )

        # ---------------- FEATURES ----------------

        X = feature_df.copy()

        # ---------------- CONVERT TO NUMERIC ----------------

        for col in X.columns:

            X[col] = pd.to_numeric(
                X[col],
                errors='coerce'
            )

        X = X.fillna(0)

        # ---------------- FEATURE SELECTION ----------------

        selector = VarianceThreshold(
            threshold=0
        )

        X = selector.fit_transform(X)

        # ---------------- FEATURE SCALING ----------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # ---------------- PROCESSED DATA ----------------

        st.subheader("Processed Dataset")

        processed_df = pd.DataFrame(X_scaled)

        st.dataframe(processed_df.head())

        # ---------------- DATA VISUALIZATION ----------------

        st.subheader("Dataset Statistics")

        st.write(df.describe())

        # ---------------- BAR GRAPH ----------------

        st.subheader("Column Distribution")

        visual_col = st.selectbox(
            "Select Column",
            original_df.columns
        )

        fig1, ax1 = plt.subplots()

        original_df[visual_col].value_counts().plot(
            kind='bar',
            ax=ax1
        )

        ax1.set_title(
            f"Distribution of {visual_col}"
        )

        st.pyplot(fig1)

        # ---------------- HEATMAP ----------------

        st.subheader("Correlation Heatmap")

        numeric_df = feature_df.select_dtypes(
            include=np.number
        )

        fig2, ax2 = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap='coolwarm',
            ax=ax2
        )

        st.pyplot(fig2)

        # ---------------- HISTOGRAM ----------------

        st.subheader("Feature Histogram")

        hist_col = st.selectbox(
            "Select Feature",
            numeric_df.columns,
            key="hist"
        )

        fig3, ax3 = plt.subplots()

        ax3.hist(
            numeric_df[hist_col],
            bins=20
        )

        ax3.set_title(hist_col)

        st.pyplot(fig3)

        # ---------------- TEST SIZE ----------------

        test_size = st.slider(
            "Select Test Size",
            0.1,
            0.5,
            0.2
        )

        # ---------------- TRAIN TEST SPLIT ----------------

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
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

        # ---------------- PARAMETERS ----------------

        if model_name == "KNN":

            k_value = st.slider(
                "Select K Value",
                1,
                15,
                5
            )

        elif model_name == "Logistic Regression":

            c_value = st.slider(
                "Select C Value",
                0.01,
                10.0,
                1.0
            )

        elif model_name == "Decision Tree":

            max_depth = st.slider(
                "Select Max Depth",
                1,
                20,
                5
            )

        elif model_name == "Random Forest":

            n_estimators = st.slider(
                "Number of Trees",
                50,
                500,
                100
            )

            rf_depth = st.slider(
                "Random Forest Max Depth",
                1,
                20,
                10
            )

        # ---------------- TRAIN BUTTON ----------------

        if st.button("Train Model"):

            # ---------------- MODEL CONFIG ----------------

            if model_name == "KNN":

                model = KNeighborsClassifier(
                    n_neighbors=k_value
                )

            elif model_name == "Logistic Regression":

                model = LogisticRegression(
                    C=c_value,
                    max_iter=5000,
                    solver='lbfgs'
                )

            elif model_name == "Decision Tree":

                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    random_state=42
                )

            elif model_name == "Random Forest":

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=rf_depth,
                    random_state=42
                )

            # ---------------- TRAIN MODEL ----------------

            model.fit(X_train, y_train)

            # ---------------- PREDICTION ----------------

            y_pred = model.predict(X_test)

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

            # ---------------- DECODE PREDICTIONS ----------------

            decoded_actual = target_encoder.inverse_transform(
                y_test.astype(int)
            )

            decoded_pred = target_encoder.inverse_transform(
                y_pred.astype(int)
            )

            # ---------------- RESULT DATAFRAME ----------------

            st.subheader("Prediction Results")

            result_df = pd.DataFrame({

                "Actual": decoded_actual,
                "Predicted": decoded_pred

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

