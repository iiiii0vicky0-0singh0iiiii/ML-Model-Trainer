
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

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Dataset Shape:", df.shape)

        # ---------------- CLEAN DATA ----------------

        df = df.dropna(axis=1, how='all')

        df = df.fillna(0)

        # ---------------- ENCODE CATEGORICAL ----------------

        label_encoders = {}

        for col in df.columns:

            if df[col].dtype == 'object':

                le = LabelEncoder()

                df[col] = le.fit_transform(
                    df[col].astype(str)
                )

                label_encoders[col] = le

        # ---------------- NUMERIC CONVERSION ----------------

        for col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors='coerce'
            )

        df = df.fillna(0)

        df = df.select_dtypes(
            include=[np.number]
        )

        # ---------------- CHECK DATASET ----------------

        if len(df.columns) < 2:

            st.error(
                "Dataset must contain at least 2 columns."
            )

            st.stop()

        st.subheader("Processed Dataset")
        st.dataframe(df.head())

        # ---------------- COLUMN INFO ----------------

        st.subheader("Column Information")

        for col in df.columns:

            st.write(
                f"{col} : {df[col].nunique()} unique values"
            )

        # ---------------- STATISTICS ----------------

        st.subheader("Dataset Statistics")

        st.write(df.describe())

        # ---------------- VISUALIZATION ----------------

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

        st.pyplot(fig1)

        # ---------------- HEATMAP ----------------

        st.subheader("Correlation Heatmap")

        fig2, ax2 = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            df.corr(),
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

        st.pyplot(fig3)

        # ---------------- TARGET DETECTION ----------------

        possible_targets = []

        for col in df.columns:

            unique_values = df[col].nunique()

            if 2 <= unique_values <= 20:

                possible_targets.append(col)

        if len(possible_targets) == 0:

            st.error(
                "No valid target column found."
            )

            st.stop()

        st.subheader("Select Target Column")

        target_column = st.selectbox(
            "Target Column",
            possible_targets
        )

        # ---------------- FEATURES ----------------

        X = df.drop(columns=[target_column])

        y = df[target_column]

        # ---------------- FEATURE SELECTION ----------------

        selector = VarianceThreshold(
            threshold=0
        )

        X = selector.fit_transform(X)

        # ---------------- FEATURE SCALING ----------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # ---------------- SMART TEST SIZE ----------------

        if len(df) < 50:
            test_size = 0.1
        else:
            test_size = 0.2

        # ---------------- TRAIN TEST SPLIT ----------------

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # ---------------- MODEL SELECTION ----------------

        st.subheader("Select Model")

        model_name = st.selectbox(
            "Choose Model",
            [
                "KNN",
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ]
        )

        # ---------------- MODEL PARAMETERS ----------------

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
                "Max Depth",
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

            # ---------------- TRAIN ----------------

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

            # ---------------- REPORT ----------------

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

