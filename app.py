import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.feature_selection import VarianceThreshold

# ---- Classifiers ----
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AutoML Studio", layout="wide")
st.title("AutoML Studio")
st.write("Advanced Machine Learning Dashboard")

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    try:

        # ============================================================
        # LOAD DATA
        # ============================================================
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.write("Dataset Shape:", df.shape)

        # ============================================================
        # MISSING VALUE STRATEGY
        # ============================================================
        st.subheader("Missing Value Strategy")
        fill_strategy = st.radio(
            "Choose how to handle missing values",
            ["Fill with 0", "Fill with Mean", "Fill with Median"],
            horizontal=True
        )

        # Remove fully empty columns
        df = df.dropna(axis=1, how='all')

        # ============================================================
        # ENCODE CATEGORICAL DATA
        # ============================================================
        label_encoders = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

        # ============================================================
        # CONVERT TO NUMERIC
        # ============================================================
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Apply missing value fill strategy
        if fill_strategy == "Fill with 0":
            df = df.fillna(0)
        elif fill_strategy == "Fill with Mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_strategy == "Fill with Median":
            df = df.fillna(df.median(numeric_only=True))

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # ============================================================
        # CHECK DATASET
        # ============================================================
        if len(df.columns) < 2:
            st.error("Dataset must contain at least 2 valid columns.")
            st.stop()

        # ============================================================
        # OUTLIER REMOVAL (OPTIONAL)
        # ============================================================
        remove_outliers = st.checkbox("Remove Outliers using IQR method")
        if remove_outliers:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
            before = len(df)
            df = df[mask].reset_index(drop=True)
            st.info(f"Removed {before - len(df)} outlier rows. Remaining: {len(df)}")

        # ============================================================
        # PROCESSED DATA
        # ============================================================
        st.subheader("Processed Dataset")
        st.dataframe(df.head())

        # ============================================================
        # COLUMN INFORMATION
        # ============================================================
        st.subheader("Column Information")
        for col in df.columns:
            st.write(f"{col} : {df[col].nunique()} unique values")

        # ============================================================
        # DATASET STATISTICS
        # ============================================================
        st.subheader("Dataset Statistics")
        st.write(df.describe())

        # ============================================================
        # VISUALIZATIONS
        # ============================================================
        st.subheader("Visualizations")

        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs(
            ["Bar Chart", "Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"]
        )

        with viz_tab1:
            visual_col = st.selectbox("Select Column for Bar Chart", df.columns, key="bar")
            fig1, ax1 = plt.subplots()
            df[visual_col].value_counts().plot(kind='bar', ax=ax1)
            ax1.set_title(f"Distribution of {visual_col}")
            st.pyplot(fig1)

        with viz_tab2:
            hist_col = st.selectbox("Select Feature for Histogram", df.columns, key="hist")
            fig3, ax3 = plt.subplots()
            ax3.hist(df[hist_col], bins=20, color='steelblue', edgecolor='black')
            ax3.set_title(hist_col)
            st.pyplot(fig3)

        with viz_tab3:
            box_col = st.selectbox("Select Feature for Box Plot", df.columns, key="box")
            fig4, ax4 = plt.subplots()
            ax4.boxplot(df[box_col].dropna())
            ax4.set_title(f"Box Plot - {box_col}")
            st.pyplot(fig4)

        with viz_tab4:
            cols_for_scatter = df.columns.tolist()
            if len(cols_for_scatter) >= 2:
                scatter_x = st.selectbox("X-axis", cols_for_scatter, key="sx")
                scatter_y = st.selectbox("Y-axis", cols_for_scatter, index=1, key="sy")
                fig5, ax5 = plt.subplots()
                ax5.scatter(df[scatter_x], df[scatter_y], alpha=0.5, s=20)
                ax5.set_xlabel(scatter_x)
                ax5.set_ylabel(scatter_y)
                ax5.set_title(f"{scatter_x} vs {scatter_y}")
                st.pyplot(fig5)

        with viz_tab5:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)

        # ============================================================
        # TARGET COLUMN SELECTION
        # ============================================================
        st.subheader("Select Target Column")
        st.write("Choose the output column for model training.")

        target_column = st.selectbox("Target Column", df.columns)

        n_classes = df[target_column].nunique()
        st.success(f"Selected Target Column: {target_column}")
        st.write("Unique Classes:", n_classes)
        st.write("Unique Values:", df[target_column].unique())

        # ---- FIX: Single-class guard ----
        if n_classes < 2:
            st.error(
                "The selected target column has only ONE class. "
                "Classification requires at least 2 distinct classes. "
                "Please choose a different target column."
            )
            st.stop()

        # ============================================================
        # FEATURES & TARGET
        # ============================================================
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ============================================================
        # FEATURE SELECTION (Variance Threshold)
        # ============================================================
        selector = VarianceThreshold(threshold=0)
        X_selected = selector.fit_transform(X)
        selected_feature_names = X.columns[selector.get_support()].tolist()

        # ============================================================
        # FEATURE SCALING
        # ============================================================
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # ============================================================
        # TEST SIZE
        # ============================================================
        test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)

        # ---- FIX: Safe stratify (disabled when any class has < 2 samples) ----
        min_class_count = y.value_counts().min()
        use_stratify = y if min_class_count >= 2 else None
        if min_class_count < 2:
            st.warning(
                "One or more classes have fewer than 2 samples. "
                "Stratified split has been disabled automatically."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=42,
            stratify=use_stratify
        )

        # ============================================================
        # MODEL SELECTION
        # ============================================================
        st.subheader("Select Machine Learning Model")

        model_name = st.selectbox(
            "Choose Model",
            [
                "KNN",
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "SVM",
                "Naive Bayes",
                "Gradient Boosting",
                "AdaBoost"
            ]
        )

        # ============================================================
        # MODEL HYPERPARAMETERS
        # ============================================================
        if model_name == "KNN":
            k_value = st.slider("K Value (Neighbors)", 1, 15, 5)

        elif model_name == "Logistic Regression":
            c_value = st.slider("C Value (Regularization)", 0.01, 10.0, 1.0)

        elif model_name == "Decision Tree":
            max_depth = st.slider("Max Depth", 1, 20, 5)

        elif model_name == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            rf_depth = st.slider("Max Depth", 1, 20, 10)

        elif model_name == "SVM":
            svm_c = st.slider("C Value", 0.01, 10.0, 1.0)
            svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])

        elif model_name == "Gradient Boosting":
            gb_estimators = st.slider("Number of Estimators", 50, 300, 100)
            gb_lr = st.slider("Learning Rate", 0.01, 0.5, 0.1)
            gb_depth = st.slider("Max Depth", 1, 10, 3)

        elif model_name == "AdaBoost":
            ab_estimators = st.slider("Number of Estimators", 10, 200, 50)
            ab_lr = st.slider("Learning Rate", 0.01, 2.0, 1.0)

        # Naive Bayes has no hyperparameters to tune

        # ============================================================
        # CROSS VALIDATION FOLDS
        # ============================================================
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)

        # ============================================================
        # TRAIN BUTTON
        # ============================================================
        if st.button("Train Model"):

            # ---- Build model ----
            if model_name == "KNN":
                model = KNeighborsClassifier(n_neighbors=k_value)

            elif model_name == "Logistic Regression":
                model = LogisticRegression(C=c_value, max_iter=5000, solver='lbfgs')

            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            elif model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=rf_depth, random_state=42
                )

            elif model_name == "SVM":
                model = SVC(C=svm_c, kernel=svm_kernel, probability=True, random_state=42)

            elif model_name == "Naive Bayes":
                model = GaussianNB()

            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(
                    n_estimators=gb_estimators,
                    learning_rate=gb_lr,
                    max_depth=gb_depth,
                    random_state=42
                )

            elif model_name == "AdaBoost":
                model = AdaBoostClassifier(
                    n_estimators=ab_estimators,
                    learning_rate=ab_lr,
                    random_state=42
                )

            # ---- Train ----
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ---- Accuracy ----
            accuracy = accuracy_score(y_test, y_pred)
            st.success("Model Trained Successfully!")
            st.metric("Test Accuracy", f"{accuracy:.4f}")

            # ---- Cross-Validation ----
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
            st.metric(
                f"{cv_folds}-Fold CV Accuracy",
                f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
            )

            # ---- ROC-AUC (binary only) ----
            if n_classes == 2:
                try:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test)[:, 1]
                    else:
                        y_prob = model.decision_function(X_test)
                    auc = roc_auc_score(y_test, y_prob)
                    st.metric("ROC-AUC Score", f"{auc:.4f}")
                except Exception:
                    pass

            # ---- Classification Report ----
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # ---- Confusion Matrix ----
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                ax=ax_cm,
                xticklabels=np.unique(y),
                yticklabels=np.unique(y)
            )
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            # ---- Feature Importance (tree-based models) ----
            if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost"]:
                st.subheader("Feature Importance")
                importances = model.feature_importances_
                feat_df = pd.DataFrame({
                    "Feature": selected_feature_names[:len(importances)],
                    "Importance": importances
                }).sort_values("Importance", ascending=False)

                fig_fi, ax_fi = plt.subplots(figsize=(8, 4))
                ax_fi.barh(feat_df["Feature"], feat_df["Importance"], color='teal')
                ax_fi.invert_yaxis()
                ax_fi.set_title("Feature Importance")
                st.pyplot(fig_fi)

            # ---- Prediction Results ----
            st.subheader("Prediction Results (first 20 rows)")
            result_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": y_pred
            })
            st.dataframe(result_df.head(20))

            # ---- Download ----
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
