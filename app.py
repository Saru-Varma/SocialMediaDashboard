
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Social Media Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Social Media Users Analysis Dashboard")
st.write("Analyze social media activity, usage categories, emotions, and unusual activity patterns.")

@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    data.columns = data.columns.str.strip()

    numeric_cols = [
        "Age",
        "Daily_Usage_Time (minutes)",
        "Posts_Per_Day",
        "Likes_Received_Per_Day",
        "Comments_Received_Per_Day",
        "Messages_Sent_Per_Day"
    ]

    categorical_cols = ["Gender", "Platform", "Dominant_Emotion"]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    data["Usage_Category"] = pd.cut(
        data["Daily_Usage_Time (minutes)"],
        bins=[-float("inf"), 60, 120, float("inf")],
        labels=["Low", "Medium", "High"],
        right=False
    )

    return data


try:
    data = load_data()

    required_cols = [
        "Age",
        "Gender",
        "Platform",
        "Daily_Usage_Time (minutes)",
        "Posts_Per_Day",
        "Likes_Received_Per_Day",
        "Comments_Received_Per_Day",
        "Messages_Sent_Per_Day",
        "Dominant_Emotion"
    ]

    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    st.sidebar.header("Dashboard Filters")

    platforms = st.sidebar.multiselect(
        "Select platforms",
        sorted(data["Platform"].dropna().unique()),
        default=sorted(data["Platform"].dropna().unique())
    )

    filtered = data[data["Platform"].isin(platforms)]

    if filtered.empty:
        st.warning("Select at least one platform.")
        st.stop()

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Users", len(filtered))
    c2.metric("Average Age", f"{filtered['Age'].mean():.1f}")
    c3.metric(
        "Average Daily Usage",
        f"{filtered['Daily_Usage_Time (minutes)'].mean():.1f} min"
    )
    c4.metric(
        "High Usage Users",
        int((filtered["Usage_Category"] == "High").sum())
    )

    st.divider()
    st.subheader("Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(filtered["Age"], bins=15, kde=True, ax=ax)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Users")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots()
        filtered["Platform"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Users by Platform")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Number of Users")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        filtered["Dominant_Emotion"].value_counts().plot(
            kind="bar",
            ax=ax
        )
        ax.set_title("Dominant Emotions")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Number of Users")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        fig, ax = plt.subplots()
        filtered["Usage_Category"].value_counts().reindex(
            ["Low", "Medium", "High"], fill_value=0
        ).plot(kind="bar", ax=ax)
        ax.set_title("Daily Usage Categories")
        ax.set_xlabel("Usage Category")
        ax.set_ylabel("Number of Users")
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("XGBoost Classification")

    st.write(
        "The model predicts Low, Medium, or High usage using demographic, "
        "platform, activity, and emotion features."
    )

    numeric_features = [
        "Age",
        "Posts_Per_Day",
        "Likes_Received_Per_Day",
        "Comments_Received_Per_Day",
        "Messages_Sent_Per_Day"
    ]

    categorical_features = [
        "Gender",
        "Platform",
        "Dominant_Emotion"
    ]

    features = numeric_features + categorical_features

    model_data = data.dropna(subset=["Usage_Category"]).copy()

    X = model_data[features]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(
        model_data["Usage_Category"].astype(str)
    )

    if len(label_encoder.classes_) < 2:
        st.error("XGBoost classification requires at least two usage categories.")
        st.stop()

    class_counts = pd.Series(y).value_counts()

    if class_counts.min() < 2:
        st.error("There are too few examples in one usage category to split the data reliably.")
        st.stop()

    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    classifier = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    st.metric("XGBoost Test Accuracy", f"{accuracy * 100:.2f}%")

    st.caption(
        "Usage categories are project-defined from daily usage minutes: "
        "Low < 60, Medium 60–119, High ≥ 120. Daily usage minutes are "
        "excluded from model inputs because they define the target."
    )

    st.write("**Classification Report**")

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose().round(3))

    st.write("**Confusion Matrix**")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(label_encoder.classes_)))
    )

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Feature Importance")

    fitted_preprocessor = classifier.named_steps["preprocessor"]
    fitted_model = classifier.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    importance = pd.Series(
        fitted_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False).head(15).sort_values()

    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Important Features")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Predict Usage Category")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=1, max_value=100, value=22)
        gender = st.selectbox(
            "Gender",
            sorted(data["Gender"].unique())
        )
        platform = st.selectbox(
            "Platform",
            sorted(data["Platform"].unique())
        )
        posts = st.number_input("Posts Per Day", min_value=0, value=2)
        likes = st.number_input("Likes Received Per Day", min_value=0, value=50)
        comments = st.number_input(
            "Comments Received Per Day",
            min_value=0,
            value=10
        )
        messages = st.number_input(
            "Messages Sent Per Day",
            min_value=0,
            value=20
        )
        emotion = st.selectbox(
            "Dominant Emotion",
            sorted(data["Dominant_Emotion"].unique())
        )

        predict_button = st.form_submit_button("Predict Usage Category")

    if predict_button:
        user_input = pd.DataFrame([{
            "Age": age,
            "Posts_Per_Day": posts,
            "Likes_Received_Per_Day": likes,
            "Comments_Received_Per_Day": comments,
            "Messages_Sent_Per_Day": messages,
            "Gender": gender,
            "Platform": platform,
            "Dominant_Emotion": emotion
        }])

        predicted_label = classifier.predict(user_input).astype(int)
        predicted_category = label_encoder.inverse_transform(predicted_label)[0]

        st.success(f"Predicted Usage Category: {predicted_category}")

    st.divider()
    st.subheader("Isolation Forest Anomaly Detection")

    st.write(
        "Isolation Forest identifies unusual activity patterns in the dataset. "
        "An anomaly is not necessarily harmful or evidence of problematic usage."
    )

    anomaly_features = [
        "Age",
        "Daily_Usage_Time (minutes)",
        "Posts_Per_Day",
        "Likes_Received_Per_Day",
        "Comments_Received_Per_Day",
        "Messages_Sent_Per_Day"
    ]

    anomaly_data = filtered[anomaly_features].copy()

    anomaly_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("isolation_forest", IsolationForest(
            contamination=0.05,
            random_state=42
        ))
    ])

    anomaly_predictions = anomaly_pipeline.fit_predict(anomaly_data)

    anomaly_results = filtered.copy()
    anomaly_results["Anomaly_Status"] = [
        "Anomaly" if result == -1 else "Normal"
        for result in anomaly_predictions
    ]

    normal_count = int((anomaly_results["Anomaly_Status"] == "Normal").sum())
    anomaly_count = int((anomaly_results["Anomaly_Status"] == "Anomaly").sum())

    a1, a2 = st.columns(2)
    a1.metric("Normal Activity Patterns", normal_count)
    a2.metric("Unusual Activity Patterns", anomaly_count)

    fig, ax = plt.subplots()
    anomaly_results["Anomaly_Status"].value_counts().plot(
        kind="bar",
        ax=ax
    )
    ax.set_title("Isolation Forest Results")
    ax.set_xlabel("Activity Status")
    ax.set_ylabel("Number of Users")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        anomaly_results[
            ["User_ID", "Platform", "Daily_Usage_Time (minutes)", "Anomaly_Status"]
        ] if "User_ID" in anomaly_results.columns else anomaly_results[
            ["Platform", "Daily_Usage_Time (minutes)", "Anomaly_Status"]
        ]
    )

    csv_data = anomaly_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Analysis Results as CSV",
        data=csv_data,
        file_name="social_media_analysis_results.csv",
        mime="text/csv"
    )

except FileNotFoundError:
    st.error("train.csv was not found. Place it in the same folder as app.py.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info("Check the error message and verify that train.csv contains the expected columns.")
