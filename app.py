import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

sns.set_theme(style='darkgrid', palette='viridis')

# ─── PAGE SETUP ───
st.set_page_config(page_title="Elimu Analytics", layout="wide", page_icon="🎓")
st.title("🎓 Elimu Analytics")
st.caption("Automated Academic Performance & ML Risk Report")

# ─── UPLOAD ───
uploaded = st.file_uploader("Upload marks CSV", type=["csv"])
if uploaded is None:
    st.info("👆 Upload a CSV to begin")
    st.stop()

# ───  CACHE DATA, NO RELOADS ON EVERY RERUNS ───
@st.cache_data
def load_and_train(file):
    df = pd.read_csv(file)
    
    # Clean
    df = df.drop_duplicates().dropna()
    
    # Features
    df["percentage"] = (df["marks"] / df["out_of"] * 100).round(1)
    df["exam_num"] = (df["term"] - 1) * 2 + df["exam_type"].map({"Opener": 1, "End Term": 2})
    
    le_dict = {}
    for col in ['class', 'gender', 'subject']:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le

    df['At_Risk'] = (df["percentage"] < 50).astype(int)

    features = ['class_encoded', 'term', 'gender_encoded', 'subject_encoded', 'exam_num']
    X = df[features]
    y = df['At_Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return df, model, features, acc

df, model, features, accuracy = load_and_train(uploaded)

# ─── KPI CARDS ───
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Class Mean", f"{df['percentage'].mean():.1f}%")
col2.metric("Students", df["student_id"].nunique())
col3.metric("Subjects", df["subject"].nunique())
col4.metric("Exams", df["exam_num"].nunique())
col5.metric("Total At Risk", df["At_Risk"].sum())
col6.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.divider()

# ─── TOP STUDENT ───
st.subheader("🏆 Top Student")
# Find the student with the highest average
top_student_name = df.groupby("name")["percentage"].mean().idxmax()
top_student_id = df[df["name"] == top_student_name]["student_id"].iloc[0]
top_student_avg = df[df["name"] == top_student_name]["percentage"].mean()

st.success(f"**{top_student_name}** (ID: {top_student_id}) - Overall Average: **{top_student_avg:.1f}%**")

# Show their specific subject breakdown
top_marks = df[df["student_id"] == top_student_id]
modified = top_marks.groupby("subject")["percentage"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 3))
sns.barplot(data=modified, x="subject", y="percentage", ax=ax, hue="subject", palette="viridis", legend=False)
st.pyplot(fig)

# ─── TABS ───
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends", "⚠️ At-Risk (ML)", "Top 10 Students", "Gender Performance"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="subject", y="percentage", ax=ax, hue="subject", palette="viridis", legend=False)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x="exam_num", y="percentage", marker="o", ax=ax, color="#4ade80")
    st.pyplot(fig)

with tab3:
    st.subheader("🚨 Machine Learning Risk Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Most Important Features (Why do students fail?)**")
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(data=imp_df, x='Importance', y='Feature', palette='rocket', ax=ax)
        st.pyplot(fig)

    with col_b:
        st.write("**Students needing help in Mathematics**")
        math_at_risk = df[(df["subject"] == 'Mathematics') & (df['At_Risk'] == 1)]
        st.dataframe(math_at_risk[['name', 'class', 'percentage']].sort_values(by='percentage').head(10), use_container_width=True)

with tab4:
    # Sort values before getting head(10)
    top_students = df.groupby(["student_id", "name"])["percentage"].mean().sort_values(ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_students, x="percentage", y="name", ax=ax, palette="viridis", errorbar=None)
    st.pyplot(fig)

with tab5:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="subject", y="percentage", ax=ax, hue="gender", palette="viridis", errorbar=None)
    ax.legend(facecolor="#0e1117", labelcolor="white")
    st.pyplot(fig)