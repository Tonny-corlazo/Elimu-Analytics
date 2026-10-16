import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    context='notebook',
    style='darkgrid',
    palette='deep',
    font='sans-serif',
    font_scale=1,
    color_codes=True,
    rc=None,
)

# ─── PAGE SETUP ───
st.set_page_config(page_title="Elimu Analytics", layout="wide")

# ─── TITLE ───
st.title("🎓 Elimu Analytics")
st.write("Upload a class marks CSV to generate the report.")

# ─── UPLOAD ───
uploaded = st.file_uploader("Upload marks CSV", type=["csv"])

if uploaded is None:
    st.info("👆 Upload a CSV to begin")
    st.stop()   # stops the script here until a file is uploaded

# ─── LOAD + ENRICH ───
df = pd.read_csv(uploaded)

# ==========  clean the dataframe ==============
df.isna().sum()

df.drop_duplicates()
df.dropna()

df["percentage"] = (df["marks"] / df["out_of"] * 100).round(1)
df["exam_num"] = (df["term"] - 1) * 2 + df["exam_type"].map(
    {"Opener": 1, "End Term": 2}
)

# ─── KPI CARDS ───
col1, col2, col3, col4 = st.columns(4)
col1.metric("Class Mean", f"{df['percentage'].mean():.1f}%")
col2.metric("Students", df["student_id"].nunique())
col3.metric("Subjects", df["subject"].nunique())
col4.metric("Exams", df["exam_num"].nunique())

st.divider()

# ─── TABS ───
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends", "⚠️ At-Risk", "Top Students", "Gender Perfomance"])

with tab1:
    st.subheader("Subject Performance")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="subject", y="percentage",
                ax=ax, errorbar=None, hue="subject", legend=False,
                palette="viridis")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

with tab2:
    st.subheader("Class Progression")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x="exam_num", y="percentage",
                 marker="o", ax=ax, color="#4ade80")
    st.pyplot(fig)

with tab3:
    st.subheader("Students Below 50% in Recent Exams")
    recent = df[df["exam_num"].isin([5, 6])]
    at_risk = (
        recent.groupby("name")["percentage"]
        .mean().sort_values()
    )
    at_risk = at_risk[at_risk < 50]
    if len(at_risk) == 0:
        st.success("No at-risk students 🎉")
    else:
        st.dataframe(at_risk.reset_index().rename(
            columns={"percentage": "Recent Mean %"}
        ))

with tab4:
    st.subheader("Top 10 Perfoming Students")

    top_students = (
        df.groupby(["student_id", "name"])["percentage"]
        .mean().head(10).reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_students, x="percentage", y="name", ax=ax, palette="viridis", errorbar=None)
    ax.set_title("Top Students")
    ax.set_ylabel("Name")
    ax.set_xlabel("Percentage")

    st.pyplot(fig)

with tab5:
    st.subheader("Gender Perfomance")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="subject", y="percentage", 
            ax=ax,
            palette="viridis",
            alpha=0.9,
            errorbar=None,
            hue="gender"
           )

    ax .set_title("Subject Perfomance Gender Based", weight="bold")
    ax.set_xlabel("Subect")
    ax.set_ylabel("Percentage")
    ax.legend(facecolor="#0e1117", labelcolor="white")
    ax.tick_params(axis="x", rotation=30)

    st.pyplot(fig)