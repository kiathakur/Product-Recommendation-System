import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS (PREMIUM UI)
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}
.card {
    padding: 18px;
    border-radius: 15px;
    background: rgba(255,255,255,0.85);
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 10px;
}
.rank {
    font-size: 13px;
    color: gray;
}
.score {
    color: #4f46e5;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    with open("train_matrix.pkl", "rb") as f:
        user_item_matrix = pickle.load(f)

    with open("item_similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    with open("user_cluster_df.pkl", "rb") as f:
        clusters = pickle.load(f)

    return user_item_matrix, similarity, clusters

user_item_matrix, similarity, clusters = load_data()

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend(user, n=5):
    user_vec = user_item_matrix.loc[user]

    scores = similarity.dot(user_vec)
    scores = scores.sort_values(ascending=False)

    seen = user_vec[user_vec > 0].index
    scores = scores.drop(seen)

    return scores.head(n)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Controls")

selected_user = st.sidebar.selectbox(
    "Select User ID",
    user_item_matrix.index
)

top_n = st.sidebar.slider("Top Recommendations", 1, 10, 5)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("## 🎯 AI Recommendation System")
st.caption("Personalized recommendations using clustering + similarity")

st.divider()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "👤 Recommendations", "📊 Insights"])

# -----------------------------
# DASHBOARD
# -----------------------------
with tab1:
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", user_item_matrix.shape[0])
    col2.metric("Total Products", user_item_matrix.shape[1])

    sparsity = (user_item_matrix == 0).sum().sum() / user_item_matrix.size
    col3.metric("Sparsity", f"{round(sparsity*100,2)}%")

    st.markdown("### 🔥 Top Products (Most Interacted)")
    top_products = user_item_matrix.sum().sort_values(ascending=False).head(10)
    st.dataframe(top_products.rename("Total Interactions"))

    # ✅ NEW: Top Active Users
    st.markdown("### 🔝 Top Active Users")
    user_activity = (user_item_matrix > 0).sum(axis=1)
    top_users = user_activity.sort_values(ascending=False).head(5)
    st.bar_chart(top_users)

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
with tab2:
    st.subheader("👤 User Profile")

    # Safe cluster extraction
    try:
        cluster = clusters[clusters['user_id'] == selected_user]['cluster'].values[0]
    except:
        cluster = "N/A"

    c1, c2 = st.columns(2)
    c1.metric("User ID", selected_user)
    c2.metric("Cluster Group", cluster)

    st.subheader("📌 Recommended Products")

    recs = recommend(selected_user, top_n)

    for i, (item, score) in enumerate(recs.items(), start=1):
        st.markdown(f"""
        <div class="card">
            <div class="rank">#{i} Recommendation</div>
            <h4>Product ID: {item}</h4>
            <div class="score">Score: {round(score,3)}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(float(score), 1.0))

# -----------------------------
# INSIGHTS
# -----------------------------
with tab3:
    st.subheader("📊 Data Insights")

    st.write("### User-Item Matrix Sample")
    st.dataframe(user_item_matrix.head())

    st.write("### Similarity Matrix Sample")
    st.dataframe(similarity.head())

    # ✅ NEW: Cluster Distribution + Count
    st.write("### 👥 Users per Cluster")
    cluster_counts = clusters['cluster'].value_counts().sort_index()

    st.bar_chart(cluster_counts)
    st.dataframe(cluster_counts.rename("User Count"))

    # ✅ NEW: Top Users Full List
    st.write("### 🔝 Most Active Users (Top 10)")
    user_activity = (user_item_matrix > 0).sum(axis=1)
    top_users_full = user_activity.sort_values(ascending=False).head(10)
    st.dataframe(top_users_full.rename("Number of Ratings"))

    # MODEL EXPLANATION
    with st.expander("🧠 Model Explanation"):
        st.write("""
        - Item-based Collaborative Filtering
        - Cosine similarity between products
        - KMeans clustering for user segmentation
        - Sparse matrix optimization for performance
        """)
