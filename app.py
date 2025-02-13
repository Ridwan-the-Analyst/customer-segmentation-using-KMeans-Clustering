import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Streamlit App Title with improved UI
st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.markdown(
    """
    <style>
        .main {background-color: #f5f5f5;}
        h1 {color: #4CAF50; text-align: center;}
    </style>
    """, unsafe_allow_html=True
)

st.title("📊 Customer Segmentation using K-Means Clustering")
st.markdown("---")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format)", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Select numerical features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        feature_1 = st.selectbox("🔹 Select first feature", numeric_features, index=0)
    with col2:
        feature_2 = st.selectbox("🔹 Select second feature", numeric_features, index=1)
    
    selected_features = df[[feature_1, feature_2]]

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)

    # Determine optimal clusters using Elbow Method
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    # Plot Elbow Method
    st.subheader("📈 Elbow Method for Optimal K")
    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bo-')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)
    
    # User selects number of clusters
    k_value = st.slider("🎯 Select the number of clusters", min_value=2, max_value=10, value=3, step=1)
    
    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=k_value, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Display Cluster Results
    st.subheader("📊 Clustered Data")
    st.dataframe(df[[feature_1, feature_2, 'Cluster']].head())
    
    # Visualize Clusters
    st.subheader("🎨 Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature_1], y=df[feature_2], hue=df['Cluster'], palette="Set1", s=100, ax=ax)
    ax.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
               kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
               s=300, c='black', marker='X', label='Centroids')
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title('Customer Segments')
    st.pyplot(fig)
    
    # Cluster Insights
    st.subheader("📊 Cluster Insights & Interpretation")
    cluster_descriptions = {
        0: "🟢 Cluster 0: High-income, high-spending customers - likely premium customers.",
        1: "🔵 Cluster 1: Low-income, high-spending customers - potential impulsive buyers.",
        2: "🟠 Cluster 2: High-income, low-spending customers - cautious or investment-focused buyers.",
        3: "🔴 Cluster 3: Low-income, low-spending customers - budget-conscious customers."
    }

    for cluster, description in cluster_descriptions.items():
        st.write(f"**Cluster {cluster}:** {description}")
    
    # Download Results
    st.subheader("📥 Download Clustered Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇ Download CSV", data=csv, file_name="customer_segments.csv", mime="text/csv")
