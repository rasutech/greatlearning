import pandas as pd
import re
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load your data into a pandas dataframe
df = pd.read_csv('tickets_data.csv')

# Inspect the data for missing values in 'ticket_description'
missing_descriptions = df['ticket_description'].isnull().sum()
print(f"Number of missing ticket descriptions: {missing_descriptions}")

# Handle missing values by replacing NaN with empty strings
df['ticket_description'] = df['ticket_description'].fillna('')

# Alternatively, drop rows with missing descriptions
# df = df.dropna(subset=['ticket_description'])

# Ensure the 'ticket_description' column is of string type
df['ticket_description'] = df['ticket_description'].astype(str)

# Function for basic text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        text = ''
    # Lowercase the text
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Apply preprocessing to the ticket descriptions
df['cleaned_description'] = df['ticket_description'].apply(preprocess_text)

# Verify preprocessing
print(df[['ticket_description', 'cleaned_description']].head())

# Load a pre-trained transformer model (Sentence-BERT is recommended for embeddings)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose a different model if preferred

# Generate embeddings for the cleaned ticket descriptions
ticket_embeddings = model.encode(df['cleaned_description'].tolist(), show_progress_bar=True)

# Normalize the embeddings for better clustering
ticket_embeddings = normalize(ticket_embeddings)

# Optional: Dimensionality Reduction using PCA
pca = PCA(n_components=50, random_state=42)
reduced_embeddings = pca.fit_transform(ticket_embeddings)

# Optional: Determine the optimal number of clusters using the Elbow Method
def plot_elbow_method(embeddings, max_clusters=20):
    sse = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10,6))
    plt.plot(range(2, max_clusters+1), sse, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

# Uncomment the line below to plot and determine the optimal number of clusters
# plot_elbow_method(reduced_embeddings, max_clusters=20)

# For demonstration, let's proceed with a predefined number of clusters
num_clusters = 10  # Adjust based on the Elbow Method or your specific needs

# Clustering using KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(reduced_embeddings)

# Function to analyze clusters
def analyze_clusters(df, num_clusters):
    for cluster_num in range(num_clusters):
        print(f"\nCluster {cluster_num}:")
        cluster_data = df[df['cluster'] == cluster_num]
        print(f"Number of tickets in this cluster: {len(cluster_data)}")
        # Display top 5 ticket descriptions in the cluster
        print(cluster_data['ticket_description'].head(5).values)
        # Optionally, display the assignment groups or resolution codes
        print("Assignment Groups:", cluster_data['assignment_group'].unique())
        print("Resolution Codes:", cluster_data['resolution_code'].unique())

# Display clustering results
analyze_clusters(df, num_clusters)

# Optional: Visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=50)
tsne_embeddings = tsne.fit_transform(reduced_embeddings)

plt.figure(figsize=(12,8))
scatter = plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c=df['cluster'], cmap='viridis')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title('t-SNE Visualization of Ticket Clusters')
plt.show()

# Optionally, save the clustered data to a CSV
df.to_csv('clustered_tickets.csv', index=False)
print("Clustered data saved to 'clustered_tickets.csv'.")

# Save the KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')
print("KMeans model saved to 'kmeans_model.pkl'.")
