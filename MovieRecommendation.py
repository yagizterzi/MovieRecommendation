import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from metrics import calculate_metrics_at_k_100_users
from Visuals import plot_metrics_at_k(


# Load the ratings and movies data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Create a sparse user-movie matrix
user_movie_matrix_sparse = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))

# Calculate cosine similarity between users using NearestNeighbors for efficiency
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix_sparse)

target_user = input("Enter the user ID: ")
target_user = int(target_user)

top_n = input("How many movies: ")
top_n = int(top_n)

n_neighbors = top_n

def recommend_movies(user_id, model_knn, user_movie_matrix_sparse, movies, top_n):
    print(f"Generating recommendations for user {user_id}...")
    user_ratings = user_movie_matrix_sparse[user_id].toarray().flatten()
    unseen_movies = (user_ratings == 0).nonzero()[0]

    distances, indices = model_knn.kneighbors(user_movie_matrix_sparse[user_id], n_neighbors)
    similar_users = pd.Series(distances.flatten(), index=indices.flatten())
    similar_users = similar_users[similar_users > 0]

    weighted_sum = user_movie_matrix_sparse[similar_users.index].T.dot(1 - similar_users.values)
    similarity_sum = (1 - similar_users.values).sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        recommendations = np.true_divide(weighted_sum, similarity_sum)
        recommendations[~np.isfinite(recommendations)] = 0

    recommendations = pd.Series(recommendations[unseen_movies], index=unseen_movies).sort_values(ascending=False)
    
    # Return only movieId as a list
    return recommendations.index.tolist()

# Extract year from the title and handle NaN values
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['year'] = movies['year'].fillna(0).astype(int)

recommended_movies = recommend_movies(target_user, model_knn, user_movie_matrix_sparse, movies, top_n)
print(recommended_movies)

# Define K values
k_values = [5, 10, 15, 20]
user_ids = range(min(10, user_movie_matrix_sparse.shape[0]))  # Only the first 100 users

# Recommendations (example)
print("Generating recommendations for the first 100 users...")
recommendations = {}
for user_id in user_ids:
    try:
        recommendations[user_id] = recommend_movies(user_id, model_knn, user_movie_matrix_sparse, movies, top_n=20)
    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {e}")
print("Recommendations generated.")

# Calculate Recall@K and Precision@K values
print("Calculating metrics...")
metrics_scores = calculate_metrics_at_k_100_users(
    user_ids=user_ids, 
    user_movie_matrix_sparse = user_movie_matrix_sparse,
    recommendations=recommendations, 
    k_values=k_values
)
print("Metrics calculated.")

# Plot the metric graph
print("Plotting metrics...")
plot_metrics_at_k(metrics_scores)
print("Plotting done.")
