import numpy as np

def recall_at_k(recommended_movies, relevant_movies, k):
    recommended_top_k = recommended_movies[:k]
    relevant_and_recommended = set(recommended_top_k) & set(relevant_movies)
    return len(relevant_and_recommended) / len(relevant_movies) if len(relevant_movies) > 0 else 0

def precision_at_k(recommended_movies, relevant_movies, k):
    recommended_top_k = recommended_movies[:k]
    relevant_and_recommended = set(recommended_top_k) & set(relevant_movies)
    return len(relevant_and_recommended) / k

def calculate_metrics_at_k_100_users(user_ids, user_movie_matrix_sparse, recommendations, k_values):
    metrics_scores = {k: {"recall": [], "precision": []} for k in k_values}

    for user_id in user_ids:
        # Get the movies watched by the user
        user_ratings = user_movie_matrix_sparse[user_id].toarray().flatten()
        relevant_movies = np.where(user_ratings > 0)[0].tolist()
        
        # Get the movies recommended to the user
        recommended_movies = (
    [recommendations[user_id]]
    if isinstance(recommendations[user_id], int)
    else [movie[0] for movie in recommendations[user_id]]
)
        
        # Calculate Recall@K and Precision@K
        for k in k_values:
            recall = recall_at_k(recommended_movies, relevant_movies, k)
            precision = precision_at_k(recommended_movies, relevant_movies, k)
            metrics_scores[k]["recall"].append(recall)
            metrics_scores[k]["precision"].append(precision)
    
    # Calculate the average Recall and Precision for each K
    mean_metrics_scores = {
        k: {
            "recall": np.mean(metrics_scores[k]["recall"]),
            "precision": np.mean(metrics_scores[k]["precision"])
        }
        for k in k_values
    }
    return mean_metrics_scores