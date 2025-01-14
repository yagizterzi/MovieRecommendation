import matplotlib.pyplot as plt

def plot_metrics_at_k(metrics_scores):

    k_values = list(metrics_scores.keys())
    recall_scores = [metrics_scores[k]["recall"] for k in k_values]
    precision_scores = [metrics_scores[k]["precision"] for k in k_values]

    plt.figure(figsize=(10, 6))

    # Recall@K plot
    plt.plot(k_values, recall_scores, marker='o', linestyle='-', color='g', label='Recall@K')

    # Precision@K plot
    plt.plot(k_values, precision_scores, marker='o', linestyle='--', color='b', label='Precision@K')

    plt.title('Recall@K and Precision@K Plot')
    plt.xlabel('K Values')
    plt.ylabel('Score')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.xticks(k_values)
    plt.show()
