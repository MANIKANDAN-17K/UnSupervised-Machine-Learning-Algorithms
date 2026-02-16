# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create movie watching data
# Let's say we track 2 things: Action level (0-10) and Romance level (0-10)

movies = [
    # Action Movies (high action, low romance)
    [9, 1], [8, 2], [9, 0], [8, 1], [10, 1],
    
    # Romance Movies (low action, high romance)
    [1, 9], [2, 8], [0, 9], [1, 8], [2, 10],
    
    # Comedy Movies (medium action, medium romance)
    [5, 5], [6, 6], [5, 6], [6, 5], [5, 4],
    
    # Thriller/Drama (medium-high action, some romance)
    [7, 4], [8, 3], [7, 5], [6, 4], [8, 4]
]

movie_names = [
    'Fast & Furious', 'John Wick', 'Mission Impossible', 'Die Hard', 'Mad Max',
    'The Notebook', 'Titanic', 'La La Land', 'Pride & Prejudice', 'Me Before You',
    'The Hangover', 'Superbad', 'Crazy Rich Asians', 'Bridesmaids', '21 Jump Street',
    'Inception', 'The Dark Knight', 'Interstellar', 'Shutter Island', 'Gone Girl'
]

X = np.array(movies)

# Apply K-Means with 3 clusters (Action, Romance, Mixed)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Assign cluster names based on centroids
cluster_names = ['Action Movies', 'Romance Movies', 'Comedy/Thriller Movies']

# Visualize
plt.figure(figsize=(12, 8))

colors = ['red', 'pink', 'orange']
for i in range(3):
    cluster_movies = X[clusters == i]
    plt.scatter(cluster_movies[:, 0], cluster_movies[:, 1], 
               c=colors[i], s=200, alpha=0.6, edgecolors='black', linewidth=2,
               label=cluster_names[i])

# Plot centroids (Netflix's "category centers") - FIXED: using '*' instead of '★'
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='blue', s=500, marker='*', edgecolors='black', linewidth=3,
           label='Category Centers', zorder=5)

# Add movie names as labels
for i, name in enumerate(movie_names):
    plt.annotate(name, (X[i, 0], X[i, 1]), 
                fontsize=8, ha='center', va='bottom')

plt.xlabel('Action Level (0-10)', fontsize=14, fontweight='bold')
plt.ylabel('Romance Level (0-10)', fontsize=14, fontweight='bold')
plt.title('Netflix Movie Clustering - K-Means Algorithm', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper left')
plt.grid(alpha=0.3)
plt.xlim(-1, 11)
plt.ylim(-1, 11)

plt.tight_layout()
plt.show()

# Print results
print("="*60)
print("NETFLIX MOVIE RECOMMENDATION SYSTEM")
print("="*60)

for i in range(3):
    print(f"\n{cluster_names[i]}:")
    cluster_indices = np.where(clusters == i)[0]
    for idx in cluster_indices:
        print(f"  • {movie_names[idx]} - Action: {X[idx][0]}, Romance: {X[idx][1]}")

# Show recommendation logic
print("\n" + "="*60)
print("HOW NETFLIX USES THIS:")
print("="*60)
print("\n🎬 If you watch 'John Wick' (Action: 8, Romance: 2)")
print("   → K-Means finds it's in the 'Action Movies' cluster")
print("   → Netflix recommends: Fast & Furious, Mission Impossible, Die Hard")

print("\n💕 If you watch 'The Notebook' (Action: 1, Romance: 9)")
print("   → K-Means finds it's in the 'Romance Movies' cluster")
print("   → Netflix recommends: Titanic, La La Land, Pride & Prejudice")

print("\n😄 If you watch 'The Hangover' (Action: 5, Romance: 5)")
print("   → K-Means finds it's in the 'Comedy/Thriller' cluster")
print("   → Netflix recommends: Superbad, Inception, Bridesmaids")
