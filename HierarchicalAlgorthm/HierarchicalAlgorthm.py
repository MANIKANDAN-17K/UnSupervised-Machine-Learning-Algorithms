# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Create song data
# Features: Energy (0-10), Danceability (0-10)

songs_data = [
    # High Energy Dance (EDM/Pop)
    [9, 9, 'Uptown Funk'],
    [10, 10, 'Blinding Lights'],
    [9, 8, 'Levitating'],
    
    # Medium Energy Pop
    [6, 7, 'Shape of You'],
    [7, 6, 'Someone Like You'],
    [6, 6, 'Perfect'],
    
    # Low Energy Ballads
    [2, 2, 'Fix You'],
    [1, 1, 'Hallelujah'],
    [2, 3, 'The Scientist'],
    
    # High Energy Rock
    [9, 4, 'Bohemian Rhapsody'],
    [10, 3, 'Sweet Child O Mine'],
    [8, 4, 'Smells Like Teen Spirit'],
    
    # Chill/Acoustic
    [3, 5, 'Wonderwall'],
    [4, 4, 'Thinking Out Loud'],
    [3, 4, 'Let Her Go']
]

# Separate features and names
songs = []
song_names = []
for item in songs_data:
    songs.append([item[0], item[1]])
    song_names.append(item[2])

X = np.array(songs)

# Perform hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Create the dendrogram
plt.figure(figsize=(14, 8))

# Plot dendrogram
dendrogram(linkage_matrix, 
          labels=song_names,
          leaf_rotation=90,
          leaf_font_size=10,
          color_threshold=15)

plt.title('Spotify Playlist Hierarchy - How Songs Group Together', 
         fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Songs', fontsize=12, fontweight='bold')
plt.ylabel('Distance (How Different Songs Are)', fontsize=12, fontweight='bold')

# Add cutting lines for different playlist options
plt.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Cut for 4 Playlists')
plt.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Cut for 6 Playlists')
plt.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Cut for 10+ Playlists')

plt.legend(fontsize=11, loc='upper right')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Apply hierarchical clustering with 4 clusters
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
clusters = hierarchical.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(12, 8))

colors = ['red', 'blue', 'green', 'purple']
cluster_names = ['Party Hits', 'Chill Vibes', 'Power Ballads', 'Rock Classics']

for i in range(4):
    cluster_songs = X[clusters == i]
    plt.scatter(cluster_songs[:, 0], cluster_songs[:, 1],
               c=colors[i], s=300, alpha=0.6, edgecolors='black', linewidth=2,
               label=cluster_names[i])

# Add song names as labels
for i, name in enumerate(song_names):
    plt.annotate(name, (X[i, 0], X[i, 1]),
                fontsize=9, ha='center', va='bottom', fontweight='bold')

plt.xlabel('Energy Level (0-10)', fontsize=14, fontweight='bold')
plt.ylabel('Danceability (0-10)', fontsize=14, fontweight='bold')
plt.title('Spotify Auto-Generated Playlists', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper left')
plt.grid(alpha=0.3)
plt.xlim(-1, 11)
plt.ylim(-1, 11)

plt.tight_layout()
plt.show()

# Print results
print("="*70)
print("SPOTIFY PLAYLIST ORGANIZATION - HIERARCHICAL CLUSTERING")
print("="*70)

for i in range(4):
    print(f"\n📱 {cluster_names[i]} Playlist:")
    cluster_indices = np.where(clusters == i)[0]
    for idx in cluster_indices:
        print(f"   ♪ {song_names[idx]} - Energy: {X[idx][0]}, Dance: {X[idx][1]}")

# Show the hierarchy
print("\n" + "="*70)
print("HOW HIERARCHICAL CLUSTERING WORKS:")
print("="*70)

print("\n🎵 STEP-BY-STEP GROUPING:")
print("\n1. START: Every song is alone")
print("   • 'Blinding Lights', 'Uptown Funk', 'Hallelujah', etc.")

print("\n2. FIRST MERGE: Most similar songs join")
print("   • 'Blinding Lights' + 'Uptown Funk' → High Energy Dance")
print("   • 'Fix You' + 'Hallelujah' → Sad Ballads")

print("\n3. KEEP MERGING: Groups combine into bigger groups")
print("   • High Energy Dance + 'Levitating' → Party Hits")
print("   • Sad Ballads + 'The Scientist' → Power Ballads")

print("\n4. FINAL RESULT: 4 main playlists created!")

print("\n" + "="*70)
print("WHY HIERARCHICAL > K-MEANS FOR SPOTIFY:")
print("="*70)
print("\n✅ See the TREE structure (dendrogram) - shows song relationships")
print("✅ Don't need to decide playlist count beforehand")
print("✅ Can cut at different levels:")
print("   • High cut (red line) = 4 broad playlists (Party, Chill, Ballads, Rock)")
print("   • Low cut (green line) = 10+ specific playlists (90s Pop, Acoustic, etc.)")
print("\n✅ Shows which songs are 'cousins' vs 'siblings'")
print("   • 'Uptown Funk' and 'Blinding Lights' are SIBLINGS (very similar)")
print("   • 'Uptown Funk' and 'Shape of You' are COUSINS (somewhat similar)")

print("\n" + "="*70)
print("REAL SPOTIFY USE:")
print("="*70)
print("\n🎧 When you create a playlist with mixed songs,")
print("   Spotify uses hierarchical clustering to:")
print("   • Group similar songs together")
print("   • Create smooth transitions between sections")
print("   • Suggest similar songs to add")
print("   • Auto-generate 'Daily Mix' playlists")
