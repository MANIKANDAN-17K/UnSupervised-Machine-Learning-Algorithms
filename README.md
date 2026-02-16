📊 Unsupervised Machine Learning Algorithms – Practical Implementation
📌 Project Overview

This project demonstrates the implementation of three major Unsupervised Machine Learning algorithms using real-world inspired examples:

🔵 K-Means Clustering (Movie Recommendation System)

🌳 Hierarchical Clustering (Spotify Playlist Generator)

🛒 Apriori Algorithm (Amazon Market Basket Analysis)

The purpose of this project is to understand how clustering and association rule mining are used in real-world recommendation systems and business analytics.

🔵 1️⃣ K-Means Clustering – Movie Recommendation
🎬 Problem Statement

Group movies based on two features:

Action Level (0–10)

Romance Level (0–10)

The algorithm automatically clusters similar movies together.

🧠 Algorithm Steps

Select number of clusters (K)

Initialize centroids

Assign each movie to nearest centroid (Euclidean distance)

Recalculate centroid as mean of cluster

Repeat until convergence

📈 Output

2D cluster visualization

Centroid representation

Automatic category grouping

💡 Real-World Applications

Movie & content recommendations

Customer segmentation

Image compression

🌳 2️⃣ Hierarchical Clustering – Spotify Playlist Generator
🎵 Problem Statement

Organize songs into playlists based on:

Energy Level

Danceability

🧠 Algorithm Type

Agglomerative Hierarchical Clustering (Ward’s Method)

⚙️ How It Works

Each song starts as an individual cluster

Closest clusters merge step-by-step

A dendrogram (tree diagram) shows the hierarchy

Cutting the tree at different heights produces different playlist counts

📈 Output

Dendrogram visualization

Clustered playlist map

Multi-level grouping

💡 Real-World Applications

Music recommendation systems

Document similarity grouping

Biological data clustering

🛒 3️⃣ Apriori Algorithm – Market Basket Analysis
🛍 Problem Statement

Analyze customer shopping carts to discover frequently bought together products.

📊 Key Metrics

Support – Frequency of item occurrence

Confidence – Probability of buying B if A is bought

Lift – Strength of relationship beyond random chance

⚙️ Process

Identify frequent itemsets (using minimum support)

Generate association rules

Evaluate rules using confidence & lift

Visualize strong buying patterns

📈 Output

Frequent item analysis

Strong association rules

Business insight visualizations

💡 Real-World Applications

Amazon “Frequently Bought Together”

Cross-selling strategies

Bundle offers

Inventory optimization

🛠 Technologies Used

Python 3

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

mlxtend

📂 Project Structure
├── kmeansAlgorithm.py
├── hierarchicalAlgorithm.py
├── aprioriAlgorithm.py
├── README.md

🚀 How to Run
1️⃣ Install Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend

2️⃣ Run Scripts
python kmeans_movie.py
python hierarchical_spotify.py
python apriori_amazon.py

📊 Algorithm Comparison
Algorithm	Category	Purpose	Requires K?
K-Means	Clustering	Group similar data	Yes
Hierarchical	Clustering	Build similarity tree	No
Apriori	Association	Discover buying patterns	No
🎯 Learning Outcomes

Understanding unsupervised learning concepts

Visualizing clustering algorithms

Implementing recommendation logic

Analyzing transaction data using association rules

Interpreting support, confidence, and lift

👨‍💻 Author

Manikandan
Computer Science Student
