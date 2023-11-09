"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import pairwise_distances
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="zelnik4.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
inerties = []
runtimes=[]
tps1 = time.time()
#INERTIE 


#on fait varier K pour avoir l'inertie en fonction des clusters 
for k in range(1,21) : 
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    inerties.append(inertie)
    runtime = (tps2 - tps1) * 1000  # Temps d'exécution en millisecondes
    runtimes.append(runtime)


# Plot the inertia values
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Inertia', color=color)
ax1.plot(range(1, 21), inerties, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Runtime (ms)', color=color)
ax2.plot(range(1, 21), runtimes, marker='x', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Inertia vs. Number of Clusters with Runtime')
plt.show()

#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
#print(dists)

#CALCUL SCORE DE REGROUPEMENT 
min_distances = []
max_distances = []
mean_distances = []

for i in range(4):
    # Get the points belonging to cluster i
    cluster_points = datanp[labels == i]
   
    # Get the corresponding centroid
    centroid = centroids[i]
   
    # Calculate distances from the centroid to points in the cluster
    distances = euclidean_distances(cluster_points, [centroid])
   
    # Calculate min, max, and mean distances
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    mean_distance = np.mean(distances)
   
    min_distances.append(min_distance)
    max_distances.append(max_distance)
    mean_distances.append(mean_distance)



# plot to visualize the scores
cluster_labels = [f"Cluster {i+1}" for i in range(4)]

plt.bar(cluster_labels, min_distances, label="Min Distance")
plt.bar(cluster_labels, max_distances, label="Max Distance", alpha=0.6)
plt.bar(cluster_labels, mean_distances, label="Mean Distance", alpha=0.6)

plt.xlabel("Clusters")
plt.ylabel("Distances")
plt.title("Scores de Regroupement")

# Ajouter des annotations pour afficher les valeurs sur le graphique
for i, v in enumerate(min_distances):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
for i, v in enumerate(max_distances):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
for i, v in enumerate(mean_distances):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

plt.legend()
plt.show()



#CALCUL SCORE DE SEPARATION 
# Calculate pairwise distances between centroids
centroid_distances = pairwise_distances(centroids)

min_separation = np.inf
max_separation = -np.inf
sum_separation = 0
count = 0
separation_values = []
for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        distance = centroid_distances[i, j]
       
        min_separation = min(min_separation, distance)
        max_separation = max(max_separation, distance)
        sum_separation += distance
        count += 1
        separation_values.append(distance)

average_separation = sum_separation / count

print(f"Minimum separation = {min_separation}, Maximum separation = {max_separation}, Average separation = {average_separation}")

# Affichage des résultats dans un graphique
plt.figure(figsize=(8, 6))
plt.bar(["Minimum", "Maximum", "Average"], [min_separation, max_separation, average_separation])
plt.xlabel("Separation Measure")
plt.ylabel("Distance")
plt.title("Centroid Separation Scores")
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# List to store scores for different values of k
silhouette_scores = []
Davies_scores = []
Calinski_scores = []
runtimes_silhouette = []
runtimes_davies = []
runtimes_calinski = []

# Iterate through different values of k

for k in range(2, 11):
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    tps1_score = time.time()
    model.fit_predict(datanp)
    tps2_score = time.time()
    labels = model.labels_
    
    # Calculate silhouette score for the current clustering
    silhouette_avg = silhouette_score(datanp, labels)
    silhouette_scores.append(silhouette_avg)
    runtime_silhouette = (tps2_score - tps1_score) * 1000 
    runtimes_silhouette.append(runtime_silhouette)

    # Calculate Davies bouldin score 
    Davies_avg = davies_bouldin_score(datanp, labels)
    Davies_scores.append(Davies_avg) 
    runtime_davies = (tps2_score - tps1_score) * 1000  # Temps d'exécution en millisecondes
    runtimes_davies.append(runtime_davies)
   
    # Calculate Calinski Harbasz score
    Calinski_avg = calinski_harabasz_score(datanp, labels)
    Calinski_scores.append(Calinski_avg)
    runtime_calinski = (tps2_score - tps1_score) * 1000  # Temps d'exécution en millisecondes
    runtimes_calinski.append(runtime_calinski)
    

    
    print(f"For k={k}, silhouette score: {silhouette_avg}")
    print(f"For k={k}, Davies score: {Davies_avg}")
    print(f"For k={k}, Calinski score: {Calinski_avg}")

# Find the value of k with the highest silhouette score
best_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 to account for the range starting from 2
print(f"The best value of k is: {best_k} (based on silhouette score)")
best_kD = Davies_scores.index(min(Davies_scores)) + 2  
print(f"The best value of k is: {best_kD} (based on Davies bouldin score)")
best_kC = Calinski_scores.index(max(Calinski_scores)) + 2  
print(f"The best value of k is: {best_kC} (based on Calinski score)")

plt.figure(figsize=(15, 5))  

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')

# Calinski Score
plt.subplot(1, 3, 2)
plt.plot(range(2, 11), Calinski_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Calinski Score')
plt.title('Calinski Score vs. Number of Clusters')

# Davies Score
plt.subplot(1, 3, 3)
plt.plot(range(2, 11), Davies_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies Score')
plt.title('Davies Score vs. Number of Clusters')

plt.tight_layout()
plt.show()
# Plot runtimes for each metric
plt.figure()
plt.plot(range(2, 11), runtimes_silhouette, marker='o', label='Silhouette Runtime (ms)')
plt.plot(range(2, 11), runtimes_davies, marker='o', label='Davies Runtime (ms)')
plt.plot(range(2, 11), runtimes_calinski, marker='o', label='Calinski Runtime (ms)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Runtime (ms)')
plt.title('Runtimes for Different Metrics vs. Number of Clusters')
plt.legend()
plt.show()

#######MINI BATCH 
from sklearn.cluster import MiniBatchKMeans

batch = [10, 50, 100, 500]
list_cluster = [2, 3, 4, 5,6,7,8,9,10,11]  
start = 10 

silhouette_mini_scores=[]  
davies_mini_scores=[]  
calinski_mini_scores=[]  
runtimes_mini=[]

# Minibatch iterations 
for nb_clusters in list_cluster:
    
    for batch_act in batch:
        model = MiniBatchKMeans(n_clusters=nb_clusters, batch_size=batch_act, n_init=start, init='k-means++') 
        start_time = time.time()
        model.fit(datanp)
        end_time = time.time() 
        runtime_mini = (end_time-start_time)*1000
        
        

        labels = model.labels_
        centroids = model.cluster_centers_

        silhouette_avg = silhouette_score(datanp, labels)
        davies_score = davies_bouldin_score(datanp, labels)
        calinski_score = calinski_harabasz_score(datanp, labels)

        silhouette_mini_scores.append(silhouette_avg)
        davies_mini_scores.append(davies_score)
        calinski_mini_scores.append(calinski_score)
        runtimes_mini.append(runtime_mini)
# Plot runtimes for KMeans and MiniBatch
plt.figure()
plt.plot(range(1, 11), runtimes, marker='o', label='KMeans Runtime (ms)', color='blue')
plt.plot(list_cluster, runtimes_mini[:len(list_cluster)], marker='o', label='MiniBatch Runtime (ms)', color='pink')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Runtime (ms)')
plt.title('Runtimes for KMeans and MiniBatch')
plt.legend()
plt.show()

plt.figure()
for i in range(len(list_cluster)):
    plt.plot(batch, silhouette_mini_scores[i*len(batch):(i+1)*len(batch)], marker='o', label=f'{list_cluster[i]} clusters')
plt.xlabel('Batch Size')
plt.ylabel('Silhouette Score')
plt.legend()
plt.title('Silhouette Score vs. Batch Size for Different Numbers of Clusters')
plt.show()
# Plot Davies-Bouldin Score
plt.figure()
for i in range(len(list_cluster)):
    plt.plot(batch, davies_mini_scores[i*len(batch):(i+1)*len(batch)], marker='o', label=f'{list_cluster[i]} clusters')
plt.xlabel('Batch Size')
plt.ylabel('Davies-Bouldin Score')
plt.legend()
plt.title('Davies-Bouldin Score vs. Batch Size for Different Numbers of Clusters')
plt.show()

# Plot Calinski-Harabasz Score
plt.figure()
for i in range(len(list_cluster)):
    plt.plot(batch, calinski_mini_scores[i*len(batch):(i+1)*len(batch)], marker='o', label=f'{list_cluster[i]} clusters')
plt.xlabel('Batch Size')
plt.ylabel('Calinski-Harabasz Score')
plt.legend()
plt.title('Calinski-Harabasz Score vs. Batch Size for Different Numbers of Clusters')
plt.show()

bs_values = [10, 50, 100, 500]
k = 9
num_plots = len(bs_values)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))  

for idx, bs in enumerate(bs_values):
    model = MiniBatchKMeans(n_clusters=k, batch_size=bs, n_init=start, init='k-means++', random_state=42)
    model.fit(datanp)

    labels = model.labels_
    centroids = model.cluster_centers_

    row, col = idx // 2, idx % 2  
    ax = axes[row, col]  

    ax.scatter(f0, f1, c=labels, s=8)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
    ax.set_title("Donnees apres clustering : " + str(name) + " - Nb clusters =" + str(k) + " - batchsize =" + str(bs))

plt.tight_layout()  # Ajuste automatiquement la mise en page
plt.show()


k=4
model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1) 
model.fit(datanp)
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_
#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Donnees apres clustering : "+ str(name) + " - Nb clusters ="+ str(k)+" kmeans")
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches=’ tight’, pad_inches=0.1)
plt.show()

###############################################
#2d-3c-no123.arff non ideal pour Kmeans car densité des points faible en plus du chevauchement des clusters ainsi qu'une densité non uniforme
#2dnormals.arff non ideal car un seul nuage de point non separé
#Carre.arff adapté car dense et cluster bien determiné
#convexe_mal_separe.arff trop condensé
#2d-4c.arff adapté car dense et bien séparé