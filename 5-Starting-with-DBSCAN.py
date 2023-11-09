import numpy as np
import matplotlib.pyplot as plt
import time
import hdbscan

from scipy.io import arff
from sklearn.cluster import DBSCAN
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="longsquare.arff"

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


# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
tps1 = time.time()
epsilon=0.11 #2  # 4
min_pts= 5 #10   # 10
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()


####################################################
# Standardisation des donnees

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()


print("------------------------------------------------------")
print("Appel DBSCAN (2) sur données standardisees ... ")

epsilon=0.3 #0.05
min_pts=5 # 10
start_time_db = time.time()
model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
model.fit(data_scaled)
end_time_db = time.time()
runtime_db = end_time_db - start_time_db


labels = model.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
plt.title("Données après clustering DBSCAN (2) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
plt.show()

#Calcler la distance moyenne entre un nombre de points donné k pour avoir la distance moyenne eps
# Distances aux k plus proches voisins # Donnees dans X
k = 5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(data_scaled)
distances, _ = neigh.kneighbors(data_scaled)

# Calculer la distance moyenne en retirant le point "origine"
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(distances.shape[0])])

# Trier par ordre croissant
distances_triees = np.sort(newDistances)

# Calculer la dérivée des distances
derivee = np.diff(distances_triees)

# Trouver le point de coude
indice_coude = np.argmax(derivee)

# La distance eps idéale est le point de coude
eps_ideal = distances_triees[indice_coude]

# Afficher le graphe
plt.plot(distances_triees)
plt.title("Plus proches voisins " + str(k))
plt.xlabel("Index")
plt.ylabel("Distance")
plt.scatter(indice_coude, distances_triees[indice_coude], c='red', marker='o', label='Point de coude')
plt.legend()
plt.show()


#############################HDBSCAN##############################
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
min_cluster_sizes = range(5, 20)  # Plage de min_cluster_size à explorer
min_samples_values = range(5, 20)  # Plage de min_samples à explorer

best_stability = -1
best_min_cluster_size = None
best_min_samples = None

stability_matrix = np.zeros((len(min_cluster_sizes), len(min_samples_values)))

for i, min_cluster_size in enumerate(min_cluster_sizes):
    for j, min_samples in enumerate(min_samples_values):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(datanp)
        
        # Calculate silhouette score for stability
        silhouette = silhouette_score(datanp, cluster_labels)
        
        stability_matrix[i, j] = silhouette

        if silhouette > best_stability:
            best_stability = silhouette
            best_min_cluster_size = min_cluster_size
            best_min_samples = min_samples

# Tracer la matrice de stabilité
plt.imshow(stability_matrix, cmap='viridis', origin='lower', extent=[min_samples_values[0], min_samples_values[-1], min_cluster_sizes[0], min_cluster_sizes[-1]])
plt.colorbar(label='Stability (Silhouette Score)')
plt.xlabel('Min Samples')
plt.ylabel('Min Cluster Size')
plt.title('Stability Matrix')
plt.show()

# Afficher la meilleure combinaison de paramètres
print("Meilleure combinaison de paramètres:")
print("Min Cluster Size:", best_min_cluster_size)
print("Min Samples:", best_min_samples)
print("Stability (Silhouette Score):", best_stability)

start_time_hd = time.time()
best_clusterer = hdbscan.HDBSCAN(min_cluster_size=best_min_cluster_size, min_samples=best_min_samples)
best_cluster_labels = best_clusterer.fit_predict(datanp)
end_time_hd = time.time()
runtime_hd = end_time_hd - start_time_hd


# Tracer le graphe de clustering
plt.scatter(datanp[:, 0], datanp[:, 1], c=best_cluster_labels, cmap='viridis', s=8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('HDBSCAN Clustering')
plt.show()

print("Runtime hd", runtime_hd)
print("Runtime db", runtime_db)