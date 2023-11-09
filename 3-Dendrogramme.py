import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import fcluster

from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
##################################################################
# Exemple :  Dendrogramme and Agglomerative Clustering



path = './artificial/'
name="square1.arff"

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

#################################################
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix) #, **kwargs)



#######################TEST##########################
linkage_methods = ["average", "ward","single", "complete"]

# Plage de valeurs pour le seuil de distance ou le nombre de clusters
threshold_values = np.arange(2, 30, 1)
num_clusters_values = np.arange(2, 21, 1)

# Dictionnaires pour stocker les résultats
silhouette_scores_cl = {}
silhouette_scores_tr = {}
davies_bouldin_scores_cl = {}
davies_bouldin_scores_tr = {}
runtimes = {}
calinski_scores_cl = {}
calinski_scores_tr = {}

# Boucle sur les méthodes de linkage
for linkage_method in linkage_methods:
    print(f"Testing linkage method: {linkage_method}")

    # Initialisation des listes de résultats
    silhouette_scores_cl[linkage_method] = []
    davies_bouldin_scores_cl[linkage_method] = []
    silhouette_scores_tr[linkage_method] = []
    davies_bouldin_scores_tr[linkage_method] = []
    runtimes[linkage_method] = []
    calinski_scores_cl[linkage_method] =[]
    calinski_scores_tr[linkage_method] = []

    # Boucle pour ajuster le seuil de distance
    for threshold in threshold_values:
        t0 = time.time()
        model = AgglomerativeClustering(distance_threshold=threshold, linkage=linkage_method, n_clusters=None)
        model.fit(datanp)
        t1 = time.time()
        num_clusters = model.n_clusters_
        if num_clusters > 1:
            silhouette = silhouette_score(datanp, model.labels_)
            davies_bouldin = davies_bouldin_score(datanp, model.labels_)
            calinski= calinski_harabasz_score(datanp, model.labels_)

        else:
            silhouette = 0.0  # Silhouette est nul s'il n'y a qu'un cluster
            davies_bouldin = 0.0
            calinski = 0.0

        silhouette_scores_tr[linkage_method].append(silhouette)
        davies_bouldin_scores_tr[linkage_method].append(davies_bouldin)
        calinski_scores_tr[linkage_method].append(calinski)
        runtime = t1 - t0
        runtimes[linkage_method].append(runtime)

        print(f"Threshold = {threshold}, Num Clusters = {num_clusters}, Silhouette Score = {silhouette:.2f}, Davies Bouldin Score = {davies_bouldin:.2f}, Runtime = {runtime:.2f} seconds")

    # Boucle pour ajuster le nombre de clusters
    for num_clusters in num_clusters_values:
        t0 = time.time()
        model = AgglomerativeClustering(linkage=linkage_method, n_clusters=num_clusters)
        model.fit(datanp)
        t1 = time.time()

        silhouette = silhouette_score(datanp, model.labels_)
        davies_bouldin = davies_bouldin_score(datanp, model.labels_)
        calinski= calinski_harabasz_score(datanp, model.labels_)

        silhouette_scores_cl[linkage_method].append(silhouette)
        davies_bouldin_scores_cl[linkage_method].append(davies_bouldin)
        calinski_scores_cl[linkage_method].append(calinski)
        runtime = t1 - t0
        runtimes[linkage_method].append(runtime)

        print(f"Num Clusters = {num_clusters}, Silhouette Score = {silhouette:.2f}, Davies Bouldin Score = {davies_bouldin:.2f}, Runtime = {runtime:.2f} seconds")

# Création de graphiques clusters
for linkage_method in linkage_methods:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(num_clusters_values, silhouette_scores_cl[linkage_method][:len(num_clusters_values)], marker='o', label='Silhouette Score')
    plt.xlabel("cluster Values")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Score - {linkage_method}")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(num_clusters_values, davies_bouldin_scores_cl[linkage_method][:len(num_clusters_values)], marker='o', label='Davies Bouldin Score')
    plt.xlabel("cluster Values")
    plt.ylabel("Davies Bouldin Score")
    plt.title(f"Davies Bouldin Score - {linkage_method}")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(num_clusters_values, calinski_scores_cl[linkage_method][:len(num_clusters_values)], marker='o', label='Calinski Score')
    plt.xlabel("cluster Values")
    plt.ylabel("Calinski Score")
    plt.title(f"Calinski Score - {linkage_method}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Création de graphiques threshold
for linkage_method in linkage_methods:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(threshold_values, silhouette_scores_tr[linkage_method][:len(threshold_values)], marker='o', label='Silhouette Score')
    plt.xlabel("Threshold Values")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Score - {linkage_method}")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(threshold_values, davies_bouldin_scores_tr[linkage_method][:len(threshold_values)], marker='o', label='Davies Bouldin Score')
    plt.xlabel("Threshold Values")
    plt.ylabel("Davies Bouldin Score")
    plt.title(f"Davies Bouldin Score - {linkage_method}")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(num_clusters_values, calinski_scores_tr[linkage_method][:len(num_clusters_values)], marker='o', label='Calinski Score')
    plt.xlabel("Threshold Values")
    plt.ylabel("Calinski Score")
    plt.title(f"Calinski Score - {linkage_method}")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Création de graphiques pour les runtimes
plt.figure(figsize=(8, 6))
for linkage_method in linkage_methods:
    plt.plot(threshold_values, runtimes[linkage_method][:len(threshold_values)], marker='o', label=f'Runtimes - {linkage_method}')

plt.xlabel("Threshold Values")
plt.ylabel("Runtime (seconds)")
plt.title("Runtimes")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

####################################################@

# setting distance_threshold=0 ensures we compute the full tree.
model = cluster.AgglomerativeClustering(distance_threshold=0, linkage='average', n_clusters=None)

#average method 
model = model.fit(datanp)
plt.figure(figsize=(9, 8))
plt.title("Hierarchical Clustering Dendrogram (average method) ")
# plot the top p levels of the dendrogram
plot_dendrogram(model) #, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#single method 
model = cluster.AgglomerativeClustering(distance_threshold=0, linkage='single', n_clusters=None)

model = model.fit(datanp)
plt.figure(figsize=(9, 8))
plt.title("Hierarchical Clustering Dendrogram (single method) ")
# plot the top p levels of the dendrogram
plot_dendrogram(model) #, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#ward method
model = cluster.AgglomerativeClustering(distance_threshold=0, linkage='ward', n_clusters=None)

model = model.fit(datanp)
plt.figure(figsize=(9, 8))
plt.title("Hierarchical Clustering Dendrogram (ward method) ")
# plot the top p levels of the dendrogram
plot_dendrogram(model) #, truncate_mode="level", p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()




### FIXER la distance
# 
tps1 = time.time()
seuil_dist = 0.75
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='ward', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
datanp = np.array(datanp, dtype=float)
print(datanp.dtype)
k=10
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")





from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

#############Score de regroupement
# 1. Calculer le centre de chaque cluster
cluster_centers = []
for i in range(k):
    cluster_points = datanp[labels == i]
    center = np.mean(cluster_points, axis=0)
    cluster_centers.append(center)

# 2. Calculer les distances entre les points d'un cluster et son centre
distances = []
for i in range(k):
    cluster_points = datanp[labels == i]
    center = cluster_centers[i]
    cluster_distances = [euclidean(point, center) for point in cluster_points]
    distances.append(cluster_distances)

# 3. Enregistrer ces distances pour chaque point (déjà fait dans 'distances')

# 4. Calculer la distance minimale, maximale et moyenne pour chaque cluster
min_distances = [np.min(d) for d in distances]
max_distances = [np.max(d) for d in distances]
avg_distances = [np.mean(d) for d in distances]

x = np.arange(k)  # Les clusters
width = 0.2  # Largeur des barres

fig, ax = plt.subplots()
bar1 = ax.bar(x - width, min_distances, width, label='Min Distance')
bar2 = ax.bar(x, max_distances, width, label='Max Distance')
bar3 = ax.bar(x + width, avg_distances, width, label='Avg Distance')

ax.set_xlabel('Cluster')
ax.set_ylabel('Distances')
ax.set_title('Distances Min/Max/Avg for Each Cluster')
ax.set_xticks(x)
ax.set_xticklabels([str(i + 1) for i in range(k)])
ax.legend()

# Ajouter les valeurs des distances sur les barres
for i in range(k):
    ax.text(x[i] - width, min_distances[i], f'{min_distances[i]:.2f}', ha='center', va='bottom')
    ax.text(x[i], max_distances[i], f'{max_distances[i]:.2f}', ha='center', va='bottom')
    ax.text(x[i] + width, avg_distances[i], f'{avg_distances[i]:.2f}', ha='center', va='bottom')

plt.show()

###############score de separation 

# 1. Calculer le centre de chaque cluster
cluster_centers = []
for i in range(k):
    cluster_points = datanp[labels == i]
    center = np.mean(cluster_points, axis=0)
    cluster_centers.append(center)

# 2. Calculer les distances entre les centres de cluster
center_distances = []
for i in range(k):
    for j in range(i + 1, k):
        distance = euclidean(cluster_centers[i], cluster_centers[j])
        center_distances.append(distance)

# 3. Calculer la distance minimale, maximale et moyenne entre les centres
min_center_distance = np.min(center_distances)
max_center_distance = np.max(center_distances)
avg_center_distance = np.mean(center_distances)

# Créer un graphique à barres pour afficher les résultats
x = ['Min Distance', 'Max Distance', 'Avg Distance']
y = [min_center_distance, max_center_distance, avg_center_distance]

fig, ax = plt.subplots()
bar = ax.bar(x, y)

ax.set_ylabel('Distance')
ax.set_title('Distances Min/Max/Avg Between Cluster Centers')

# Ajouter les valeurs des distances sur les barres
for i, v in enumerate(y):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.show()
#######################################################################
