import os
import logging
logger = logging.getLogger(__name__)
import pickle
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from skimage import measure, segmentation, morphology
from skimage.measure import regionprops
import math
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
import geopandas as gpd

# -- Dummy root paths (from arborescence.py) --
rootDisk = Path('./')
root_target = Path('./')


# --- From tools.py --- 

import datetime as dt

def find_dates_between(start, end):
    start_date = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(end, "%Y-%m-%d").date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date <= end_date:
        res.append(date.strftime("%Y-%m-%d"))
        date += delta
    return res

allDates = find_dates_between('2017-06-12', '2025-12-31')

def save_object(obj, filename: str, path: Path):
    check_and_create_path(path)
    with open(path / filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename: str, path: Path):
    if not (path / filename).is_file():
        logger.info(f"{path / filename} not found")
        return None
    return pickle.load(open(path / filename, "rb"))

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def order_class(predictor, pred, min_values=0):
    res = np.zeros(pred[~np.isnan(pred)].shape[0], dtype=int)
    cc = predictor.cluster_centers_.reshape(-1)
    classes = np.arange(cc.shape[0])
    ind = np.lexsort([cc])
    cc = cc[ind]
    classes = classes[ind]
    for c in range(cc.shape[0]):
        mask = np.argwhere(pred == classes[c])
        res[mask] = c
    return res + min_values

def count_pixels_in_france_deg_square(res_km=2, deg_size=0.25, lat_deg=46.5):
    """
    Calcule le nombre de pixels (res_km x res_km) dans un carré deg_size x deg_size degrés,
    situé au centre de la France (latitude 46.5°N par défaut).

    Args:
        res_km (float): Taille d’un pixel en kilomètres (par défaut 2 km).
        deg_size (float): Taille du carré en degrés (par défaut 0.25°).
        lat_deg (float): Latitude (par défaut 46.5°N, centre de la France).

    Returns:
        tuple: (n_rows, n_cols, total_pixels)
    """
    
    print('kjehfkjzefknef')
    
    # Longueur d’un degré de latitude (quasi constant)
    km_per_deg_lat = 111.32

    # Longueur d’un degré de longitude dépendant de la latitude
    # km_per_deg_lon = 111.32 * math.cos(math.radians(lat_deg))
    km_per_deg_lon = 111.32

    # Dimensions du carré en km
    height_km = deg_size * km_per_deg_lat
    width_km = deg_size * km_per_deg_lon

    print(deg_size)
    
    # Nombre de pixels
    n_rows = int(height_km // res_km)
    n_cols = int(width_km // res_km)
    total_pixels = n_rows * n_cols

    return n_rows, n_cols, total_pixels

def merge_adjacent_clusters(
    image,
    mode="size",
    min_cluster_size=0,
    max_cluster_size=math.inf,
    exclude_label=None,
    background=-1,
    features=None,
    nb_attempt=3,
):
    """
    Fusionne les clusters adjacents dans une image en fonction de critères définis.

    Paramètres :
    - image : Image labellisée contenant des clusters.
    - mode : Critère de fusion ('size', 'time_series_similarity', 'time_series_similarity_fast').
    - min_cluster_size : Taille minimale d'un cluster avant fusion.
    - max_cluster_size : Taille maximale autorisée après fusion.
    - oridata : Données supplémentaires utilisées pour la fusion basée sur des séries temporelles (facultatif).
    - exclude_label : Label à exclure de la fusion.
    - background : Label représentant le fond (par défaut -1).
    """

    # Copie de l'image d'entrée pour éviter de la modifier directement
    labeled_image = np.copy(image)

    # Obtenir les propriétés des régions labellisées
    regions = measure.regionprops(labeled_image)
    # Trier les régions par taille croissante
    regions = sorted(regions, key=lambda r: r.area)

    # Masque pour stocker les labels mis à jour après fusion
    res = np.copy(labeled_image)

    # Liste des labels qui ont été modifiés
    changed_labels = []

    fix_label = []

    # Longueur initiale des régions
    len_regions = len(regions)
    i = 0

    # Boucle pour traiter chaque région
    while i < len_regions:
        region = regions[i]

        # Vérifier si le cluster est à exclure ou est un fond
        if region.label == exclude_label or region.label == background:
            # On conserve ces clusters tels quels
            res[labeled_image == region.label] = region.label
            i += 1
            continue

        label = region.label
        if label in fix_label:
            i += 1
            continue

        # Si le label a déjà été modifié, passer au suivant
        # if label in changed_labels:
        #    i += 1
        #    continue

        # Vérifier la taille du cluster actuel
        ones = np.argwhere(res == label).shape[0]
        if ones < min_cluster_size:
            # Si la taille est inférieure au minimum, essayer de fusionner avec un voisin
            nb_test = 0
            find_neighbor = False
            dilated_image = np.copy(res)
            while nb_test < nb_attempt and not find_neighbor:
                print(f"DEBUG inner: nb_test={nb_test}, label={label}", flush=True)

                # Trouver les voisins du cluster actuel
                mask_label = dilated_image == label
                mask_label_ori = res == label
                neighbors = segmentation.find_boundaries(
                    mask_label, connectivity=1, mode="outer", background=background
                )
                neighbor_labels = np.unique(dilated_image[neighbors])
                # Exclure les labels indésirables
                neighbor_labels = neighbor_labels[
                    (neighbor_labels != exclude_label)
                    & (neighbor_labels != background)
                    & (neighbor_labels != label)
                ]
                dilate = True
                changed_labels.append(label)

                if len(neighbor_labels) > 0:
                    # Trier les voisins par taille
                    neighbors_size = sorted(
                        [
                            [neighbor_label, np.sum(res == neighbor_label)]
                            for neighbor_label in neighbor_labels
                        ],
                        key=lambda x: x[1],  # trie par la somme (ordre croissant)
                    )

                    best_neighbor = None

                    if mode == "size":
                        # Mode basé sur la taille des clusters
                        max_neighbor_size = -math.inf
                        for nei, neighbor in enumerate(neighbors_size):
                            if neighbor[0] == label:
                                continue
                            neighbor_size = neighbor[1] + np.sum(res == label)

                            # Vérifier si le voisin satisfait min_cluster_size
                            if neighbor_size > min_cluster_size:
                                # Vérifier si la taille reste sous max_cluster_size
                                if neighbor_size < max_cluster_size:
                                    dilate = False
                                    res[mask_label_ori] = neighbor[0]
                                    dilated_image[mask_label] = neighbor[0]
                                    logger.info(
                                        f"Use neighbord label {label} -> {neighbor[0]}"
                                    )
                                    label = neighbor[0]
                                    find_neighbor = True
                                    break

                                best_neighbor = neighbor[0]
                                max_neighbor_size = neighbor_size
                                break

                            # Enregistrer le plus grand voisin si min_cluster_size n'est pas atteint
                            if neighbor_size > max_neighbor_size:
                                best_neighbor = neighbor[0]
                                max_neighbor_size = neighbor_size

                    elif mode == "timeSeriesSimilarity":
                        # Mode basé sur la similarité de séries temporelles (DTW)
                        assert features is not None
                        time_series_data = np.nansum(
                            features[dilated_image == label], axis=0
                        ).reshape(-1, 1)
                        best_neighbord = None
                        min_dst = math.inf  # Cherche à minimiser la distance
                        dst_thresh = 1000

                        for neighbor in neighbors_size:
                            if neighbor[0] == label:
                                continue

                            time_series_data_neighbor = np.nansum(
                                features[dilated_image == neighbor[0]], axis=0
                            ).reshape(-1, 1)
                            distance = dtw.distance(
                                time_series_data, time_series_data_neighbor
                            )

                            if distance < min_dst and distance < dst_thresh:
                                best_neighbord = neighbor[0]
                                min_dst = distance

                        if best_neighbord is not None:
                            dilate = False
                            res[mask_label_ori] = best_neighbord
                            logger.info(f"label {label} -> {best_neighbord}")
                            changed_labels.append(label)
                            label = best_neighbord
                            find_neighbor = True

                    elif mode == "time_series_similarity_fast":
                        # Mode basé sur une version rapide de DTW
                        assert features is not None
                        time_series_data = np.nansum(
                            features[dilated_image == label], axis=0
                        ).reshape(-1, 1)
                        best_neighbord = None
                        min_simi = math.inf  # Cherche à minimiser la similarité
                        dst_thresh = 100

                        for neighbor in neighbors_size:
                            time_series_data_neighbor = np.nansum(
                                features[dilated_image == neighbor[0]], axis=0
                            ).reshape(-1, 1)
                            _, simi = dtw_functions.dtw(
                                time_series_data,
                                time_series_data_neighbor,
                                local_dissimilarity=d.euclidean,
                            )

                            if simi < min_simi and simi < dst_thresh:
                                best_neighbord = neighbor[0]
                                min_simi = simi

                        if best_neighbord is not None:
                            dilate = False
                            res[mask_label_ori] = best_neighbord
                            changed_labels.append(label)
                            label = best_neighbord
                            find_neighbor = True

                    elif mode == "BrayCurtis":
                        assert features is not None
                        best_neighbor, max_neighbor_size, find_neighbor, dilate = (
                            find_neighbor_by_BrayCurtis_similarity(
                                res,
                                features,
                                label,
                                min_cluster_size,
                                max_cluster_size,
                                mask_label_ori,
                                dilated_image,
                                mask_label,
                                neighbor_labels,
                            )
                        )

                    # Si aucun voisin ne satisfait les critères, utiliser le plus grand
                    if not find_neighbor and best_neighbor is not None:
                        if max_neighbor_size < max_cluster_size:
                            res[mask_label] = best_neighbor
                            dilated_image[mask_label] = best_neighbor
                            dilate = False
                            logger.info(
                                f"Use biggest neighbord label {label} -> {best_neighbor}"
                            )
                            label = best_neighbor
                            find_neighbor = True
                            # Si la taille après fusion dépasse la taille maximal, appliquer l'érosion (peut être ne pas fusionner)
                            if max_neighbor_size < max_cluster_size:
                                mask_label = dilated_image == label
                                ones = np.argwhere(mask_label == 1).shape[0]
                                while ones > max_cluster_size:
                                    mask_label = morphology.erosion(
                                        mask_label, morphology.disk(3)
                                    )
                                    ones = np.argwhere(mask_label == 1).shape[0]

                # Si aucun voisin trouvé, dilater la région
                if dilate:
                    mask_label = morphology.dilation(mask_label, morphology.square(3))
                    dilated_image[(mask_label)] = label
                    nb_test += 1

                if not dilate:
                    break

            # Si aucun voisin trouvé après nb_attempt, supprimer ou conserver la région
            if not find_neighbor:
                if ones < min_cluster_size:
                    mask_label = dilated_image == label
                    ones = np.argwhere(mask_label == 1).shape[0]
                    # Si l'objet dilaté ne vérifie pas la condition minimum
                    if ones < min_cluster_size:
                        res[mask_label] = 0
                        logger.info(f"Remove label {region.label}")
                    else:
                        # Si l'objet dilaté ne vérifie pas la condition maximum
                        while ones > max_cluster_size:
                            mask_label = morphology.erosion(
                                mask_label, morphology.square(3)
                            )
                            ones = np.argwhere(mask_label == 1).shape[0]

                        res[mask_label] = region.label
                        logger.info(f"Keep label dilated {region.label}")
                        fix_label.append(region.label)

            # Mettre à jour les régions pour tenir compte des changements
            regions = measure.regionprops(res)
            regions = sorted(regions, key=lambda r: r.area)
            len_regions = len(regions)
            i = 0
            continue

        else:
            mask_label = res == region.label
            mask_before_erosion = np.copy(mask_label)
            while ones > max_cluster_size:
                print(f"DEBUG erode2: ones={ones}", flush=True)
                mask_label = morphology.erosion(mask_label, morphology.square(3))
                ones = np.argwhere(mask_label == 1).shape[0]

            res[mask_before_erosion & ~mask_label] = 0

            # Si le cluster est assez grand, on le conserve tel quel
            logger.info(f"Keep label {region.label}")

        i += 1

    return res

def find_clusters(image, threshold, clusters_to_ignore=None, background=0):
    """
    Traverse the clusters in an image and return the clusters whose size is greater than a given threshold.

    :param image: np.array, 2D image with values representing the clusters
    :param threshold: int, minimum size of the cluster to be considered
    :param background: int, value representing the background (default: 0)
    :param clusters_to_ignore: list, list of clusters to ignore (default: None)
    :return: list, list of cluster IDs whose size is greater than the threshold
    """
    # Initialize the list of valid clusters to return
    valid_clusters = []

    # If no clusters to ignore are provided, initialize with an empty list
    if clusters_to_ignore is None:
        clusters_to_ignore = []

    # Create a mask where the background is ignored
    mask = image != background

    # Label the clusters in the image
    cluster_ids = np.unique(image[mask])
    cluster_ids = cluster_ids[~np.isnan(cluster_ids)]

    # Traverse each cluster and check its size
    for cluster_id in cluster_ids:
        # Skip the cluster if it's in the ignore list
        if cluster_id == clusters_to_ignore:
            continue

        # Calculate the size of the cluster
        cluster_size = np.sum(image == cluster_id)

        # If the cluster size exceeds the threshold, add it to the list
        if cluster_size > threshold:
            valid_clusters.append(cluster_id)

    return valid_clusters

def split_large_clusters(
    image, size_threshold, min_cluster_size, wanted_size, background
):
    labeled_image = np.copy(image)

    regions = measure.regionprops(labeled_image)
    new_labeled_image = np.copy(labeled_image)
    changes_made = False

    for region in regions:
        if region.label in background:
            continue

        original_size = region.area

        if original_size > size_threshold:
            minr, minc, maxr, maxc = region.bbox
            region_mask = labeled_image[minr:maxr, minc:maxc] == region.label
            coords = np.column_stack(np.nonzero(region_mask))

            if len(coords) > 1:
                # Appliquer KMeans
                clusterer = KMeans(n_clusters=2, random_state=42, n_init=10).fit(coords)
                labels = clusterer.labels_

                # Calcul des tailles des deux sous-clusters
                size_1 = np.sum(labels == 0)
                size_2 = np.sum(labels == 1)

                # Vérifier si le split améliore la proximité aux tailles voulues
                original_diff = abs(original_size - wanted_size)
                split_diff = abs(size_1 - wanted_size) + abs(size_2 - wanted_size)

                # if split_diff < original_diff and size_1 >= min_cluster_size and size_2 >= min_cluster_size:
                if split_diff < original_diff:
                    # Appliquer le split
                    new_label_1 = new_labeled_image.max() + 1
                    new_label_2 = new_label_1 + 1
                    full_region = new_labeled_image[minr:maxr, minc:maxc]

                    new_region = np.where(region_mask, full_region, 0)
                    # On assigne en place dans la région uniquement là où le masque est actif
                    new_region[region_mask] = np.where(
                        labels == 0, new_label_1, new_label_2
                    )
                    new_labeled_image[minr:maxr, minc:maxc][region_mask] = new_region[
                        region_mask
                    ]

                    changes_made = True

    if changes_made:
        # Appel récursif pour gérer les divisions multiples
        new_labeled_image = split_large_clusters(
            new_labeled_image, size_threshold, min_cluster_size, wanted_size, background
        )

    return new_labeled_image

def relabel_clusters(cluster_labels, started):
    """
    Réorganise les labels des clusters pour qu'ils soient séquentiels et croissants.
    """
    unique_labels = np.sort(np.unique(cluster_labels[~np.isnan(cluster_labels)]))

    relabeled_image = np.copy(cluster_labels)
    for ncl, cl in enumerate(unique_labels):
        relabeled_image[cluster_labels == cl] = ncl + started
    return relabeled_image

def frequency_ratio(values: np.array, mask: np.array):
    FF_t = np.sum(values)
    Area_t = len(values)
    FF_i = np.sum(values[mask])

    if FF_t == 0 or Area_t == 0:
        return 0

    # Calculer Area_i (le nombre total de pixels pour la classe c)
    Area_i = mask.shape[0]

    # Calculer FireOcc et Area pour la classe c
    FireOcc = FF_i / FF_t
    Area = Area_i / Area_t
    if Area == 0:
        return 0

    # Calculer le ratio de fréquence (FR) pour la classe c
    FR = FireOcc / Area

    return round(FR, 3)

# --- From graph_structure.py --- 

def to_binary_mask(S):
    """
    Converts an array to a binary mask where values > 0 become 1 and others become 0.
    """
    if S is None:
        return None
    return (np.asarray(S) > 0).astype(int)

def iou_binary(maskA, maskB):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.
    """
    maskA = np.asarray(maskA) > 0
    maskB = np.asarray(maskB) > 0
    
    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

# --- From weigh_predictor.py --- 

class Predictor():
    def __init__(self, n_clusters, name='', type='kmeans', eps=0.5, binary=False):
        if type == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif type == 'dbscan':
            self.model = DBSCAN(eps=eps)
        elif type == 'fix':
            self.model = FixedThresholdPredictor(n_clusters, name, 1 if binary else None)
        elif type == 'jenks':
            self.model = JenksThresholdPredictor(n_clusters=n_clusters, name=name)

        self.binary = binary
        self.bounds = None
        self.n_clusters = n_clusters
        self.type = type
        self.name = name

    def fit(self, X : np.array):
        self.train_X = X
        if len(X.shape) == 1:
            self.model.fit(X.reshape(-1,1))
            Xpre = self.model.predict(X.reshape(-1,1))
        else:
            self.model.fit(X)
            Xpre = self.model.predict(X)

        if self.type == 'kmeans':
            self.cluster_centers_ = self.model.cluster_centers_
        else:
            self.cluster_centers_= []
            cls = np.unique(Xpre)
            for c in cls:
                mask = np.argwhere(Xpre == c)
                self.cluster_centers_.append(np.mean(X[mask]))
            self.cluster_centers_ = np.asarray(self.cluster_centers_)
                
        self.histogram = np.bincount(Xpre)
        self.highNumberClass = np.argwhere(self.histogram == np.max(self.histogram))

    def predict(self, X : np.array, min_class : int = 0):
        X = X.astype(float)
        if len(X.shape) == 1:
            pred = self.model.predict(X.reshape(-1,1)) + min_class
        else:
            pred = self.model.predict(X) + min_class
        mask = self.train_X > np.nanmax(self.train_X)
        if True in np.unique(mask):
            pred[mask] += 1
        return pred
    
    def weight(self, c : int):
        #valueOfClass = self.histogram[c]
        #valueOfMinClass = self.histogram[self.highNumberClass][:,0][0]
        #return valueOfMinClass / valueOfClass
        return c + 1
    
    def weight_array(self, array : np.array):
        lis = [self.weight(c) for c in array]
        return np.asarray(lis)
    
    def get_centroid(self, c : int):
        if self.binary:
            return self.bounds[c]
        return self.model.cluster_centers_[c]

    def log(self, logger = None):
        if logger is not None:
            logger.info(f'############# Predictor {self.name} ###############')
            logger.info('Histogram')
            logger.info(self.histogram)
            logger.info('Cluster Centers')
            logger.info(self.model.cluster_centers_)
            logger.info(f'####################################')
        else:
            print(f'############# Predictor {self.name} ###############')
            print('Histogram')
            print(self.histogram)
            print('Cluster Centers')
            print(self.model.cluster_centers_)
            print(f'####################################')

class FixedThresholdPredictor:
    def __init__(self, n_clusters=5, name='', max_value=None):
        """
        Initialise un modèle avec 5 seuils fixes pour les clusters.
        
        Args:
            n_clusters (int): Le nombre de clusters, fixé à 5 pour ce modèle.
            name (str): Nom du modèle.
            binary (bool): Toujours True pour ce type de modèle.
        """
        self.bounds = None
        self.n_clusters = n_clusters
        self.name = name
        self.cluster_centers_ = None
        self.histogram = None
        self.highNumberClass = None
        self.max_value = max_value

    def fit(self, X: np.array):
        """
        Calcule les seuils et génère les clusters en fonction des données X.
        
        Args:
            X (np.array): Tableau 1D de données pour l'entraînement.
        """
        # Définir des bornes fixes pour 5 clusters
        if self.max_value is None:
            self.max_value = np.max(X)

        self.bounds = np.array([0.0, 0.1, 0.40, 0.64, 0.95, 1.0])

        # Clusteriser les valeurs selon les bornes
        Xpre = np.zeros(X.shape[0], dtype=int)

        # Appliquer les bornes pour chaque classe
        for c in range(self.n_clusters):
            lower_bound = self.bounds[c]
            upper_bound = self.bounds[c + 1]
            mask = np.argwhere((X >= self.max_value * lower_bound) & (X < self.max_value * upper_bound))[:, 0]
            Xpre[mask] = c

        # Pour les valeurs exactement égales au max, assigner au dernier cluster
        max_mask = np.argwhere(X == self.max_value)[:, 0]
        Xpre[max_mask] = self.n_clusters - 1

        self.cluster_centers_ = np.zeros(self.n_clusters)
        for c in range(self.n_clusters):
            self.cluster_centers_[c] = (self.bounds[c] * self.max_value + self.bounds[c + 1] * self.max_value) / 2

        # Histogramme des points dans chaque cluster
        self.histogram = np.bincount(Xpre, minlength=self.n_clusters)
        self.highNumberClass = np.argwhere(self.histogram == np.max(self.histogram))

    def predict(self, X: np.array):
        """
        Prédire les classes pour les nouvelles données X.
        
        Args:
            X (np.array): Données à classer.
        
        Returns:
            np.array: Tableau des classes prédites.
        """
        pred = np.zeros(X.shape[0], dtype=int)

        # Appliquer les bornes pour chaque classe
        for c in range(self.n_clusters):
            lower_bound = self.bounds[c]
            upper_bound = self.bounds[c + 1]
            mask = np.argwhere((X >= self.max_value * lower_bound) & (X < self.max_value * upper_bound))[:, 0]
            pred[mask] = c

        # Pour les valeurs exactement égales au max, assigner au dernier cluster
        max_mask = np.argwhere(X == self.max_value)[:, 0]
        pred[max_mask] = self.n_clusters - 1

        return pred

class JenksThresholdPredictor:
    def __init__(self, n_clusters=5, name='Jenks Natural Breaks'):
        """
        Initialise un modèle basé sur la méthode Jenks Natural Breaks.
        
        Args:
            n_clusters (int): Nombre de clusters/classes souhaités.
            name (str): Nom du modèle.
        """
        self.n_clusters = n_clusters
        self.name = name
        self.bounds = None
        self.cluster_centers_ = None
        self.histogram = None

    def _calculate_jenks_breaks(self, X, n_clusters):
        """
        Implémente l'algorithme de Jenks Natural Breaks.
        
        Args:
            X (np.array): Données triées (1D).
            n_clusters (int): Nombre de classes souhaitées.
        
        Returns:
            np.array: Tableau des bornes calculées (breaks).
        """
        data = np.sort(X)
        n = len(data)

        # Matrices pour la minimisation de la variance intra-classe
        lower_class_limits = np.zeros((n + 1, n_clusters + 1), dtype=np.float64)
        variance_combinations = np.zeros((n + 1, n_clusters + 1), dtype=np.float64)

        # Initialisation
        for i in range(1, n + 1):
            lower_class_limits[i][1] = 1
            variance_combinations[i][1] = np.sum((data[:i] - np.mean(data[:i]))**2)

        for k in range(2, n_clusters + 1):
            for i in range(2, n + 1):
                best_variance = float("inf")
                for j in range(1, i):
                    variance = variance_combinations[j][k - 1] + np.sum((data[j:i] - np.mean(data[j:i]))**2)
                    if variance < best_variance:
                        best_variance = variance
                        lower_class_limits[i][k] = j
                variance_combinations[i][k] = best_variance

        # Récupération des bornes optimales
        k = n_clusters
        breaks = np.zeros(n_clusters + 1)
        breaks[-1] = data[-1]
        for i in range(n_clusters - 1, 0, -1):
            breaks[i] = data[int(lower_class_limits[int(breaks[i + 1])][k]) - 1]
            k -= 1
        breaks[0] = data[0]

        return breaks

    def fit(self, X: np.array):
        """
        Calcule les seuils Jenks et prépare le modèle pour la prédiction.
        
        Args:
            X (np.array): Tableau 1D de données pour l'entraînement.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Les données doivent être un tableau numpy (np.array).")

        # Calculer les bornes via la méthode de Jenks
        self.bounds = self._calculate_jenks_breaks(X, self.n_clusters)

        # Calcul des centres des clusters
        self.cluster_centers_ = []
        for i in range(self.n_clusters):
            lower_bound = self.bounds[i]
            upper_bound = self.bounds[i + 1]
            mask = (X >= lower_bound) & (X < upper_bound)
            if np.any(mask):
                self.cluster_centers_.append(np.mean(X[mask]))
            else:
                self.cluster_centers_.append((lower_bound + upper_bound) / 2)

        # Histogramme des points dans chaque cluster
        labels = self.predict(X)
        self.histogram = np.bincount(labels, minlength=self.n_clusters)

    def predict(self, X: np.array):
        """
        Prédire les classes pour les nouvelles données X.
        
        Args:
            X (np.array): Données à classer.
        
        Returns:
            np.array: Tableau des classes prédites.
        """
        if self.bounds is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Appelez 'fit' d'abord.")
        
        pred = np.zeros(X.shape[0], dtype=int)
        for i in range(self.n_clusters):
            lower_bound = self.bounds[i]
            upper_bound = self.bounds[i + 1]
            mask = (X >= lower_bound) & (X < upper_bound)
            pred[mask] = i
        
        # Assignation des valeurs égales au max dans le dernier cluster
        pred[X == self.bounds[-1]] = self.n_clusters - 1
        return pred