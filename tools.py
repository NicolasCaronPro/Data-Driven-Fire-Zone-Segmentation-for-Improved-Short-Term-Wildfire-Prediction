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
                mask_label = morphology.erosion(mask_label, morphology.square(3))
                ones = np.argwhere(mask_label == 1).shape[0]

            res[mask_before_erosion & ~mask_label] = 0

            # Si le cluster est assez grand, on le conserve tel quel
            logger.info(f"Keep label {region.label}")

        i += 1

    return res


def variance_threshold(df, th):
    var_thres = VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:, new_cols]


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


def most_frequent_neighbor(image, mask, i, j, non_cluster):
    """
    Retourne la valeur la plus fréquente des voisins d'un pixel, en tenant compte d'un masque.
    """
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if (
                0 <= ni < image.shape[0]
                and 0 <= nj < image.shape[1]
                and not mask[ni, nj]
            ):
                if non_cluster is not None:
                    if len(image[ni, nj][image[ni, nj] != non_cluster]) > 0:
                        neighbors.append(image[ni, nj][image[ni, nj] != non_cluster][0])
                else:
                    neighbors.append(image[ni, nj])
    if neighbors:
        return Counter(neighbors).most_common(1)[0][0]
    else:
        return image[i, j]  # En cas d'absence de voisins valides


def merge_small_clusters(image, min_size, non_cluster=None):
    """
    Fusionne les petits clusters en remplaçant leurs pixels par la valeur la plus fréquente de leurs voisins.
    """
    output_image = np.copy(image)
    unique_clusters, counts = np.unique(image, return_counts=True)
    counts = counts[~np.isnan(unique_clusters)]
    unique_clusters = unique_clusters[~np.isnan(unique_clusters)]
    for cluster_id, count in zip(unique_clusters, counts):
        if cluster_id == -1:
            continue
        if count < min_size:
            # Trouver tous les pixels appartenant au cluster
            mask = image == cluster_id
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if mask[i, j]:
                        output_image[i, j] = most_frequent_neighbor(
                            output_image, mask, i, j, non_cluster
                        )

    return output_image


def relabel_clusters(cluster_labels, started):
    """
    Réorganise les labels des clusters pour qu'ils soient séquentiels et croissants.
    """
    unique_labels = np.sort(np.unique(cluster_labels[~np.isnan(cluster_labels)]))

    relabeled_image = np.copy(cluster_labels)
    for ncl, cl in enumerate(unique_labels):
        relabeled_image[cluster_labels == cl] = ncl + started
    return relabeled_image
