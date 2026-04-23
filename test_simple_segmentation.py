import matplotlib
matplotlib.use('Agg')
print("DEBUG: test_simple_segmentation.py start", flush=True)
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Pour importer segmentation depuis le répertoire courant
sys.path.append(str(Path(__file__).resolve().parent))

from segmentation import Segmentation
import segmentation

def test_segmentation_with_create_geometry():
    print("DEBUG: Starting test_segmentation_with_create_geometry", flush=True)

    
    # 1. Création de données fictives (Image de risque 64x64)
    # Résolution 2km par pixel (exprimée en degrés)
    res_x = 0.02875215641173088
    res_y = 0.020721094073767096

    data = np.zeros((64, 64))
    y_idx, x_idx = np.indices((64, 64))
    x = x_idx * res_x
    y = y_idx * res_y

    # Pic 1 autour de l'indice (20, 20)
    sigma_x1, sigma_y1 = 5 * res_x, 5 * res_y
    data += 10 * np.exp(-((x - 20*res_x)**2 / (2*sigma_x1**2) + (y - 20*res_y)**2 / (2*sigma_y1**2)))
    
    # Pic 2 autour de l'indice (45, 45)
    sigma_x2, sigma_y2 = 7 * res_x, 7 * res_y
    data += 15 * np.exp(-((x - 45*res_x)**2 / (2*sigma_x2**2) + (y - 45*res_y)**2 / (2*sigma_y2**2)))
    
    # Raster factice : 1 partout sauf un bord à NaN
    raster_2d = np.ones((64, 64))
    raster_2d[0:3, :] = np.nan
    raster = [raster_2d] # create_geometry_with_watershed fait raster = raster[0]

    # nbsinister factice (requis par create_geometry_with_watershed)
    data_bin = np.zeros((64, 64, 1))
    data_bin[18:23, 18:23, 0] = 1 # Un feu au centre du pic 1

    # 2. Ajout des imports pour calculer l'IoU
    import itertools
    from tools import iou_binary, to_binary_mask
    
    B_bin = to_binary_mask(data_bin)
    
    # 3. Paramètres de Grid Search
    scales = [1, 2, 3, 4, 5]
    attempts = [1, 2, 3, 4, 5]
    reduces = [2, 3, 4, 5, 6]
    
    best_iou = -1
    best_params = None
    
    path_output = Path("./test_output")
    path_output.mkdir(exist_ok=True)
    
    print("Lancement de la grid search pour create_geometry_with_watershed...")
    
    # Fichier pour sauvegarder tous les résultats
    csv_file = path_output / "grid_search_results.csv"
    with open(csv_file, "w") as f:
        f.write("scale,attempt,reduce,iou\n")
    
    for scale, attempt, reduce in itertools.product(scales, attempts, reduces):
        seg = Segmentation(scale=scale, base="degree", attempt=attempt, reduce=reduce, tol=0.3)
        
        # Mocks pour éviter les lectures fichiers réelles et calculs géo complexes
        with patch('segmentation.read_object') as mock_read, \
             patch.object(Segmentation, 'process_input_data') as mock_process:
            
            # Le premier appel est pour le raster, le deuxième pour data_bin
            mock_read.side_effect = [raster, data_bin] 
            
            # process_input_data est appelé pour 'nbsinister'
            mock_process.return_value = data_bin.squeeze(-1)
            
            pred, pred_fz = seg.create_geometry_with_watershed(
                dept="test_dept",
                vec_base=("risk", "size"),
                path=Path("."),
                sinister="firepoint",
                dataset_name="fire_risk",
                sinister_encoding="occurence",
                resolution="2x2",
                node_already_predicted=0,
                train_date=["2024-01-01"],
                data=data,
                GT=None
            )
            
            S = to_binary_mask(np.asarray(pred))
            iou = iou_binary(B_bin, S)
            
            print(f"Scale: {scale}, Attempt: {attempt}, Reduce: {reduce} -> IoU: {iou:.4f}")
            
            # Sauvegarder les métriques
            with open(csv_file, "a") as f:
                f.write(f"{scale},{attempt},{reduce},{iou:.4f}\n")
                
            # Sauvegarder la prédiction en image de manière unique
            plt.figure(figsize=(8, 8))
            plt.imshow(pred)
            plt.title(f"Scale={scale}, Att={attempt}, Red={reduce} | IoU={iou:.4f}")
            plt.colorbar()
            plt.savefig(path_output / f"pred_s{scale}_a{attempt}_r{reduce}.png")
            plt.close('all')
            
            if iou > best_iou:
                best_iou = iou
                best_params = (scale, attempt, reduce)

    if best_params:
        print(f"\nGrid Search terminé.")
        print(f"Meilleurs paramètres globaux : scale={best_params[0]}, attempt={best_params[1]}, reduce={best_params[2]} avec IoU={best_iou:.4f}")

    print("Traitement complet terminé.")

if __name__ == "__main__":
    test_segmentation_with_create_geometry()
