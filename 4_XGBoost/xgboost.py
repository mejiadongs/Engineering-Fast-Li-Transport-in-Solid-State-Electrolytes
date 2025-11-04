import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN, JmolNN
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import glob
import os
import psutil
import gc

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def load_data(cif_file_path, y_data_path):
    structures = []
    y_values = pd.read_csv(y_data_path, index_col=0)
    y_values = y_values.select_dtypes(include=[np.number])
    
    for cif_file in glob.glob(cif_file_path):
        structure = Structure.from_file(cif_file)
        structures.append(structure)
    
    min_length = min(len(structures), len(y_values))
    structures = structures[:min_length]
    y_values = y_values.iloc[:min_length]
    
    return structures, y_values.values.ravel()

def extract_features(structures):
    features = []
    voronoi_nn = VoronoiNN()
    jmol_nn = JmolNN()
    
    electronegativity = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Cl': 3.16,
        'Br': 2.96, 'I': 2.66, 'S': 2.58, 'P': 2.19, 'Si': 1.90
    }
    
    for structure in structures:
        avg_coord_number = np.mean([len(voronoi_nn.get_nn_info(structure, i)) for i in range(len(structure))])
        density = structure.density
        
        element_counts = structure.composition.get_el_amt_dict()
        num_elements = len(element_counts)
        
        total_mass = structure.composition.weight
        volume = structure.volume
        
        avg_atomic_mass = total_mass / len(structure)
        
        polar_atoms = sum(element_counts.get(el, 0) for el in ['O', 'N', 'F', 'Cl'])
        polar_ratio = polar_atoms / len(structure)
        
        avg_electronegativity = np.mean([electronegativity.get(str(site.specie), 0) for site in structure])
        
        bonds = []
        angles = []
        for i in range(len(structure)):
            nn_info = jmol_nn.get_nn_info(structure, i)
            bonds.extend([nn.get('distance', 0) for nn in nn_info])
            if len(nn_info) >= 2:
                for j in range(len(nn_info)):
                    for k in range(j+1, len(nn_info)):
                        angle = structure.get_angle(nn_info[j]['site'].index, i, nn_info[k]['site'].index)
                        angles.append(angle)
        avg_bond_length = np.mean(bonds) if bonds else 0
        avg_bond_angle = np.mean(angles) if angles else 0
        
        max_distance = np.max(structure.distance_matrix)
        
        charge_distribution = np.linalg.norm(sum((electronegativity.get(str(site.specie), 0) - avg_electronegativity) * site.coords for site in structure))
        
        polarizability = {
            'H': 0.666793, 'C': 1.76, 'N': 1.1, 'O': 0.802, 'F': 0.557, 'Cl': 2.18,
            'Br': 3.05, 'I': 5.35, 'S': 2.90, 'P': 3.63, 'Si': 5.38
        }
        total_polarizability = sum(polarizability.get(str(site.specie), 0) for site in structure)
        
        feature = [
            avg_coord_number,
            density,
            num_elements,
            total_mass,
            volume,
            avg_atomic_mass,
            polar_ratio,
            avg_electronegativity,
            avg_bond_length,
            avg_bond_angle,
            max_distance,
            charge_distribution,
            total_polarizability
        ]
        
        for element in ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']:
            feature.append(element_counts.get(element, 0))
        
        features.append(feature)
    
    columns = [
        'avg_coord_number', 'density', 'num_elements', 'total_mass', 'volume',
        'avg_atomic_mass', 'polar_ratio', 'avg_electronegativity', 'avg_bond_length',
        'avg_bond_angle', 'max_distance', 'charge_distribution', 'total_polarizability',
        'H_count', 'C_count', 'N_count', 'O_count', 'F_count', 'Cl_count', 'Br_count', 'I_count', 'S_count', 'P_count', 'Si_count'
    ]
    
    return pd.DataFrame(features, columns=columns)

def train_xgboost_model_bootstrap(X, y, n_bootstrap=10):
    print_memory_usage()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = []
    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_train_boot = X_train.iloc[bootstrap_idx]
        y_train_boot = y_train[bootstrap_idx]
        
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,          
            'random_state': 42 + i,
            'tree_method': 'hist'
        }
        
        model = XGBRegressor(**params)
        print(f"Training bootstrap model {i+1}/{n_bootstrap}...")
        model.fit(X_train_boot, y_train_boot)
        models.append(model)
    
    print("All bootstrap models trained.")
    return models, X_test, y_test

def predict_with_uncertainty(models, X_test, lower_percentile=5, upper_percentile=95):

    predictions = np.array([model.predict(X_test) for model in models])
    y_pred_mean = predictions.mean(axis=0)
    y_pred_lower = np.percentile(predictions, lower_percentile, axis=0)
    y_pred_upper = np.percentile(predictions, upper_percentile, axis=0)
    return y_pred_mean, y_pred_lower, y_pred_upper

def evaluate_model_with_uncertainty(models, X_test, y_test):
    y_pred_mean, y_pred_lower, y_pred_upper = predict_with_uncertainty(models, X_test)
    mae = mean_absolute_error(y_test, y_pred_mean)
    r2 = r2_score(y_test, y_pred_mean)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(y_test, y_pred_mean, yerr=[y_pred_mean - y_pred_lower, y_pred_upper - y_pred_mean],
                 fmt='o', alpha=0.5, label='Predictions with uncertainty')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted with Uncertainty")
    plt.legend()
    plt.savefig("true_vs_predicted_uncertainty.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    
    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.savefig("feature_importance.png")
    plt.close()

def main():
    print_memory_usage()
    cif_file_path = r"\*.cif"
    y_data_path = r"id_prop.csv"
    
    structures, y_values = load_data(cif_file_path, y_data_path)
    print_memory_usage()
    
    X = extract_features(structures)
    print_memory_usage()
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y_values: {y_values.shape}")
    print(f"NaN values in X: {X.isna().sum().sum()}")
    print(f"NaN values in y_values: {np.isnan(y_values).sum()}")
    
    print("First few rows of X:")
    print(X.head())
    print("First few values of y_values:")
    print(y_values[:5])
    
    gc.collect()
    print_memory_usage()
    
    models, X_test, y_test = train_xgboost_model_bootstrap(X, y_values, n_bootstrap=10)
    print_memory_usage()
    
    evaluate_model_with_uncertainty(models, X_test, y_test)
    plot_feature_importance(models[0], X.columns)
    print_memory_usage()

if __name__ == "__main__":
    main()
