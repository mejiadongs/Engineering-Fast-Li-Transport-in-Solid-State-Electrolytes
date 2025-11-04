import pandas as pd
import numpy as np
from pymatgen.core import Structure, Molecule
from xgboost import XGBRegressor
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, cross_val_score,
    learning_curve, GridSearchCV
)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import glob
import os
import psutil
import gc
from pymatgen.core.periodic_table import Element
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.tree import DecisionTreeRegressor, plot_tree
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import shap  # SHAP explanation
from sklearn.preprocessing import StandardScaler

# =============== 不确定性分析所需的函数（新增） ===============

def train_xgboost_model_bootstrap(X, y, n_bootstrap=10):
    print_memory_usage()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = []
    for i in range(n_bootstrap):
        bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        
        if isinstance(X_train, pd.DataFrame):
            X_train_boot = X_train.iloc[bootstrap_idx]
        else:
            X_train_boot = X_train[bootstrap_idx]
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
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        y_test, y_pred_mean,
        yerr=[y_pred_mean - y_pred_lower, y_pred_upper - y_pred_mean],
        fmt='o', alpha=0.5, label='Predictions with uncertainty'
    )
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted with Uncertainty")
    plt.legend()
    plt.savefig("true_vs_predicted_uncertainty.png")
    plt.close()
    
    # Save prediction data with uncertainty bounds to CSV
    uncertainty_df = pd.DataFrame({
        'True_Values': y_test,
        'Predicted_Mean': y_pred_mean,
        'Prediction_Lower_Bound': y_pred_lower,
        'Prediction_Upper_Bound': y_pred_upper,
        'Uncertainty_Range': y_pred_upper - y_pred_lower
    })
    uncertainty_df.to_csv('predictions_with_uncertainty.csv', index=False)
    print("Saved predictions with uncertainty data to 'predictions_with_uncertainty.csv'")

# ===========================================================

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {mem:.2f} MB")

def load_data(cif_file_path, y_data_path, homo_lumo_path):
    structures = []
    y_values = pd.read_csv(y_data_path, header=None, names=['id', 'target'], sep='\s+')
    y_values[['id', 'target']] = y_values['id'].str.split(',', expand=True)
    y_values['target'] = pd.to_numeric(y_values['target'], errors='coerce')
    
    homo_lumo_data = pd.read_csv(homo_lumo_path)
    
    # Extract IDs from CIF filenames
    cif_files = sorted(glob.glob(cif_file_path))
    ids = [os.path.basename(f).split('.')[0] for f in cif_files]

    # Update structures and IDs
    valid_structures = []
    valid_ids = []
    for idx, cif_file in enumerate(cif_files):
        id_ = ids[idx]
        if id_ in y_values['id'].values and id_ in homo_lumo_data['Model'].values:
            structure = Structure.from_file(cif_file)
            valid_structures.append(structure)
            valid_ids.append(id_)
        else:
            print(f"ID {id_} not found in y_values or homo_lumo_data. Skipping.")

    # Update y_values and homo_lumo_data
    y_values = y_values[y_values['id'].isin(valid_ids)]
    homo_lumo_data = homo_lumo_data[homo_lumo_data['Model'].isin(valid_ids)]

    # Reset index
    y_values = y_values.reset_index(drop=True)
    homo_lumo_data = homo_lumo_data.reset_index(drop=True)

    # Merge data
    combined_data = pd.merge(y_values, homo_lumo_data, left_on='id', right_on='Model')
    combined_data = combined_data.drop(columns=['Model'])

    print(f"Number of structures: {len(valid_structures)}")
    print(f"Shape of combined_data: {combined_data.shape}")
    print(f"Columns in combined_data: {combined_data.columns}")
    
    return valid_structures, combined_data

def extract_features(structures, combined_data):
    features = []

    # Elemental properties
    electronegativity = {el.symbol: el.X for el in Element}
    atomic_radii = {el.symbol: el.atomic_radius for el in Element}

    # Manually defined polarizability values
    polarizability_values = {
        'H': 0.6668, 'C': 1.76, 'N': 1.10, 'O': 0.802,
        'F': 0.557, 'Cl': 2.18, 'Br': 3.05, 'I': 5.35,
        'S': 2.90, 'P': 3.63, 'Si': 5.38
    }

    # Polar atoms for polar surface area calculation
    polar_atoms = ['O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'P']

    # Neighbor cutoff radius (in Å)
    neighbor_cutoff = 1.2  # Adjust as appropriate

    for i, structure in enumerate(structures):
        if i >= len(combined_data):
            break

        try:
            molecule = Molecule.from_sites(structure.sites)
        except Exception as e:
            print(f"Error processing structure {i}: {e}")
            continue
        
        num_atoms = len(molecule)
        total_mass = molecule.composition.weight
        avg_atomic_mass = (total_mass / num_atoms) if num_atoms else 0
        
        element_counts = molecule.composition.get_el_amt_dict()
        atom_type_fractions = {
            el: count / num_atoms for el, count in element_counts.items()
        }

        avg_electronegativity = np.mean([
            electronegativity.get(site.specie.symbol, 0)
            for site in molecule
        ])

        charges = np.array([
            electronegativity.get(site.specie.symbol, 0)
            for site in molecule
        ])
        coords = np.array(molecule.cart_coords)
        dipole_vector = np.sum(charges[:, np.newaxis] * coords, axis=0)
        dipole_moment = np.linalg.norm(dipole_vector)

        total_polarizability = sum(
            polarizability_values.get(site.specie.symbol, 0)
            for site in molecule
        )

        h_bond_donors = 0
        h_bond_acceptors = 0
        for site in molecule:
            symbol = site.specie.symbol
            neighbors = molecule.get_neighbors(site, r=neighbor_cutoff)
            if symbol == 'H':
                if any(neigh.specie.symbol in ['O', 'N', 'F']
                       for neigh in neighbors):
                    h_bond_donors += 1
            elif symbol in ['O', 'N', 'F']:
                h_bond_acceptors += 1

        rotatable_bonds = 0
        for site in molecule:
            symbol = site.specie.symbol
            if symbol != 'H':
                neighbors = molecule.get_neighbors(site, r=neighbor_cutoff)
                non_h_neighbors = [
                    neigh for neigh in neighbors
                    if neigh.specie.symbol != 'H'
                ]
                if len(non_h_neighbors) > 1:
                    rotatable_bonds += len(non_h_neighbors) - 1

        psa = sum(
            atomic_radii.get(site.specie.symbol, 0) ** 2
            for site in molecule if site.specie.symbol in polar_atoms
        )

        molecular_volume = sum(
            (4 / 3) * np.pi *
            (atomic_radii.get(site.specie.symbol, 0) ** 3)
            for site in molecule
        )
        molecular_surface_area = sum(
            4 * np.pi *
            (atomic_radii.get(site.specie.symbol, 0) ** 2)
            for site in molecule
        )

        hydroxyl_groups = 0
        for site in molecule:
            if site.specie.symbol == 'O':
                neighbors = molecule.get_neighbors(site, r=neighbor_cutoff)
                h_neighbors = [
                    neigh for neigh in neighbors
                    if neigh.specie.symbol == 'H'
                ]
                if h_neighbors:
                    hydroxyl_groups += 1

        feature = [
            num_atoms, total_mass, avg_atomic_mass,
            avg_electronegativity, dipole_moment,
            total_polarizability, h_bond_donors, h_bond_acceptors,
            rotatable_bonds, psa, molecular_volume,
            molecular_surface_area, hydroxyl_groups
        ]

        elements = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
        for element in elements:
            fraction = atom_type_fractions.get(element, 0)
            feature.append(fraction)

        homo = combined_data.iloc[i].get('HOMO', np.nan)
        lumo = combined_data.iloc[i].get('LUMO', np.nan)
        homo_lumo_gap = combined_data.iloc[i].get('HOMO-LUMO_gap', np.nan)
        feature.extend([homo, lumo, homo_lumo_gap])
        
        features.append(feature)
    
    columns = [
        'num_atoms', 'total_mass', 'avg_atomic_mass', 'avg_electronegativity',
        'dipole_moment', 'total_polarizability', 'h_bond_donors', 'h_bond_acceptors',
        'rotatable_bonds', 'polar_surface_area', 'molecular_volume', 'molecular_surface_area',
        'hydroxyl_groups'
    ]
    columns.extend([f'{el}_fraction' for el in ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']])
    columns.extend(['HOMO', 'LUMO', 'HOMO-LUMO_gap'])
    
    result = pd.DataFrame(features, columns=columns)
    print(f"Shape of extracted features: {result.shape}")
    return result

def plot_partial_dependence(model, X, features):
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        model, X, features=features, ax=ax
    )
    plt.tight_layout()
    plt.savefig('partial_dependence.png')
    plt.close()

    # Save data to CSV
    for feature in features:
        disp = PartialDependenceDisplay.from_estimator(
            model, X, features=[feature]
        )
        values = disp.pd_results[0]['values'][0]
        average = disp.pd_results[0]['average'][0]
        df = pd.DataFrame({
            feature: values,
            'Partial_Dependence': average
        })
        df.to_csv(f'partial_dependence_{feature}.csv', index=False)

def plot_shap_summary(model, X, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_values_df.to_csv('shap_values.csv', index=False)

    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=10
    )
    plt.tight_layout()
    plt.savefig('shap_summary_swarm.png', dpi=300)
    plt.close()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.savefig('residuals_vs_predicted.png')
    plt.close()

    pd.DataFrame({
        'Predicted_Values': y_pred,
        'Residuals': residuals
    }).to_csv('residuals_vs_predicted.csv', index=False)

def plot_correlation_matrix(X_scaled, feature_names):
    corr_matrix = pd.DataFrame(
        X_scaled, columns=feature_names
    ).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f"
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

    corr_matrix.to_csv('correlation_matrix.csv')

def plot_prediction_distribution(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_test, label='Actual', shade=True)
    sns.kdeplot(y_pred, label='Predicted', shade=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Actual vs Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    plt.close()
    
    pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    }).to_csv('prediction_distribution.csv', index=False)

def train_xgboost_model(X_scaled, y, X_unscaled=None):
    print_memory_usage()
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    if X_unscaled is not None:
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
            X_unscaled, y, test_size=0.2, random_state=42
        )

        X_test_u.to_csv("extracted_features_test.csv", index=False)

        X_test.to_csv("scaled_features_test.csv", index=False)

    param_dist = {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    model = XGBRegressor(random_state=42, tree_method='hist')
    print("Starting hyperparameter search...")
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=50,
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print("Hyperparameter search completed.")

    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")

    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=5,
        scoring='neg_mean_squared_error'
    )
    rmse_scores = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE scores: {rmse_scores}")
    print(f"Average RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test set RMSE: {rmse:.4f}")
    print(f"Test set MAE: {mae:.4f}")
    print(f"Test set R² Score: {r2:.4f}")

    return best_model, X_test, y_test, y_pred, random_search
# ===========================================================

def train_decision_tree_model(X_scaled, y):
    print_memory_usage()
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    param_grid = {'max_depth': [3, 5, 7, 9, None]}
    tree = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        tree, param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_tree = grid_search.best_estimator_

    y_pred = best_tree.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Decision Tree - Test set RMSE: {rmse:.4f}")
    print(f"Decision Tree - Test set MAE: {mae:.4f}")
    print(f"Decision Tree - Test set R² Score: {r2:.4f}")

    return best_tree, X_test, y_test

def plot_decision_tree(tree_model, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_model, feature_names=feature_names, filled=True,
        rounded=True, fontsize=10
    )
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree_visualization.png")
    plt.close()

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', lw=2
    )
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted")
    plt.savefig("true_vs_predicted.png")
    plt.close()

    pd.DataFrame({
        'True_Values': y_test,
        'Predicted_Values': y_pred
    }).to_csv('true_vs_predicted.csv', index=False)

def plot_learning_curve(model, X_scaled, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    test_rmse = np.sqrt(-test_scores.mean(axis=1))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, label='Training RMSE')
    plt.plot(train_sizes, test_rmse, label='Cross-validation RMSE')
    plt.legend()
    plt.xlabel('Training examples')
    plt.ylabel('RMSE')
    plt.title('Learning Curve')
    plt.savefig('learning_curve.png')
    plt.close()

    pd.DataFrame({
        'Training_Examples': train_sizes,
        'Training_RMSE': train_rmse,
        'Cross_Validation_RMSE': test_rmse
    }).to_csv('learning_curve.csv', index=False)

def train_ensemble_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    rf_params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42)
    rf_search = RandomizedSearchCV(
        rf, rf_params, n_iter=10, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=42
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    lgb_params = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100]
    }
    lgb = LGBMRegressor(random_state=42)
    lgb_search = RandomizedSearchCV(
        lgb, lgb_params, n_iter=10, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=42
    )
    lgb_search.fit(X_train, y_train)
    best_lgb = lgb_search.best_estimator_

    xgb_params = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    xgb = XGBRegressor(random_state=42, tree_method='hist')
    xgb_search = RandomizedSearchCV(
        xgb, xgb_params, n_iter=10, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1,
        random_state=42
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    ensemble = VotingRegressor([
        ('rf', best_rf),
        ('lgb', best_lgb),
        ('xgb', best_xgb)
    ])
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Ensemble Model - Test set RMSE: {rmse:.4f}")
    print(f"Ensemble Model - Test set MAE: {mae:.4f}")
    print(f"Ensemble Model - Test set R² Score: {r2:.4f}")

    return ensemble, X_test, y_test

def plot_feature_importance(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        print("The model does not have feature_importances_ attribute.")
        return
    
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

    pd.DataFrame({
        'Feature': np.array(feature_names)[sorted_idx],
        'Importance': importance[sorted_idx]
    }).to_csv('feature_importance.csv', index=False)

def plot_hyperparameter_performance(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    if ('param_n_estimators' not in results.columns) or ('param_max_depth' not in results.columns):
        print("Cannot plot hyperparameter performance: param_n_estimators or param_max_depth not in cv_results_.")
        return
    
    grouped = results.groupby(['param_n_estimators', 'param_max_depth'])['mean_test_score'].mean()
    pivot_table = grouped.reset_index().pivot(index='param_n_estimators', columns='param_max_depth', values='mean_test_score')
    pivot_table = -pivot_table  # Convert negative MSE to positive MSE

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f")
    plt.title('Hyperparameter Grid Search Results (MSE)')
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.tight_layout()
    plt.savefig('hyperparameter_heatmap.png')
    plt.close()

    pivot_table.to_csv('hyperparameter_heatmap_data.csv')

def main():
    print_memory_usage()
    cif_file_path = r"/*.cif"
    y_data_path = r"/id_prop.csv"
    homo_lumo_path = r".csv"
    
    structures, combined_data = load_data(cif_file_path, y_data_path, homo_lumo_path)
    print_memory_usage()
    
    X = extract_features(structures, combined_data)
    X.to_csv("extracted_features.csv", index=False) 
    y_values = combined_data['target'].values
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y_values: {y_values.shape}")
    print_memory_usage()

    X.fillna(X.mean(), inplace=True)

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    X_scaled_df.to_csv("scaled_features.csv", index=False)
    
    gc.collect()
    print_memory_usage()


    xgb_model, X_test_scaled, y_test, y_pred, random_search = train_xgboost_model(
        X_scaled_df,
        y_values,
        X_unscaled=X 
    )
    print_memory_usage()

    evaluate_model(xgb_model, X_test_scaled, y_test)
    plot_feature_importance(xgb_model, X.columns)
    plot_learning_curve(xgb_model, X_scaled_df, y_values)
    plot_correlation_matrix(X_scaled_array, X.columns.tolist())

    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    selected_features = ['avg_electronegativity', 'dipole_moment']
    missing_features = [feat for feat in selected_features if feat not in X_test_df.columns]
    if missing_features:
        print(f"Error: Missing features: {missing_features}")
    else:
        plot_partial_dependence(xgb_model, X_test_df, selected_features)

    plot_shap_summary(xgb_model, X_test_df, X.columns)
    plot_hyperparameter_performance(random_search)
    plot_residuals(y_test, y_pred)
    plot_prediction_distribution(y_test, y_pred)


    tree_model, X_test_dt, y_test_dt = train_decision_tree_model(X_scaled_df, y_values)
    plot_decision_tree(tree_model, X.columns.tolist())


    ensemble_model, X_test_ens, y_test_ens = train_ensemble_model(X_scaled_df, y_values)
    evaluate_model(ensemble_model, X_test_ens, y_test_ens)

    print_memory_usage()


    print("\n=== Starting Bootstrap-based XGBoost training for uncertainty analysis ===")
    X_df_for_bootstrap = pd.DataFrame(X_scaled_array, columns=X.columns)
    models_bootstrap, X_test_bs, y_test_bs = train_xgboost_model_bootstrap(X_df_for_bootstrap, y_values, n_bootstrap=10)
    evaluate_model_with_uncertainty(models_bootstrap, X_test_bs, y_test_bs)
    plot_feature_importance(models_bootstrap[0], X.columns)
    print_memory_usage()

if __name__ == "__main__":
    main()