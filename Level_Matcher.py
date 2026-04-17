import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Feature_Engineer import extract_features, generate_synthetic_training_data, parse_json_datasets
from Level_Clusterer import perform_clustering_and_output
from Training_Metrics_Visualizer import visualize_training_metrics
import warnings

"""
# High-level Structure and Workflow Explanation:
======================================

Workflow Diagram:
[Start]
   |
   v
[Step 1: Synthetic Data Generation] --> [Physics Rules]
   |                                  |
   v                                  v
[Step 2: Model Training] <---- [Synthetic Labels]
   |
   | (Model Ready: XGBoost)
   v
[Step 3: Test Data Ingestion] --> [Data Standardization (A, B, C)]
   |
   v
[Step 4: Pairwise Inference] --> [Feature Extraction] --> [Model Prediction]
   |
   v
[Step 5: Clustering] --> [Level_Clusterer Module] --> [Final Reconciled Scheme]
   |
   v
[End]

Workflow Steps (5-Step Pipeline):

Step 1: Synthetic Training Data Generation
  Input: Physics rules and configuration from Feature_Engineer.py
  Output: 580+ synthetic balanced labeled examples covering physics scenarios
  Function: generate_synthetic_training_data()

Step 2: Model Training
  Input: Training features and labels from Step 1
  Data Partitioning: 80% training, 20% validation
  Process: Train XGBoost with monotonic constraints (Physics Prior)
    - XGBoost: Gradient boosting with monotonic constraints enforcing physics priors
  Evaluation: RMSE, MAE, LogLoss, Feature Importance (Gain metric)
  Output: Trained XGBoost model with diagnostic visualizations

Step 3: Test Data Ingestion
  Source: Real-world datasets (K, L) in data/json/
  Input: Local JSON files (test_dataset_K.json, test_dataset_L.json) containing levels and gamma transitions
  Process: Standardizes energy, uncertainty, Spin-Parity strings, and gamma decay records into a unified DataFrame
  Constraint: Strictly for inference; these levels are never seen during the training phase.
  Function: parse_json_datasets()

Step 4: Pairwise Inference
  Input: Standardized test data (Step 3) and trained XGBoost model (Step 2)
  Process:
    - Generate all cross-dataset pairs
    - Extract feature vectors (Feature_Engineer.extract_features)
    - Apply XGBoost to calculate matching probabilities
  Output: Ranked list of candidate matches

Step 5: Constrained Clustering (Graph Partitioning)
  Input: Candidate match probabilities from Step 4
  Process: Enforce clique consistency and dataset uniqueness constraints
  Function: Level_Clusterer.perform_clustering_and_output()
  Output: Final unified level scheme reconciling datasets A, B, and C
"""

# Configuration Parameters
pairwise_output_threshold = 0.001  # Minimum probability for outputting level pairs (0.1%)
clustering_merge_threshold = 0.15  # Minimum probability for cluster merging (15%)

# Feature names for explicit labeling (prevents sklearn warnings)
feature_names = ['Energy_Similarity', 'Spin_Similarity', 'Parity_Similarity', 
                 'Specificity', 'Gamma_Decay_Pattern_Similarity']

if __name__ == "__main__":
    # ==========================================
    # Main Execution Pipeline:
    # 1. Synthetic Training Data Generation
    # 2. Model Training (XGBoost)
    # 3. Test Data Ingestion
    # 4. Pairwise Inference
    # 5. Graph-Based Clustering
    # ==========================================

    # ==========================================
    # Step 1: Synthetic Training Data Generation
    # ==========================================
    # Get training data from physics parser
    training_features, training_labels = generate_synthetic_training_data()
    
    # Convert to DataFrame with explicit feature names to prevent sklearn warnings
    training_dataframe = pd.DataFrame(training_features, columns=feature_names)
    
    # Split synthetic data into training and validation subsets (80/20 split)
    # Purpose of this split:
    #   Training Set (80%): Model learns patterns from these samples (gradient descent optimization)
    #   Validation Set (20%): Held-out samples used to monitor generalization WITHOUT updating model weights
    #                         Enables early stopping when validation performance stops improving (prevents overfitting)
    # Note: This is NOT the final test data (A, B, C datasets are completely separate real test data)
    X_train, X_validation, y_train, y_validation = train_test_split(
        training_dataframe, training_labels, test_size=0.2, random_state=42, stratify=None
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_validation)}")

    # Train XGBoost regressor with monotonic constraints enforcing physics rules
    # All five features designed as higher value → better match probability
    # ==========================================
    # Step 2: Model Training
    # ==========================================
    
    # 2a: Train XGBoost (Primary Model)
    print("\nTraining XGBoost Model...")
    model_xgboost = XGBRegressor(objective='binary:logistic',
                                       # Learning Objective Function: Training Loss + Regularization.
                                       # The loss function computes the difference between the true y value and the predicted y value.
                                       # The regularization term discourages overly complex trees.
                                       monotone_constraints='(1, 1, 1, 1, 1)',
                                       # Enforce increasing constraint on all five features.
                                       # Physics prior: higher feature values always indicate better matches.
                                       # Monotonic constraints improve predictive performance when strong prior beliefs exist.
                                       n_estimators=1000,
                                       # Number of gradient boosted trees or boosting rounds. Typical ranges for n_estimators are between 50 and 1000, with higher values generally leading to better performance but longer training times.
                                       max_depth=10,
                                       # Maximum tree depth. Lower values prevent overfitting but may underfit, while larger values allow the model to capture more complex relationships but may lead to overfitting.
                                       # Value of 3 balances model complexity with generalization.
                                       learning_rate=0.05,
                                       # Step size; determines the contribution of each tree to the final outcome by scaling the weights of the features.
                                       # Impact on model performance:
                                       # Lower values slow down learning but can improve generalization
                                       # Higher values speed up learning but may lead to suboptimal results
                                       # Interaction with the number of trees:
                                       # Lower learning rates typically require more trees
                                       # Higher learning rates may converge faster but require careful tuning of other parameters
                                       early_stopping_rounds=50,
                                       # Stop training if validation metric does not improve for 50 consecutive rounds
                                       # Prevents overfitting and reduces training time
                                       random_state=42)
                                       # Random number seed for reproducibility.
    
    # Train model with validation monitoring
    # Training process:
    #   1. Model sees X_train samples and adjusts internal weights to minimize prediction error (learns)
    #   2. After each boosting round, model predicts on X_validation (does NOT update weights)
    #   3. If validation error stops improving for 50 consecutive rounds, training halts (early stopping)
    # Result: Model learns from 80% of data while 20% acts as independent quality check
    # XGBoost automatically tracks multiple evaluation metrics during training
    evaluation_results = {}
    model_xgboost.fit(X_train, y_train, 
                     eval_set=[(X_train, y_train), (X_validation, y_validation)],
                     verbose=False)
    
    # Compute diagnostic metrics for regression quality
    train_predictions = model_xgboost.predict(X_train)
    validation_predictions = model_xgboost.predict(X_validation)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    validation_rmse = np.sqrt(mean_squared_error(y_validation, validation_predictions))
    train_mae = mean_absolute_error(y_train, train_predictions)
    validation_mae = mean_absolute_error(y_validation, validation_predictions)
    
    # Manual binary cross-entropy computation (sklearn log_loss incompatible with continuous probabilities)
    # Convert labels to numpy array to ensure safe element-wise multiplication with prediction arrays
    y_train_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_validation_np = y_validation.values if hasattr(y_validation, 'values') else np.array(y_validation)
    
    train_predictions_clipped = np.clip(train_predictions, 1e-15, 1 - 1e-15)
    validation_predictions_clipped = np.clip(validation_predictions, 1e-15, 1 - 1e-15)
    train_logloss = -np.mean(y_train_np * np.log(train_predictions_clipped) + (1 - y_train_np) * np.log(1 - train_predictions_clipped))
    validation_logloss = -np.mean(y_validation_np * np.log(validation_predictions_clipped) + (1 - y_validation_np) * np.log(1 - validation_predictions_clipped))
    
    # Extract feature importance (Gain metric: average loss reduction per feature)
    xgboost_feature_importance = model_xgboost.get_booster().get_score(importance_type='gain')
    
    print(f"XGBoost Training Complete (stopped at iteration {model_xgboost.best_iteration})")
    print(f"  Train RMSE: {train_rmse:.4f} | Validation RMSE: {validation_rmse:.4f}")
    print(f"  Train MAE:  {train_mae:.4f} | Validation MAE:  {validation_mae:.4f}")
    print(f"  Train LogLoss: {train_logloss:.4f} | Validation LogLoss: {validation_logloss:.4f}")

    # ==========================================
    # Step 2b: Visualize Training Diagnostics
    # ==========================================
    # Generate comprehensive diagnostic plots for XGBoost model
    print("\nGenerating Training Metrics Visualization...")
    xgboost_metrics_data = {
        'train_rmse': train_rmse,
        'validation_rmse': validation_rmse,
        'train_mae': train_mae,
        'validation_mae': validation_mae,
        'train_logloss': train_logloss,
        'validation_logloss': validation_logloss,
        'feature_importance': xgboost_feature_importance,
        'best_iteration': model_xgboost.best_iteration,
        'maximum_iterations': 1000
    }
    visualize_training_metrics(xgboost_metrics_data)

    # ==========================================
    # Step 3: Test Data Ingestion
    # ==========================================
    # Load level and gamma test data from JSON files for matching.
    # Logic: These datasets (A, B, C) are used solely for inference and were not seen during training.
    # Logic: These datasets (K, L) are the real-world inference targets; never seen during the synthetic training phase.

    print("Ingesting Test Data...")
    levels = parse_json_datasets(['K', 'L'])
    dataframe = pd.DataFrame(levels)
    dataframe['level_id'] = dataframe.apply(lambda row: f"{row['dataset_code']}_{int(row['energy_value'])}", axis=1)

    # ==========================================
    # Step 4: Pairwise Inference
    # ==========================================
    # Predict match probabilities for all cross-dataset level pairs using XGBoost
    matching_level_pairs_xgboost = []
    all_pairwise_results_for_display = []

    level_dataframe_rows_list = list(dataframe.iterrows())
    
    print("\nRunning Pairwise Inference (XGBoost)...")
    for i in range(len(level_dataframe_rows_list)):
        for j in range(i + 1, len(level_dataframe_rows_list)):
            _, level_1 = level_dataframe_rows_list[i]
            _, level_2 = level_dataframe_rows_list[j]
            
            # Skip same-dataset pairs (only match across different datasets)
            if level_1['dataset_code'] == level_2['dataset_code']:
                continue

            # Extract the input features for each level pair
            feature_vector = extract_features(level_1, level_2)
            
            # Prediction: Use XGBoost to predict match probability for this input feature vector
            # Convert to DataFrame with explicit feature names
            feature_dataframe = pd.DataFrame([feature_vector], columns=feature_names)
            xgboost_probability = model_xgboost.predict(feature_dataframe)[0]

            # Record level pairs above output threshold
            if xgboost_probability > pairwise_output_threshold:
                all_pairwise_results_for_display.append({
                    'level_1_id': level_1['level_id'], 'level_2_id': level_2['level_id'],
                    'dataset_code_1': level_1['dataset_code'], 'dataset_code_2': level_2['dataset_code'],
                    'xgboost_probability': xgboost_probability,
                    'features': feature_vector
                })
                # Prepare list for clustering
                matching_level_pairs_xgboost.append({
                    'level_1_id': level_1['level_id'],
                    'level_2_id': level_2['level_id'],
                    'dataset_code_1': level_1['dataset_code'],
                    'dataset_code_2': level_2['dataset_code'],
                    'probability': xgboost_probability,
                    'features': feature_vector
                })

    # Sort by XGBoost probability descending
    all_pairwise_results_for_display.sort(key=lambda x: x['xgboost_probability'], reverse=True)
    matching_level_pairs_xgboost.sort(key=lambda x: x['probability'], reverse=True)

    # Write pairwise inference results to file (Side-by-Side Comparison)
    threshold_percent = pairwise_output_threshold * 100
    with open('outputs/pairwise/Output_Level_Pairwise_Inference.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(f"=== PAIRWISE INFERENCE RESULTS (>{threshold_percent:.1f}%) ===\n")
        output_file.write(f"Model: XGBoost\n")
        output_file.write(f"Total Level Pairs Found: {len(all_pairwise_results_for_display)}\n\n")
        
        for item in all_pairwise_results_for_display:
            energy_similarity, spin_similarity, parity_similarity, specificity, gamma_decay_pattern_similarity = item['features']
            output_file.write(
                f"{item['level_1_id']} <-> {item['level_2_id']} | "
                f"XGBoost: {item['xgboost_probability']:.1%}\n"
                f"  Features: Energy_Similarity={energy_similarity:.2f}, Spin_Similarity={spin_similarity:.2f}, "
                f"Parity_Similarity={parity_similarity:.2f}, Specificity={specificity:.2f}, Gamma_Pattern_Similarity={gamma_decay_pattern_similarity:.2f}\n\n"
            )
    
    print(f"\n[INFO] Pairwise Inference Complete: {len(all_pairwise_results_for_display)} level pairs (>{threshold_percent:.1f}%) written to 'outputs/pairwise/Output_Level_Pairwise_Inference.txt'")

    # ==========================================
    # Step 5: Graph-Based Clustering
    # ==========================================
    
    # Execute clustering with XGBoost results
    perform_clustering_and_output(matching_level_pairs_xgboost, model_xgboost, "outputs/clustering/Output_Clustering_Results_XGBoost.txt", "XGBoost", dataframe)
