import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, brier_score_loss
from pathlib import Path

def evaluate_and_plot_nn(model, X_scaled, y_ohe, dataset_name, save_dir):
    print(f"--- Evaluating on {dataset_name} Set ---")
    
    # Predictions
    y_prob = model.predict(X_scaled)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_ohe, axis=1)
    
    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob[:, 1])

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confidence MSE (Brier Score): {brier:.4f}\n")

    # For plotting
    X_unscaled = scaler.inverse_transform(X_scaled)
    
    # Accuracy Plot
    plt.figure(figsize=(10, 8))
    title1 = f'NN_Prediction_Correctness_{dataset_name}_Set'
    correct_predictions = (y_true == y_pred)
    plt.scatter(X_unscaled[correct_predictions, 0], X_unscaled[correct_predictions, 1], 
                c='green', label='Correct', alpha=0.6, s=10)
    plt.scatter(X_unscaled[~correct_predictions, 0], X_unscaled[~correct_predictions, 1], 
                c='red', label='Incorrect', alpha=0.6, s=10)
    plt.title(f'NN Prediction Correctness ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    save_path1 = save_dir / f"{title1}.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path1}")

    # Confidence Plot
    plt.figure(figsize=(10, 8))
    title2 = f'NN_Prediction_Confidence_{dataset_name}_Set'
    scatter = plt.scatter(X_unscaled[:, 0], X_unscaled[:, 1], c=y_prob[:, 1], 
                          cmap='coolwarm', vmin=0, vmax=1, s=10)
    plt.colorbar(scatter, label='Probability of being Valid (Label=1)')
    plt.title(f'NN Prediction Confidence ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    save_path2 = save_dir / f"{title2}.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path2}\n")

def main():
    seed_value = 42
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    script_dir = Path(__file__).parent.resolve()
    data_file_path = script_dir / 'classification_data.csv'
    
    try:
        data = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"Error: '{data_file_path}' not found.")
        return

    X = data[['longitude', 'latitude']]
    y = data['label']

    # 1. One-hot Encode
    y_ohe = to_categorical(y, num_classes=2)

    # 2. Data Splitting
    X_train, X_temp, y_train_ohe, y_temp_ohe = train_test_split(X, y_ohe, test_size=0.3, random_state=42, stratify=y_ohe)
    X_val, X_test, y_val_ohe, y_test_ohe = train_test_split(X_temp, y_temp_ohe, test_size=(1/3), random_state=42, stratify=y_temp_ohe)

    # 3. Data Scaling
    global scaler 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(2,)), # Input => Hidden Layer 1 
        Dense(32, activation='relu'),                  # Hidden Layer 1 => Hidden Layer 2
        Dense(2, activation='softmax')                 # Output Layer (Softmax)
    ])

    # 5. Compile
    model.compile(optimizer='adam',
                  
                  # Option 1: Euclidean Distance
                  #loss='mean_squared_error',
                  
                  # Option 2: Cosine Similarity
                  #loss='cosine_similarity',
                  
                  # Option 3: Categorical Crossentropy
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary() 

    # 6. Train with Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\nTraining the Neural Network model...")
    history = model.fit(X_train_scaled, y_train_ohe,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val_scaled, y_val_ohe),
                        callbacks=[early_stopping],
                        verbose=2) 
    print("Model training complete.\n")

    # 7. Evaluate and Plot
    evaluate_and_plot_nn(model, X_train_scaled, y_train_ohe, 'Training', script_dir)
    evaluate_and_plot_nn(model, X_val_scaled, y_val_ohe, 'Validation', script_dir)
    evaluate_and_plot_nn(model, X_test_scaled, y_test_ohe, 'Test', script_dir)

if __name__ == '__main__':
    main()