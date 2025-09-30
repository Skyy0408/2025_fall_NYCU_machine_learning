import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from pathlib import Path

def evaluate_and_plot_regression_nn(model, X_scaled, y, dataset_name, save_dir):
    """
    Evaluates the regression neural network and saves the resulting plots.
    """
    print(f"--- Evaluating on {dataset_name} Set ---")

    # Predictions
    y_pred = model.predict(X_scaled).flatten() # Flatten to make it a 1D array

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    max_err = max_error(y, y_pred)
    
    print(f"Temperature MSE: {mse:.4f}")
    print(f"Temperature RMSE: {rmse:.4f} (°C)")
    print(f"Temperature MAE: {mae:.4f} (°C)")
    print(f"Max Temperature Error: {max_err:.4f} (°C)\n")

    # For plotting, unscale the features
    X_unscaled = scaler.inverse_transform(X_scaled)
    
    # --- Plotting and Saving ---
    vmin = min(y.min(), y_pred.min())
    vmax = max(y.max(), y_pred.max())
    
    # Plot 1: Actual Temperature Distribution
    plt.figure(figsize=(10, 8))
    title1 = f'NN_Actual_Temperature_{dataset_name}_Set'
    scatter1 = plt.scatter(X_unscaled[:, 0], X_unscaled[:, 1], c=y, cmap='viridis', 
                           vmin=vmin, vmax=vmax, s=15)
    plt.colorbar(scatter1, label='Actual Temperature (°C)')
    plt.title(f'NN Actual Temperature Distribution ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    save_path1 = save_dir / f"{title1}.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path1}")

    # Plot 2: Predicted Temperature Distribution
    plt.figure(figsize=(10, 8))
    title2 = f'NN_Predicted_Temperature_{dataset_name}_Set'
    scatter2 = plt.scatter(X_unscaled[:, 0], X_unscaled[:, 1], c=y_pred, cmap='viridis',
                           vmin=vmin, vmax=vmax, s=15)
    plt.colorbar(scatter2, label='Predicted Temperature (°C)')
    plt.title(f'NN Predicted Temperature Distribution ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    save_path2 = save_dir / f"{title2}.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path2}")

    # Plot 3: Temperature Prediction Error
    plt.figure(figsize=(10, 8))
    title3 = f'NN_Temperature_Error_{dataset_name}_Set'
    errors = y_pred - y
    error_max_abs = np.abs(errors).max()
    scatter3 = plt.scatter(X_unscaled[:, 0], X_unscaled[:, 1], c=errors, cmap='coolwarm',
                           vmin=-error_max_abs, vmax=error_max_abs, s=15)
    plt.colorbar(scatter3, label='Prediction Error (°C)')
    plt.title(f'NN Temperature Prediction Error ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    save_path3 = save_dir / f"{title3}.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path3}\n")


def main():
    seed_value = 42
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    script_dir = Path(__file__).parent.resolve()
    data_file_path = script_dir / 'regression_data.csv'
    
    try:
        data = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"Error: '{data_file_path}' not found.")
        return

    X = data[['longitude', 'latitude']]
    y = data['value']

    # 1. Data Splitting
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1/3), random_state=42)

    # 2. Data Scaling
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)), # Input => Hidden Layer 1
        Dense(64, activation='relu'),                  # Hidden Layer 1 => Hidden Layer 2
        Dense(1)                                       # Output Layer (1 neuron, linear activation)
    ])

    # 4. Compile Model
    model.compile(optimizer='adam',
                  loss='mean_squared_error', # Standard loss for regression
                  metrics=['mae', 'mse']) # Track Mean Absolute Error and Mean Squared Error
    
    model.summary()

    # 5. Train with Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\nTraining the Neural Network model for regression...")
    history = model.fit(X_train_scaled, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val_scaled, y_val),
                        callbacks=[early_stopping],
                        verbose=2)
    print("Model training complete.\n")

    # 6. Evaluate and Plot
    evaluate_and_plot_regression_nn(model, X_train_scaled, y_train, 'Training', script_dir)
    evaluate_and_plot_regression_nn(model, X_val_scaled, y_val, 'Validation', script_dir)
    evaluate_and_plot_regression_nn(model, X_test_scaled, y_test, 'Test', script_dir)

if __name__ == '__main__':
    main()