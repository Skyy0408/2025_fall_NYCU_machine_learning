# --- 1. 匯入函式庫 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# JAX 和相關函式庫
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

# Sklearn 函式庫 (我們保留大部分的預處理和評估工具)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

# --- 2. 定義 Flax 模型 ---
# 我們定義一個繼承 nn.Module 的類別，來描述神經網路的架構
class MLP(nn.Module):
    # 在 setup 中定義模型的層
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# --- 3. 定義訓練狀態 ---
# TrainState 是一個方便的容器，用來存放所有與訓練相關的狀態
class TrainState(train_state.TrainState):
    pass

# --- 4. 定義訓練和評估步驟 ---
# @jax.jit 是一個非常重要的裝飾器，它會將 Python 函式編譯成高效能的 JAX 可執行碼
@jax.jit
def train_step(state, batch_x, batch_y):
    """單一訓練步驟的函式"""
    def loss_fn(params):
        # 使用 state.apply_fn 來執行模型的前向傳播
        predictions = state.apply_fn({'params': params}, batch_x).flatten()
        # 計算損失 (MSE)
        loss = jnp.mean((predictions - batch_y) ** 2)
        return loss, predictions

    # 計算損失和梯度
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # 更新 state
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch_x, batch_y):
    """單一評估步驟的函式"""
    predictions = state.apply_fn({'params': state.params}, batch_x).flatten()
    loss = jnp.mean((predictions - batch_y) ** 2)
    return loss, predictions

# --- 5. 評估與繪圖函式 (與之前版本幾乎相同) ---
# 唯一的差別是 y_pred 現在是直接傳入，而不是在函式內部計算
def evaluate_and_plot_regression(y_true, y_pred, X_coords, dataset_name, save_dir):
    """評估回歸模型並儲存圖表"""
    print(f"--- Evaluating on {dataset_name} Set ---")

    # Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    
    print(f"Temperature MSE: {mse:.4f}")
    print(f"Temperature RMSE: {rmse:.4f} (°C)")
    print(f"Temperature MAE: {mae:.4f} (°C)")
    print(f"Max Temperature Error: {max_err:.4f} (°C)\n")

    # (繪圖部分與 TensorFlow 版本完全相同，故省略以保持簡潔)
    # ... 您可以將之前版本中的繪圖程式碼貼到這裡 ...
    # --- Plotting and Saving ---
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    
    # Plot 1: Actual Temperature Distribution
    plt.figure(figsize=(10, 8))
    title1 = f'JAX_Actual_Temperature_{dataset_name}_Set'
    scatter1 = plt.scatter(X_coords['longitude'], X_coords['latitude'], c=y_true, cmap='viridis', 
                           vmin=vmin, vmax=vmax, s=15)
    plt.colorbar(scatter1, label='Actual Temperature (°C)')
    plt.title(f'JAX Actual Temperature Distribution ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    save_path1 = save_dir / f"{title1}.png"
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path1}")

    # Plot 2: Predicted Temperature Distribution
    plt.figure(figsize=(10, 8))
    title2 = f'JAX_Predicted_Temperature_{dataset_name}_Set'
    scatter2 = plt.scatter(X_coords['longitude'], X_coords['latitude'], c=y_pred, cmap='viridis',
                           vmin=vmin, vmax=vmax, s=15)
    plt.colorbar(scatter2, label='Predicted Temperature (°C)')
    plt.title(f'JAX Predicted Temperature Distribution ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    save_path2 = save_dir / f"{title2}.png"
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path2}")

    # Plot 3: Temperature Prediction Error
    plt.figure(figsize=(10, 8))
    title3 = f'JAX_Temperature_Error_{dataset_name}_Set'
    errors = y_pred - y_true
    error_max_abs = np.abs(errors).max()
    scatter3 = plt.scatter(X_coords['longitude'], X_coords['latitude'], c=errors, cmap='coolwarm',
                           vmin=-error_max_abs, vmax=error_max_abs, s=15)
    plt.colorbar(scatter3, label='Prediction Error (°C)')
    plt.title(f'JAX Temperature Prediction Error ({dataset_name} Set)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    save_path3 = save_dir / f"{title3}.png"
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path3}\n")


# --- 6. 主要執行函式 ---
def main():
    # 設定隨機種子以確保可重現性
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    # 讀取和準備資料 (與之前相同)
    script_dir = Path(__file__).parent.resolve()
    data_file_path = script_dir / 'regression_data.csv'
    data = pd.read_csv(data_file_path)
    X = data[['longitude', 'latitude']]
    y = data['value'].values # 將 y 轉為 numpy array

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1/3), random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 初始化模型和訓練狀態
    model = MLP()
    key, init_key = jax.random.split(key)
    # JAX 需要一個假的輸入來初始化模型的參數形狀
    params = model.init(init_key, jnp.ones((1, X_train_scaled.shape[1])))['params']
    
    optimizer = optax.adam(learning_rate=1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # 手動訓練迴圈與 Early Stopping 實作
    epochs = 10000
    batch_size = 32
    patience = 3000 # 與您 TensorFlow 版本設定相同
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    print("\nTraining the JAX/Flax model for regression...")
    for epoch in range(epochs):
        # 訓練步驟
        epoch_loss = 0
        num_batches = len(X_train_scaled) // batch_size
        perms = jax.random.permutation(key, len(X_train_scaled))
        key, _ = jax.random.split(key)
        X_train_shuffled = X_train_scaled[perms]
        y_train_shuffled = y_train[perms]

        for i in range(num_batches):
            batch_x = X_train_shuffled[i*batch_size : (i+1)*batch_size]
            batch_y = y_train_shuffled[i*batch_size : (i+1)*batch_size]
            state, loss = train_step(state, batch_x, batch_y)
            epoch_loss += loss
        
        train_loss = epoch_loss / num_batches

        # 驗證步驟
        val_loss, _ = eval_step(state, X_val_scaled, y_val)
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        # Early Stopping 邏輯
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = state # 儲存最佳的模型狀態
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # 使用驗證集上最好的模型狀態
    if best_state:
        state = best_state
        
    print("Model training complete.\n")

    # 評估與繪圖
    # 我們需要從 NumPy 陣列重新取得 DataFrame 以便繪圖
    X_train_coords = pd.DataFrame(X_train, columns=['longitude', 'latitude'])
    X_val_coords = pd.DataFrame(X_val, columns=['longitude', 'latitude'])
    X_test_coords = pd.DataFrame(X_test, columns=['longitude', 'latitude'])
    
    _, y_train_pred = eval_step(state, X_train_scaled, y_train)
    _, y_val_pred = eval_step(state, X_val_scaled, y_val)
    _, y_test_pred = eval_step(state, X_test_scaled, y_test)

    evaluate_and_plot_regression(np.array(y_train), np.array(y_train_pred), X_train_coords, 'Training', script_dir)
    evaluate_and_plot_regression(np.array(y_val), np.array(y_val_pred), X_val_coords, 'Validation', script_dir)
    evaluate_and_plot_regression(np.array(y_test), np.array(y_test_pred), X_test_coords, 'Test', script_dir)

if __name__ == '__main__':
    main()