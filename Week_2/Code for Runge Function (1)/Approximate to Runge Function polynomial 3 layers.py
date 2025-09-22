import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# 目標函式 (Runge function)
def f(x):
    return 1 / (1 + 25 * x**2)

# --- 主要修改部分 (1): 新增神經網路的啟動函式 ---
def sigmoid(x):
    """Sigmoid 啟動函式"""
    return 1 / (1 + jnp.exp(-x))

# --- 主要修改部分 (2): 定義一個有三層隱藏層的深度神經網路 ---
def deep_3_layer_model(params, x):
    """一個 1 -> 16 -> 16 -> 16 -> 1 的三層隱藏層神經網路"""
    x = x.reshape(-1, 1) # 確保輸入是二維的
    
    # 第一層隱藏層
    hidden1 = sigmoid(x @ params['w1'] + params['b1'])
    # 第二層隱藏層
    hidden2 = sigmoid(hidden1 @ params['w2'] + params['b2'])
    # 第三層隱藏層
    hidden3 = sigmoid(hidden2 @ params['w3'] + params['b3'])
    # 輸出層 (迴歸任務通常不用啟動函式)
    output = hidden3 @ params['w4'] + params['b4']
    
    return output

# --- 主要修改部分 (3): 為新的三層神經網路初始化參數 ---
key = jax.random.PRNGKey(0)
# 我們需要為每一層都建立權重(w)和偏置(b)
keys = jax.random.split(key, 9) # 總共需要 4 組 w, b + 1 個 subkey
params = {
    'w1': jax.random.normal(keys[0], (1, 16)), 'b1': jax.random.normal(keys[1], (16,)),
    'w2': jax.random.normal(keys[2], (16, 16)), 'b2': jax.random.normal(keys[3], (16,)),
    'w3': jax.random.normal(keys[4], (16, 16)), 'b3': jax.random.normal(keys[5], (16,)),
    'w4': jax.random.normal(keys[6], (16, 1)), 'b4': jax.random.normal(keys[7], (1,))
}

# --- 主要修改部分 (4): 更新損失函式，讓它呼叫新的模型 ---
def loss_fn(params, x, y):
    predictions = deep_3_layer_model(params, x).squeeze() 
    return jnp.mean((predictions - y)**2)

# JIT 編譯的更新函式 (這部分不需要修改)
@jax.jit
def update_step(params, x, y, learning_rate):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.util.tree_map(lambda p, g: p - learning_rate * g, params, grads)

# --- 訓練設定 (這部分不需要修改) ---
learning_rate = 0.01
epochs = 40000
datanum = 1000

# 產生訓練資料 (使用均勻分布的點)
x_train = jnp.linspace(-1.0, 1.0, datanum)
y_train = f(x_train)

loss_history = []

# --- 訓練迴圈 (這部分不需要修改) ---
# 使用 Mini-batch 來訓練會更穩定
batch_size = 64
num_train = len(x_train)
steps_per_epoch = num_train // batch_size

for epoch in range(epochs):
    # 每個 epoch 都重新洗牌資料
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, num_train)
    
    for step in range(steps_per_epoch):
        batch_idx = perm[step * batch_size : (step + 1) * batch_size]
        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
        params = update_step(params, x_batch, y_batch, learning_rate)
    
    if epoch % 1000 == 0:
        loss = loss_fn(params, x_train, y_train)
        loss_history.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# --- 繪圖與結果展示 (這部分不需要修改) ---
x_plot = jnp.linspace(-1, 1, 500)
y_true = f(x_plot)
y_pred = deep_3_layer_model(params, x_plot).squeeze()

final_mse = loss_fn(params, x_train, y_train)
max_error = jnp.max(jnp.abs(deep_3_layer_model(params, x_train).squeeze() - y_train))

print("\n--- 最終結果 ---")
print(f"訓練完成後的 MSE (3-Layer NN): {final_mse:.6f}")
print(f"訓練完成後的 Max Error: {max_error:.6f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, label='True Runge Function', color='blue')
plt.plot(x_plot, y_pred, label='3-Layer NN Prediction', color='red', linestyle='--')
plt.scatter(x_train, y_train, s=10, color='gray', alpha=0.5, label='Training Data')
plt.title('Function Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(0, epochs, 1000), loss_history)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()