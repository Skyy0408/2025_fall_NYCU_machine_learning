import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import jax.lax # 引入 jax.lax 以使用 while_loop

# --- The Runge Function ---
def f(x):
    return 1/(1+25*x**2)

grad_f_true = jax.grad(f)
vmap_grad_f_true = jax.vmap(grad_f_true)

# --- Hyperparameters ---
epochs = 40000  # Armijo 可能收斂較慢，可以先用較少 epoch 測試
datanum = 10001
batch_size = 128 # 較大的 batch size 對 line search 更穩定
n = 16
r = 0.5
rho = 0.6

# --- Data Generation ---
x_center = jnp.linspace(-0.5, 0.5, int(datanum * r))
x_sides = jnp.concatenate([jnp.linspace(-1.0, -0.5, int(datanum * (1-r)/2)), jnp.linspace(0.5, 1.0, int(datanum *(1-r)/2))])
x_train = jnp.concatenate([x_center, x_sides])
x_train = jnp.sort(x_train)
y_train = f(x_train)

key = jax.random.PRNGKey(0)

# Validation Data
key, validation_key = jax.random.split(key)
x_validation = jax.random.uniform(validation_key, shape=(4000,), minval=-1.0, maxval=1.0)
y_validation = f(x_validation)

# --- Initialize Parameters ---
key, w1_key, b1_key, w2_key, b2_key, w3_key, b3_key = jax.random.split(key, 7)
glorot_normal_initializer = jax.nn.initializers.glorot_normal()
params = {
    'w1': glorot_normal_initializer(w1_key, (1, n)), 'b1': jnp.zeros(n,),
    'w2': glorot_normal_initializer(w2_key, (n, n)), 'b2': jnp.zeros(n,),
    'w3': glorot_normal_initializer(w3_key, (n, 1)), 'b3': jnp.zeros(1,)
}

# --- Model Definition ---
def deep_model(params, x):
    x = x.reshape(-1, 1)
    hidden1 = jax.nn.tanh(x @ params['w1'] + params['b1'])
    hidden2 = jax.nn.tanh(hidden1 @ params['w2'] + params['b2'])
    output = hidden2 @ params['w3'] + params['b3']
    return output

def mse(params, x, y):
    predictions = deep_model(params, x).squeeze()
    return jnp.mean((predictions - y)**2)

# --- Derivative Calculation ---
def model_for_grad(x, model_params):
    return deep_model(model_params, x).squeeze()[()]

model_derivative_fn = jax.grad(model_for_grad, argnums=0)
vmap_model_derivative = jax.vmap(model_derivative_fn, in_axes=(0, None))

# --- Loss Function ---
@jax.jit
def loss_fn(params, x, y):
    pred_y = deep_model(params, x).squeeze()
    loss_f = jnp.mean((pred_y - y)**2)
    true_dy = vmap_grad_f_true(x)
    pred_dy = vmap_model_derivative(x, params)
    weights = jnp.square(true_dy) + 1.0
    loss_df = jnp.mean(weights * (pred_dy - true_dy)**2)
    return (1-rho)*loss_f + rho * loss_df

# --- Armijo Line Search (已修正) ---
def armijo_line_search(params, grads, x_batch, y_batch, sigma=0.1, beta=0.5, alpha_init=1.0):
    p = jax.tree_util.tree_map(lambda g: -g, grads)
    
    # <<< 關鍵修改：使用 tree_map 和 tree_sum 計算點積 >>>
    # 這樣做更高效且為 JAX 公開 API
    grad_p_products = jax.tree_util.tree_map(lambda g, p_leaf: jnp.sum(g * p_leaf), grads, p)
    
    # <<< 關鍵修改：先取出所有葉節點，再加總 >>>
    leaves = jax.tree_util.tree_leaves(grad_p_products)
    grad_dot_p_scalar = jnp.sum(jnp.array(leaves))
    
    current_loss = loss_fn(params, x_batch, y_batch)

    def cond_fun(alpha):
        new_params = jax.tree_util.tree_map(lambda p_old, p_dir: p_old + alpha * p_dir, params, p)
        return loss_fn(new_params, x_batch, y_batch) > current_loss + sigma * alpha * grad_dot_p_scalar

    def body_fun(alpha):
        return alpha * beta

    final_alpha = jax.lax.while_loop(cond_fun, body_fun, alpha_init)
    return final_alpha

# --- JIT-compiled Training Epoch (已修正) ---
@jax.jit
def train_epoch(params, train_data, permutation):
    x_train, y_train = train_data
    steps_per_epoch = len(x_train) // batch_size
    
    # 取得 loss_and_grad 函數
    loss_and_grad_fn = jax.value_and_grad(loss_fn)

    def body_fun(step, current_params):
        start_idx = step * batch_size
        batch_idx = jax.lax.dynamic_slice_in_dim(permutation, start_idx, batch_size)
        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
        
        # 1. 計算梯度
        # 這裡不需重複呼叫 jax.grad，直接用 grad_fn 即可
        grads = jax.grad(loss_fn)(current_params, x_batch, y_batch)
        
        # 2. 使用 Armijo Line Search 找到學習率
        learning_rate = armijo_line_search(current_params, grads, x_batch, y_batch, alpha_init=1.0)
        
        # 3. 用找到的 learning_rate 更新參數
        return jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, current_params, grads)

    params = jax.lax.fori_loop(0, steps_per_epoch, body_fun, params)
    return params

# --- Training Loop ---
loss_history = []
key, shuffle_key = jax.random.split(key)
num_train = len(x_train)
print("\n---Start Training with Armijo Line Search---")
start_time = time.time()

for epoch in range(epochs):
    shuffle_key, perm_key = jax.random.split(shuffle_key)
    perm = jax.random.permutation(perm_key, num_train)
    params = train_epoch(params, (x_train, y_train), perm)
    
    if epoch % 1000 == 0: # 增加打印頻率以觀察收斂情況
        loss = loss_fn(params, x_train, y_train)
        loss_history.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- Evaluation and Plotting (與你的版本相同) ---
x_plot = jnp.linspace(-1, 1, 500, dtype=jnp.float32)
y_true = f(x_plot)
y_pred = deep_model(params, x_plot).squeeze()
y_pred_train = deep_model(params, x_train).squeeze()
final_mse = mse(params, x_train, y_train)
max_error = jnp.max(jnp.abs(y_pred_train - y_train))
y_pred_validation= deep_model(params, x_validation).squeeze()
validation_mse = mse(params, x_validation, y_validation)
validation_max_error = jnp.max(jnp.abs(y_pred_validation - y_validation))

print(f"\n--- Final Result ---")
print(f"Final Training MSE: {final_mse:.6e}")
print(f"Final Training Max Error: {max_error:.6e}")
print(f"Final Validation MSE: {validation_mse:.6e}")
print(f"Final Validation Max Error: {validation_max_error:.6e}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, label='True Runge Function')
plt.plot(x_plot, y_pred, label='Deep Tanh NN', linestyle='--')
plt.scatter(x_validation, y_validation, label='Validation', color='orange', s=10, alpha=0.6)
plt.title('Deep Network Approximation')
plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(0, epochs, 1000), loss_history) # 匹配上面的打印頻率
plt.title('Training Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log'); plt.grid(True)
plt.tight_layout()
plt.show()

y_d_true = vmap_grad_f_true(x_plot)
y_d_pred = vmap_model_derivative(x_plot, params)
final_derivative_mse = jnp.mean((vmap_model_derivative(x_train, params) - vmap_grad_f_true(x_train))**2)
max_derivative_error = jnp.max(jnp.abs(y_d_pred-y_d_true))
validation_derivative_mse = jnp.mean((vmap_model_derivative(x_validation, params) - vmap_grad_f_true(x_validation))**2)
validation_max_derivative_error = jnp.max(jnp.abs(vmap_model_derivative(x_validation, params) - vmap_grad_f_true(x_validation)))

print(f"\n--- Final Result (Derivatives) ---")
print(f"Final Training MSE (Derivative): {final_derivative_mse:.6e}")
print(f"Final Training Max Error (Derivative): {max_derivative_error:.6e}")
print(f"Final Validation MSE (Derivative): {validation_derivative_mse:.6e}")
print(f"Final Validation Max Error (Derivative): {validation_max_derivative_error:.6e}")

plt.figure(figsize=(8,6))
plt.plot(x_plot, y_d_true, label='True Derivative of Runge Function')
plt.plot(x_plot, y_d_pred, label='Deep Tanh NN Derivative', linestyle='--')
plt.scatter(x_validation, vmap_grad_f_true(x_validation), label='Validation (True)', color='orange', s=10, alpha=0.6)
plt.title("Comparison of Derivatives")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()