import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time # Import time for performance comparison

# The Runge Function
def f(x):
    return 1/(1+25*x**2)
grad_f = jax.grad(f)
vmap_grad_f = jax.vmap(grad_f)
# Hyperparameters
learning_rate = 0.01
epochs = 40000 # Stop Criterion
datanum = 10001
batch_size = 32
n = 32
rho = 0.6 # Weight for derivative loss.

# Data
x_train = jnp.linspace(-1.0, 1.0, datanum, dtype=jnp.float32)
y_train = f(x_train)


key = jax.random.PRNGKey(0)

# Validation Data
key, validation_key = jax.random.split(key)
x_validation = jax.random.uniform(validation_key, shape=(4000,), minval=-1.0, maxval=1.0)
y_validation = f(x_validation)

key, w1_key, b1_key, w2_key, b2_key, w3_key, b3_key, w4_key, b4_key, w5_key, b5_key = jax.random.split(key, 11)
# Initialize parameters
params = {
    'w1': jax.random.normal(w1_key, (1, n)), 'b1': jax.random.normal(b1_key, (n,)),
    'w2': jax.random.normal(w2_key, (n, n)), 'b2': jax.random.normal(b2_key, (n,)),
    'w3': jax.random.normal(w3_key, (n, n)), 'b3': jax.random.normal(b3_key, (n,)),
    'w4': jax.random.normal(w4_key, (n, n)), 'b4': jax.random.normal(b4_key, (n,)),
    'w5': jax.random.normal(w5_key, (n, 1)), 'b5': jax.random.normal(b5_key, (1,)),
}

def deep_model(params, x):
    x = x.reshape(-1, 1)
    hidden1 = jax.nn.tanh(x @ params['w1'] + params['b1'])       # tanh and sigmoid are equivalent
    hidden2 = jax.nn.tanh(hidden1 @ params['w2'] + params['b2']) # Plus, tanh passes the origin.
    hidden3 = jax.nn.tanh(hidden2 @ params['w3'] + params['b3'])
    hidden4 = jax.nn.tanh(hidden3 @ params['w4'] + params['b4'])
    output = hidden4 @ params['w5'] + params['b5']
    return output
def mse(params, x, y):
    predictions = deep_model(params, x).squeeze()
    return jnp.mean((predictions - y)**2)
# Derivative calculation
def model_for_grad(x, model_params):
    return deep_model(model_params, x).squeeze()
model_derivative_fn = jax.grad(model_for_grad, argnums=0)
vmap_model_derivative = jax.vmap(model_derivative_fn, in_axes=(0, None))
@jax.jit
def loss_fn(params, x, y):
    pred_y = deep_model(params, x).squeeze()
    loss_f = jnp.mean((pred_y - y)**2)

    true_dy = vmap_grad_f(x)
    pred_dy = vmap_model_derivative(x, params)
    loss_df = jnp.mean((pred_dy - true_dy)**2)

    return (1-rho)*loss_f + rho * loss_df

# --- Create a JIT-compiled function for the ENTIRE epoch ---
@jax.jit
def train_epoch(params, train_data, permutation):
    x_train, y_train = train_data
    steps_per_epoch = len(x_train) // batch_size

    def body_fun(step, current_params):
        start_idx = step * batch_size
        batch_idx = jax.lax.dynamic_slice_in_dim(permutation, start_idx, batch_size)
        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
        grads = jax.grad(loss_fn)(current_params, x_batch, y_batch)
        return jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, current_params, grads)

    params = jax.lax.fori_loop(0, steps_per_epoch, body_fun, params)
    return params

# --- Modified Training Loop ---
loss_history = []
key, shuffle_key = jax.random.split(key)
num_train = len(x_train)
print("\n---Start Training---")
start_time = time.time()

for epoch in range(epochs):
    shuffle_key, perm_key = jax.random.split(shuffle_key)
    perm = jax.random.permutation(perm_key, num_train)
    
    params = train_epoch(params, (x_train, y_train), perm)
    
    if epoch % 1000 == 0:
        loss = loss_fn(params, x_train, y_train)
        loss_history.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- Evaluation ---
x_plot = jnp.linspace(-1, 1, 500, dtype=jnp.float32)
y_true = f(x_plot)
y_pred = deep_model(params, x_plot).squeeze()
y_pred_train = deep_model(params, x_train).squeeze()
final_loss = loss_fn(params, x_train, y_train)
final_mse = mse(params, x_train, y_train)
max_error = jnp.max(jnp.abs(y_pred_train - y_train))

y_pred_validation= deep_model(params, x_validation).squeeze()
validation_mse = mse(params, x_validation, y_validation)
validation_max_error = jnp.max(jnp.abs(y_pred_validation - y_validation))


print(f"\n--- Final Result ---")
print(f"Final Loss: {final_loss:.6e}")
print(f"Final Training MSE: {final_mse:.6e}")
print(f"Final Training Max Error: {max_error:.6e}")
print(f"Final Validation MSE: {validation_mse:.6e}")
print(f"Final Training Max Error: {validation_max_error:.6e}")

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, label='True Runge Function')
plt.plot(x_plot, y_pred, label='Deep Sigmoid NN', linestyle='--')
plt.scatter(x_validation, y_validation, label='Validation', color='orange', s=10, alpha=0.6)
plt.title('Deep Sigmoid Network Approximation')
plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(0, epochs, 1000), loss_history)
plt.title('Training Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log'); plt.grid(True)
plt.tight_layout()
plt.show()


y_d_true = vmap_grad_f(x_plot) 
y_d_true_validation = vmap_grad_f(x_validation) 
model_derivative_fn = jax.grad(model_for_grad, argnums=0)
vectorized_model_derivative = jax.vmap(model_derivative_fn, in_axes=(0, None))
y_d_pred = vectorized_model_derivative(x_plot, params)
y_d_pred_validation = vectorized_model_derivative(x_validation, params)

pred_derivative_on_train = vectorized_model_derivative(x_train, params)
pred_derivative_on_validation = vectorized_model_derivative(x_validation, params)
true_derivative_on_train = vmap_grad_f(x_train)
true_derivative_on_validation = vmap_grad_f(x_validation)
final_derivative_mse = jnp.mean((pred_derivative_on_train - true_derivative_on_train)**2)
final_derivative_mse_validation = jnp.mean((pred_derivative_on_validation - true_derivative_on_validation)**2)
max_derivative_error = jnp.max(jnp.abs(y_d_pred-y_d_true))
max_derivative_error_validation = jnp.max(jnp.abs(y_d_pred_validation-y_d_true_validation))


print(f"\n--- Final Result (Derivatives) ---")
print(f"Final Training MSE (Derivative): {final_derivative_mse:.6e}")
print(f"Final Training Max Error (Derivative): {max_derivative_error:.6e}")
print(f"Final Validation MSE (Derivative): {final_derivative_mse_validation:.6e}")
print(f"Final Validation Max Error (Derivative): {max_derivative_error_validation:.6e}")

plt.figure(figsize=(8,6))
plt.plot(x_plot, y_d_true, label='True Derivative of Runge Function')
plt.plot(x_plot, y_d_pred, label='Deep Sigmoid NN Derivative', linestyle='--')
plt.scatter(x_validation, y_validation, label='Validation', color='orange', s=10, alpha=0.6)
plt.title("Comparison of Derivatives")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()