import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import optax 

# --- The Runge Function ---
def f(x):
    return 1 / (1 + 25 * x**2)

grad_f_true = jax.grad(f)
vmap_grad_f_true = jax.vmap(grad_f_true)

# --- Hyperparameters ---
initial_learning_rate = 0.005 
epochs = 40000  
datanum = 10000
batch_size = 32
n = 32        
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
x_validation = jax.random.uniform(validation_key, shape=(6000,), minval=-1.0, maxval=1.0)
y_validation = f(x_validation)

# --- Optimizer Setup ---
lr_schedule = optax.exponential_decay(
    init_value=initial_learning_rate,
    transition_steps=2000,
    decay_rate=0.9
)
optimizer = optax.adam(learning_rate=lr_schedule)


# --- Initialize Parameters ---
key, w1_key, b1_key, w2_key, b2_key, w3_key, b3_key = jax.random.split(key, 7)
glorot_normal_initializer = jax.nn.initializers.glorot_normal()
params = {
    'w1': glorot_normal_initializer(w1_key, (1, n)), 'b1': jnp.zeros(n,),
    'w2': glorot_normal_initializer(w2_key, (n, n)), 'b2': jnp.zeros(n,),
    'w3': glorot_normal_initializer(w3_key, (n, 1)), 'b3': jnp.zeros(1,)
}
opt_state = optimizer.init(params) 

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

# --- Derivative Calculation  ---
def model_for_grad(x, model_params):
    return deep_model(model_params, x).squeeze()[()]
model_derivative_fn = jax.grad(model_for_grad, argnums=0)
vmap_model_derivative = jax.vmap(model_derivative_fn, in_axes=(0, None))

# --- Loss Function with Adaptive Weights ---
@jax.jit
def loss_fn(params, x, y):
    pred_y = deep_model(params, x).squeeze()
    loss_f = jnp.mean((pred_y - y)**2)
    
    true_dy = vmap_grad_f_true(x)
    pred_dy = vmap_model_derivative(x, params)
    
    weights = jnp.square(true_dy) + 1.0
    loss_df = jnp.mean(weights * (pred_dy - true_dy)**2)

    return (1-rho)*loss_f + rho * loss_df

# --- JIT-compiled Training Step for Optax ---
@jax.jit
def train_epoch(epoch_state, train_data):
    params, opt_state, key = epoch_state
    x_train, y_train = train_data
    
    key, perm_key = jax.random.split(key)
    perm = jax.random.permutation(perm_key, len(x_train))
    
    steps_per_epoch = len(x_train) // batch_size
    
    def body_fun(step, val):
        params, opt_state = val
        batch_idx = jax.lax.dynamic_slice_in_dim(perm, step * batch_size, batch_size)
        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
        
        loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params) # params needed for weight decay etc.
        params = optax.apply_updates(params, updates)
        
        return (params, opt_state)

    final_params, final_opt_state = jax.lax.fori_loop(0, steps_per_epoch, body_fun, (params, opt_state))
    
    return final_params, final_opt_state, key

# --- Training Loop ---
loss_history = []
print("\n--- Start Training with Optax and All Optimizations ---")
start_time = time.time()

epoch_state = (params, opt_state, key)

for epoch in range(epochs):
    params, opt_state, key = train_epoch(epoch_state, (x_train, y_train))
    epoch_state = (params, opt_state, key)
    
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
print(f"Final Validation Max Error: {validation_max_error:.6e}")

# --- Plotting ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, label='True Runge Function')
plt.plot(x_plot, y_pred, label='Deep Tanh NN', linestyle='--')
plt.scatter(x_validation, y_validation, label='Validation', color='orange', s=10, alpha=0.6)
plt.title('Deep Network Approximation')
plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
# Plot both training and validation loss
plt.plot(range(0, epochs, 1000), loss_history, label='Training Loss')
plt.title('Training Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log'); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

# --- Derivatives Evaluation ---
y_d_true = vmap_grad_f_true(x_plot) 
y_d_true_validation = vmap_grad_f_true(x_validation) 
y_d_pred = vmap_model_derivative(x_plot, params)
y_d_pred_validation = vmap_model_derivative(x_validation, params)

pred_derivative_on_train = vmap_model_derivative(x_train, params)
pred_derivative_on_validation = vmap_model_derivative(x_validation, params)
true_derivative_on_train = vmap_grad_f_true(x_train)
true_derivative_on_validation = vmap_grad_f_true(x_validation)
final_derivative_mse = jnp.mean((pred_derivative_on_train - true_derivative_on_train)**2)
final_derivative_mse_validation = jnp.mean((pred_derivative_on_validation - true_derivative_on_validation)**2)
max_derivative_error = jnp.max(jnp.abs(y_d_pred - y_d_true))
max_derivative_error_validation = jnp.max(jnp.abs(y_d_pred_validation - y_d_true_validation))


print(f"\n--- Final Result (Derivatives) ---")
print(f"Final Training MSE (Derivative): {final_derivative_mse:.6e}")
print(f"Final Training Max Error (Derivative): {max_derivative_error:.6e}")
print(f"Final Validation MSE (Derivative): {final_derivative_mse_validation:.6e}")
print(f"Final Validation Max Error (Derivative): {max_derivative_error_validation:.6e}")

plt.figure(figsize=(8,6))
plt.plot(x_plot, y_d_true, label='True Derivative of Runge Function')
plt.plot(x_plot, y_d_pred, label='Deep Tanh NN Derivative', linestyle='--')
plt.scatter(x_validation, y_d_true_validation, label='Validation (True)', color='orange', s=10, alpha=0.6) # Scatter plot for derivative
plt.title("Comparison of Derivatives")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True)
plt.show()