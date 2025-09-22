import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# The Runge Function
def f(x):
    return 1 / (1 + 25 * x**2)
    
# The Sigmoid Function
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

# Hyperparameters
learning_rate = 0.01
epochs = 20000
datanum = 10001
batch_size = 32 

# Data
x_train = jnp.linspace(-1.0, 1.0, datanum)
y_train = f(x_train)

# Construct the Sigmoid Model
def sigmoid_model(params, x):
    x = x.reshape(-1, 1)
    hidden = sigmoid(x @ params['w1'] + params['b1'])
    return hidden @ params['w2'] + params['b2']
    
key = jax.random.PRNGKey(0)
key, w1_key, b1_key, w2_key, b2_key = jax.random.split(key, 5)
params = {
    'w1': jax.random.normal(w1_key, (1, 16)),
    'b1': jax.random.normal(b1_key, (16,)),
    
    'w2': jax.random.normal(w2_key, (16, 1)),
    'b2': jax.random.normal(b2_key, (1,))
}

def loss_fn(params, x, y):
    predictions = sigmoid_model(params, x).squeeze()
    return jnp.mean((predictions - y)**2)
loss_history = []

@jax.jit
# Gradient Descent
def update_step(params, x, y, learning_rate):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)

num_train = len(x_train)
steps_per_epoch = num_train // batch_size

key, shuffle_key = jax.random.split(key)
for epoch in range(epochs):
    shuffle_key, perm_key = jax.random.split(shuffle_key)
    perm = jax.random.permutation(perm_key, num_train)
    
    for step in range(steps_per_epoch):
        batch_idx = perm[step * batch_size : (step + 1) * batch_size]
        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
        params = update_step(params, x_batch, y_batch, learning_rate)
    
    if epoch % 1000 == 0:
        loss = loss_fn(params, x_train, y_train)
        loss_history.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")


x_plot = jnp.linspace(-1, 1, 500)
y_true = f(x_plot)
y_pred = sigmoid_model(params, x_plot).squeeze()
y_pred_train = sigmoid_model(params, x_train).squeeze()
final_mse = loss_fn(params, x_train, y_train)
max_error = jnp.max(jnp.abs(y_pred_train - y_train))

# Result
print(f"\n--- Final Result ---")
print(f"Final MSE: {final_mse:.6f}")
print(f"Final Max Error: {max_error:.6f}")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, label='True Runge Function')
plt.plot(x_plot, y_pred, label='1-Hidden-Layer NN', linestyle='--')
plt.title('Single Hidden Layer Network')
plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(0, epochs, 1000), loss_history)
plt.title('Training Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log'); plt.grid(True)
plt.tight_layout()
plt.show()