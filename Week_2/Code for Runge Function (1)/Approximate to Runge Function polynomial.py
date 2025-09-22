import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# The Runge Function
def f(x):
    return 1/(1+25*x**2)

# Hyperparameters
POLYNOMIAL_DEGREE = 12
learning_rate = 0.01
epochs = 40000
datanum = 1000

# Data
x_train = jnp.linspace(-1.0, 1.0, datanum)
y_train = f(x_train)

key=jax.random.PRNGKey(0)
key, w_key, b_key = jax.random.split(key, 3)
params = {
    'w': jax.random.normal(w_key, (POLYNOMIAL_DEGREE, 1)), 
    'b': jax.random.normal(b_key, (1,))
}

# Construct the Polynomial Model
def polynomial_model(params, x):
    x_col = x.reshape(-1, 1)
    exponents = jnp.arange(1, POLYNOMIAL_DEGREE + 1)
    features = jnp.power(x_col, exponents)
    return features @ params['w'] + params['b']

def loss_fn(params, x, y):
    predictions = polynomial_model(params, x).squeeze() 
    return jnp.mean((predictions - y)**2)
loss_history = []

@jax.jit
# Gradient Descent
def update_step(params, x, y, learning_rate):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)


for epoch in range(epochs):
    params = update_step(params, x_train, y_train, learning_rate)
    
    if epoch % 1000 == 0:
        loss = loss_fn(params, x_train, y_train)
        loss_history.append(loss)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
x_plot = jnp.linspace(-1, 1, 500)
y_true = f(x_plot)
y_pred = polynomial_model(params, x_plot).squeeze()

final_mse = loss_fn(params, x_train, y_train)
max_error = jnp.max(jnp.abs(polynomial_model(params, x_train).squeeze() - y_train))

# Result
print("\n--- Final Result ---")
print(f"Final MSE (Degree={POLYNOMIAL_DEGREE}): {final_mse:.6f}")
print(f"Final Max Error: {max_error:.6f}")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_plot, y_true, label='True Runge Function', color='blue')
plt.plot(x_plot, y_pred, label='Neural Network Prediction', color='red', linestyle='--')
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