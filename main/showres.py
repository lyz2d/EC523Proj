from tensorboard.backend.event_processing import event_accumulator

# Path to the TensorBoard log file directory
log_dir = "/simple_VIT"

# Load the TensorBoard logs
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()  # Load events from the log file

# Get the scalar keys (e.g., loss, accuracy)
scalar_keys = ea.Tags()["scalars"]

# Print all available scalar keys
print("Available scalar keys:", scalar_keys)

# Access specific scalar data
scalars = ea.Scalars("loss")  # Replace 'loss' with the key you want

# Extract epoch and value data
epochs = [scalar.step for scalar in scalars]
values = [scalar.value for scalar in scalars]

# Plot the results
import matplotlib.pyplot as plt

plt.plot(epochs, values, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss")
plt.legend()
plt.show()