import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# Paths to the TensorBoard log file directories
log_dirs = [
    ("main/training_logs/laf_img128_p64", "Our method, 64 patches"),
    ("main/training_logs/laf_img256_p256", "Our method, 256 patches"),
    ("main/training_logs/vit_img128", "ViT, 64 patches"),
    ("main/training_logs/vit_img256", "ViT, 256 patches"),
    ("main/training_logs/ViT_img128_withDA", "ViT with DA, 64 patches"),
    ("main/training_logs/ViT_img256_withDA", "ViT with DA, 256 patches")
]

# Initialize plot
plt.figure(figsize=(10, 6))

for log_dir, label in log_dirs:
    # Load the TensorBoard logs
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Extract scalar data
    scalars_val = ea.Scalars("val_acc")
    scalars_epoch = ea.Scalars("epoch")
    
    epochs_step = [scalar.step for scalar in scalars_epoch]
    acc_step = [scalar.step for scalar in scalars_val] 
    #Get epochs and Acc
    epochs = [scalar.value for scalar in scalars_epoch]
    Acc = [scalar.value for scalar in scalars_val]
    # print(f"acc:  {len(Acc)}")
    # print(f"epochs: {len(epochs)}")

    # idx=find(epochs==124)


    

    epoch  = np.linspace(epochs[0], epochs[-1], len(Acc))
    Acc = [i*100 for i in Acc]
    # Plot data
    plt.plot(epoch[0:125], Acc[0:125], label=label,linewidth=3)
    # print(Acc[125])

# Customize plot
font = {
        'weight' : 'bold',
        'size'   : 16}
plt.xlabel("Epochs", fontdict=font)
plt.ylabel("Validation Accuracy (%)", fontdict=font)
plt.title("Validation Accuracy Across Experiments", fontdict=font)
plt.legend(prop={'size': 14})
plt.grid(True)
plt.show()