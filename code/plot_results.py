"""
EPL 442 - Assignment 1: plots errors.txt and successrate.txt (current folder).
Name: <YOUR NAME>   AM: <YOUR AM>
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Errors ---
if os.path.exists("errors.txt"):
    data = np.loadtxt("errors.txt")
    plt.figure(figsize=(10, 5))
    plt.plot(data[:, 0], data[:, 1], label="Training error (MSE)", color="blue")
    plt.plot(data[:, 0], data[:, 2], label="Testing error (MSE)", color="orange", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Training vs Testing Error Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("error_plot.png")
    plt.show()
else:
    print("errors.txt not found. Run the training script first!")

# --- 2. Success rate ---
if os.path.exists("successrate.txt"):
    data = np.loadtxt("successrate.txt")
    plt.figure(figsize=(10, 5))
    plt.plot(data[:, 0], data[:, 1], label="Training success rate (%)", color="yellow")
    plt.plot(data[:, 0], data[:, 2], label="Testing success rate (%)", color="purple", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Success Rate (%)")
    plt.title("Training vs Testing Success Rate Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("success_plot.png")
    plt.show()
else:
    print("successrate.txt not found. Run the training script first!")