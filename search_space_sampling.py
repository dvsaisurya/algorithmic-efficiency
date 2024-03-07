import numpy as np
import json

# Define the search space for each hyperparameter
search_space = {
    "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],  # Example values
    "label_smoothing": [0.0, 0.1, 0.2],  # Example values
    "learning_rate": np.logspace(-4, -1, num=10),  # Log scale from 1e-4 to 1e-1
    "one_minus_beta1": np.linspace(0.01, 0.1, num=10),
    "beta2": np.linspace(0.99, 0.999, num=10),
    "weight_decay": np.linspace(0.01, 0.1, num=10),
    "warmup_factor": [0.01, 0.02, 0.05],  # Example values
    "beta3": [0.9, 0.99, 0.999],  # Example values
    "lamb_eps": [1e-6, 1e-8]  # Example values
}

def sample_hyperparameters():
    sampled_params = {}
    for param, values in search_space.items():
        sampled_params[param] = np.random.choice(values)
    return sampled_params

def generate_json_files(num_files):
    for i in range(num_files):
        params = sample_hyperparameters()
        file_name = f"hyperparameters_{i}.json"
        with open(file_name, 'w') as f:
            json.dump(params, f, indent=4)

# Adjust the `num_files` according to the number of TPUs you want to use
num_tpus = 7  # Example, adjust as necessary
generate_json_files(num_tpus)