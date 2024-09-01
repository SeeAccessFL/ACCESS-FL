import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import hashlib
import pickle
import time
from tqdm import tqdm
import sys
import os
import logging


# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

num_clients = 100
num_rounds = 100


# Configure logging to output to a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("output_ac_drop_cifar10.log"),  # Log to a file
        logging.StreamHandler()             # Also log to console
    ]
)

def log_print(message):
    logging.info(message)  # Log the message to both console and file



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def preprocess(dataset):
    #dataset = dataset.map(lambda x, y: (tf.reshape(x, [-1]), y))
    return dataset.shuffle(1000).batch(64)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# train_data = preprocess(train_data)
test_data = preprocess(test_data)


# Create CNN Model
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model



# Function to calculate the total number of parameters in a model
def calculate_model_size(model):
    return np.sum([np.prod(v.shape) for v in model.trainable_weights])

# def calculate_total_parameters(model):
#     """
#     Calculates the total number of parameters (weights and biases) in the model.
#     """
#     return np.sum([np.prod(v.shape) for v in model.trainable_weights])


start_inimodel_time = time.time()
global_model = create_keras_model()
global_model.load_weights('cnn.h5')

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
global_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

end_inimodel_time = time.time()
server_inimodel_time = (end_inimodel_time - start_inimodel_time) * 1000

# Get the shapes of all layers' weights
model_shape = [w.shape for w in global_model.get_weights()]


# Display the global model architecture
log_print("\n===== Global Model Architecture (2NN) =====")
global_model.summary(print_fn=log_print)




def distribute_data(x, y, num_clients):
    # Ensure the number of clients is divisible by 10
    assert num_clients % 10 == 0, "Number of clients should be divisible by 10"

    # Number of clients handling each digit
    clients_per_digit = num_clients // 10

    # Divide samples of each digit among corresponding client groups
    client_data = []
    num_samples_per_client = {}  # Dictionary to store the number of samples per client
    client_id = 0


    # Group data by labels
    for digit in range(10):
        indices = np.where(y == digit)[0]
        x_digit = x[indices]
        y_digit = y[indices]

        # Split digit data among corresponding clients in the group
        x_digit_splits = np.array_split(x_digit, clients_per_digit)
        y_digit_splits = np.array_split(y_digit, clients_per_digit)

        for x_split, y_split in zip(x_digit_splits, y_digit_splits):
            client_data.append((client_id, x_split, y_split))

            # Calculate the number of samples for the current client and store it in the dictionary
            num_samples_per_client[client_id] = len(x_split)

            client_id += 1  # Increment client ID for each new client

    return client_data, num_samples_per_client


# Distribute data among 100 clients
client_data, num_samples_per_client = distribute_data(x_train, y_train, num_clients)

log_print("\n===== NonIID Data Distribution =====")
log_print(f"Data distributed among {num_clients} clients \n")
# Verify client order and data distribution
for client_id, x_split, y_split in client_data:
    log_print(f"Client ID: {client_id}, Data Size (Number of Samples): {len(x_split)}, Unique Labels: {np.unique(y_split)}")


participating_list = sorted([client_id for client_id, _, _ in client_data])  # Sort all clients


public_parameters = ec.SECP256R1()

def generate_key_pairs(participating_list, public_parameters):
    """
    Generates ECC key pairs and initializes shared secrets and masks for all clients.
    """
    private_keys = {}
    public_keys = {}
    client_genkeypair_time = 0
    client_pubkey_count = 0
    client_pubkey_size = 0
    server_pubkeys_count = 0
    server_pubkeys_size = 0
    

    for client_id in participating_list:
        start_time = time.time()
        private_key = ec.generate_private_key(public_parameters, default_backend())
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        end_time = time.time()
        client_genkeypair_time += (end_time - start_time) * 1000  # convert to milliseconds
        private_keys[client_id] = private_key
        public_keys[client_id] = public_key
        
        client_pubkey_count += 1
        
        # Calculate the size of the public key in kilobytes
        pubkey_size_kb = sys.getsizeof(public_key) / 1024
        client_pubkey_size += pubkey_size_kb
        
        # Add the size of the public key to the total broadcast size
        server_pubkeys_size += pubkey_size_kb

    server_pubkeys_count = 1 # Broad list of all public keys

    return private_keys, public_keys, client_genkeypair_time, client_pubkey_count, client_pubkey_size, server_pubkeys_count, server_pubkeys_size


def calculate_distance(public_parameters, round_number, num_participants, previous_distance, dropout_count):
    start_time = time.time() 
    public_parameters_bytes = public_parameters.name.encode('utf-8')  # Using the curve name as bytes
    hash_value = int(hashlib.sha256(public_parameters_bytes).hexdigest(), 16)

    # Adjust seed with dropout count
    seed = (hash_value + round_number + dropout_count) % (2 ** 32 - 1)
    np.random.seed(seed)

    possible_distances = list(range(1, (num_participants - 1) // 2 + 1))
    if previous_distance is not None and previous_distance in possible_distances:
        possible_distances.remove(previous_distance)


    end_time = time.time()
    client_caldis_time = (end_time - start_time) * 1000

    return np.random.choice(possible_distances), client_caldis_time


def generate_shared_secrets(participating_list, private_keys, public_keys, distance):
    """
    Generates shared keys for each client based on the calculated distance.
    """
    shared_secrets = {}

    start_time = time.time()
    for client_id in participating_list:
        i = participating_list.index(client_id)
        n = len(participating_list)
        pair1_idx = (i + distance) % n
        pair2_idx = (i - distance + n) % n
        pair1_id, pair2_id = participating_list[pair1_idx], participating_list[pair2_idx]

        shared_secrets[client_id] = {}

        for pair_id in [pair1_id, pair2_id]:
            # Generate shared key with each pair
            peer_public_key = serialization.load_pem_public_key(public_keys[pair_id], backend=default_backend())
            shared_secret = private_keys[client_id].exchange(ec.ECDH(), peer_public_key)

            shared_secrets[client_id][pair_id] = shared_secret
    
    end_time = time.time()
    client_shsec_time = (end_time - start_time) * 1000

    return shared_secrets, client_shsec_time



def generate_shared_masks(shared_secrets, model_shape):
    """
    Expands shared secrets into masks for each client using SHA256
    The masks are within a stable numeric range and shape compatible with the model weights.
    """
    masks = {}

    start_time = time.time()

    for client_id, secrets in shared_secrets.items():
        masks[client_id] = {}
        for pair_id, shared_secret in secrets.items():
            # Use a secure hash function (SHA-256) to generate a sequence of pseudo-random numbers
            hasher = hashlib.sha256()
            hasher.update(shared_secret)
            hash_digest = hasher.digest()

            # Initialize a pseudo-random number generator with the hash digest as the seed
            rng = np.random.default_rng(int.from_bytes(hash_digest, byteorder='big'))

            mask = []
            for shape in model_shape:
                # Generate random numbers within a stable range (e.g., [0, 1])
                random_array = rng.uniform(low=0.0, high=1.0, size=shape)

                # Scale the random numbers to a desired range if necessary (e.g., [0, 1e3])
                random_array *= 1e3

                mask.append(random_array)

            masks[client_id][pair_id] = mask

    end_time = time.time()
    client_genshmsk_time = (end_time - start_time) * 1000

    return masks, client_genshmsk_time



def train_local_model(global_model, client_data, participating_list, global_weight):
    """
    Trains local models on each client's data.
    """
    local_models = []
    num_samples = []

    for client_id, x_split, y_split in client_data:
        if client_id in participating_list:
            # Clear any previous model sessions
            tf.keras.backend.clear_session()


            start_time = time.time()
            # Set the global model weights to the provided global weights
            global_model.set_weights(global_weight)

            # Prepare the local dataset
            #local_data = tf.data.Dataset.from_tensor_slices((x_split, y_split)).map(lambda x, y: (tf.reshape(x, [-1]), y)).batch(64)
            local_data = tf.data.Dataset.from_tensor_slices((x_split, y_split)).batch(64)

            # Train the global model locally on the client's data
            global_model.fit(local_data, epochs=1, verbose=0)  # Epochs set to 1

            # Retrieve the model weights after training
            trained_weights = global_model.get_weights()
            
            end_time = time.time()
            client_trainw_time = (end_time - start_time) * 1000

            # Append the trained weights to the local models list
            local_models.append(trained_weights)

            # Append the number of samples to the list
            num_samples.append(len(x_split))

    return local_models, num_samples, client_trainw_time



def generate_masked_weights(local_models, masks, participating_list, num_samples_per_client):
    """
    Applies masks to each participating client's local model weights based on the client_id and pair_id.
    """
    masked_weights = []

    client_maskw_time = 0
    client_maskw_count = 0
    client_maskw_size = 0

    for client_id in participating_list:
        local_weights = local_models[participating_list.index(client_id)]

        scaling_factor = 1.0 / num_samples_per_client[client_id] if num_samples_per_client[client_id] != 0 else 0.0

        masked_weights_for_client = []

        start_time = time.time()
        for weight, layer_index in zip(local_weights, range(len(local_weights))):
            mask = np.zeros_like(weight)

            for pair_id, secret_mask in masks[client_id].items():
                if pair_id in participating_list:
                    mask_to_apply = np.maximum(secret_mask[layer_index] * scaling_factor, 0)  # Ensuring mask is non-negative

                    mask += mask_to_apply if pair_id < client_id else -mask_to_apply

            masked_weight_layer = weight + mask
            masked_weight_layer = np.clip(masked_weight_layer, -1e3, 1e3)  # Adjust clipping as needed

            end_time = time.time()
            client_maskw_time += (end_time - start_time) * 1000
            
            #client_maskw_size += sys.getsizeof(public_key) / 1024
            # Calculate size of the masked weight layer in bytes
            masked_layer_size = masked_weight_layer.nbytes
            client_maskw_size += masked_layer_size / 1024  # Convert to KB

            
            masked_weights_for_client.append(masked_weight_layer)
        
        
        masked_weights.append(masked_weights_for_client)
        client_maskw_count += 1

    return masked_weights, client_maskw_time, client_maskw_count, client_maskw_size






def aggregate_weights(local_models, num_samples):
    # Calculate the total number of samples across all clients
    total_samples = np.sum(num_samples)

    # Initialize the new weights as zero arrays with the same shape as the weights of the first local model
    new_weights = [np.zeros_like(weights) for weights in local_models[0]]

    start_time = time.time()
    # Aggregate weights using FedAvg
    for client_weights, num in zip(local_models, num_samples):
        for i, weights in enumerate(client_weights):
            # Accumulate the weighted sum of the client's weights
            new_weights[i] += weights * num / total_samples

    end_time = time.time()
    server_aggw_time = (end_time - start_time) * 1000
    # Calculate the size of aggregated weights in bytes
    aggregated_weights_size_bytes = sum(weight.nbytes for weight in new_weights)
    
    server_aggw_size = aggregated_weights_size_bytes / 1024  # Convert to kilobytes
    server_aggw_count = 1  # Broadcast msg   

            
    # Return the aggregated weights
    return new_weights, server_aggw_time, server_aggw_size, server_aggw_count




# main function
def federated_learning(global_model, client_data, num_rounds, participating_list, public_parameters, test_data):

    """
    Main function to run federated learning.
    """
    
    accuracy_log = []

    client_time = 0
    server_time = 0
    
    client_msg_count = 0
    server_msg_count = 0

    client_msg_size = 0
    server_msg_size = 0

    client_accumulated_time = [0] * num_rounds
    server_accumulated_time = [0] * num_rounds

    client_msg_counts = [0] * num_rounds
    server_msg_counts = [0] * num_rounds

    client_msg_sizes = [0] * num_rounds
    server_msg_sizes = [0] * num_rounds


    previous_distance = None
    dropout_count = 0

    # Calculate the size of the initial global model
    initial_model_weights = global_model.get_weights()
    initial_model_weights_bytes = sum(weight.nbytes for weight in initial_model_weights)
    
    server_inimodelw_size = initial_model_weights_bytes / 1024  # Convert to KB
    server_inimodelw_count = 1 # Broadcast initial global model
    
    server_time += server_inimodel_time
    server_msg_size += server_inimodelw_size
    server_msg_count += server_inimodelw_count

    
    log_print(f"\n===== Starting ACCESS-FL (dropped out scenario) on CIFAR10 for {num_rounds} Training Rounds =====")

    start_fl_time = time.time()
    # Generate key pairs and initialize masks
    private_keys, public_keys, client_genkeypair_time, client_pubkey_count, client_pubkey_size, server_pubkeys_count, server_pubkeys_size = generate_key_pairs(participating_list, public_parameters)
    
    client_time += client_genkeypair_time
    client_msg_count += client_pubkey_count
    server_msg_count += server_pubkeys_count
    client_msg_size += client_pubkey_size
    server_msg_size += server_pubkeys_size

    for round_num in tqdm(range(num_rounds)):
        log_print(f'Round {round_num+1}/{num_rounds}')
    
        start_round_time = time.time()
        # Calculate distance for finding peers
        distance, client_caldis_time = calculate_distance(public_parameters, round_num, len(participating_list), previous_distance, dropout_count)
        previous_distance = distance
        log_print(f"distance: {distance}")
        # log_print(f"participating_list: {participating_list}")

        # Generate shared secrets
        shared_secrets, client_shsec_time = generate_shared_secrets(participating_list, private_keys, public_keys, distance)

        # Expand shared secrets into masks
        shared_masks, client_genshmsk_time = generate_shared_masks(shared_secrets, model_shape)
        
        # Train local models
        global_weight = global_model.get_weights()
        local_models, num_samples, client_trainw_time = train_local_model(global_model, client_data, participating_list, global_weight)

        # Apply scaled masks to local models
        masked_weights, client_maskw_time, client_maskw_count, client_maskw_size = generate_masked_weights(local_models, shared_masks, participating_list, num_samples_per_client)

       

        #start_dropoutcheck_time = time.time()
        if (round_num + 1) % 10 == 0:

            dropout_client_id = participating_list.pop(np.random.randint(len(participating_list)))
            log_print(f"Client {dropout_client_id} dropped out")
            dropout_count += 1
            log_print(f"participating_list: {participating_list}")
            #participating_list = [client_id for client_id, _ in enumerate(masked_weights)]
            log_print(f"dropout_count: {dropout_count}")

            
            #Calculate distance to find new peers
            distance, client_caldis_time = calculate_distance(public_parameters, round_num, len(participating_list), previous_distance, dropout_count)
            previous_distance = distance
            log_print(f"new distance: {distance}")
            #log_print(f"participating_list: {participating_list}")

            # Mask model again
            # Generate shared secrets with new pairs
            shared_secrets, client_shsec_time = generate_shared_secrets(participating_list, private_keys, public_keys, distance)

            # Expand shared secrets into masks
            shared_masks, client_genshmsk_time = generate_shared_masks(shared_secrets, model_shape)
        
            # Apply scaled masks to local models
            masked_weights, client_maskw_time, client_maskw_count, client_maskw_size = generate_masked_weights(local_models, shared_masks, participating_list, num_samples_per_client)

            # server broadcast the new participating list
            server_broadpartilist_count = 1
            server_broadpartilist_size = (len(participating_list) * 4) / 1024  # assuming each participant ID is an integer (4 bytes), size in kilobytes
            server_msg_count += server_broadpartilist_count
            server_msg_size += server_broadpartilist_size






        # Aggregate masked updates using FedAvg
        new_weights, server_aggw_time, server_aggw_size, server_aggw_count = aggregate_weights(masked_weights, num_samples)

        
        # Update global model with averaged weights
        global_model.set_weights(new_weights)
        
        # dropout_count = 0
        end_round_time = time.time()
        round_time = (end_round_time - start_round_time) * 1000


        # Evaluate global model on test data
        loss, accuracy = global_model.evaluate(test_data, verbose=0)
        log_print(f"Accuracy: {accuracy:.4f}, \t Round Time: {round_time: .3f} msec")
       

        # Log accuracy
        accuracy_log.append(accuracy)

        # Update accumulated time and message counts/sizes
        client_time +=  client_caldis_time + client_shsec_time + client_genshmsk_time + client_trainw_time + client_maskw_time
     
        server_time += server_aggw_time

        client_accumulated_time[round_num] = client_time
        server_accumulated_time[round_num] = server_time


        client_msg_count += client_maskw_count
        server_msg_count += server_aggw_count 

        client_msg_counts[round_num] = client_msg_count
        server_msg_counts[round_num] = server_msg_count


        client_msg_size += client_maskw_size
        server_msg_size += server_aggw_size

        client_msg_sizes[round_num] = client_msg_size
        server_msg_sizes[round_num] = server_msg_size

    
    end_fl_time = time.time()
    total_fl_time = (end_fl_time - start_fl_time) * 1000  
    
    # Final evaluation on the test data
    final_loss, final_accuracy = global_model.evaluate(test_data)
    log_print(f"Final test accuracy: {final_accuracy:.4f}, \t Total FL Time: {total_fl_time: .3f} msec")
    accuracy_log.append(final_accuracy)

   


    return accuracy_log, client_accumulated_time, server_accumulated_time, client_msg_counts, server_msg_counts, client_msg_sizes, server_msg_sizes


####### Run federated learning #######
accuracy_log, client_accumulated_time, server_accumulated_time, client_msg_counts, server_msg_counts, client_msg_sizes, server_msg_sizes = federated_learning(global_model, client_data, num_rounds, participating_list, public_parameters, test_data=test_data)

# Ensure all lists have the same length
accuracy_log = accuracy_log[:num_rounds]
client_accumulated_time = client_accumulated_time[:num_rounds]
server_accumulated_time = server_accumulated_time[:num_rounds]
client_msg_counts = client_msg_counts[:num_rounds]
server_msg_counts = server_msg_counts[:num_rounds]
client_msg_sizes = client_msg_sizes[:num_rounds]
server_msg_sizes = server_msg_sizes[:num_rounds]

# Save accuracy log to CSV
df = pd.DataFrame({
    "Round": list(range(1, num_rounds + 1)), 
    "Accuracy": accuracy_log, 
    "Accumulated Client Time": client_accumulated_time,
    "Accumulated Server Time": server_accumulated_time,
    "Accumulated Client Message Counts": client_msg_counts,
    "Accumulated Server Message Counts": server_msg_counts,
    "Accumulated Client Message Sizes": client_msg_sizes,
    "Accumulated Server Message Sizes": server_msg_sizes
})
df.to_csv("accuracy_costs_ac_drop_cifar10.csv", index=False)

# Read the CSV file back into a DataFrame
df_output = pd.read_csv("accuracy_costs_ac_drop_cifar10.csv")

# # # Print the DataFrame to the terminal
# log_print(df_output.to_string(index=False))


