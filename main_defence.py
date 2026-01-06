"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import datetime
import uuid
import torch
import constants
from federated_network.network import FederatedNetwork
from logs.analysis_functions import plot_average_performance
import matplotlib.pyplot as plt

def main():
    print("torch threads:", torch.get_num_threads())

    async_drift_specs = dict(
        num_drift_groups=1,  # Number of groups of clients that are affected by the drift asynchronously
        drift_groups=None,  # Groups of clients that are affected by the drift asynchronously
        drift_split_round=1,  # Times at which the drift is split into multiple asynchronous drifts,
    )
    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.5,
        # clients_fraction=0.7,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        drift_localization_factor=1,  # Factor to localize the drift to a certain concentrated group of clients
        is_synchronous=True,  # If the drift is synchronous or asynchronous
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        drift_pattern=constants.DriftPatterns.ABRUPT,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.LABEL_SWAPPING,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.25,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.75,  # Round at which the drift ends as a fraction of the total number of rounds
        max_rotation=45,  # Maximum rotation angle for the drift created by rotations
        class_pairs_to_swap=[],  # Classes to be swapped in the label-swapping drift method
        classes_to_rotate=[],  # Classes to be rotated in the rotation drift method
        # class_pairs_to_swap=[('Sandal', 'Shirt'), ('Trouser', 'Bag')],  # Classes to be swapped in F_MNIST
        num_drift_cycles=2, # Number of drift cycles
        swap_direction='bidirectional', # Direction of label swapping ('unidirectional' or 'bidirectional')
    )

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=8,  # Number of clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # #################################
    # Async - UTA - D=0.375, L=1
    # #################################
    base_global_model_uuid = str(uuid.uuid4())
    if constants.UpdateDataSettings.SAVE_DATASET_UPDATES:
        type_of_run = "update_datasets"
    else:
        type_of_run = "testing"

    class_drift_info = ""
    if drift_specifications['drift_method'] == constants.DriftCreationMethods.LABEL_SWAPPING:
        if drift_specifications['class_pairs_to_swap']:
            for class_pair in drift_specifications['class_pairs_to_swap']:
                class_drift_info += f"{class_pair[0]}-{class_pair[1]}_"
        else:
            class_drift_info = "all_classes_"
        class_drift_info += drift_specifications["swap_direction"]
    elif drift_specifications['drift_method'] == constants.DriftCreationMethods.ROTATION:
        if drift_specifications['classes_to_rotate']:
            for class_label in drift_specifications['classes_to_rotate']:
                class_drift_info += f"{class_label}_"
        else:
            class_drift_info = "all_classes_"
        class_drift_info += f"rot_{drift_specifications['max_rotation']}"

    base_save_path = f"{constants.UpdateDataSettings.DATA_SAVE_PATH}/{fed_net.dataset_name}/{fed_net.drift.drift_method}/{fed_net.drift.drift_pattern}/{class_drift_info}/{type_of_run}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_fedavg_{base_global_model_uuid}/"

    # Running the simulation
    fed_net.run_simulation(
        # file_save_path='./plots/saved_plots/rotation_tests/cifar_test/incremental_drift_after_160_r/',
        update_save_path=base_save_path + "updates/",
        plot_save_path=base_save_path + "plots/",
        # log_save_path='./logs/saved_logs/rotation_tests/cifar_test/incremental_drift_after_160_r/',)
        log_save_path=base_save_path + "logs/",)

    print("Plotting simulation complete...")
    if constants.ModelSettings.SAVE_MODEL_STATE:
        fed_net.save_model(f"./root_server_model_{fed_net.num_training_rounds}_rounds_{fed_net.dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

    plt.figure()
    plt.plot(fed_net.drift._applied_drift_logging)
    plt.title("Drift Transition Progress")
    plt.xlabel("Training Round")
    plt.ylabel("Transition Progress")
    import os
    os.makedirs(base_save_path + "plots/", exist_ok=True)
    filename = base_save_path + "plots/" + f"{fed_net.drift.drift_pattern}_{class_drift_info}.png"
    plt.savefig(filename)


if __name__ == "__main__":
    main()
