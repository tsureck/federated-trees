"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
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
        drift_pattern=constants.DriftPatterns.INCREMENTAL,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.LABEL_SWAPPING,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.1,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.9,  # Round at which the drift ends as a fraction of the total number of rounds
        max_rotation=45,  # Maximum rotation angle for the drift created by rotations
        class_pairs_to_swap=[(1, 2), (5, 6)],  # Classes to be swapped in the label-swapping drift method
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
        num_training_rounds=20,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # #################################
    # Async - UTA - D=0.375, L=1
    # #################################
    # if not constants.ModelSettings.SAVE_MODEL_STATE:
    #     fed_net.load_model('./root_server_model.pth')

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='./plots/saved_plots/rotation_tests/server_1/incremental_bidirectional/',
        log_save_path='./logs/saved_logs/rotation_tests/server_1/incremental_bidirectional/',)

    print("Plotting simulation complete...")
    # torch.save(fed_net.server_hierarchy[0][0].model.state_dict(),'./models/saved_models/tests/server_1/drift_rotation_22.5/root_server_model.pth')
    if constants.ModelSettings.SAVE_MODEL_STATE:
        fed_net.save_model('./root_server_model3.pth')
    # fed_net.save_model('./models/saved_models/tests/server_1/base/root_server_model.pth')

    plt.figure()
    plt.plot(fed_net.drift._applied_drift_logging)
    plt.title("Drift Transition Progress")
    plt.xlabel("Training Round")
    plt.ylabel("Transition Progress")
    import os
    os.makedirs("./plots/saved_plots/ls_tests/drifts/", exist_ok=True)
    plt.savefig(f"./plots/saved_plots/ls_tests/drifts/drift_{fed_net.drift.drift_pattern}_{str(fed_net.drift.max_rotation)}.png")

if __name__ == "__main__":
    main()
