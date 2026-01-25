"""
Description: This module defines a federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import os
import random
import time
from typing import List
import uuid

import torch

import constants
from data.dataset_loader import load_datasets
from data.update_data import ClientUpdateRecord, ModelSpec
from data.utils import split_dataset, convert_dataset_to_loader
from drift_concepts.drift import drift_fn, modify_drifted_client_groups
from federated_network.client import client_fn, Client, client_initial_training
from federated_network.server import server_fn, aggregate_client_models, downward_link_aggregate_server_models
from federated_network.utils import update_progress, link_server_hierarchy, train_client_models, link_clients_to_servers
from logs.analysis_functions import compute_client_average_metrics, compute_server_average_metrics, \
    split_clients_loss_and_accuracy
from logs.logging import write_logs
from plots.plot_from_logs import plot_and_read_from_logs

class FederatedNetwork:
    def __init__(self, num_client_instances, server_tree_layout, num_training_rounds, dataset_name, drift_specs,
                 simulation_parameters, client_select_fraction=0.5, minibatch_size=32, num_local_epochs=4):
        # Dataset name
        self.dataset_name = dataset_name

        # Fraction of clients to be selected for each round (represented by C in originally by McMahan et al. 2017)
        self.client_select_fraction = client_select_fraction

        # Minibatch size for each client (represented by B in originally by McMahan et al. 2017)
        self.minibatch_size = minibatch_size

        # Number of local epochs for each client (represented by E in originally by McMahan et al. 2017)
        self.num_local_epochs = num_local_epochs

        # Number of training rounds
        self.num_training_rounds = num_training_rounds

        # Number of client instances
        self.num_client_instances = num_client_instances

        # Load the dataset
        self.trainset, self.testset, self._original_trainset, self._original_testset = load_datasets(dataset_name)

        # Partition the data set into subsets for each client
        partitioned_trainsets = split_dataset(self.trainset, self.num_client_instances)
        partitioned_testsets = split_dataset(self.testset, self.num_client_instances)

        # Create client instances
        self.clients = [
            client_fn(i, self.num_local_epochs, self.minibatch_size, self.dataset_name,
                      [partitioned_trainsets[i], partitioned_testsets[i]]) for i in range(num_client_instances)]

        # Concept drift properties
        self.drift = drift_fn(num_client_instances, num_training_rounds, drift_specs)

        # Simulation parameters
        self.simulation_parameters = simulation_parameters

        # Create instances for servers at each level of the server tree
        server_hierarchy = []
        absolute_index = 0

        for depth_level in range(len(server_tree_layout)):
            # For each level in the tree, create a list of server instances
            servers_at_level = [server_fn(server_id, self.dataset_name, absolute_index + i)  # Pass the absolute index
                                for i, server_id in enumerate(range(server_tree_layout[depth_level]))]

            server_hierarchy.append(servers_at_level)
            absolute_index += server_tree_layout[depth_level]

        self.server_hierarchy = server_hierarchy

        # Link servers in the hierarchical structure
        link_server_hierarchy(self.server_hierarchy)

        # Distribute the clients to the leaf servers
        link_clients_to_servers(self.server_hierarchy[-1], self.clients, self.num_client_instances)

    def sample_clients(self) -> List[Client]:
        """ Sample clients from the client pool and returns a list of client instances """
        return random.sample(self.clients, int(self.client_select_fraction * len(self.clients)))

    def run_simulation(self, update_save_path=None, plot_save_path=None, log_save_path=None, base_global_model_uuid: str = None) -> None:
        """
        Run the simulation for the specified number of rounds
        :param file_save_path: Path to save the logs
        :param log_save_path: Path to save the logs
        :return: None
        """
        clients_loss_and_accuracy = []  # Store the loss and accuracy of the all the clients at each round
        sampled_clients_in_each_round = []  # To keep track of the client IDs sampled in each round
        server_loss_and_accuracy = []  # Store the loss and accuracy at each level of the server hierarchy
        base_global_model_round = 0

        # Start the timer
        start_time = time.time()
        # Train the clients initially using their local data
        initial_client_loss_and_accuracy = client_initial_training(self.clients)
        # clients_loss_and_accuracy.append(initial_client_loss_and_accuracy)

        # Start the timer
        end_initial_training_time = time.time()
        print("End of initial training took : " + str(end_initial_training_time - start_time) + " seconds")
        # Load the test set for server evaluation
        server_test_set = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.minibatch_size)

        for _round in range(self.num_training_rounds):
            # Add drift to the clients, if within the drift period
            self.drift.current_round = _round
            if self.drift.drift_pattern == constants.DriftPatterns.GRADUAL_REOCCURRING:
                if self.drift.drift_start_round <= _round <= self.drift.drift_end_round:
                    self.drift.is_drift = True
            elif self.drift.drift_start_round <= _round:
                self.drift.is_drift = True

                # Modify the client groups if the drift is asynchronous
                if not self.drift.is_synchronous:
                    modify_drifted_client_groups(self.drift, _round)
            else:
                self.drift.is_drift = False
            print(f"--- Training Round {_round} with drift: {self.drift.is_drift} ---")
            # Clients sampled for a single round
            sampled_clients = self.sample_clients()

            # Extract the sampled client IDs and store them
            sampled_client_ids = [client.client_id for client in sampled_clients]
            sampled_clients_in_each_round.append(sampled_client_ids)

            # Implement the clustering algorithm here. That is, which clients to be selected for each server

            # As an example, only one server is considered
            sampled_clients_model_parameters = [sampled_client.model.state_dict() for sampled_client in sampled_clients]

            # Server downward-aggregator-link
            if self.simulation_parameters['is_server_downward_aggregation']:
                # Aggregate the models of the clients to the server model
                _ = aggregate_client_models(self.server_hierarchy, sampled_clients_model_parameters, server_test_set)
                round_server_loss_and_accuracy = downward_link_aggregate_server_models(self.server_hierarchy,
                                                                                       server_test_set)
                server_loss_and_accuracy.append(round_server_loss_and_accuracy)
            else:
                # Aggregate the models of the clients to the server model
                round_server_loss_and_accuracy = aggregate_client_models(self.server_hierarchy,
                                                                         sampled_clients_model_parameters,
                                                                         server_test_set)
                server_loss_and_accuracy.append(round_server_loss_and_accuracy)

            # Implement local training for every client and evaluate the client models
            if self.simulation_parameters['is_download_from_root_server']:
                # If the clients download the model from the root server of the hierarchy
                server_depth = 0
            elif self.simulation_parameters['is_download_from_level1_server']:
                # If the clients download the model from the root server of the hierarchy
                server_depth = len(self.server_hierarchy) - 3
            elif self.simulation_parameters['is_download_from_level2_server']:
                # If the clients download the model from the root server of the hierarchy
                server_depth = len(self.server_hierarchy) - 2
            else:
                # If the clients download the model from the leaf servers of the hierarchy
                server_depth = len(self.server_hierarchy) - 1

            round_client_loss_and_accuracy = train_client_models(self.clients,
                                                                 sampled_client_ids,
                                                                 self.server_hierarchy[server_depth],
                                                                 self.drift,
                                                                 self.simulation_parameters,
                                                                 plot_save_path)
            clients_loss_and_accuracy.append(round_client_loss_and_accuracy)

            if constants.UpdateDataSettings.SAVE_DATASET_UPDATES:
                self.save_client_updates(
                    _round,
                    update_save_path,
                    base_global_model_round,
                    base_global_model_uuid,
                    round_client_loss_and_accuracy,
                )

            if self._original_testset is not None and self._original_trainset is not None:
                for client in self.clients:
                    client.restore_original_data(self._original_trainset, self._original_testset)

            # Update the progress of the simulation
            update_progress(_round=_round + 1, num_training_rounds=self.num_training_rounds)

        # Stop the timer
        end_time = time.time()

        print(f"Runtime: {end_time - start_time} seconds")

        # Split the client performance to drifted and non-drifted clients
        if self.drift.is_synchronous:
            non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                clients_loss_and_accuracy, self.drift.drifted_client_indices, None)
        else:
            non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                clients_loss_and_accuracy, self.drift.drifted_client_indices,
                self.drift.async_drift_specs['drift_groups'])

        # Get average performance of the clients
        non_drifted_client_averages = compute_client_average_metrics(non_drifted_clients_loss_and_accuracy)
        if self.drift.is_synchronous:
            drifted_client_averages = compute_client_average_metrics(drifted_clients_loss_and_accuracy)
        else:
            drifted_client_averages = []
            for drited_groups in drifted_clients_loss_and_accuracy:
                drifted_client_averages.append(compute_client_average_metrics(drited_groups))

        if log_save_path is None:
            log_save_path = constants.Paths.LOG_SAVE_PATH

        # Log the performance of the clients
        write_logs(clients_loss_and_accuracy, file_name=log_save_path + constants.Logs.CLIENT_LOG)
        # Log the performance of the clients separated by drifted and non-drifted
        # write_logs(non_drifted_clients_loss_and_accuracy,
        #            file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_LOG)
        # write_logs(drifted_clients_loss_and_accuracy,
        #            file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_LOG)
        # Average performance of the clients
        write_logs(non_drifted_client_averages,
                   file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_AVG_LOG)
        write_logs(drifted_client_averages,
                   file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_AVG_LOG)

        # Get average performance of the servers
        server_level_averages, server_overall_averages = compute_server_average_metrics(server_loss_and_accuracy)

        # Log the performance of the server hierarchy
        write_logs(server_loss_and_accuracy, file_name=log_save_path + constants.Logs.SERVER_LOG)
        write_logs(server_level_averages, file_name=log_save_path + constants.Logs.SERVER_LVL_AVG_LOG)
        write_logs(server_overall_averages, file_name=log_save_path + constants.Logs.SERVER_OVERALL_AVG_LOG)

        if not constants.AnalysisSettings.STORE_PLOTS:
            return
        plot_and_read_from_logs(dir_name=plot_save_path.split('/plots/')[0])

    def save_client_updates(
        self,
        _round: int,
        file_save_path: str,
        base_global_model_round: int,
        base_global_model_uuid: uuid.UUID,
        round_client_loss_and_accuracy: list,
    ) -> None:
        """
        Save the client updates to the specified path.
        :param path: Path to save the client updates
        :return: None
        """
        count = len(round_client_loss_and_accuracy)
        mean_loss = sum(loss for loss, _ in round_client_loss_and_accuracy) / count
        mean_accuracy = sum(acc for _, acc in round_client_loss_and_accuracy) / count

        if _round % 10 == 0:
            if _round > 0:
                base_global_model_uuid = str(uuid.uuid4())
            base_global_model_round = _round
            os.makedirs(f"{file_save_path}base_models/", exist_ok=True)
            self.save_model(
                path=f"{file_save_path}base_models/server_model_round_{_round}_{base_global_model_uuid}.pt")

        base_global_model_uuid = get_latest_base_global_model_uuid(f"{file_save_path}base_models/")

        for idx, client in enumerate(self.clients):
            malicious_drift = False
            if idx in self.drift.drifted_client_indices:
                if self.drift.drift_method == constants.DriftCreationMethods.LABEL_SWAPPING:
                    malicious_drift = True

            update_record = ClientUpdateRecord.build_update_record(
                        client_id=client.client_id,
                        round_id=_round,
                        global_sd=self.server_hierarchy[0][0].model.state_dict(),
                        client_sd=client.model.state_dict(),
                        spec=ModelSpec.from_model(client.model),
                        base_global_round=base_global_model_round,
                        base_global_hash=base_global_model_uuid,
                        num_examples=client.local_trainset.dataset.targets.shape[0],
                        dtype_for_storage=torch.float32,
                        local_steps=client.epochs * (client.local_trainset.dataset.targets.shape[0] // client.mini_batch_size),
                        local_epochs=client.epochs,
                        hyper={
                            "learning_rate": constants.TrainingSettings.LEARNING_RATE,
                            "optimizer": constants.TrainingSettings.OPTIMIZER,
                            "loss": constants.TrainingSettings.LOSS,
                            "eps": constants.TrainingSettings.EPS,
                            "betas": constants.TrainingSettings.BETAS,
                        },
                        metrics={"train_loss": round_client_loss_and_accuracy[idx][0],
                                 "train_accuracy": round_client_loss_and_accuracy[idx][1],
                                 "train_deviation_loss": round_client_loss_and_accuracy[idx][0] - mean_loss,
                                 "train_deviation_accuracy": round_client_loss_and_accuracy[idx][1] - mean_accuracy},
                        store_flat_vector=True,
                        malicious=malicious_drift,
                        drift_applied=self.drift.is_drift,
                        is_drifted_client=idx in self.drift.drifted_client_indices,
                        )
            os.makedirs(f"{file_save_path}client_updates/", exist_ok=True)
            update_record.save_torch(path=f"{file_save_path}client_updates/client_{client.client_id}_round_{_round}.pt")

    def save_model(self, path: str) -> None:
        """
        Save the model of the root server to the specified path.
        :param path: Path to save the model
        :return: None
        """
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.server_hierarchy[0][0].model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Load the model of the root server from the specified path.
        :param path: Path to load the model
        :return: None
        """
        import os

        if not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(f"Model file not found at {path}")
        weights = torch.load(path)
        self.server_hierarchy[0][0].model.load_state_dict(weights)

def get_latest_base_global_model_uuid(path_to_models: str) -> str | None:
    import os
    import re

    best_round = None
    best_uuid = None

    uuid_pat = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    pattern = re.compile(rf"^server_model_round_(\d+)_({uuid_pat})\.pt$")

    model_names = os.listdir(path_to_models)
    for name in model_names:
        m = pattern.match(name)
        if not m:
            continue
        rnd = int(m.group(1))
        uid = m.group(2)

        if best_round is None or rnd > best_round:
            best_round = rnd
            best_uuid = uid

    return best_uuid