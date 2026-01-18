import sys
sys.path.append(".")

import constants
from logs.logging import read_logs
from plots.plotting import (
    plot_client_avg_performance_vs_rounds,
    plot_client_performance_vs_rounds,
    plot_server_lvl_avg_performance_vs_rounds,
    plot_server_overall_avg_performance_vs_rounds,
    plot_server_performance_vs_rounds,
)


def plot_and_read_from_logs(dir_name: str):
    """Create plots from saved log files in the specified directory."""
    path_logs_suffix = "/logs/"
    clients_loss_and_accuracy = read_logs(
        file_name=dir_name + path_logs_suffix + constants.Logs.CLIENT_LOG + constants.FileExtesions.PKL
    )
    # non_drifted_clients_loss_and_accuracy = read_logs(file_name=dir_name + path_logs_suffix + constants.Logs.NON_DRIFTED_CLIENT_LOG)
    # drifted_clients_loss_and_accuracy = read_logs(file_name=dir_name + path_logs_suffix + constants.Logs.DRIFTED_CLIENT_LOG)

    # Average performance of the clients
    non_drifted_client_averages = read_logs(
        file_name=dir_name
        + path_logs_suffix
        + constants.Logs.NON_DRIFTED_CLIENT_AVG_LOG + constants.FileExtesions.PKL
    )
    drifted_client_averages = read_logs(
        file_name=dir_name + path_logs_suffix + constants.Logs.DRIFTED_CLIENT_AVG_LOG + constants.FileExtesions.PKL
    )

    # Log the performance of the server hierarchy
    server_loss_and_accuracy = read_logs(
        file_name=dir_name + path_logs_suffix + constants.Logs.SERVER_LOG + constants.FileExtesions.PKL
    )
    server_level_averages = read_logs(
        file_name=dir_name + path_logs_suffix + constants.Logs.SERVER_LVL_AVG_LOG + constants.FileExtesions.PKL
    )
    server_overall_averages = read_logs(
        file_name=dir_name + path_logs_suffix + constants.Logs.SERVER_OVERALL_AVG_LOG + constants.FileExtesions.PKL
    )

    path_plots_suffix = "/plots/"
    # Plot the performance of the clients
    plot_client_performance_vs_rounds(
        clients_loss_and_accuracy, file_save_path=dir_name + path_plots_suffix
    )

    # Plot the performance of the server hierarchy
    plot_server_performance_vs_rounds(
        server_loss_and_accuracy, file_save_path=dir_name + path_plots_suffix
    )

    # Plot average performances
    plot_client_avg_performance_vs_rounds(
        [non_drifted_client_averages, drifted_client_averages],
        True,
        file_save_path=dir_name + path_plots_suffix,
    )
    plot_server_lvl_avg_performance_vs_rounds(
        server_level_averages, file_save_path=dir_name + path_plots_suffix
    )
    plot_server_overall_avg_performance_vs_rounds(
        server_overall_averages, file_save_path=dir_name + path_plots_suffix
    )

if __name__ == "__main__":
    dir_name = "fl_runs/MNIST/label_swapping/incremental/1-2_5-6_bidirectional/update_datasets_run_2026-01-18_20-45-05_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1"
    plot_and_read_from_logs(dir_name=dir_name)
