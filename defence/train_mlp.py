from strategies.mlp import MLPDefense, MLPDefenseConfig

mlp = MLPDefense(MLPDefenseConfig(threshold=0.8, epochs=300))

dir_name_ls = "./fl_runs/fl_runs/MNIST/label_swapping_cf_0.5/incremental/5-6_bidirectional/update_datasets_run_2026-01-26_09-27-18_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501
dir_name_rot = "./fl_runs/fl_runs/MNIST/rotation_cf_0.5/incremental/all_classes_rot_65/update_datasets_run_2026-01-26_09-29-33_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501

# Train on label swapping + rotation (IMPORTANT so it learns "rotation drift benign")
for r in range(10, 20):
    mlp.add_training_key(dir_name_ls, r)
    mlp.add_training_key(dir_name_rot, r)

mlp.fit(label_mode="malicious_only")
mlp.save("defence/mlp_defense.pt")
pass
