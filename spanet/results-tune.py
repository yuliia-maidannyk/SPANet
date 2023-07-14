# https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html
from ray import tune, air
from ray.tune import ResultGrid
from ray.air import Result
import os
from spanet.tune import spanet_trial

base_options_file = "options_files/my_tth.json"
storage_path = "spanet_output/gpu_spanet_asha_tune"
exp_name = "'spanet_trial_18be2_00004_4_dropout=0.0696,hidden_dim=96,l2_penalty=0.0003,learning_rate=0.0001,num_attention_heads=2,num_branch_em_2023-07-12_17-02-19'"

exp_path = storage_path + exp_name
print(f"Loading results from {storage_path}...")

train_fn_with_parameters = tune.with_parameters(
    spanet_trial,
    base_options_file=base_options_file,
    home_dir=os.getcwd(),
    num_epochs=40,
    gpus_per_trial=0
)

restored_tuner = tune.Tuner.restore(storage_path, train_fn_with_parameters)
result_grid = restored_tuner.get_results()

num_results = len(result_grid)
print("Number of results:", num_results)

# Iterate over results
for i, result in enumerate(result_grid):
    if result.error:
        print(f"Trial #{i} had an error:", result.error)
        continue
    print(result.metrics.keys())
    print(
        f"Trial #{i} finished successfully with a mean accuracy metric of:",
        result.metrics["mean_accuracy"] # i.e. validation accuracy. also available: loss, val_loss
    )

results_df = result_grid.get_dataframe()
print(results_df[["training_iteration", "loss", "val_loss", "mean_accuracy"]])

# Get the result with the maximum test set `mean_accuracy`
best_result: Result = result_grid.get_best_result()
print(best_result.config)

best_result.metrics_dataframe.plot("training_iteration", "mean_accuracy")

ax = None
for result in result_grid:
    label = f"lr={result.config['lr']:.3f}, momentum={result.config['momentum']}"
    if ax is None:
        ax = result.metrics_dataframe.plot("training_iteration", "mean_accuracy", label=label)
    else:
        result.metrics_dataframe.plot("training_iteration", "mean_accuracy", ax=ax, label=label)
ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
ax.set_ylabel("Mean Test Accuracy")
