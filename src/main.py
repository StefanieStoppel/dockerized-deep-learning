import mlflow
import optuna
from optimization import objective


def main():
    # Experiment options
    options = {
        "tracking_uri": "file:/mlruns",
        "experiment_name": "debugging-optuna-mlflow",
        "epochs": 2,
        "use_cuda": False,
        "log_interval": 10,
        "save_model": True,
    }

    # Create mlflow experiment if it doesn't exist already
    experiment_name = options["experiment_name"]
    mlflow.set_tracking_uri(options["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Propagate logs to the root logger.
    optuna.logging.set_verbosity(verbosity=optuna.logging.INFO)

    # Create the optuna study which shares the experiment name
    study = optuna.create_study(study_name=experiment_name, direction="minimize")
    study.optimize(lambda trial: objective(trial, experiment, options), n_trials=5)

    # Filter optuna trials by state
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Loss (trial value): ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
