# =========================================================
# OPTUNA OBJECTIVES
# =========================================================

bo_trials = []
aa_bo_trials = []

def bo_objective(trial):
    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "heads": trial.suggest_categorical("heads", [4, 8]),
        "key_dim": trial.suggest_categorical("key_dim", [32, 64]),
        "momentum": trial.suggest_float("momentum", 0.4, 0.95),
        "alpha_scale": trial.suggest_float("alpha_scale", 0.5, 2.0),
    }

    result = train_and_eval(hparams, "DWAF_BO")
    bo_trials.append({**hparams, **result})

    # BOAUC objective: validation AUC only
    return result["auc"]


def aa_bo_objective(trial):
    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "heads": trial.suggest_categorical("heads", [4, 8]),
        "key_dim": trial.suggest_categorical("key_dim", [32, 64]),
        "momentum": trial.suggest_float("momentum", 0.4, 0.95),
        "alpha_scale": trial.suggest_float("alpha_scale", 0.5, 2.0),
    }

    result = train_and_eval(hparams, "DWAF_AA_BO")

    # AABO objective: validation AUC + attention discriminability
    obj = 0.7 * result["auc"] + 0.3 * result["attn_quality"]

    aa_bo_trials.append({**hparams, **result, "objective": obj})

    return obj


# =========================================================
# RUN STUDIES
# =========================================================

study_bo = optuna.create_study(direction="maximize")
study_bo.optimize(bo_objective, n_trials=15)

study_aa = optuna.create_study(direction="maximize")
study_aa.optimize(aa_bo_objective, n_trials=15)


# =========================================================
# SAVE RESULTS
# =========================================================

pd.DataFrame(bo_trials).to_csv("dwaf_bo_trials.csv", index=False)
pd.DataFrame(aa_bo_trials).to_csv("dwaf_attention_aware_bo_trials.csv", index=False)

print("Best BOAUC parameters:", study_bo.best_params)
print("Best BOAUC objective value:", study_bo.best_value)

print("Best AABO parameters:", study_aa.best_params)
print("Best AABO objective value:", study_aa.best_value)
