# hybrid_bayesian_maml_sequential.py
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

import pathlib, shutil

for cache_dir in pathlib.Path('.').rglob('__pycache__'):
    shutil.rmtree(cache_dir)


# -----------------------
# Helper functions
# -----------------------
def parse_embedding(emb):
    """Convert string '[0.1, 0.2]' or list to numpy array."""
    if isinstance(emb, str):
        return np.fromstring(emb.strip("[]"), sep=",")
    return np.asarray(emb)


def concat_emb(task_emb, dataset_emb):
    """Concatenate embeddings to make a joint feature vector."""
    return np.concatenate([task_emb, dataset_emb], axis=-1)


def softmax(x, temp=1.0):
    ex = np.exp((x - np.max(x)) / (temp + 1e-12))
    return ex / (np.sum(ex) + 1e-12)


# -----------------------
# Meta-prior (running mean of per-task coefficients)
# -----------------------
class ContinualMetaPrior:
    def __init__(self):
        self.meta_w = None  # shape: (n_targets, feat_dim)
        self.meta_b = None  # shape: (n_targets,)
        self.total_weight = 0.0

    def update_with_task(self, W_task, b_task, task_weight=1.0):
        """Weighted running mean update of meta prior."""
        if self.meta_w is None:
            self.meta_w = W_task.copy()
            self.meta_b = b_task.copy()
            self.total_weight = float(task_weight)
        else:
            tw = self.total_weight
            tw_new = tw + float(task_weight)
            self.meta_w = (tw * self.meta_w + task_weight * W_task) / tw_new
            self.meta_b = (tw * self.meta_b + task_weight * b_task) / tw_new
            self.total_weight = tw_new

    def get_prior(self):
        return (self.meta_w.copy(), self.meta_b.copy()) if self.meta_w is not None else (None, None)


# -----------------------
# Core predict function (uses training_df as source of neighbors)
# -----------------------
def predict_bayes_with_maml_prior(training_df, query_task_emb, query_dataset_emb,
                                 numeric_cols, top_k=20, neighbor_scale=5.0,
                                 temp=0.1, meta_prior=(None, None)):
    """
    - training_df: DataFrame containing 'task_embedding','dataset_embedding' and numeric_cols;
      embeddings are numpy arrays (not strings).
    - meta_prior: tuple (meta_w, meta_b) or (None, None). meta_w shape = (n_targets, feat_dim)
    Returns: preds dict, uncerts dict, adapted_coeffs dict (w,b) for each numeric col.
    """

    task_embs = np.stack(training_df["task_embedding"].values)
    dataset_embs = np.stack(training_df["dataset_embedding"].values)
    X_all = np.concatenate([task_embs, dataset_embs], axis=1)
    query_raw = concat_emb(query_task_emb, query_dataset_emb)

    # scale embeddings (fit on training + query)
    scaler = StandardScaler().fit(np.vstack([X_all, query_raw.reshape(1, -1)]))
    X_all_s = scaler.transform(X_all)
    query_s = scaler.transform(query_raw.reshape(1, -1))

    # top-K by cosine similarity
    sims = cosine_similarity(query_s, X_all_s).flatten()
    k = min(top_k, len(sims))
    top_idx = np.argsort(sims)[-k:]
    sims_top = sims[top_idx]
    weights = softmax(sims_top * neighbor_scale, temp=1.0)
    # USE THESE INSTEAD OF TASK WEIGHTS?

    X = X_all_s[top_idx]        # features for local regression
    preds, uncerts, coeffs = {}, {}, {}

    meta_w, meta_b = meta_prior

    for i, col in enumerate(numeric_cols):
        y = training_df[col].astype(float).values[top_idx]

        # If meta prior exists for this target, fit on residuals (y - X@w0 - b0)
        if meta_w is not None:
            w0 = meta_w[i]
            b0 = meta_b[i]
            # residual targets relative to prior mean
            # print("X stats:", np.nanmin(X), np.nanmax(X), np.isnan(X).any(), np.isinf(X).any())
            # print("w0 stats:", np.nanmin(w0), np.nanmax(w0), np.isnan(w0).any(), np.isinf(w0).any())
            # print("b0:", b0)
            # print("y stats:", np.nanmin(y), np.nanmax(y), np.isnan(y).any(), np.isinf(y).any())

            y_resid = y - (X @ w0 + b0)
            bayes = BayesianRidge()
            # fit residual model (this is the inner adaptation)
            bayes.fit(X, y_resid)

            # adapted model parameters = prior + learned residual coefficients
            mean_resid, std_resid = bayes.predict(query_s, return_std=True)
            pred = float(mean_resid[0] + (query_s @ w0 + b0).item())
            print("MEAN_RESID - ",mean_resid)
            print("W and B - ",w0,b0)
            std = float(std_resid[0])

            w_adapt = w0 + bayes.coef_
            b_adapt = b0 + bayes.intercept_
            # predict residual mean/std for query, then add prior mean back
            coeffs[col] = (w_adapt.copy(), float(b_adapt))

        preds[col] = pred
        uncerts[col] = std

    return preds, uncerts, coeffs


# -----------------------
# Main sequential run
# -----------------------
def sequential_bayesian_maml(df, numeric_cols, task_order=None,
                             top_k=20, neighbor_scale=5.0, temp=0.1,
                             predictions_csv_path="predictions_sequential.csv",
                             dummy_uncertainty=1e-6):
    """
    Processes tasks in `task_order` sequentially.
      - Tasks 1 & 2: append actual CSV rows to predictions file with dummy uncertainty.
      - From task 3 onward: use predictions file as training data, apply lightweight MAML + BayesianRidge.
    Returns: results_df (per-row preds), rmse_df (per-task RMSE), and final meta prior.
    """
    if task_order is None:
        task_order = list(df["task_name"].unique())

    # ensure embeddings are arrays
    df = df.copy()
    df["task_embedding"] = df["task_embedding"].apply(parse_embedding)
    df["dataset_embedding"] = df["dataset_embedding"].apply(parse_embedding)
    # inside sequential_bayesian_maml, after parsing embeddings
    task_mat = np.stack(df["task_embedding"].values)
    data_mat = np.stack(df["dataset_embedding"].values)

    scaler_all = StandardScaler()
    all_feats = np.concatenate([task_mat, data_mat], axis=1)
    all_feats = scaler_all.fit_transform(all_feats)

    df["task_embedding"] = list(all_feats[:, :task_mat.shape[1]])
    df["dataset_embedding"] = list(all_feats[:, task_mat.shape[1]:])


    # predictions_df: stores rows (embedding arrays + predicted numeric cols + uncertainty cols)
    preds_cols = ["task_name", "ltm_id", "task_embedding", "dataset_embedding"] + numeric_cols + \
                 [f"unc_{c}" for c in numeric_cols]
    predictions_df = pd.DataFrame(columns=preds_cols)

    meta = ContinualMetaPrior()
    records = []     # per-row predictions for output
    rmse_rows = []

    for t_idx, task in enumerate(task_order, start=1):
        df_task = df[df["task_name"] == task].reset_index(drop=True)
        print(f"\n=== Task {t_idx}/{len(task_order)}: {task} (rows={len(df_task)}) ===")

        # --- TASK 1 & 2: no model predictions, just append actual data as "predictions" with dummy uncertainty ---
        if t_idx <= 2:
            for _, row in df_task.iterrows():
                # use actual values as "predicted" values
                pred_entry = {
                    "task_name": task,
                    "ltm_id": row.get("ltm_id", None),
                    "task_embedding": row["task_embedding"],
                    "dataset_embedding": row["dataset_embedding"],
                }
                for c in numeric_cols:
                    pred_entry[c] = float(row[c])               # actual used as prediction
                    pred_entry[f"unc_{c}"] = float(dummy_uncertainty)  # dummy uncertainty
                    # record for results (actual vs pred)
                    records.append({
                        "task_name": task,
                        "ltm_id": row.get("ltm_id", None),
                        f"actual_{c}": float(row[c]),
                        f"pred_{c}": float(row[c]),
                        f"unc_{c}": float(dummy_uncertainty),
                        f"lower_{c}": float(row[c] - 1.96 * dummy_uncertainty),
                        f"upper_{c}": float(row[c] + 1.96 * dummy_uncertainty),
                    })
                predictions_df = pd.concat([predictions_df, pd.DataFrame([pred_entry])], ignore_index=True)

            # After finishing task 1/2, we can build per-task Bayesian weights (from the predictions_df rows for this task)
            # and update the meta prior so later MAML has an initial prior built from tasks 1&2.
            # Fit BayesianRidge on the task's embedding -> numeric mapping to get W_task, b_task
            X_task_feats = np.stack(df_task["task_embedding"].values)
            D_task_feats = np.stack(df_task["dataset_embedding"].values)
            X_task = np.concatenate([X_task_feats, D_task_feats], axis=1)
            W_task, b_task = [], []
            for c in numeric_cols:
                y_task = df_task[c].astype(float).values
                bayes = BayesianRidge()
                bayes.fit(X_task, y_task)
                W_task.append(bayes.coef_.copy())
                b_task.append(float(bayes.intercept_))
            W_task, b_task = np.vstack(W_task), np.array(b_task)
            meta.update_with_task(W_task, b_task, task_weight=len(df_task))

        # --- TASK >= 3: use predictions_df as the ONLY training source; incorporate lightweight MAML prior ---
        else:
            # training data is predictions_df (which contains previous tasks' predicted means)
            # ensure the columns are proper types (embeddings as arrays)
            # only use *previous* tasks for neighbor pool (exclude current task)
            training_df = predictions_df[predictions_df["task_name"] != task].copy()
            adapted_W, adapted_b = [], []
            # convert embedding columns back to arrays if they were saved as strings (this code assumes arrays already)
            # now predict for each row in current task using training_df and meta_prior
            meta_prior = meta.get_prior()
            for _, row in df_task.iterrows():
                preds, uncerts, adapted_coeffs = predict_bayes_with_maml_prior(
                    training_df=training_df,
                    query_task_emb=row["task_embedding"],
                    query_dataset_emb=row["dataset_embedding"],
                    numeric_cols=numeric_cols,
                    top_k=top_k,
                    neighbor_scale=neighbor_scale,
                    temp=temp,
                    meta_prior=meta_prior
                )

                pred_entry = {
                    "task_name": task,
                    "ltm_id": row.get("ltm_id", None),
                    "task_embedding": row["task_embedding"],
                    "dataset_embedding": row["dataset_embedding"],
                }
                for c in numeric_cols:
                    pred_entry[c] = preds[c]
                    pred_entry[f"unc_{c}"] = uncerts[c]
                    adapted_W.append(adapted_coeffs[c][0])
                    adapted_b.append(adapted_coeffs[c][1])
                    records.append({
                        "task_name": task,
                        "ltm_id": row.get("ltm_id", None),
                        f"actual_{c}": float(row[c]),
                        f"pred_{c}": float(preds[c]),
                        f"unc_{c}": float(uncerts[c]),
                    })
                predictions_df = pd.concat([predictions_df, pd.DataFrame([pred_entry])], ignore_index=True)

            # Use adapted coefficients directly to update meta-prior
            W_task = np.vstack(adapted_W)
            b_task = np.array(adapted_b)
            meta.update_with_task(W_task, b_task, task_weight=len(df_task))

        # End of task loop: compute task RMSE from `records` entries for this task
        # Convert records entries for this task into DataFrame (they were appended per numeric col; combine)
        # We'll compute RMSE per numeric column using the records we appended.
        # Build a small helper DataFrame:
        rows_for_task = [r for r in records if r["task_name"] == task]
        if len(rows_for_task) > 0:
            # reconstruct per-row actual/pred for rmse calculation
            actuals = {c: [] for c in numeric_cols}
            preds_list = {c: [] for c in numeric_cols}
            # records appended include keys like 'actual_col' and 'pred_col' per numeric col; but since we appended one small dict per numeric col,
            # they may be multiple dicts per row. Instead, recompute by looking up predictions_df last N rows for this task.
            task_rows_in_preds = predictions_df[predictions_df["task_name"] == task]
            # If first two tasks we stored actuals directly; for others we stored predicted values too.
            rmse_dict = {}
            for c in numeric_cols:
                try:
                    y_true = df_task[c].astype(float).values
                    # For predicted values, read from predictions_df rows corresponding to this task in insertion order
                    y_pred = task_rows_in_preds[c].astype(float).values[: len(y_true)]
                    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                except Exception:
                    rmse_val = float("nan")
                rmse_dict[c] = rmse_val
            rmse_dict["task"] = task
            rmse_rows.append(rmse_dict)
        else:
            rmse_rows.append({"task": task, **{c: float("nan") for c in numeric_cols}})

    # Final assemble results DataFrames
    # Build per-row results_df from `records` structure: records currently has one dict per appended numeric col per row.
    # We'll pivot records into one row per sample by grouping on 'task_name' & 'ltm_id' and extracting fields.
    # Simpler: reconstruct results_df from predictions_df joined with original df to get actuals.
    results_rows = []
    for _, pred_row in predictions_df.iterrows():
        # find matching original actual row (if exists) to get actuals
        matching = df[
            (df["task_name"] == pred_row["task_name"]) &
            (df["task_embedding"].apply(lambda x: np.array_equal(x, pred_row["task_embedding"]))) &
            (df["dataset_embedding"].apply(lambda x: np.array_equal(x, pred_row["dataset_embedding"])))
        ]
        actual_vals = None
        if len(matching) > 0:
            actual_vals = matching.iloc[0]
        # construct result entry
        ent = {
            "task_name": pred_row["task_name"],
            "ltm_id": pred_row.get("ltm_id", None)
        }
        for c in numeric_cols:
            ent[f"pred_{c}"] = float(pred_row[c])
            ent[f"unc_{c}"] = float(pred_row[f"unc_{c}"])
            if actual_vals is not None:
                a = float(actual_vals[c])
                ent[f"actual_{c}"] = a
                ent[f"lower_{c}"] = float(ent[f"pred_{c}"] - 1.96 * ent[f"unc_{c}"])
                ent[f"upper_{c}"] = float(ent[f"pred_{c}"] + 1.96 * ent[f"unc_{c}"])
            else:
                ent[f"actual_{c}"] = float("nan")
                ent[f"lower_{c}"] = float("nan")
                ent[f"upper_{c}"] = float("nan")
        results_rows.append(ent)

    results_df = pd.DataFrame(results_rows)
    rmse_df = pd.DataFrame(rmse_rows)

    # Save predictions_df (the one used as training data for later tasks) to CSV
    # Convert embeddings to strings for CSV storage
    save_df = predictions_df.copy()
    save_df["task_embedding"] = save_df["task_embedding"].apply(lambda x: "[" + ",".join(map(str, x)) + "]")
    save_df["dataset_embedding"] = save_df["dataset_embedding"].apply(lambda x: "[" + ",".join(map(str, x)) + "]")
    save_df.to_csv(predictions_csv_path, index=False)

    return results_df, rmse_df, meta, predictions_df


# -----------------------
# Example usage (entry point)
# -----------------------
if __name__ == "__main__":
    csv_path = "/Users/nehabhaskar/Documents/test/best_configs.csv"
    df = pd.read_csv(csv_path)
    numeric_cols = ["learning_rate", "weight_decay", "warmup_ratio","dropout_rate"]

    task_order = [
        "Dance Classification: Categorize different dance styles from video clips.",
        "Emotion Classification: Categorize human emotions from movie or series clips.",
        "Sports Classification: Categorize different sports activities from video clips.",
        "Gesture Recognition: Recognize and classify human gestures from video clips.",
        "Sport Recognition: Identify and categorize different sports or athletic activities present within video content.",
        # "Emotion Classification: Identify and categorize human emotions from clips extracted from movies or TV series."
    ]

    # task_order = [
    #     "Multi-class classification: each text belongs to one category from a fixed set.",
    #     "Each document is classified into one, and only one, category from the set",
    #     "Sentiment Analysis: Classify the text into the respective sentiments."
    # ]

    results_df, rmse_df, meta, preds_df = sequential_bayesian_maml(
        df, numeric_cols, task_order=task_order,
        top_k=20, neighbor_scale=5.0, temp=0.1,
        predictions_csv_path="bayesian_maml_best.csv",
        dummy_uncertainty=1e-6
    )

    print("\nPer-task RMSE:")
    print(rmse_df)
    print("\nSaved predictions to 'predictions_sequential.csv'")
