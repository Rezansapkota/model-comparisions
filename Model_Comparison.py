# Exported from the fully executed verification notebook.
# Cell 1
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path
import json, platform, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

SEED = 42
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
versions = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "tensorflow": tf.__version__}
print(versions)
from sklearn.model_selection import StratifiedGroupKFold
import hashlib, urllib.request, zipfile, io
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# Cell 3
DATA_PATH = Path("data/dataset.csv")
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(exist_ok=True)
    payload = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/942/rt-iot2022.zip", timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        DATA_PATH.write_bytes(archive.read("RT_IOT2022"))
dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
raw = pd.read_csv(DATA_PATH)
TARGET = "Attack_type"
df = raw.drop(columns=[c for c in raw if c.startswith("Unnamed:")]).copy()
assert df[TARGET].notna().all()
duplicate_count = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
base_features = [c for c in df if c != TARGET]
numeric = df[base_features].select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in base_features if c not in numeric]
df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
groups = pd.util.hash_pandas_object(df[base_features], index=False)
conflicts = df.groupby(groups)[TARGET].nunique().gt(1).sum()
print("Raw shape:", raw.shape, "Predictors:", len(base_features))
print("Exact duplicate records removed:", duplicate_count, "Remaining:", len(df))
print("Feature groups with conflicting labels:", conflicts)
print("SHA256:", dataset_sha256)
display(df[TARGET].value_counts().rename("records").to_frame())
audit = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique()})
display(audit)
audit.to_csv(OUTPUT / "data_audit.csv")

# Cell 5
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
dev_idx, test_idx = next(splitter.split(df[base_features], df[TARGET], groups))
train_df, test_df = df.iloc[dev_idx].copy(), df.iloc[test_idx].copy()
fit_idx, val_idx = next(splitter.split(train_df[base_features], train_df[TARGET], groups.loc[train_df.index]))
fit_df, val_df = train_df.iloc[fit_idx].copy(), train_df.iloc[val_idx].copy()
parts = {"fit": fit_df, "validation": val_df, "test": test_df}
for name, part in parts.items():
    assert set(part[TARGET]) == set(df[TARGET]), f"Missing class in {name}"
    print(name, part.shape)
for a, b in [(fit_df, val_df), (fit_df, test_df), (val_df, test_df)]:
    assert set(groups.loc[a.index]).isdisjoint(groups.loc[b.index])
split_counts = pd.DataFrame({name: part[TARGET].value_counts() for name, part in parts.items()})
display(split_counts)
split_counts.to_csv(OUTPUT / "split_class_counts.csv")
pd.concat([pd.DataFrame({"source_row": part.index, "split": name}) for name, part in parts.items()]).to_csv(OUTPUT / "split_membership.csv", index=False)

# Cell 7
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fit_df[TARGET].value_counts().sort_values().plot.barh(ax=axes[0])
axes[0].set(title="Fitting class counts", xlabel="Records (log scale)", xscale="log")
fit_df["proto"].value_counts().plot.bar(ax=axes[1])
axes[1].set(title="Fitting protocols", ylabel="Records")
fig.tight_layout(); fig.savefig(OUTPUT / "class_distribution.png", dpi=150); plt.show()
display(fit_df[["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot"]].describe())
print("Missing fitting values:", int(fit_df[base_features].isna().sum().sum()))
# Week 6: skewness, bivariate class differences and multivariate correlation.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
values = fit_df["flow_duration"].clip(lower=0)
axes[0].hist(values, bins=50); axes[0].set(title="Raw flow duration", xlabel="Dataset units")
axes[1].hist(np.log1p(values), bins=50); axes[1].set(title="log1p flow duration", xlabel="log1p(dataset units)")
fig.tight_layout(); fig.savefig(OUTPUT / "duration_transformation.png", dpi=160); plt.show()
crosstab = pd.crosstab(fit_df[TARGET], fit_df["proto"], normalize="index")
crosstab.plot.barh(stacked=True, figsize=(10, 6))
plt.xlabel("Within-class proportion"); plt.title("Protocol and traffic class: fitting data")
plt.tight_layout(); plt.savefig(OUTPUT / "protocol_by_class.png", dpi=160); plt.show()
corr_columns = ["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "fwd_pkts_payload.tot", "bwd_pkts_payload.tot", "flow_SYN_flag_count", "flow_ACK_flag_count"]
correlations = fit_df[corr_columns].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(corr_columns)), corr_columns, rotation=90)
ax.set_yticks(range(len(corr_columns)), corr_columns)
fig.colorbar(im, ax=ax, label="Spearman correlation")
ax.set_title("Traffic-feature correlation: fitting data")
fig.tight_layout(); fig.savefig(OUTPUT / "feature_correlations.png", dpi=160); plt.show()
correlations.to_csv(OUTPUT / "feature_correlations.csv")


# Cell 9
def make_features(frame, feature_set):
    result = frame[base_features].copy()
    if feature_set == "log_numeric":
        result[numeric] = np.sign(result[numeric]) * np.log1p(np.abs(result[numeric]))
    return result

def make_preprocessor(X):
    categories = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("category", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                               ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categories)
    ])

encoder = LabelEncoder().fit(fit_df[TARGET])
classes = encoder.classes_
y_fit = encoder.transform(fit_df[TARGET])
y_val = encoder.transform(val_df[TARGET])
y_train = encoder.transform(train_df[TARGET])
def scores(y, predictions):
    return {"accuracy": accuracy_score(y, predictions),
            "precision_macro": precision_score(y, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y, predictions, average="macro", zero_division=0)}
print("Classes:", dict(enumerate(classes)))

# Cell 11
specs = {
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [10, None], "min_samples_leaf": [1, 5]}),
    "Random Forest": (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
                      {"max_depth": [16, None], "min_samples_leaf": [1, 5]}),
    "Naive Bayes": (GaussianNB(), {"var_smoothing": [1e-9, 1e-5, 1e-2]})
}
search_rows, chosen = [], {}
for name, (estimator, grid) in specs.items():
    best = None
    for feature_set in ["base", "log_numeric"]:
        Xf, Xv = make_features(fit_df, feature_set), make_features(val_df, feature_set)
        for params in ParameterGrid(grid):
            model = Pipeline([("preprocessing", make_preprocessor(Xf)),
                              ("classifier", clone(estimator).set_params(**params))])
            model.fit(Xf, y_fit)
            metrics = scores(y_val, model.predict(Xv))
            row = {"model": name, "features": feature_set, "parameters": json.dumps(params), **metrics}
            search_rows.append(row)
            if best is None or metrics["f1_macro"] > best["f1_macro"]:
                best = {**row, "pipeline": model}
    chosen[name] = best
    print(name, "validation macro F1:", round(best["f1_macro"], 4), best["features"])
dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_fit), 1)), y_fit)
dummy_val = scores(y_val, dummy.predict(np.zeros((len(y_val), 1))))
search_table = pd.DataFrame(search_rows)
display(search_table.sort_values("f1_macro", ascending=False))
print("Baseline validation:", dummy_val)

# Cell 13
def build_ann(widths, input_width):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_width,)),
        tf.keras.layers.Dense(widths[0], activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(widths[1], activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(classes), activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

ann_best, ann_rows = None, []
for feature_set in ["base", "log_numeric"]:
    Xf, Xv = make_features(fit_df, feature_set), make_features(val_df, feature_set)
    prep = make_preprocessor(Xf)
    Zf = prep.fit_transform(Xf).astype("float32")
    Zv = prep.transform(Xv).astype("float32")
    for widths in [(32, 16), (64, 32)]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        ann = build_ann(widths, Zf.shape[1])
        history = ann.fit(Zf, y_fit, validation_data=(Zv, y_val), epochs=40,
                          batch_size=256, verbose=0,
                          callbacks=[tf.keras.callbacks.EarlyStopping(
                              monitor="val_loss", patience=5, restore_best_weights=True)])
        predictions = ann.predict(Zv, verbose=0).argmax(axis=1)
        metrics = scores(y_val, predictions)
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        row = {"model": "MLP", "features": feature_set, "widths": list(widths),
               "best_epoch": best_epoch, **metrics}
        ann_rows.append(row)
        if ann_best is None or metrics["f1_macro"] > ann_best["f1_macro"]:
            ann_best = {**row, "history": history.history}
        print("MLP", feature_set, widths, "validation macro F1:", round(metrics["f1_macro"], 4))
display(pd.DataFrame(ann_rows))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["loss", "accuracy"]):
    h = ann_best["history"]
    ax.plot(range(1, len(h[metric]) + 1), h[metric], label="Training")
    ax.plot(range(1, len(h[metric]) + 1), h["val_" + metric], label="Validation")
    ax.set(xlabel="Epoch", ylabel=metric, title="Selected MLP: " + metric)
    ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT / "ann_learning_curves.png", dpi=160); plt.show()

# Cell 15
ablation = pd.concat([search_table, pd.DataFrame(ann_rows)], ignore_index=True)
ablation = ablation.groupby(["model", "features"])["f1_macro"].max().unstack()
display(ablation)
ablation.to_csv(OUTPUT / "feature_comparison.csv")

# Cell 17
from sklearn.feature_selection import SelectFromModel
rf_item = chosen["Random Forest"]
rf_params = json.loads(rf_item["parameters"])
rf_factory = lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED, **rf_params)
Xfit_rf = make_features(fit_df, rf_item["features"])
Xval_rf = make_features(val_df, rf_item["features"])
prep_rf = make_preprocessor(Xfit_rf)
Zfit_rf = prep_rf.fit_transform(Xfit_rf)
Zval_rf = prep_rf.transform(Xval_rf)
selector = SelectFromModel(rf_factory(), threshold=-np.inf, max_features=20)
Zfit_selected = selector.fit_transform(Zfit_rf, y_fit)
Zval_selected = selector.transform(Zval_rf)
selected_rf = rf_factory().fit(Zfit_selected, y_fit)
encoded_names = prep_rf.get_feature_names_out()
importance_table = pd.DataFrame({"feature": encoded_names, "importance": selector.estimator_.feature_importances_, "selected": selector.get_support()}).sort_values("importance", ascending=False)
importance_table.to_csv(OUTPUT / "feature_importance.csv", index=False)
display(importance_table.head(20))
importance_table.head(15).sort_values("importance").plot.barh(x="feature", y="importance", legend=False, figsize=(10, 6))
plt.title("Fitting-only forest impurity importance"); plt.tight_layout()
plt.savefig(OUTPUT / "feature_importance.png", dpi=160); plt.show()

def engineered_features(frame):
    X = make_features(frame, rf_item["features"])
    fwd, bwd = frame["fwd_pkts_tot"], frame["bwd_pkts_tot"]
    X["packet_direction_balance"] = (fwd - bwd) / (fwd + bwd + 1)
    fbytes, bbytes = frame["fwd_pkts_payload.tot"], frame["bwd_pkts_payload.tot"]
    X["backward_payload_share"] = bbytes / (fbytes + bbytes + 1)
    return X
Xfit_eng, Xval_eng = engineered_features(fit_df), engineered_features(val_df)
eng_numeric = numeric + ["packet_direction_balance", "backward_payload_share"]
eng_prep = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), eng_numeric),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
])
engineered_rf = Pipeline([("preprocessing", eng_prep), ("classifier", rf_factory())]).fit(Xfit_eng, y_fit)
feature_experiments = pd.DataFrame([
    {"experiment": "Full-feature selected RF", "encoded_features": Zfit_rf.shape[1], **scores(y_val, rf_item["pipeline"].predict(Xval_rf))},
    {"experiment": "Top-20 encoded features", "encoded_features": Zfit_selected.shape[1], **scores(y_val, selected_rf.predict(Zval_selected))},
    {"experiment": "Two engineered ratios", "encoded_features": len(engineered_rf.named_steps["preprocessing"].get_feature_names_out()), **scores(y_val, engineered_rf.predict(Xval_eng))}
])
display(feature_experiments)
feature_experiments.to_csv(OUTPUT / "supplementary_validation.csv", index=False)
print("Supplementary validation only; primary test model choices remain fixed.")


# Cell 19
selection = {name: {k: v for k, v in item.items() if k != "pipeline"}
             for name, item in chosen.items()}
selection["MLP"] = {k: v for k, v in ann_best.items() if k != "history"}
validation_scores = {name: item["f1_macro"] for name, item in selection.items()}
validation_scores["Majority baseline"] = dummy_val["f1_macro"]
recommended = max(validation_scores, key=validation_scores.get)
y_test = encoder.transform(test_df[TARGET])
final_predictions, test_rows, fitted_models = {}, [], {}
for name, item in chosen.items():
    start = time.perf_counter()
    model = clone(item["pipeline"])
    model.fit(make_features(train_df, item["features"]), y_train)
    elapsed = time.perf_counter() - start
    predictions = model.predict(make_features(test_df, item["features"]))
    final_predictions[name] = predictions
    fitted_models[name] = model
    test_rows.append({"model": name, **scores(y_test, predictions), "fit_seconds": elapsed})

start = time.perf_counter()
ann_prep = make_preprocessor(make_features(train_df, ann_best["features"]))
Ztrain = ann_prep.fit_transform(make_features(train_df, ann_best["features"])).astype("float32")
Ztest = ann_prep.transform(make_features(test_df, ann_best["features"])).astype("float32")
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
final_ann = build_ann(ann_best["widths"], Ztrain.shape[1])
final_ann.fit(Ztrain, y_train, epochs=ann_best["best_epoch"], batch_size=256, verbose=0)
elapsed = time.perf_counter() - start
final_predictions["MLP"] = final_ann.predict(Ztest, verbose=0).argmax(axis=1)
test_rows.append({"model": "MLP", **scores(y_test, final_predictions["MLP"]), "fit_seconds": elapsed})
baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
final_predictions["Majority baseline"] = baseline.predict(np.zeros((len(y_test), 1)))
test_rows.append({"model": "Majority baseline", **scores(y_test, final_predictions["Majority baseline"]),
                  "fit_seconds": np.nan})
results = pd.DataFrame(test_rows).set_index("model")
display(results.round(4))
print("Model selected using validation macro F1:", recommended)

# Cell 20
fig, axes = plt.subplots(2, 3, figsize=(24, 17))
reports = []
for ax, (name, predictions) in zip(axes.flat, final_predictions.items()):
    assert len(predictions) == len(y_test)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=classes,
        ax=ax, colorbar=False, cmap="Blues", xticks_rotation=90, values_format="d")
    ax.set_title(name)
    report = pd.DataFrame(classification_report(y_test, predictions, target_names=classes,
                        output_dict=True, zero_division=0)).T
    report["model"] = name
    reports.append(report)
    print(name); display(report.round(4))
axes.flat[-1].axis("off")
fig.tight_layout(); fig.savefig(OUTPUT / "confusion_matrices.png", dpi=140); plt.show()
ax = results[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot.bar(figsize=(11, 5))
plt.ylim(0, 1); plt.ylabel("Score"); plt.title("Final test comparison")
plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(OUTPUT / "test_comparison.png", dpi=150); plt.show()
results.to_csv(OUTPUT / "test_metrics.csv")
pd.concat(reports).to_csv(OUTPUT / "classification_reports.csv")

# Cell 22
summary = (
    f"Validation-selected model: {recommended}\n"
    f"Selected model test accuracy: {results.loc[recommended, 'accuracy']:.4f}\n"
    f"Selected model test macro F1: {results.loc[recommended, 'f1_macro']:.4f}\n"
    f"Test majority baseline accuracy: {results.loc['Majority baseline', 'accuracy']:.4f}\n"
    "These are within-dataset benchmark results, not deployment performance.\n"
)
print(summary)
(OUTPUT / "summary.txt").write_text(summary, encoding="utf-8")
search_table.to_csv(OUTPUT / "classical_validation.csv", index=False)
pd.DataFrame(ann_rows).to_csv(OUTPUT / "ann_validation.csv", index=False)
pd.DataFrame(ann_best["history"]).to_csv(OUTPUT / "selected_ann_history.csv", index=False)
prediction_table = pd.DataFrame({"source_row": test_df.index, "actual": classes[y_test]})
for name, predictions in final_predictions.items():
    prediction_table[name] = classes[predictions]
prediction_table.to_csv(OUTPUT / "test_predictions.csv", index=False)
manifest = {"dataset": "UCI RT-IoT2022", "doi": "10.24432/C5P338", "sha256": dataset_sha256,
            "seed": SEED, "versions": versions, "selection": selection, "recommended": recommended,
            "raw_rows": len(raw), "deduplicated_rows": len(df), "predictors": base_features,
            "split": "first fold of seeded 5-fold StratifiedGroupKFold, repeated on development",
            "partition_sizes": {name: len(part) for name, part in parts.items()}}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
# Save fitted artifacts and the feature/label metadata needed to reproduce inference.
import joblib
model_dir = OUTPUT / "models"
model_dir.mkdir(exist_ok=True)
for name, fitted in fitted_models.items():
    joblib.dump(fitted, model_dir / (name.lower().replace(" ", "_") + ".joblib"))
joblib.dump(ann_prep, model_dir / "ann_preprocessor.joblib")
joblib.dump(encoder, model_dir / "label_encoder.joblib")
final_ann.save(model_dir / "ann.keras")
(model_dir / "feature_notes.txt").write_text("Apply make_features using the selected feature set from manifest.json before the saved pipeline or ANN preprocessor. Use label_encoder.joblib to decode classes. Only load trusted joblib files.", encoding="utf-8")
