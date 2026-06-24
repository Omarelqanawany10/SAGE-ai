# %% [markdown]
# # Crop Water Quantity Prediction - Regression Notebook
#
# **Objective:** Build a supervised Machine Learning regression pipeline that predicts
# the daily crop water requirement from crop, soil, climate, irrigation, and field
# management variables.
#
# The target used in this dataset is `Water_Need_mm_day`, which represents the
# water depth required per day in millimeters. When field area is available,
# this can be converted to a daily water volume:
#
# `liters_per_day = predicted_mm_day * field_area_hectare * 10000`
#
# This notebook follows the requested production-style pipeline:
#
# 1. Complete library imports
# 2. Data loading
# 3. Exploratory Data Analysis and visualization
# 4. Data preprocessing and feature engineering
# 5. Model building and training
# 6. Model evaluation
# 7. Model artifact saving for the Streamlit dashboard

# %% [markdown]
# ## 1. Complete Library Imports
#
# Keep imports in one place so the environment requirements are explicit.

# %%
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from crop_water_model_utils import (
    IQRClipper,
    add_engineered_features,
    make_one_hot_encoder,
    to_jsonable,
)

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "Water_Need_mm_day"
MAX_REASONABLE_TEST_R2 = 0.95

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titleweight"] = "bold"

print("XGBoost available:", XGBOOST_AVAILABLE)

# %% [markdown]
# ## 2. Data Loading
#
# The first path is the requested placeholder (`crop_water_data.csv`). The second
# path points to the dataset location used in this project workspace.

# %%
candidate_paths = [
    Path("crop_water_data.csv"),
    Path("../data sets/crop_water_prediction_dataset_with_source.csv"),
]

DATA_PATH = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Place the CSV as 'crop_water_data.csv' in this "
        "notebook folder, or update DATA_PATH to your dataset path."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print(f"Loaded dataset from: {DATA_PATH.resolve()}")
print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
display(df.head())

# %%
if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' was not found. Available columns: {df.columns.tolist()}"
    )

display(
    pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "unique_values": df.nunique(dropna=True).values,
        }
    )
)

# %% [markdown]
# ## 3. Exploratory Data Analysis and Visualization
#
# This section examines data quality, distributions, feature relationships, target
# behavior, and potential outliers before any modeling decisions are made.

# %%
print("Dataset information:")
df.info()

# %%
numeric_cols_eda = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols_eda = df.select_dtypes(exclude=np.number).columns.tolist()

print(f"Numeric columns ({len(numeric_cols_eda)}): {numeric_cols_eda}")
print(f"Categorical columns ({len(categorical_cols_eda)}): {categorical_cols_eda}")

display(df[numeric_cols_eda].describe().T)

# %%
missing_summary = (
    df.isna()
    .sum()
    .to_frame("missing_count")
    .assign(missing_percent=lambda x: x["missing_count"] / len(df) * 100)
    .query("missing_count > 0")
    .sort_values("missing_percent", ascending=False)
)

if missing_summary.empty:
    print("No missing values were detected.")
else:
    display(missing_summary)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df[TARGET_COL], kde=True, bins=40, ax=axes[0])
axes[0].set_title("Target Distribution: Water Need (mm/day)")
axes[0].set_xlabel("Water need (mm/day)")

sns.boxplot(x=df[TARGET_COL], ax=axes[1])
axes[1].set_title("Target Boxplot: Outlier Check")
axes[1].set_xlabel("Water need (mm/day)")

plt.tight_layout()
plt.show()

# %%
target_summary = df[TARGET_COL].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
display(target_summary.to_frame("Water_Need_mm_day"))

non_positive_target_rate = (df[TARGET_COL] <= 0).mean()
print(f"Non-positive target rate: {non_positive_target_rate:.2%}")

# %%
correlation_matrix = df[numeric_cols_eda].corr(numeric_only=True)

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot=False,
)
plt.title("Correlation Heatmap for Numerical Features")
plt.tight_layout()
plt.show()

# %%
target_correlations = (
    correlation_matrix[TARGET_COL]
    .drop(labels=[TARGET_COL], errors="ignore")
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

display(target_correlations.to_frame("correlation_with_target"))

top_numeric_features = target_correlations.head(6).index.tolist()
print("Top numeric features for scatter plots:", top_numeric_features)

# %%
if top_numeric_features:
    n_cols = 3
    n_rows = int(np.ceil(len(top_numeric_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for axis, feature in zip(axes, top_numeric_features):
        sns.scatterplot(
            data=df.sample(min(len(df), 5000), random_state=RANDOM_STATE),
            x=feature,
            y=TARGET_COL,
            alpha=0.35,
            ax=axis,
        )
        axis.set_title(f"{feature} vs {TARGET_COL}")

    for axis in axes[len(top_numeric_features) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Matplotlib Scatter Diagnostics
#
# These plots use plain Matplotlib so the relationship between important
# numerical features and the target is easy to inspect without relying only on
# seaborn defaults. The color scale represents the target value, which helps
# reveal clusters, non-linear behavior, and suspicious target-proxy patterns.

# %%
matplotlib_features = top_numeric_features[:6]

if matplotlib_features:
    plot_sample = df.sample(min(len(df), 5000), random_state=RANDOM_STATE)
    n_cols = 3
    n_rows = int(np.ceil(len(matplotlib_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for axis, feature in zip(axes, matplotlib_features):
        scatter = axis.scatter(
            plot_sample[feature],
            plot_sample[TARGET_COL],
            c=plot_sample[TARGET_COL],
            cmap="viridis",
            alpha=0.35,
            s=16,
            edgecolors="none",
        )
        axis.set_title(f"{feature} vs {TARGET_COL}")
        axis.set_xlabel(feature)
        axis.set_ylabel("Water need (mm/day)")
        axis.grid(True, alpha=0.25)

    for axis in axes[len(matplotlib_features) :]:
        axis.axis("off")

    fig.colorbar(scatter, ax=axes.tolist(), label="Water need (mm/day)", shrink=0.85)
    fig.suptitle("Matplotlib Scatter Views for Target Relationships", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

    corr_subset = target_correlations.head(10).sort_values()
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(corr_subset.index, corr_subset.values, color="#3B82F6")
    axis.axvline(0, color="black", linewidth=1)
    axis.set_title("Top Numeric Correlations With Target")
    axis.set_xlabel("Pearson correlation")
    axis.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()

# %%
categorical_target_columns = [
    column
    for column in categorical_cols_eda
    if column not in {"Irrigation_Need_Label", "Source"}
]

for column in categorical_target_columns[:6]:
    category_summary = (
        df.groupby(column, dropna=False)[TARGET_COL]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
    )
    display(category_summary)

    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=category_summary.reset_index().head(20),
        x=column,
        y="mean",
    )
    plt.title(f"Average Water Need by {column}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average water need (mm/day)")
    plt.tight_layout()
    plt.show()

# %%
def iqr_outlier_summary(data: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Estimate outlier counts using the common 1.5 * IQR rule."""
    rows = []

    for column in numeric_columns:
        series = data[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()

        rows.append(
            {
                "feature": column,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_count / len(series) * 100,
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_percent", ascending=False)


outlier_report = iqr_outlier_summary(df, numeric_cols_eda)
display(outlier_report.head(20))

# %% [markdown]
# ## 4. Data Preprocessing and Feature Engineering
#
# The model pipeline handles:
#
# - missing numerical values with median imputation
# - missing categorical values with most-frequent imputation
# - numerical outliers with an IQR clipping transformer fitted on training data only
# - numerical scaling with `RobustScaler`
# - categorical encoding with one-hot encoding
# - feature engineering using domain-inspired variables
#
# `Irrigation_Need_Label` is dropped because it is a label derived from water need,
# so it can leak target information. `Source` is dropped because it is a dataset
# provenance field, not an operational input expected from a farmer or sensor.

# %%
model_df = add_engineered_features(df)
engineered_columns = sorted(set(model_df.columns) - set(df.columns))

print("Engineered feature columns:")
display(pd.DataFrame({"engineered_feature": engineered_columns}))

# Columns removed from model training:
# - Irrigation_Need_Label is a target-derived class label.
# - Previous_Irrigation_mm is a previous/farmer irrigation decision that can
#   encode the answer we want to predict.
# - Source is dataset provenance and should not be available in production.
# - Field_Area_hectare is not needed to predict water depth in mm/day. It is
#   kept only after prediction to convert depth into total volume.
# - Engineered features based on previous irrigation are also removed.
FORBIDDEN_LEAKAGE_COLUMNS = [
    "Irrigation_Need_Label",
    "Previous_Irrigation_mm",
]
label_or_provenance_columns = ["Source"]
leakage_risk_columns = [
    *FORBIDDEN_LEAKAGE_COLUMNS,
    "Recent_Water_Available_mm",
    "Previous_Irrigation_Volume_m3",
]
volume_only_columns = ["Field_Area_hectare"]

drop_from_model_columns = list(
    dict.fromkeys(label_or_provenance_columns + leakage_risk_columns + volume_only_columns)
)
drop_columns = [TARGET_COL] + [
    column for column in drop_from_model_columns if column in model_df.columns
]

y = pd.to_numeric(model_df[TARGET_COL], errors="coerce")
X = model_df.drop(columns=drop_columns, errors="ignore")

remaining_forbidden_columns = [
    column for column in FORBIDDEN_LEAKAGE_COLUMNS if column in X.columns
]
if remaining_forbidden_columns:
    raise ValueError(
        "Leakage guard failed. These forbidden columns are still in X: "
        f"{remaining_forbidden_columns}"
    )

valid_target_mask = y.notna()
X = X.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].copy()

categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_features = [column for column in X.columns if column not in categorical_features]
model_engineered_columns = [column for column in engineered_columns if column in X.columns]

leakage_audit = pd.DataFrame(
    {
        "column": drop_from_model_columns,
        "reason": [
            "Dataset provenance; not a production feature.",
            "Target-derived label; direct leakage.",
            "Previous action can encode the target decision.",
            "Engineered from previous irrigation; leakage-prone.",
            "Engineered from previous irrigation and field area; leakage-prone.",
            "Used only to convert predicted depth to volume, not to predict mm/day.",
        ],
    }
)

print(f"Training rows after removing missing targets: {len(X):,}")
print(f"Numerical model features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical model features ({len(categorical_features)}): {categorical_features}")
display(leakage_audit)

# %%
def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a fresh preprocessing pipeline for each model."""
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("iqr_clipper", IQRClipper(factor=1.5)),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


preprocessor_preview = build_preprocessor(numeric_features, categorical_features)
preprocessor_preview

# %% [markdown]
# ## 5. Model Building and Training
#
# We train and compare at least two regression models:
#
# - Random Forest Regressor
# - XGBoost Regressor, when `xgboost` is installed
#
# If XGBoost is not installed, the notebook falls back to sklearn's
# `HistGradientBoostingRegressor` so the comparison still contains a tree-based
# boosting model. To install XGBoost locally, run `pip install xgboost`.

# %%
split_strategy = "random train_test_split"

if "Source" in df.columns and df.loc[valid_target_mask, "Source"].nunique() > 1:
    # Holding out an entire data source is a stricter test than random splitting.
    # It reduces the chance that source-specific synthetic patterns appear in
    # both train and test sets.
    groups = df.loc[valid_target_mask, "Source"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    split_strategy = "GroupShuffleSplit by Source"
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

print(f"Training set: {X_train.shape[0]:,} rows")
print(f"Testing set: {X_test.shape[0]:,} rows")
print(f"Split strategy: {split_strategy}")

# %%
model_registry = {
    "Ridge Regressor": Ridge(
        alpha=25.0,
        random_state=RANDOM_STATE,
    ),
    "ElasticNet Regressor": ElasticNet(
        alpha=0.03,
        l1_ratio=0.20,
        max_iter=10000,
        random_state=RANDOM_STATE,
    ),
    "Regularized Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=25,
        min_samples_split=50,
        max_features=0.60,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
}

if XGBOOST_AVAILABLE:
    model_registry["Regularized XGBoost Regressor"] = XGBRegressor(
        n_estimators=250,
        learning_rate=0.025,
        max_depth=2,
        min_child_weight=15,
        subsample=0.65,
        colsample_bytree=0.65,
        reg_alpha=1.0,
        reg_lambda=10.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
else:
    model_registry["Regularized HistGradientBoosting Regressor"] = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.03,
        max_leaf_nodes=7,
        min_samples_leaf=60,
        l2_regularization=5.0,
        random_state=RANDOM_STATE,
    )

trained_models = {}

for model_name, estimator in model_registry.items():
    print(f"Training {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    trained_models[model_name] = pipeline

print("Training complete.")

# %%
RUN_CROSS_VALIDATION = False

if RUN_CROSS_VALIDATION:
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []

    for model_name, estimator in model_registry.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "MSE": "neg_mean_squared_error",
                "R2": "r2",
            },
            cv=cv,
            n_jobs=1,
        )

        cv_rows.append(
            {
                "model": model_name,
                "cv_mae_mean": -scores["test_MAE"].mean(),
                "cv_rmse_mean": np.sqrt(-scores["test_MSE"].mean()),
                "cv_r2_mean": scores["test_R2"].mean(),
            }
        )

    display(pd.DataFrame(cv_rows).sort_values("cv_rmse_mean"))

# %% [markdown]
# ## 6. Model Evaluation
#
# Regression performance is evaluated with:
#
# - **MAE:** average absolute prediction error in mm/day
# - **RMSE:** stronger penalty for large errors, also in mm/day
# - **R2 Score:** explained variance, where 1.0 is best

# %%
def evaluate_regression_model(
    model_name: str,
    model: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> dict[str, float]:
    """Generate standard regression metrics for a fitted model."""
    predictions = model.predict(X_eval)

    mae = mean_absolute_error(y_eval, predictions)
    rmse = np.sqrt(mean_squared_error(y_eval, predictions))
    r2 = r2_score(y_eval, predictions)

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


evaluation_rows = []
test_predictions = {}

for model_name, model in trained_models.items():
    train_metrics = evaluate_regression_model(model_name, model, X_train, y_train)
    train_metrics["split"] = "train"
    evaluation_rows.append(train_metrics)

    test_metrics = evaluate_regression_model(model_name, model, X_test, y_test)
    test_metrics["split"] = "test"
    evaluation_rows.append(test_metrics)

    test_predictions[model_name] = model.predict(X_test)

performance_df = (
    pd.DataFrame(evaluation_rows)
    .sort_values(["split", "RMSE"], ascending=[True, True])
    .reset_index(drop=True)
)

evaluation_df = (
    performance_df.query("split == 'test'")
    .drop(columns=["split"])
    .sort_values("RMSE", ascending=True)
    .reset_index(drop=True)
)

train_metrics_df = performance_df.query("split == 'train'").set_index("model")
test_metrics_df = performance_df.query("split == 'test'").set_index("model")
overfit_summary = test_metrics_df.join(
    train_metrics_df,
    lsuffix="_test",
    rsuffix="_train",
)
overfit_summary["R2_gap_train_minus_test"] = (
    overfit_summary["R2_train"] - overfit_summary["R2_test"]
)
overfit_summary["RMSE_gap_test_minus_train"] = (
    overfit_summary["RMSE_test"] - overfit_summary["RMSE_train"]
)

display(performance_df)
display(overfit_summary.sort_values("RMSE_test"))
display(evaluation_df)

# %%
metric_plot_df = performance_df.melt(
    id_vars=["model", "split"],
    value_vars=["MAE", "RMSE", "R2"],
    var_name="metric",
    value_name="value",
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for axis, metric in zip(axes, ["MAE", "RMSE", "R2"]):
    plot_df = metric_plot_df[metric_plot_df["metric"] == metric]
    pivot_df = plot_df.pivot(index="model", columns="split", values="value")
    x_positions = np.arange(len(pivot_df.index))
    width = 0.35

    axis.bar(
        x_positions - width / 2,
        pivot_df.get("train", pd.Series(index=pivot_df.index, dtype=float)),
        width,
        label="Train",
        color="#60A5FA",
    )
    axis.bar(
        x_positions + width / 2,
        pivot_df.get("test", pd.Series(index=pivot_df.index, dtype=float)),
        width,
        label="Test",
        color="#F97316",
    )
    axis.set_title(metric)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(pivot_df.index, rotation=25, ha="right")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

plt.tight_layout()
plt.show()

# %%
n_models = len(trained_models)
fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)
axes = axes.reshape(-1)

for axis, (model_name, predictions) in zip(axes, test_predictions.items()):
    residuals = y_test - predictions
    scatter = axis.scatter(
        y_test,
        predictions,
        c=residuals,
        cmap="coolwarm",
        alpha=0.45,
        s=16,
        edgecolors="none",
    )

    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    axis.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")

    axis.set_title(f"Predicted vs Actual - {model_name}")
    axis.set_xlabel("Actual water need (mm/day)")
    axis.set_ylabel("Predicted water need (mm/day)")
    axis.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axis, label="Residual")

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)
axes = axes.reshape(-1)

for axis, (model_name, predictions) in zip(axes, test_predictions.items()):
    residuals = y_test - predictions
    axis.hist(residuals, bins=40, color="#3B82F6", alpha=0.75, edgecolor="white")
    axis.axvline(0, color="red", linestyle="--")
    axis.set_title(f"Residual Distribution - {model_name}")
    axis.set_xlabel("Actual - Predicted (mm/day)")
    axis.set_ylabel("Count")
    axis.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# %%
def plot_feature_importance(fitted_pipeline: Pipeline, top_n: int = 20) -> None:
    """Plot feature importances for tree-based models that expose them."""
    model = fitted_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print("The selected model does not expose feature_importances_.")
        return

    feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()

    display(importance_df)


overfit_rejected_models = evaluation_df[evaluation_df["R2"] > MAX_REASONABLE_TEST_R2].copy()
accepted_models = evaluation_df[evaluation_df["R2"] <= MAX_REASONABLE_TEST_R2].copy()

if not overfit_rejected_models.empty:
    print(
        f"Rejected {len(overfit_rejected_models)} model(s) with test R2 above "
        f"{MAX_REASONABLE_TEST_R2:.2f}. Treat these as overfit or leakage-suspect."
    )
    display(overfit_rejected_models)

if accepted_models.empty:
    raise RuntimeError(
        "All trained models exceeded the maximum reasonable test R2 threshold "
        f"({MAX_REASONABLE_TEST_R2:.2f}). Do not save a deployment model until "
        "more leakage/proxy features are removed or the validation split is made stricter."
    )

model_selection_df = (
    accepted_models
    .assign(distance_to_target_r2=lambda data: (MAX_REASONABLE_TEST_R2 - data["R2"]).abs())
    .sort_values(["distance_to_target_r2", "RMSE"], ascending=[True, True])
    .reset_index(drop=True)
)

display(model_selection_df)

best_model_name = model_selection_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print(
    f"Selected deployment model: {best_model_name} "
    f"(test R2 <= {MAX_REASONABLE_TEST_R2:.2f})"
)
plot_feature_importance(best_model, top_n=20)

# %% [markdown]
# ## Save the Best Model for Production Use
#
# The saved artifact is the full sklearn pipeline, including preprocessing,
# categorical encoding, outlier clipping, scaling, and the fitted regression
# model. The metadata file stores the feature schema and UI defaults used by the
# Streamlit dashboard.

# %%
model_path = ARTIFACT_DIR / "best_crop_water_model.joblib"
metadata_path = ARTIFACT_DIR / "model_metadata.json"

joblib.dump(best_model, model_path)

prediction_raw_input_columns = [
    column
    for column in df.columns
    if column not in [TARGET_COL, *label_or_provenance_columns, *leakage_risk_columns, *volume_only_columns]
]
optional_context_columns = [column for column in volume_only_columns if column in df.columns]
raw_input_columns = prediction_raw_input_columns + optional_context_columns
raw_numeric_inputs = [
    column
    for column in raw_input_columns
    if pd.api.types.is_numeric_dtype(df[column])
]
raw_categorical_inputs = [
    column
    for column in raw_input_columns
    if column not in raw_numeric_inputs
]

numeric_input_stats = {}
for column in raw_numeric_inputs:
    series = pd.to_numeric(df[column], errors="coerce")
    numeric_input_stats[column] = {
        "min": to_jsonable(series.min()),
        "median": to_jsonable(series.median()),
        "max": to_jsonable(series.max()),
        "mean": to_jsonable(series.mean()),
    }

categorical_options = {}
for column in raw_categorical_inputs:
    values = (
        df[column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    categorical_options[column] = values

metadata = {
    "target_col": TARGET_COL,
    "target_unit": "mm/day",
    "model_name": best_model_name,
    "model_path": str(model_path),
    "max_reasonable_test_r2": MAX_REASONABLE_TEST_R2,
    "raw_feature_columns": raw_input_columns,
    "required_feature_columns": prediction_raw_input_columns,
    "optional_context_columns": optional_context_columns,
    "model_feature_columns": X.columns.tolist(),
    "numeric_input_features": raw_numeric_inputs,
    "categorical_input_features": raw_categorical_inputs,
    "numeric_input_stats": numeric_input_stats,
    "categorical_options": categorical_options,
    "dropped_columns": drop_from_model_columns,
    "engineered_columns": model_engineered_columns,
    "all_engineered_columns": engineered_columns,
    "split_strategy": split_strategy,
    "selected_model_policy": (
        "Choose the model closest to the maximum allowed test R2 from below; "
        "reject models above the threshold as overfit/leakage-suspect."
    ),
    "rejected_overfit_models": overfit_rejected_models.to_dict(orient="records"),
    "test_metrics": {
        row["model"]: {
            "MAE": to_jsonable(row["MAE"]),
            "RMSE": to_jsonable(row["RMSE"]),
            "R2": to_jsonable(row["R2"]),
        }
        for row in evaluation_df.to_dict(orient="records")
    },
    "train_test_metrics": {
        f"{row['model']}__{row['split']}": {
            "MAE": to_jsonable(row["MAE"]),
            "RMSE": to_jsonable(row["RMSE"]),
            "R2": to_jsonable(row["R2"]),
        }
        for row in performance_df.to_dict(orient="records")
    },
}

with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print(f"Saved model pipeline to: {model_path.resolve()}")
print(f"Saved model metadata to: {metadata_path.resolve()}")

# %% [markdown]
# ## 7. Local Dashboard Creation
#
# A complete Streamlit dashboard script is provided separately as:
#
# `streamlit_crop_water_dashboard.py`
#
# Run the notebook first so `artifacts/best_crop_water_model.joblib` and
# `artifacts/model_metadata.json` are created, then start the dashboard with:
#
# `streamlit run streamlit_crop_water_dashboard.py`

# %%
dashboard_path = Path("streamlit_crop_water_dashboard.py")

print("Dashboard script:", dashboard_path.resolve())
print("Run command: streamlit run streamlit_crop_water_dashboard.py")
