from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from reptrace.decoding.classifiers import CorrelationPrototypeClassifier, get_default_classifier_param
from reptrace.metrics import brier_score_multiclass, expected_calibration_error
from reptrace.decoding.sampling import (
    CLASS_LIMIT_SELECTION_MODES as CLASS_LIMIT_SELECTION_MODES,
    DEFAULT_CLASS_LIMIT_SEED as DEFAULT_CLASS_LIMIT_SEED,
    DEFAULT_CLASS_LIMIT_SELECTION as DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed as normalize_class_limit_seed,
    normalize_class_limit_selection as normalize_class_limit_selection,
    select_class_limited_indices as select_class_limited_indices,
)

STANDARD_DECODER_CHOICES = ("logistic", "lda", "shrinkage_lda", "linear_svm")
REGISTRY_DECODER_CHOICES = (
    "correlation_prototype",
    "multinomial_logistic",
    "multiclass_svm",
    "multiclass_svm_weighted",
    "random_forest",
    "gradient_boosting",
    "knn",
    "scikit_mlp",
)
DECODER_CHOICES = (*STANDARD_DECODER_CHOICES, *REGISTRY_DECODER_CHOICES)
DECODER_CLI_CHOICES = tuple(
    dict.fromkeys(
        (
            *DECODER_CHOICES,
            "svm",
            "lda-shrinkage",
            *(decoder.replace("_", "-") for decoder in DECODER_CHOICES if "_" in decoder),
        )
    )
)
REGISTRY_DECODER_CLASSIFIERS = {
    decoder: decoder.replace("_", "-") for decoder in REGISTRY_DECODER_CHOICES
}
DECODERS_REQUIRING_CALIBRATION = ("multiclass_svm", "multiclass_svm_weighted")
DEFAULT_DECODER_RANDOM_STATE = 13
EMISSION_MODE_CHOICES = ("calibrated", "uncalibrated")
FEATURE_PREPROCESSOR_CHOICES = ("none", "pca", "pca_whiten")
TUNING_SCORING_CHOICES = ("accuracy", "balanced_accuracy", "neg_log_loss", "neg_brier", "neg_ece")
DEFAULT_TUNING_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_TUNING_PCA_COMPONENTS_GRID = (None, 0.8, 0.95)


def make_logistic_decoder(
    max_iter: int = 1000,
    *,
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
):
    """Create the default calibrated-probability baseline decoder."""
    return make_decoder(
        "logistic",
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
    )


def make_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    tuning_pca_components_grid: Sequence[int | float | str | None] | str | None = None,
    calibration_cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
):
    """Create a standard probability-producing decoder by name.

    Optional feature preprocessing is inserted after fold-local standardization
    and before the classifier. This keeps low-rank transforms such as PCA inside
    each cross-validation fold and prevents train/test leakage.

    When ``tune_hyperparameters`` is enabled, the returned estimator is a
    ``GridSearchCV`` wrapper around the same decoder family. The caller can pass
    an integer CV count or precomputed inner-CV splits via ``tuning_cv``.
    If PCA preprocessing is enabled, ``tuning_pca_components_grid`` additionally
    searches ``pca__n_components`` in the same inner CV, keeping dimensionality
    selection inside the outer training fold.

    ``calibration_cv`` controls calibrated linear-SVM emissions and may be an
    integer or precomputed train/test splits.
    """
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)

    if tune_hyperparameters:
        return make_tuned_decoder(
            normalized,
            max_iter=max_iter,
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            cv=tuning_cv,
            scoring=tuning_scoring,
            c_grid=tuning_c_grid,
            pca_components_grid=tuning_pca_components_grid,
            calibration_cv=calibration_cv,
        )

    if normalized == "logistic":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                solver="lbfgs",
            ),
        )
    if normalized == "lda":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="svd"),
        )
    if normalized == "shrinkage_lda":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )

    if normalized in REGISTRY_DECODER_CHOICES:
        registry_decoder = _make_registry_decoder(
            normalized,
            max_iter=max_iter,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
        )
        if emission_mode == "calibrated" and normalized in DECODERS_REQUIRING_CALIBRATION:
            return _make_calibrated_classifier(registry_decoder, method="sigmoid", cv=3)
        return registry_decoder

    linear_svm = make_pipeline(
        StandardScaler(),
        *feature_steps,
        LinearSVC(
            class_weight="balanced",
            max_iter=max_iter,
        ),
    )
    if emission_mode == "uncalibrated":
        return linear_svm
    return _make_calibrated_classifier(
        linear_svm,
        method="sigmoid",
        cv=calibration_cv,
    )


def make_tuned_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
    scoring: str = "accuracy",
    c_grid: Sequence[float] | str | None = None,
    pca_components_grid: Sequence[int | float | str | None] | str | None = None,
    calibration_cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
):
    """Create a decoder with inner-CV hyperparameter selection.

    Logistic regression and linear SVM tune the regularization strength ``C``.
    LDA compares the default SVD solver with shrinkage LDA
    (``solver='lsqr', shrinkage='auto'``), which is often better conditioned for
    high-dimensional M/EEG windows.
    When PCA preprocessing is active, PCA component count or explained-variance
    fraction is tuned in the same inner-CV loop via ``pca_components_grid``.

    For calibrated linear SVMs, ``calibration_cv`` is forwarded to
    ``CalibratedClassifierCV`` so grouped callers can keep calibration folds
    group-disjoint.
    """
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    scoring = normalize_tuning_scoring(scoring)
    c_grid = parse_c_grid(c_grid)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)

    if normalized == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                solver="lbfgs",
            ),
        )
        param_grid = {"logisticregression__C": c_grid}
    elif normalized == "lda":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(),
        )
        param_grid = [
            {
                "lineardiscriminantanalysis__solver": ["svd"],
                "lineardiscriminantanalysis__shrinkage": [None],
            },
            {
                "lineardiscriminantanalysis__solver": ["lsqr"],
                "lineardiscriminantanalysis__shrinkage": ["auto"],
            },
        ]
    elif normalized == "shrinkage_lda":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="lsqr"),
        )
        param_grid = {"lineardiscriminantanalysis__shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7, 0.9]}
    elif normalized in REGISTRY_DECODER_CHOICES:
        estimator = _make_registry_decoder(
            normalized,
            max_iter=max_iter,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
        )
        param_grid = _registry_decoder_param_grid(normalized, c_grid)
        if emission_mode == "calibrated" and normalized in DECODERS_REQUIRING_CALIBRATION:
            estimator = _make_calibrated_classifier(estimator, method="sigmoid", cv=3)
            param_grid = _calibrated_param_grid(estimator, param_grid)
    elif normalized == "linear_svm":
        linear_svm = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearSVC(
                class_weight="balanced",
                max_iter=max_iter,
            ),
        )
        if emission_mode == "uncalibrated":
            estimator = linear_svm
            param_grid = {"linearsvc__C": c_grid}
        else:
            estimator = _make_calibrated_classifier(linear_svm, method="sigmoid", cv=calibration_cv)
            param_grid = {_calibrated_estimator_param(estimator, "linearsvc__C"): c_grid}
    else:
        raise ValueError(f"Unsupported tuned decoder: {name}")

    param_grid = _with_tuning_pca_components_grid(
        estimator,
        param_grid,
        feature_preprocessor=feature_preprocessor,
        pca_components_grid=pca_components_grid,
    )

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=make_tuning_scorer(scoring, emission_mode=emission_mode),
        cv=cv,
        refit=True,
    )


def _make_registry_decoder(
    name: str,
    *,
    max_iter: int,
    feature_preprocessor: str,
    pca_components: int | float | str | None,
):
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)
    return make_pipeline(
        StandardScaler(),
        *feature_steps,
        _make_registry_classifier(name, max_iter=max_iter),
    )


def _make_registry_classifier(name: str, *, max_iter: int):
    classifier = REGISTRY_DECODER_CLASSIFIERS[name]
    classifier_param = get_default_classifier_param(classifier)
    if name == "correlation_prototype":
        return CorrelationPrototypeClassifier()
    if name == "multinomial_logistic":
        return LogisticRegression(
            C=float(classifier_param),
            class_weight="balanced",
            max_iter=max_iter,
            random_state=DEFAULT_DECODER_RANDOM_STATE,
            solver="lbfgs",
        )
    if name == "multiclass_svm":
        return SVC(C=float(classifier_param), kernel="linear", random_state=DEFAULT_DECODER_RANDOM_STATE)
    if name == "multiclass_svm_weighted":
        return SVC(
            C=float(classifier_param),
            class_weight="balanced",
            kernel="linear",
            random_state=DEFAULT_DECODER_RANDOM_STATE,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(classifier_param),
            min_samples_leaf=5,
            random_state=DEFAULT_DECODER_RANDOM_STATE,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=int(classifier_param), random_state=DEFAULT_DECODER_RANDOM_STATE)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=int(classifier_param))
    if name == "scikit_mlp":
        hidden_units, default_max_iter = classifier_param
        return MLPClassifier(
            hidden_layer_sizes=int(hidden_units),
            max_iter=max(max_iter, int(default_max_iter)),
            random_state=DEFAULT_DECODER_RANDOM_STATE,
        )
    raise ValueError(f"Unsupported registry decoder: {name}")


def _registry_decoder_param_grid(name: str, c_grid: Sequence[float]) -> dict[str, Sequence[object]]:
    if name == "multinomial_logistic":
        return {"logisticregression__C": c_grid}
    if name in {"multiclass_svm", "multiclass_svm_weighted"}:
        return {"svc__C": c_grid}
    if name == "random_forest":
        return {"randomforestclassifier__min_samples_leaf": [1, 3, 5, 10]}
    if name == "gradient_boosting":
        return {"gradientboostingclassifier__learning_rate": [0.03, 0.1], "gradientboostingclassifier__n_estimators": [50, 100]}
    if name == "knn":
        return {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7]}
    if name == "scikit_mlp":
        return {"mlpclassifier__alpha": [0.0001, 0.001, 0.01]}
    return {}


def _calibrated_param_grid(estimator, param_grid: dict[str, Sequence[object]]) -> dict[str, Sequence[object]]:
    return {_calibrated_estimator_param(estimator, parameter): values for parameter, values in param_grid.items()}


def _make_calibrated_classifier(
    estimator,
    *,
    method: str,
    cv: int | Sequence[tuple[np.ndarray, np.ndarray]],
):
    """Construct CalibratedClassifierCV across sklearn estimator/base_estimator APIs."""
    kwargs = {"method": method, "cv": cv}
    if "estimator" in inspect.signature(CalibratedClassifierCV).parameters:
        kwargs["estimator"] = estimator
    else:
        kwargs["base_estimator"] = estimator
    return CalibratedClassifierCV(**kwargs)


def _calibrated_estimator_param(estimator, nested_parameter: str) -> str:
    return _estimator_param(estimator, nested_parameter)


def _estimator_param(estimator, nested_parameter: str) -> str:
    """Return direct or calibrated-wrapper parameter name for a nested estimator parameter."""

    params = estimator.get_params()
    if nested_parameter in params:
        return nested_parameter
    for prefix in ("estimator", "base_estimator"):
        candidate = f"{prefix}__{nested_parameter}"
        if candidate in params:
            return candidate
    raise ValueError(f"Could not find calibrated-estimator parameter for '{nested_parameter}'.")


def _with_tuning_pca_components_grid(
    estimator,
    param_grid: dict[str, object] | list[dict[str, object]],
    *,
    feature_preprocessor: str | None,
    pca_components_grid: Sequence[int | float | str | None] | str | None,
):
    normalized = normalize_feature_preprocessor(feature_preprocessor)
    if normalized == "none":
        if pca_components_grid is not None:
            raise ValueError(
                "tuning_pca_components_grid can only be set when feature_preprocessor is 'pca' or 'pca_whiten'."
            )
        return param_grid

    parsed_grid = parse_pca_components_grid(pca_components_grid)
    parameter = _estimator_param(estimator, "pca__n_components")
    if isinstance(param_grid, list):
        return [{**entry, parameter: list(parsed_grid)} for entry in param_grid]
    return {**param_grid, parameter: list(parsed_grid)}


def parse_c_grid(values: Sequence[float] | str | None) -> tuple[float, ...]:
    """Normalize a regularization-strength grid for CLI and API callers."""
    if values is None:
        return DEFAULT_TUNING_C_GRID
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("At least one C value is required for hyperparameter tuning.")
    if any(value <= 0 for value in grid):
        raise ValueError("All C values must be positive.")
    return grid


def parse_pca_components_grid(
    values: Sequence[int | float | str | None] | str | None,
) -> tuple[int | float | None, ...]:
    """Normalize PCA component candidates for nested hyperparameter tuning."""
    if values is None:
        return DEFAULT_TUNING_PCA_COMPONENTS_GRID
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    grid = tuple(normalize_pca_components(value) for value in values)
    if not grid:
        raise ValueError("At least one PCA component candidate is required for PCA tuning.")
    return grid


def normalize_tuning_scoring(scoring: str) -> str:
    """Normalize inner-CV scoring names."""
    normalized = scoring.lower().replace("-", "_")
    if normalized not in TUNING_SCORING_CHOICES:
        raise ValueError(f"Unknown tuning scoring '{scoring}'. Available values: {', '.join(TUNING_SCORING_CHOICES)}.")
    return normalized


def make_tuning_scorer(scoring: str, *, emission_mode: str = "calibrated") -> str | Callable:
    """Return a GridSearchCV scorer for decoder hyperparameter tuning.

    Accuracy-oriented objectives are forwarded to scikit-learn by name. Probability
    objectives are implemented here so they use the same calibrated or
    score-derived emissions that RepTrace writes to the held-out observation
    tables. This keeps model selection aligned with downstream temporal-state
    inference, where probability quality matters more than the hard class label.
    """
    normalized = normalize_tuning_scoring(scoring)
    emission_mode = normalize_emission_mode(emission_mode)
    if normalized in {"accuracy", "balanced_accuracy"}:
        return normalized
    return _make_probability_tuning_scorer(normalized, emission_mode=emission_mode)


def _make_probability_tuning_scorer(scoring: str, *, emission_mode: str) -> Callable:
    def scorer(estimator, features: np.ndarray, labels: np.ndarray) -> float:
        probabilities = predict_emission_probabilities(estimator, features, emission_mode=emission_mode)
        label_indices = _labels_to_probability_columns(labels, estimator=estimator, n_classes=probabilities.shape[1])
        if scoring == "neg_log_loss":
            return -float(log_loss(label_indices, probabilities, labels=np.arange(probabilities.shape[1])))
        if scoring == "neg_brier":
            return -brier_score_multiclass(probabilities, label_indices)
        if scoring == "neg_ece":
            return -expected_calibration_error(probabilities, label_indices)
        raise ValueError(f"Unknown probability tuning scoring '{scoring}'.")

    return scorer


def _labels_to_probability_columns(
    labels: np.ndarray,
    *,
    estimator,
    n_classes: int,
) -> np.ndarray:
    """Map estimator labels to probability-column indices for multiclass metrics."""
    labels = np.asarray(labels)
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes)
        if len(classes) != n_classes:
            raise ValueError(
                f"Estimator reports {len(classes)} classes but predicted {n_classes} probability columns."
            )
        class_to_index = {class_label: class_index for class_index, class_label in enumerate(classes.tolist())}
        try:
            return np.asarray([class_to_index[label] for label in labels.tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Validation label {exc.args[0]!r} was not seen by the fitted estimator.") from exc

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Probability tuning metrics require fitted estimator classes for non-integer labels.")
    label_indices = labels.astype(int, copy=False)
    if np.any((label_indices < 0) | (label_indices >= n_classes)):
        raise ValueError("Integer labels must be valid probability-column indices.")
    return label_indices


def normalize_decoder_name(name: str) -> str:
    """Normalize decoder aliases to the names used in result tables."""
    normalized = name.lower().replace("-", "_")
    if normalized == "svm":
        return "linear_svm"
    if normalized in {"lda_shrinkage", "shrinkage_lda", "shrinkagelda"}:
        return "shrinkage_lda"
    if normalized not in DECODER_CHOICES:
        raise ValueError(f"Unknown decoder '{name}'. Available decoders: {', '.join(DECODER_CHOICES)}.")
    return normalized


def normalize_emission_mode(mode: str) -> str:
    """Normalize calibrated/uncalibrated emission mode names."""
    normalized = mode.lower().replace("-", "_")
    if normalized not in EMISSION_MODE_CHOICES:
        raise ValueError(f"Unknown emission mode '{mode}'. Available modes: {', '.join(EMISSION_MODE_CHOICES)}.")
    return normalized


def normalize_feature_preprocessor(name: str | None) -> str:
    """Normalize feature-preprocessor aliases to canonical result-table names."""
    normalized = "none" if name is None else name.lower().replace("-", "_")
    if normalized in {"identity", "standard", "standardize", "scaler", "standard_scaler"}:
        return "none"
    if normalized in {"pca_whitened", "whitened_pca", "whiten_pca"}:
        return "pca_whiten"
    if normalized not in FEATURE_PREPROCESSOR_CHOICES:
        raise ValueError(
            f"Unknown feature preprocessor '{name}'. Available preprocessors: {', '.join(FEATURE_PREPROCESSOR_CHOICES)}."
        )
    return normalized


def normalize_pca_components(n_components: int | float | str | None) -> int | float | None:
    """Normalize PCA component specifications for sklearn.

    Integers select an explicit component count. Floats in ``(0, 1)`` select an
    explained-variance fraction. ``None``, ``auto``, or an empty string keep
    sklearn's default ``PCA(n_components=None)`` behavior.
    """
    if n_components is None:
        return None
    if isinstance(n_components, str):
        stripped = n_components.strip()
        if stripped == "" or stripped.lower() in {"none", "auto", "default"}:
            return None
        try:
            parsed: int | float = float(stripped) if any(marker in stripped for marker in (".", "e", "E")) else int(stripped)
        except ValueError as exc:
            raise ValueError("pca_components must be an integer count, a variance fraction in (0, 1), or None.") from exc
        return normalize_pca_components(parsed)
    if isinstance(n_components, (np.integer,)):
        n_components = int(n_components)
    if isinstance(n_components, (np.floating,)):
        n_components = float(n_components)
    if isinstance(n_components, bool):
        raise ValueError("pca_components must be numeric, not boolean.")
    if isinstance(n_components, int):
        if n_components < 1:
            raise ValueError("Integer pca_components must be at least 1.")
        return n_components
    if isinstance(n_components, float):
        if not np.isfinite(n_components) or n_components <= 0.0:
            raise ValueError("Float pca_components must be finite and positive.")
        if n_components < 1.0:
            return float(n_components)
        if n_components.is_integer():
            return int(n_components)
    raise ValueError("pca_components must be an integer count, a variance fraction in (0, 1), or None.")


def _feature_preprocessor_steps(feature_preprocessor: str | None, pca_components: int | float | str | None) -> list[PCA]:
    normalized = normalize_feature_preprocessor(feature_preprocessor)
    if normalized == "none":
        if pca_components is not None:
            raise ValueError("pca_components can only be set when feature_preprocessor is 'pca' or 'pca_whiten'.")
        return []
    return [
        PCA(
            n_components=normalize_pca_components(pca_components),
            whiten=normalized == "pca_whiten",
            svd_solver="full",
        )
    ]


def score_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert uncalibrated decision scores into pseudo-probability emissions."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        clipped = np.clip(scores, -50.0, 50.0)
        positive = 1.0 / (1.0 + np.exp(-clipped))
        return np.column_stack([1.0 - positive, positive])
    if scores.ndim != 2:
        raise ValueError("Decision scores must be one- or two-dimensional.")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def predict_emission_probabilities(model, features: np.ndarray, *, emission_mode: str = "calibrated") -> np.ndarray:
    """Predict calibrated probabilities or uncalibrated score-derived emissions."""
    emission_mode = normalize_emission_mode(emission_mode)
    if emission_mode == "uncalibrated" and hasattr(model, "decision_function"):
        return score_to_probabilities(model.decision_function(features))
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features), dtype=float)
    if hasattr(model, "decision_function"):
        return score_to_probabilities(model.decision_function(features))
    raise ValueError("Decoder does not provide predict_proba or decision_function.")


def make_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
    """Create stratified CV splits, optionally preserving group boundaries."""
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for decoding.")
    if np.min(class_counts) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} examples per class; smallest class has {np.min(class_counts)}."
        )
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < n_splits:
            raise ValueError(
                f"Need at least {n_splits} groups for grouped CV, found {len(unique_groups)}."
            )
        return StratifiedGroupKFold(n_splits=n_splits).split(
            np.zeros_like(labels),
            labels,
            groups,
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13).split(
        np.zeros_like(labels),
        labels,
    )


def make_tuning_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
    """Create feasible inner-CV splits for nested decoder hyperparameter tuning."""
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for decoder hyperparameter tuning.")
    feasible_splits = min(int(n_splits), int(np.min(class_counts)))
    if groups is not None:
        feasible_splits = min(feasible_splits, len(np.unique(groups)))
    if feasible_splits < 2:
        raise ValueError("Need at least two examples per class and two groups when grouped to tune decoder hyperparameters.")
    return list(make_cross_validator(labels, groups, feasible_splits))


def make_calibration_cross_validator(labels: np.ndarray, groups: np.ndarray | None, n_splits: int):
    """Create feasible CV splits for probability calibration.

    Group labels, when present, are kept disjoint between calibration-train and
    calibration-test folds to avoid calibrating on held-out groups.
    """
    return make_tuning_cross_validator(labels, groups, n_splits)


def time_windows(times: np.ndarray, window_ms: float, step_ms: float) -> list[tuple[int, int, float]]:
    """Return sample index windows and their center times for time-resolved decoding."""
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if len(times) < 2:
        raise ValueError("times must contain at least two samples")
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive")

    sfreq = 1000.0 / np.median(np.diff(times * 1000.0))
    window_samples = max(1, int(round((window_ms / 1000.0) * sfreq)))
    step_samples = max(1, int(round((step_ms / 1000.0) * sfreq)))
    windows = []
    for start in range(0, len(times) - window_samples + 1, step_samples):
        stop = start + window_samples
        center = float(np.mean(times[start:stop]))
        windows.append((start, stop, center))
    return windows
