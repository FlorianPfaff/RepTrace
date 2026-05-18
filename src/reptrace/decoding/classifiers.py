from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_CLASSIFIER_PARAMS = {
    "lasso": 0.005,
    "multiclass-svm": 0.5,
    "multiclass-svm-weighted": 0.5,
    "svm-binary": 0.5,
    "binary-svm": 0.5,
    "random-forest": 100,
    "gradient-boosting": 100,
    "knn": 5,
    "mostFrequentDummy": None,
    "always1Dummy": None,
    "scikit-mlp": (150, 1000),
    "correlation-prototype": None,
    "multinomial-logistic": 1.0,
    "shrinkage-lda": None,
    "xgboost": 100,
    "pytorch-mlp": {
        "hidden_dim": 720,
        "max_epochs": 500,
        "learning_rate": 1e-3,
        "dropout_rate": 0.2,
        "random_seed": 0,
    },
}


@dataclass(frozen=True)
class ClassifierSpec:
    """Factory metadata for classifiers that may or may not fit in builder."""

    builder: Callable[[np.ndarray, np.ndarray, Any, int | None], Any]
    fits_in_builder: bool = False


class CorrelationPrototypeClassifier:
    """Classify rows by correlation to class-average feature prototypes."""

    def __init__(self):
        self.classes_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None
        self.normalized_prototypes_: np.ndarray | None = None

    def fit(self, features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence | np.ndarray):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels).ravel()
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional feature matrix.")
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels must contain one label per feature row.")
        self.classes_ = np.unique(labels)
        if self.classes_.size == 0:
            raise ValueError("At least one class is required.")
        self.prototypes_ = np.vstack([np.mean(features[labels == class_label], axis=0) for class_label in self.classes_])
        self.normalized_prototypes_ = self._row_center_normalize(self.prototypes_)
        return self

    def decision_function(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if self.normalized_prototypes_ is None:
            raise RuntimeError("CorrelationPrototypeClassifier must be fitted before scoring.")
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional feature matrix.")
        return self._row_center_normalize(features) @ self.normalized_prototypes_.T

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("CorrelationPrototypeClassifier must be fitted before prediction.")
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]

    @staticmethod
    def _row_center_normalize(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError("values must be a two-dimensional matrix.")
        centered = values - np.mean(values, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return centered / norms


class DecodedLabelClassifier:
    """Expose original labels while fitting the wrapped model on dense integer labels."""

    def __init__(self, model: Any, classes: Sequence | np.ndarray):
        self.model = model
        self.classes_ = np.asarray(classes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def __getitem__(self, key: Any) -> Any:
        return self.model[key]

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return self._decode(np.asarray(self.model.predict(features), dtype=int))

    def decision_function(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(features), dtype=float)
            if scores.ndim == 1 and self.classes_.shape[0] == 2:
                return np.column_stack((-scores, scores))
            return scores
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(features), dtype=float)
        if hasattr(self.model, "forward"):
            return self._torch_logits(features)
        predictions = np.asarray(self.model.predict(features), dtype=int)
        scores = np.zeros((predictions.shape[0], self.classes_.shape[0]), dtype=float)
        for row_index, encoded_label in enumerate(predictions):
            if 0 <= int(encoded_label) < self.classes_.shape[0]:
                scores[row_index, int(encoded_label)] = 1.0
        return scores

    def predict_proba(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.model.__class__.__name__!r} object has no attribute 'predict_proba'")
        return np.asarray(self.model.predict_proba(features), dtype=float)

    def _decode(self, encoded_labels: Sequence[int] | np.ndarray) -> np.ndarray:
        encoded_labels = np.asarray(encoded_labels, dtype=int)
        if np.any(encoded_labels < 0) or np.any(encoded_labels >= self.classes_.shape[0]):
            raise ValueError("Classifier returned an encoded label outside the fitted class range.")
        return self.classes_[encoded_labels]

    def _torch_logits(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install RepTrace with the torch extra to score classifier='pytorch-mlp'.") from exc
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32)
            logits = self.model.forward(tensor)
        return logits.detach().cpu().numpy()


def __getattr__(name: str) -> Any:
    if name == "MLPClassifierTorch":
        from reptrace.decoding.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def should_use_default_classifier_param(classifier_param: Any) -> bool:
    """Return true for legacy NaN placeholders that request default params."""

    try:
        return bool(np.all(np.isnan(classifier_param)))
    except TypeError:
        return False


def get_default_classifier_param(classifier: str) -> Any:
    """Return a defensive copy of the configured default classifier parameter."""

    if classifier in DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    raise ValueError(f"Unsupported classifier: {classifier}")


def encode_classifier_labels(labels: Sequence | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode labels as dense integer class ids and return ``(classes, encoded)``."""

    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        raise ValueError("At least one class label is required.")
    classes = np.unique(labels)
    encoded = np.searchsorted(classes, labels).astype(int, copy=False)
    return classes, encoded


def _build_multiclass_svm(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return make_pipeline(StandardScaler(), SVC(C=classifier_param, kernel="linear", random_state=random_state))


def _build_multiclass_svm_weighted(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return make_pipeline(
        StandardScaler(),
        SVC(C=classifier_param, kernel="linear", class_weight="balanced", random_state=random_state),
    )


def _build_random_forest(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return RandomForestClassifier(n_estimators=int(classifier_param), min_samples_leaf=5, random_state=random_state)


def _build_gradient_boosting(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return GradientBoostingClassifier(n_estimators=int(classifier_param), random_state=random_state)


def _build_knn(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, _random_state: int | None):
    return KNeighborsClassifier(n_neighbors=int(classifier_param))


def _build_most_frequent_dummy(_features: np.ndarray, _labels: np.ndarray, _classifier_param: Any, _random_state: int | None):
    return DummyClassifier(strategy="most_frequent")


def _build_always_one_dummy(_features: np.ndarray, _labels: np.ndarray, _classifier_param: Any, _random_state: int | None):
    return DummyClassifier(strategy="constant", constant=1)


def _build_scikit_mlp(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return MLPClassifier(hidden_layer_sizes=int(classifier_param[0]), max_iter=int(classifier_param[1]), random_state=random_state)


def _build_correlation_prototype(_features: np.ndarray, _labels: np.ndarray, _classifier_param: Any, _random_state: int | None):
    return CorrelationPrototypeClassifier()


def _build_multinomial_logistic(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return LogisticRegression(C=float(classifier_param), max_iter=1000, random_state=random_state)


def _normalize_lda_shrinkage(classifier_param: Any):
    if classifier_param is None:
        return "auto"
    if isinstance(classifier_param, str) and classifier_param.strip().lower() == "auto":
        return "auto"
    shrinkage = float(classifier_param)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage-lda classifier_param must be 'auto' or a numeric shrinkage in [0, 1].")
    return shrinkage


def _build_shrinkage_lda(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, _random_state: int | None):
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=_normalize_lda_shrinkage(classifier_param))


def _build_xgboost(_features: np.ndarray, _labels: np.ndarray, classifier_param: Any, random_state: int | None):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install RepTrace with the xgboost extra to use classifier='xgboost'.") from exc
    return xgb.XGBClassifier(n_estimators=int(classifier_param), eval_metric="mlogloss", random_state=random_state)


def _build_pytorch_mlp_classifier(features: np.ndarray, labels: np.ndarray, classifier_param: dict[str, Any], random_state: int | None):
    return _train_pytorch_mlp(features, labels, classifier_param, random_state=random_state)


CLASSIFIER_REGISTRY = {
    "multiclass-svm": ClassifierSpec(_build_multiclass_svm),
    "multiclass-svm-weighted": ClassifierSpec(_build_multiclass_svm_weighted),
    "random-forest": ClassifierSpec(_build_random_forest),
    "gradient-boosting": ClassifierSpec(_build_gradient_boosting),
    "knn": ClassifierSpec(_build_knn),
    "mostFrequentDummy": ClassifierSpec(_build_most_frequent_dummy),
    "always1Dummy": ClassifierSpec(_build_always_one_dummy),
    "scikit-mlp": ClassifierSpec(_build_scikit_mlp),
    "correlation-prototype": ClassifierSpec(_build_correlation_prototype),
    "multinomial-logistic": ClassifierSpec(_build_multinomial_logistic),
    "shrinkage-lda": ClassifierSpec(_build_shrinkage_lda),
    "xgboost": ClassifierSpec(_build_xgboost),
    "pytorch-mlp": ClassifierSpec(_build_pytorch_mlp_classifier, fits_in_builder=True),
}


def train_classifier(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
):
    """Build and fit a classifier from a registry entry."""

    registry = CLASSIFIER_REGISTRY if registry is None else registry
    features = np.asarray(features)
    labels = np.asarray(labels).ravel()
    try:
        classifier_spec = registry[classifier]
    except KeyError as exc:
        supported_classifiers = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported classifier: {classifier}. Supported classifiers: {supported_classifiers}") from exc
    model = classifier_spec.builder(features, labels, classifier_param, random_state)
    if classifier_spec.fits_in_builder:
        return model
    model.fit(features, labels)
    return model


def train_multiclass_classifier(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    classifier: str,
    classifier_param: Any,
    random_state: int | None = None,
    *,
    registry: dict[str, ClassifierSpec] | None = None,
):
    """Train a classifier on dense labels while exposing the original labels."""

    classes, encoded_labels = encode_classifier_labels(labels)
    model = train_classifier(features, encoded_labels, classifier, classifier_param, random_state=random_state, registry=registry)
    return DecodedLabelClassifier(model, classes)


def train_gradient_boosting(train_features, train_labels, classifier_param: Any, random_state: int | None = None):
    """Train the legacy binary gradient boosting helper used by one-vs-rest decoding."""

    model = GradientBoostingClassifier(n_estimators=int(classifier_param), max_leaf_nodes=21, learning_rate=0.1, random_state=random_state)
    model.fit(train_features, train_labels)
    return model


def train_lasso_logistic(train_features, train_labels, lambda_: float, random_state: int | None = None):
    """Train an L1-regularized logistic model for binary one-vs-rest decoding."""

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1", C=1 / lambda_, solver="liblinear", max_iter=1000, random_state=random_state),
    )
    model.fit(train_features, train_labels)
    return model


def train_for_stimulus_lasso_glm(train_features, train_labels, lambda_: float, random_state: int | None = None):
    """Backward-compatible alias for the legacy one-vs-rest lasso helper."""

    return train_lasso_logistic(train_features, train_labels, lambda_, random_state=random_state)


def train_binary_svm(train_features, train_labels, box_constraint: float, random_state: int | None = None):
    """Train a linear binary SVM helper for one-vs-rest decoding."""

    model = make_pipeline(StandardScaler(), SVC(C=box_constraint, kernel="linear", random_state=random_state))
    model.fit(train_features, train_labels)
    return model


def _train_pytorch_mlp(features: np.ndarray, labels: np.ndarray, classifier_param: dict[str, Any], random_state: int | None = None):
    random_seed = _resolve_pytorch_random_seed(classifier_param, random_state)
    if random_seed is not None:
        _seed_pytorch_training(random_seed)
    model = _build_pytorch_mlp(features, labels, classifier_param)
    train_loader, val_loader = _build_pytorch_data_loaders(features, labels, random_seed=random_seed)
    trainer = _build_pytorch_trainer(classifier_param, random_seed=random_seed)
    trainer.fit(model, train_loader, val_loader)
    return model


def _resolve_pytorch_random_seed(classifier_param: dict[str, Any], random_state: int | None) -> int | None:
    random_seed = random_state
    if random_seed is None:
        random_seed = classifier_param.get("random_seed")
    return None if random_seed is None else int(random_seed)


def _seed_pytorch_training(random_seed: int) -> None:
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ImportError("Install RepTrace with the torch extra to use classifier='pytorch-mlp'.") from exc
    pl.seed_everything(random_seed, workers=True)


def _build_pytorch_mlp(features: np.ndarray, labels: np.ndarray, classifier_param: dict[str, Any]):
    try:
        from reptrace.decoding.torch_models import MLPClassifierTorch
    except ImportError as exc:
        raise ImportError("Install RepTrace with the torch extra to use classifier='pytorch-mlp'.") from exc
    return MLPClassifierTorch(
        features.shape[1],
        int(classifier_param["hidden_dim"]),
        len(np.unique(labels)),
        learning_rate=classifier_param["learning_rate"],
        dropout_rate=classifier_param["dropout_rate"],
    )


def _build_pytorch_data_loaders(features, labels, *, random_seed: int | None = None):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install RepTrace with the torch extra to use classifier='pytorch-mlp'.") from exc
    train_dataset, val_dataset = _split_pytorch_dataset(torch, features, labels, random_seed)
    train_generator = _build_torch_generator(torch, random_seed)
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, generator=train_generator),
        torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False),
    )


def _split_pytorch_dataset(torch: Any, features, labels, random_seed: int | None):
    full_dataset = torch.utils.data.TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    return torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=_build_torch_generator(torch, random_seed))


def _build_torch_generator(torch: Any, random_seed: int | None):
    if random_seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    return generator


def _build_pytorch_trainer(classifier_param: dict[str, Any], *, random_seed: int | None = None):
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ImportError("Install RepTrace with the torch extra to use classifier='pytorch-mlp'.") from exc
    return pl.Trainer(
        max_epochs=int(classifier_param["max_epochs"]),
        default_root_dir=r"lightning_logs",
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
        deterministic=random_seed is not None,
    )


def prediction_scores(model: Any, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return one confidence-like score per row from common classifier APIs."""

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional feature matrix.")
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            return np.abs(scores)
        return np.max(scores, axis=1)
    if hasattr(model, "predict_proba"):
        return np.max(np.asarray(model.predict_proba(features), dtype=float), axis=1)
    return np.full(features.shape[0], np.nan, dtype=float)


def positive_class_score(model: Any, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a binary model's score for the positive class."""

    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional feature matrix.")
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features), dtype=float)[:, 1]
    return np.asarray(model.predict(features), dtype=float)
