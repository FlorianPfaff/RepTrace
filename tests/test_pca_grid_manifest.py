from pathlib import Path

from reptrace.pca_grid_manifest import expand_pca_grid_manifest


def test_expand_pca_grid_manifest_writes_baseline_and_pca_rows(tmp_path: Path):
    base_manifest = tmp_path / "base.csv"
    base_manifest.write_text(
        "subject,epochs,metadata_csv,label_column,group_column,tmin,tmax,window_ms,step_ms,n_splits\n"
        "sub-01,data/sub-01_epo.fif,data/sub-01_metadata.csv,condition,session,-0.1,0.8,20,10,2\n",
        encoding="utf-8",
    )
    out = tmp_path / "expanded.csv"

    expanded = expand_pca_grid_manifest(
        base_manifest,
        out,
        decoders=("logistic", "lda"),
        feature_preprocessors=("none", "pca_whiten"),
        pca_components=("0.8", "16"),
        tune_hyperparameters=True,
        tuning_cv_splits=2,
        tuning_scoring="balanced_accuracy",
        tuning_c_grid="0.1,1,10",
    )

    assert out.exists()
    assert len(expanded) == 6
    assert expanded["variant"].tolist() == [
        "logistic_none_tuned",
        "logistic_pca_whiten_pca0p8_tuned",
        "logistic_pca_whiten_pca16_tuned",
        "lda_none_tuned",
        "lda_pca_whiten_pca0p8_tuned",
        "lda_pca_whiten_pca16_tuned",
    ]
    assert expanded.loc[expanded["feature_preprocessor"] == "none", "pca_components"].tolist() == ["", ""]
    assert expanded.loc[expanded["feature_preprocessor"] == "pca_whiten", "pca_components"].tolist() == ["0.8", "16", "0.8", "16"]
    assert expanded["tune_hyperparameters"].unique().tolist() == ["true"]
    assert expanded["tuning_c_grid"].unique().tolist() == ["0.1,1,10"]
