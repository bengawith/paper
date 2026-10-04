from robust_airfoil.modelling.train import train_model


def test_training_smoke_writes_reloadable_artifacts(tmp_path, model_frame):
    train = model_frame[model_frame["airfoil_id"].isin(["a", "b", "c"])]
    validation = model_frame[model_frame["airfoil_id"] == "d"]
    summary = train_model(train, validation, tmp_path, hidden_width=32, hidden_layers=1, dropout=0, batch_size=64, maximum_epochs=3, patience=3)
    assert summary["epochs"] == 3
    assert (tmp_path / "checkpoint.pt").is_file()
    assert (tmp_path / "scaling.joblib").is_file()
