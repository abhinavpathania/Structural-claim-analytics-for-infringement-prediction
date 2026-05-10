"""Configuration settings for the project."""
from __future__ import annotations
from pydantic import BaseModel, Field
from pathlib import Path


class Paths(BaseModel):
    root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    model_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "models")
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "reports")


class ModelConfig(BaseModel):
    spacy_model: str = "en_core_web_sm"
    st_model: str = "all-MiniLM-L6-v2"
    random_state: int = 42


class TrainConfig(BaseModel):
    test_size: float = 0.2
    n_folds: int = 5
    model_type: str = "regression"  # "regression" | "classification"
    save_name: str = "risk_model.joblib"


class DataConfig(BaseModel):
    min_label: float = 0.0
    max_label: float = 100.0
    n_samples: int = 5000
    seed: int = 42


class AppConfig(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)


CFG = AppConfig()