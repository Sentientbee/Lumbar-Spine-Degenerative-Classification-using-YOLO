"""
Pipeline subpackage for Spine Degeneration Classifier.
Provides CLI and programmatic APIs for training, evaluation, and study inference.
"""

def __getattr__(name: str):
    if name == "predict_study":
        from src.pipeline.predict_study import predict_study
        return predict_study
    elif name == "train_severity_classifier":
        from src.pipeline.train_severity_classifier import train_severity_classifier
        return train_severity_classifier
    elif name == "evaluate_classifier":
        from src.pipeline.evaluate_classifier import evaluate_classifier
        return evaluate_classifier
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["predict_study", "train_severity_classifier", "evaluate_classifier"]
