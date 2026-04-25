from src.evaluation.metrics import EvaluationResult, compute_classification_metrics


def evaluate_predictions(
    y_true,
    y_pred,
    y_score=None,
) -> EvaluationResult:
    return compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)


