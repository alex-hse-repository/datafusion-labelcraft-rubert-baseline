import numpy as np

from src.metrics.hierarchical_accuracy import hierarchical_accuracy_with_branch_check


def hierarchical_accuracy(eval_pred, category_tree: dict[int | None, int]):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "hierarchical_accuracy": hierarchical_accuracy_with_branch_check(
            predicted_ids=predictions, true_ids=labels, category_tree=category_tree
        )
    }
