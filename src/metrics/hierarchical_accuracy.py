import math

import numpy as np


def build_category_tree_path(category_tree):
    """
    Строит дерево категорий, добавляя для каждой категории её цепочку предков.
    :param category_tree: Словарь вида {cat_id: parent_id}, где у корневых категорий parent_id = None.
    :return: Словарь {cat_id: {"level": уровень, "ancestors": [предки в порядке от корня до родителя]}}.
    """
    category_info = {}

    def get_path_and_level(cat_id):
        if cat_id in category_info:
            return category_info[cat_id]["level"], category_info[cat_id]["ancestors"]
        parent_id = category_tree.get(cat_id)
        if np.isnan(parent_id):
            level = 1
            ancestors = []
        else:
            parent_level, parent_ancestors = get_path_and_level(parent_id)
            level = parent_level + 1
            ancestors = parent_ancestors + [parent_id]
        category_info[cat_id] = {"level": level, "ancestors": ancestors}
        return level, ancestors

    for cat_id in category_tree:
        get_path_and_level(cat_id)

    return category_info


def find_lowest_common_ancestor(true_id, pred_id, category_info):
    """
    Находит наибольшего общего предка (Lowest Common Ancestor - LCA) между предсказанной
    и истинной категорией.

    :param true_id: Истинная категория.
    :param pred_id: Предсказанная категория.
    :param category_info: Словарь с уровнями категорий.
    :return: (LCA, уровень LCA).
    """
    true_info = category_info.get(true_id, {"level": 0, "ancestors": []})
    pred_info = category_info.get(pred_id, {"level": 0, "ancestors": []})

    # Совпадает — нет необходимости искать предка
    if true_id == pred_id:
        return true_id, true_info["level"]

    # Собираем множества предков
    true_ancestors = set(true_info["ancestors"] + [true_id])
    pred_ancestors = set(pred_info["ancestors"] + [pred_id])

    # Ищем наибольшего общего предка
    common_ancestors = true_ancestors.intersection(pred_ancestors)
    if not common_ancestors:
        return None, 0  # Категории не связаны — полный штраф

    # Выбираем самого глубокого предка
    lca = max(common_ancestors, key=lambda cat: category_info[cat]["level"])

    return lca, category_info[lca]["level"]


def hierarchical_accuracy_with_branch_check(predicted_ids, true_ids, category_tree):
    """
    Рассчитывает метрику, учитывая иерархию категорий и наибольшего общего предка (LCA).

    :param predicted_ids: Список предсказанных категорий.
    :param true_ids: Список правильных категорий.
    :param category_tree: Словарь {cat_id: parent_id}, описывающий иерархию категорий.
    :return: Средняя метрика по всем примерам.
    """
    assert len(true_ids) == len(predicted_ids), "Длина списков не совпадает"  # noqa: S101

    # Словарь {cat_id: {"level": level, "ancestors": ancestors}}
    category_info = build_category_tree_path(category_tree)

    total_score = 0

    for true_id, pred_id in zip(true_ids, predicted_ids, strict=False):
        # Находим LCA для истинного и предсказанного значения
        lca, lca_level = find_lowest_common_ancestor(true_id, pred_id, category_info)

        if lca is None:
            score = 0  # Если совпадений нет, штрафуем по максимуму
        else:
            true_level = category_info.get(true_id, {"level": 0})["level"]
            level_difference = max(0, true_level - lca_level)  # LCA сравниваем с истиной

            # Дисконтируем на разницу уровней
            score = 1 / math.exp(level_difference)

        total_score += score

    return total_score / len(true_ids)
