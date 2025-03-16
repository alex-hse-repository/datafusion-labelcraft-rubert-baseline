import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from src.category_tree.category_tree import CategoryTree

#########################

CAT_PATH = "../category_tree.csv"
TRAIN_LABELED_PATH = "../labeled_train.parquet"

EXPERIMENTS_DATASET_PATH = "dataset_for_experiments.parquet"
SUBMIT_DATASET_PATH = "submit_dataset.parquet"

#########################

CAT_ID_COL = "cat_id"
TITLE_COL = "source_name"
PART_TYPE_COL = "part_type"
PART_COL = "part"
RANDOM_STATE = 42
TEST_PART_SIZE = 0.1

#########################

df = pd.read_parquet(TRAIN_LABELED_PATH, columns=[TITLE_COL, CAT_ID_COL])
category_tree = CategoryTree(category_tree_path=CAT_PATH)

# Дропнем не листовы лэйблы
df = df[df[CAT_ID_COL].isin(category_tree.leaf_nodes)]
df_submit = df.copy()

# Дропнем категоии только с ОДНИМ примером
cat_id_samples_cnt = df[CAT_ID_COL].value_counts()
one_sample_cats = cat_id_samples_cnt[cat_id_samples_cnt == 1].index.values
df = df[~df[CAT_ID_COL].isin(one_sample_cats)]

# Разделим на обучение и валидацию
train_idx, test_idx = next(GroupKFold(n_splits=int(1 / TEST_PART_SIZE), shuffle=True, random_state=RANDOM_STATE).split(df, groups=df[CAT_ID_COL]))
df_train, df_oos = df.iloc[train_idx], df.iloc[test_idx]
df_train, df_is = train_test_split(df_train, test_size=TEST_PART_SIZE, stratify=df_train[CAT_ID_COL], random_state=RANDOM_STATE)

df_train[PART_TYPE_COL] = "is"
df_is[PART_TYPE_COL] = "is"
df_oos[PART_TYPE_COL] = "oos"

df_val = pd.concat([df_is, df_oos], axis=0)

df_train[PART_COL] = "train"
df_val[PART_COL] = "val"
df = pd.concat([df_train, df_val], axis=0)

assert len(set(df_is[CAT_ID_COL].unique()) & set(df_train[CAT_ID_COL])) == len(set(df_is[CAT_ID_COL]))
assert len(set(df_oos[CAT_ID_COL].unique()) & set(df_train[CAT_ID_COL])) == 0

# Сохраняем результаты
df_submit.to_parquet(SUBMIT_DATASET_PATH, index=False)
df.to_parquet(EXPERIMENTS_DATASET_PATH, index=False)