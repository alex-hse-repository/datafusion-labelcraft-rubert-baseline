import argparse
import os

os.environ["TRANSFORMERS_CACHE"] = "tmp"
os.environ["HF_HOME"] = "tmp"

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.category_tree.category_tree import CategoryTree

MODEL_PATH = "models/rubert_label_smoothing_1_epoch"
CAT_TREE_PATH = "data/category_tree.csv"

CAT_ID_COL = "cat_id"
TITLE_COL = "source_name"

TITLE_MODEL_COL = "text"
CAT_ID_MODEL_COL = "label"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, help="test data path")
    parser.add_argument("--output_path", type=str, help="output file")
    args = parser.parse_args()

    test_data = pd.read_parquet(args.test_data_path)
    category_tree = CategoryTree(CAT_TREE_PATH)

    ############
    # Tokenize dataset
    ############

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def tokenize_function(examples):
        return tokenizer(examples[TITLE_MODEL_COL], truncation=True)

    test_data = test_data.rename(
        columns={TITLE_COL: TITLE_MODEL_COL, CAT_ID_COL: CAT_ID_MODEL_COL}
    )
    dataset = Dataset.from_pandas(test_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    ############
    # Inference
    ############

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=len(category_tree.leaf_nodes)
    )

    training_args = TrainingArguments(
        output_dir="tmp",
        per_device_eval_batch_size=64,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,  # Automatic DataCollatorWithPadding
    )

    predictions = trainer.predict(tokenized_dataset)
    predictions = predictions.predictions.argmax(axis=1)
    predictions = category_tree.label_encoder.inverse_transform(predictions)

    test_data["predicted_cat"] = predictions
    test_data[["hash_id", "predicted_cat"]].to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
