import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline


def parse_args():
  parser = argparse.ArgumentParser(description="Train or test a T5 summarization model")
  parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                      help="Train or test the model. Test essnetially summarizes a single entry and print to console for inspection")
  parser.add_argument("--train_data", type=str, default="./train.csv",
                      help="Path to the training CSV file")
  parser.add_argument("--model_name", type=str, default="t5-small",
                      help="HF Model name")

  args = parser.parse_args()
  return args


def preprocess_function(examples):
  inputs = ["summarize: " + doc for doc in examples["original text"]]
  model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
  labels = tokenizer(text_target=examples["summarized text"], max_length=128, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)
  return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
  args = parse_args()

  tokenizer = AutoTokenizer.from_pretrained("t5-small")
  rouge = evaluate.load("rouge")

  if args.mode == "train":
    # Load training data
    data = pd.read_csv(args.train_data)
    print(data.head())
    ds = Dataset.from_pandas(data)
    ds = ds.train_test_split(test_size=0.2)

    tokenized_data = ds.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir="my_fine_tuned_t5_small_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("my_fine_tuned_" + args.model_name)

  elif args.mode == "test":
    #print model ouput on an entry
    data = pd.read_csv(args.train_data)
    print(data.head())
    ds = Dataset.from_pandas(data)
    ds = ds.train_test_split(test_size=0.2)
    text = ds['test'][1]['original text']
    text = "summarize: " + text
    print("Text",text)

    summarizer = pipeline("summarization", model="my_fine_tuned_t5_small_model")
    pred = summarizer(text)
    print("Summarized",pred)