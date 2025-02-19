import argparse
import pandas as pd
from datasets import load_dataset
from nltk import word_tokenize
from llama_index.core import Settings
from llama_index.llms.llama_cpp import LlamaCPP

def summarize_entry(llm, text, percent=50):
    question = f"""
    Instruction : Summarize the following text, the length of the summary result is {percent} percent of the original text,
     keep the first sentence, and directly output your answer: \n\n {text}
    """
    resp = llm.complete(question)
    summary = resp.text.replace("Summary: ", "")
    summary_len = len(word_tokenize(summary))
    original_len = len(word_tokenize(text))
    percentage_reduction = (1 - (summary_len / original_len)) * 100

    print("\nSummarization Output on Single Entry:")
    print(summary)
    print("OG len : ", original_len)
    print("Summarized len : ", summary_len)
    print("Percentage reduction: ", percentage_reduction)

def summarize_dataset(dataset_name, percent=50, output_file="train.csv", single_entry=False):
    # Load dataset
    dataset = load_dataset(dataset_name, 'question-answer-passages')
    data = load_dataset(dataset_name, 'text-corpus')
    text_list = [x['passage'] for x in data['passages']]

    # Load summarization model
    llm = LlamaCPP(
        model_path="./mistral-7b-instruct-v0.2.Q2_K.gguf",
        temperature=0.2,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        verbose=True,
    )
    Settings.llm = llm
    Settings.chunk_size = 1024

    if single_entry:
        summarize_entry(llm, text_list[0], percent)
        return

    reduced_text_list = []

    for text in text_list:
        original_value = len(word_tokenize(text))
        reduced_value = original_value
        ftext = text
        if original_value <= 40:
            reduced_text_list.append(ftext)
            continue
        cnt = 0
        while cnt < 3 and reduced_value > original_value * (percent/100):
            question = f"""
            Instruction : Summarize the following text, the length of the summary result is {percent} percent of the original text,
            keep the first sentence, and directly output your answer please make sure the length of summary is around 50% of input length
            it is very important also don't remove important factual information: \n\n {ftext}
            """
            resp = llm.complete(question)
            summary = resp.text.replace("Summary: ", "")
            ftext = summary
            reduced_value = len(word_tokenize(summary))
            cnt = cnt + 1
        reduced_text_list.append(summary)

    # Save to DataFrame and export to CSV
    df = pd.DataFrame({'original text': text_list, 'summarized text': reduced_text_list})
    df = df[~df['original text'].str.contains('nan')]
    df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Generate summarized version of a dataset for fine-tuning summarization models.")
    parser.add_argument("--dataset_name", type=str,default="rag-datasets/mini-bioasq", help="Name of the dataset to be summarized")
    parser.add_argument("--percent", type=int, default=50, help="Percentage of summarization (default: 50)")
    parser.add_argument("--output_file", type=str, default="train.csv", help="Output file name (default: train.csv)")
    parser.add_argument("--single_entry", action="store_true", help="Summarize a single entry from the dataset to test model output")
    args = parser.parse_args()

    summarize_dataset(args.dataset_name, percent=args.percent, output_file=args.output_file, single_entry=args.single_entry)

if __name__ == "__main__":
    main()
