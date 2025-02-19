## Project Overview

The goal of this project is to reproduce the system descibed in ["TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction."](https://arxiv.org/abs/2310.15556) The project aims to implement and evaluate a token compression technique for large language models (LLMs), focusing on Retrieval Augmented Generation (RAG) systems which will allow to reduce the operation costs when using a thirdy part LLM that charges by a per token rate.

## Files

### `generate_data.py`

This script generates summarized versions of datasets for fine-tuning summarization models. It provides functionality to summarize datasets and export the results to CSV files. Users can specify the percentage of summarization and choose between summarizing the entire dataset or a single entry for testing model output.

**Components**
  * **Functions:**
      * `summarize_dataset`: This function loads a specified dataset, loads a summarization model (like LLAMA), and generates summaries for each entry in the dataset.
          * **Parameters:**
              * `dataset_name`: Name of the dataset (e.g., "rag-datasets/mini-bioasq" (this is the default dataset and what I  used for my exxperiements)).
              * `percent`: Percentage of the original text length for the summary (e.g., 50 for half the length).
              * `output_file`: Name of the file to store the summarized dataset.
              * `single_entry` (optional): Flag indicating whether to summarize a single entry (True) or the entire dataset (False). Defaults to False.
      * `summarize_entry`: This function summarizes a single entry from the dataset.
          * **Parameters:**
              * `llm`: The loaded summarization model (for this project I have used "Mistral 7B instruct' for creating summarizated version of data, can easily swap and experiment with others ).
              * `text`: The input text to be summarized.
              * `percent`: Percentage of the original text length for the summary (e.g., 50 for half the length).

**Example Usage**

* **Summarize a dataset:**

```bash
python generate_data.py --dataset_name rag-datasets/mini-bioasq --percent 50 --output_file train.csv
```

This command summarizes the `rag-datasets/mini-bioasq` dataset with summaries at 50% of the original text length and saves the summarized dataset to `train.csv` (this can be used for debugging and testing mainly).

* **Summarize a single entry:**

```bash
python summarization.py --dataset_name rag-datasets/mini-bioasq --single_entry
```

This command prompts you to provide the text for a single entry from the `rag-datasets/mini-bioasq` dataset and then summarizes it using the loaded model.



### `finetune_model.py`

This script fine-tunes T5 summarization models. It loads training data, preprocesses it, and trains the model with specified hyperparameters. It also offers functions to calculate evaluation metrics like ROUGE scores.

**Usage**

Run the script using the following command-line options:

```bash
python finetune_model.py --mode train --train_data <path_to_train_data>
```

- `--mode`: Specify whether to "train" or "test" the model.
- `--train_data`: Path to the training CSV file.

**Components**

* **Indexing:**
    - Loads data from a CSV file.
    - Splits data into training and testing sets using `Dataset.from_pandas` the HF Datasets library.
    - Tokenizes the data for T5 model training.
* **Model Setup:**
    - Uses `AutoModelForSeq2SeqLM` from Hugging Face Transformers to load the T5 model.
    - Configures training arguments (learning rate, batch size, epochs) using `Seq2SeqTrainingArguments`.
* **Preprocessing:**
    - The `preprocess_function` tokenizes input text and generates labels for training.
* **Evaluation Metrics:**
    - The `compute_metrics` function calculates metrics like ROUGE score to evaluate model performance.
* **Training:** (In `train` mode)
    - Loads training data from the specified CSV file (`--train_data`).
    - Preprocesses the data.
    - Trains the T5 model using the specified training arguments.
    - Saves the trained model to the directory specified by `output_dir` in `Seq2SeqTrainingArguments`.
* **Testing:** (In `test` mode)
    - Loads test data from a CSV file.
    - Preprocesses the data.
    - Summarizes a single entry from the test dataset using the trained model.
    - Prints the summarized text to the console.

**Output**

- After training, the script saves the trained model to the specified output directory.
- In testing mode, the script prints the summarized text to the console.


### `rag.py`

This script sets up the components for the Retrieval Augmented Generation (RAG) system. It includes functions to load the index, configure the LLM, set up the RAG query engine, and evaluate the system using predefined benchmarks.You can use --custom_rag flag to use the RAG with summarized context or without it to use RAG with full context. 

### Components

1. **Indexing**

   The indexing process creates a search index from a text corpus. It involves:

   - Loading dataset using `load_dataset`.
   - Building a vector store index from the documents.
   - Persisting the index to disk for later use.

2. **LLM Setup**

   LLM (Language, Logic, and More) is the generative model used in RAG. This step initializes the LLM model with specific parameters.
   For my experiments I have used a LLaMA 2 7B model, this can be changed to any other model of choice in the code.

3. **RAG Query Engine Setup**

   This sets up the query engine for the RAG model. It includes:

   - Retrieval of relevant passages using the retriever.
   - Summarization of passages.
   - Generation of responses using LLM based on provided context and query.

4. **Scorer Setup**

   The scorer is responsible for evaluating the RAG model's performance using various metrics like answer similarity, consistency, retrieval precision, and augmentation accuracy.

   1. **Answer Similarity Score**
        - **Input**: Question, Reference answer, LLM answer
        - **Formula**: Score between 0 and 5 measuring how well the reference answer matches the LLM answer.
        - **What does it measure?**: Overall similarity between the reference answer and the LLM-generated answer.
        - **Evaluated components**: All components of the RAG system

    2. **Retrieval Precision**
        - **Input**: Question, Retrieved context
        - **Formula**: (Count of relevant retrieved context) / (Count of retrieved context)
        - **What does it measure?**: Whether the context retrieved is relevant to answer the given question.
        - **Evaluated components**: Chunker, Embedder, Retriever

    3. **Augmentation Accuracy**
        - **Input**: Retrieved context, LLM answer
        - **Formula**: (Count of retrieved context in LLM answer) / (Count of retrieved context)
        - **What does it measure?**: Whether all the context is included in the LLM-generated answer.
        - **Evaluated components**: Prompt builder, LLM

    4. **Answer Consistency**
        - **Input**: Retrieved context, LLM answer
        - **Formula**: (Count of main points in answer that can be attributed to context) / (Count of main points in the answer)
        - **What does it measure?**: Whether the LLM-generated answer contains information from the retrieved context.
        - **Evaluated components**: Prompt builder, LLM


### `test_rag.py`

Similar to `rag.py`, this script sets up and evaluates the RAG system. However, it provides an interactive mode where users can input questions and receive responses from either the custom RAG i.e. one with summarized context or the default RAG. 

## Usage

To use the provided scripts:

1. Ensure all dependencies are installed. You may need to install additional packages specified in the `requirements.txt` file.
```bash
pip install -r equirements.txt
```
2. First run the `generate_data.py` script to generate different summarized version of datasets.
```bash
python summarization.py --dataset_name rag-datasets/mini-bioasq --percent 50 --output_file train50.csv
```
3. Then finetune the T5 model on the generated dataset and the finetuned model will be saved with name my_fine_tuned_t5_small_model

```bash
python finetune_model.py --mode train --train_data train50.csv --model t5-small
```

4. Then use the `rag.py` to evaluate the rag on QA datasets.
```bash
python rag.py --custom_rag 
```
--custom_rag flag uses the RAG system with te summarizer and without htis flag the RAG without the context compressor is used, print results of both to see the difference.
Additional flags like --dataset_name , --create_index can be used if needed.

Note : You would need to update your OPENAPI key in this file as the evaluation framework uses gpt-3.5-turbo (as of now using a custom llm doesn't seem to work with the library).

5. Alternatively you can test a question of your own and see the retrieved context using the `test_rag.py` 

## Results

| Model | Answer similarity (0-5) | Answer consistency (0-1) | Retrieval precision (0-1) | Augmentation accuracy (0-1) |
|---|---|---|---|---|
| Llama 2 7b without context retrieval | 2.0 | - | - | - |
| Llama 2 7b with context retrieval (no summarization) | 4.5 | 0.92 | 0.95 | 0.86 |
| Llama 2 7b with context summarization (no finetuning) | 3.0 | 0.67 | 0.70 | 0.78 |
| Llama 2 7b with context summarization finetuned(50%) | 3.8 | 0.76 | 0.85 | 0.80 |
| Llama 2 7b with context summarization finetuned (70%) | 4.3 | 0.90 | 0.88 | 0.83 |




