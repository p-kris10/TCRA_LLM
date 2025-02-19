from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.llama_cpp import LlamaCPP
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from ctransformers import AutoModelForCausalLM
import llama_index.llms.huggingface
from llama_index.llms.huggingface import HuggingFaceLLM
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from tonic_validate.metrics import RetrievalPrecisionMetric
from tonic_validate.metrics import AnswerConsistencyMetric, AnswerSimilarityMetric
from tonic_validate.metrics import AugmentationAccuracyMetric
import json
from tonic_validate import Benchmark, ValidateApi, ValidateScorer
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.llama_cpp import LlamaCPP 
from llama_index.core import PromptTemplate
from transformers import pipeline
import os
import argparse


def load_index(index_path):
    index_path = "./full_index"  # Replace with your actual path
    storage_context = StorageContext.from_defaults(
     persist_dir=index_path
    )   
    index = load_index_from_storage(storage_context=storage_context)
    return index

def setup_llm():
    #setup llm for the RAG
    #choose llm of choice here I am using LLama7B
    llm = LlamaCPP(
        model_url="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf",
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        verbose=True,
    )
    return llm

def setup_rag_query_engine(index, llm):
    summarizer = pipeline("summarization", model="my_fine_tuned_t5_small_model")
    retriever = index.as_retriever()
    synthesizer = get_response_synthesizer(response_mode="compact")
    qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    class RAGStringQueryEngine(CustomQueryEngine):
        retriever: BaseRetriever
        response_synthesizer: BaseSynthesizer
        llm: LlamaCPP 
        qa_prompt: PromptTemplate

        def custom_query(self, query_str: str):
            nodes = self.retriever.retrieve(query_str)
            context = [summarizer(n.node.get_content())[0]['summary_text'] for n in nodes]
            context_str = "\n\n".join(context)
            response = self.llm.complete(
                qa_prompt.format(context_str=context_str, query_str=query_str)
            )

            return {"llm_answer": response,"llm_context_list": context}
            

    query_engine = RAGStringQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        llm=llm,
        qa_prompt=qa_prompt,
    )
    return query_engine

def setup_scorer():
    scorer = ValidateScorer([AnswerSimilarityMetric(), AnswerConsistencyMetric(),RetrievalPrecisionMetric(),AugmentationAccuracyMetric()])
    return scorer

def get_llama_response(prompt):
    response = query_engine.query(prompt)
    context = [x.text for x in response.source_nodes]
    return {
        "llm_answer": response.response,
        "llm_context_list": context
    }

def get_custom_response(prompt):
    response,context = query_engine.query(prompt)
    
    return {
        "llm_answer": response,
        "llm_context_list": context
    }


def evaluate(benchmark, query_fn):
    run = scorer.score(benchmark, query_fn)
    print("Overall Scores")
    print(run.overall_scores)
    print("------")
    for item in run.run_data:
        print("Question: ", item.reference_question)
        print("Answer: ", item.reference_answer)
        print("LLM Answer: ", item.llm_answer)
        print("LLM Context: ", item.llm_context)
        print("Scores: ", item.scores)
        print("------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Evaluation')
    parser.add_argument('--custom_rag', action='store_true', help='Use custom RAG')
    parser.add_argument('--create_index', action='store_true', help='Download dataset and create index')
    parser.add_argument("--dataset_name", type=str,default="rag-datasets/mini-bioasq", help="Name of the dataset used (make sure it is a QA dataset with both QA and text corpus)")
    args = parser.parse_args()
    index_path = "full_index"  # Replace with your actual path
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    index = None
    dataset = load_dataset(args.dataset_name,'question-answer-passages')
    if args.create_index:
        data = load_dataset(args.dataset_name,'text-corpus')
        text_list = [x['passage'] for x in data['passages']]
        documents = [Document(text=t) for t in text_list]
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist("full_index")
    else:
        index = load_index(index_path)

    llm = setup_llm()
    Settings.llm = llm
    scorer = setup_scorer()

    
    
    question = input("Enter question : \n")

    
    if args.custom_rag:
        query_engine = setup_rag_query_engine(index, llm)
        print(query_engine.custom_query(question))
        
    else:
        query_engine = index.as_query_engine()
        print(get_llama_response(question))
        
