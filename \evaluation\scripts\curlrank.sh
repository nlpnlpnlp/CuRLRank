#!/bin/bash
ROOT_DIR="YOUR_ROOT_DIR"

# nohup bash ./curlrank.sh > curlrank.log 2>&1 &

benchmark_name="r2med"
splits="examples"
dataset_names="r2med_Biology r2med_Bioinformatics r2med_Medical-Sciences r2med_MedXpertQA-Exam r2med_MedQA-Diag r2med_PMC-Treatment r2med_PMC-Clinical r2med_IIYi-Clinical"

dataset_path="$ROOT_DIR/$benchmark_name/data"
retriever_result_path="$ROOT_DIR/$benchmark_name/retriever_result/$splits"
reranker_result_path="$ROOT_DIR/$benchmark_name/reranker_result/$splits"

embedder_model_path="YOUR_EMBEDDER_MODEL"
reranker_model_path="CURLRANK_MODEL"

############################### R2MED ################################
python $ROOT_DIR/main.py \
    --dataset_dir $dataset_path \
    --benchmark_name $benchmark_name \
    --dataset_names $dataset_names \
    --search_top_k 2000 \
    --k_values 1 10 100 \
    --splits $splits \
    --retriever_result_path=$retriever_result_path \
    --reranker_result_path=$reranker_result_path \
    --embedder_name_or_path $embedder_model_path \
    --embedder_model_class custom \
    --reranker_name_or_path $reranker_model_path \
    --reranker_model_class sglang-reasoning \
    --rerank_top_k 100 \
    --reranker_sample_k 1 \
    --reranker_batch_size 10240 \
    --reranker_sglang_tp_size 1 \
    --reranker_sglang_dp_size 2 \
    --reranker_sglang_seed 42 \
    --reranker_sglang_context_length 10240 \
    --reranker_max_new_tokens 2048 \
    --reranker_enable_thinking False \
    --reranker_enable_full_parallelism False \
    --reranker_enable_cache True \
    --overwrite True \


benchmark_name="bright"
splits="examples"
dataset_names="biology earth_science economics psychology robotics stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems"

dataset_path="$ROOT_DIR/$benchmark_name/data"
retriever_result_path="$ROOT_DIR/$benchmark_name/retriever_result/$splits"
reranker_result_path="$ROOT_DIR/$benchmark_name/reranker_result/$splits"

################################ BRIGHT ################################
python $ROOT_DIR/main.py \
    --dataset_dir $dataset_path \
    --benchmark_name $benchmark_name \
    --dataset_names $dataset_names \
    --search_top_k 2000 \
    --k_values 1 10 100 \
    --splits $splits \
    --retriever_result_path=$retriever_result_path \
    --reranker_result_path=$reranker_result_path \
    --embedder_name_or_path $embedder_model_path \
    --embedder_model_class custom \
    --reranker_name_or_path $reranker_model_path \
    --reranker_model_class sglang-reasoning \
    --rerank_top_k 100 \
    --reranker_sample_k 1 \
    --reranker_batch_size 10240 \
    --reranker_sglang_tp_size 1 \
    --reranker_sglang_dp_size 2 \
    --reranker_sglang_seed 42 \
    --reranker_sglang_context_length 10240 \
    --reranker_max_new_tokens 2048 \
    --reranker_enable_thinking False \
    --reranker_enable_full_parallelism False \
    --reranker_enable_cache True \
    --overwrite True \


benchmark_name="beir"
splits="examples"
dataset_names="trec-covid nfcorpus dbpedia-entity scifact signal robust04 news"
dataset_path="$ROOT_DIR/$benchmark_name/data"
retriever_result_path="$ROOT_DIR/$benchmark_name/retriever_result/$splits"
reranker_result_path="$ROOT_DIR/$benchmark_name/reranker_result/$splits"

############################### BEIR ################################
python $ROOT_DIR/main.py \
    --dataset_dir $dataset_path \
    --benchmark_name $benchmark_name \
    --dataset_names $dataset_names \
    --search_top_k 2000 \
    --k_values 1 10 100 \
    --splits $splits \
    --retriever_result_path=$retriever_result_path \
    --reranker_result_path=$reranker_result_path \
    --embedder_name_or_path $embedder_model_path \
    --embedder_model_class custom \
    --reranker_name_or_path $reranker_model_path \
    --reranker_model_class sglang-reasoning \
    --rerank_top_k 100 \
    --reranker_sample_k 1 \
    --reranker_batch_size 10240 \
    --reranker_sglang_tp_size 1 \
    --reranker_sglang_dp_size 2 \
    --reranker_sglang_seed 42 \
    --reranker_sglang_context_length 10240 \
    --reranker_max_new_tokens 2048 \
    --reranker_enable_thinking False \
    --reranker_enable_full_parallelism False \
    --reranker_enable_cache True \
    --overwrite True \
