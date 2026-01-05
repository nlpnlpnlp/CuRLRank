import os
import json
import logging
import random
import deepspeed
random.seed(42)
from typing import Dict, Any, List, Tuple
from FlagEmbedding.abc.evaluation import EvalReranker
from reranker_prompts import WINDOW_SIZE

logger = logging.getLogger(__name__)


class ReasoningEvalReranker(EvalReranker):
    def __init__(
            self,
            reranker,
            rerank_top_k: int = 100,
            reranker_enable_cache: bool = True,
        ):
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.reranker_enable_cache = reranker_enable_cache

        self.reranker_sample_k = reranker.sample_k
        self.reranker_base_name = os.path.basename(self.reranker.model_name_or_path)
        
        self.base_reranker_name = self.reranker_base_name
    
    def __str__(self) -> str:
        return self.base_reranker_name
    
    def __call__(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        search_results: Dict[str, Dict[str, float]],
        ignore_identical_ids: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Dict[str, float]], float, float]:
        
        dataset_name = kwargs['dataset_name']
        ########### excluded document ids ###########
        excluded_ids = {}
        qrels = kwargs.get("reranker_qrels", None)
        if qrels is not None:
            for qid in qrels:
                excluded_ids[qid] = []
                for docid, score in qrels[qid].items():
                    if score == 0:
                        excluded_ids[qid].append(docid)
        else:
            logger.warning("No qrels provided, so no documents will be excluded.")
        
        ########### truncate search results to top_k ###########
        for qid in search_results:
            # Filter out documents with ids in excluded_ids
            for doc_id in set(excluded_ids[qid]):
                if doc_id != "N/A":
                    search_results[qid].pop(doc_id, None)
            
            search_results[qid] = dict(
                sorted(search_results[qid].items(), key=lambda x: x[1], reverse=True)[
                    :self.rerank_top_k
                ]
            )
        ########### generate query-documents pairs ###########
        reranker_data = []
        for qid in search_results:
            single_re = {}
            single_re["q_id"] = qid
            single_re["q_content"] = queries[qid]
            top_100_doc = []
            for docid, score in search_results[qid].items():
                if ignore_identical_ids and qid == docid:
                    continue
                doc_content = corpus[docid]["text"]
                top_100_doc.append({
                    "d_id": docid,
                    "d_content": doc_content,
                    "score": score
                })
            single_re["top_docs"] = top_100_doc
            reranker_data.append(single_re)
        
        ########### reranking ###########
        reranker_data, reranker_cache, all_run_time, all_completion_tokens = self.tournament_scoring_strategy(reranker_data, dataset_name)
        
        reranked_results = {qid: {} for qid in search_results}
        for data in reranker_data:
            for doc in data["top_docs"]:
                reranked_results[data["q_id"]][doc["d_id"]] = float(doc["score"])
        
        return reranked_results, reranker_cache, all_run_time, all_completion_tokens

    def tournament_scoring_strategy(self, reranker_data, dataset_name,):
        reranker_cache = {}
        round_number = 1
        rank_start = 0
        rank_end = 100
        all_run_time = 0.0
        all_completion_tokens = 0.0
        step = WINDOW_SIZE[dataset_name]

        while (rank_end - rank_start) > 10 and round_number < 5:
            round_cache = {}
            logger.info(f"round_number: {round_number}, rank_start: {rank_start}, rank_end: {rank_end}")
            print(f"round_number: {round_number}, rank_start: {rank_start}, rank_end: {rank_end}")
            ############ get reranker data ############
            ## 这里要把所有的数据都放上去
            query_docs_pairs = []
            for single in reranker_data:
                qid = single["q_id"]
                qtext = single["q_content"]
                # 防止越界
                # rank_end = min(rank_end, len(single["top_docs"]))
                limit_end = min(rank_end, len(single["top_docs"]))
                for start in range(rank_start, limit_end, step):
                    end = min(start + step, limit_end)
                    sub_docs = single["top_docs"][start:end]
                    doc_ids = [d["d_id"] for d in sub_docs]
                    doc_texts = [d["d_content"] for d in sub_docs]
                    doc_scores = [d["score"] for d in sub_docs]
                
                    query_docs_pairs.append({
                        "qid": qid,
                        "docid": doc_ids,     # List[str]
                        "query": qtext,       # str
                        "docs": doc_texts,     # List[str]
                        "score": doc_scores,  # List[float]
                        "num_docs": len(sub_docs) # int
                    })
            ########### reranking ############
            # List[Bool], List[List[float]], float, float 
            flag_list, final_scores_list, running_time, completion_tokens = self.reranker.compute_score(query_docs_pairs, dataset_name)

            ########## update reranker_cache and reranker_data ##########
            # updata query_docs_pairs and reranker_data
            all_run_time += running_time
            all_completion_tokens += completion_tokens
            
            # get cache_score
            cache_score = 0.0
            # if have cache before this round
            if str(round_number - 1) in reranker_cache:
                doc_num = 0
                for qid, rerank_list in reranker_cache[str(round_number - 1)].items():
                    for d in rerank_list:
                        cache_score += sum(d["scores"])
                        doc_num += len(d["scores"])
                cache_score /= doc_num
            else:
                # if do not have cache before this round
                cache_score = 65
            # update reranker cache and get current round score
            current_score = {}
            for i, pair_data in enumerate(query_docs_pairs):
                qid = pair_data["qid"]
                doc_ids = pair_data["docid"] # 当前窗口内的文档ID列表
                scores = final_scores_list[i] # LLM打出的新分数列表
                flag = flag_list[i]
                # if error generation we use cache_score replace
                if flag == False:
                    scores = [cache_score] * len(doc_ids)

                if qid not in current_score:
                    current_score[qid] = {}
                for key, value in zip(doc_ids, scores):
                    current_score[qid][key] = value

                #  use current score update cache
                rerank_detail = {
                    'doc_ids': doc_ids,
                    'scores': scores,
                    'flag': flag
                }
                if qid not in round_cache:
                    round_cache[qid] = []
                round_cache[qid].append(rerank_detail)
            reranker_cache[str(round_number)] = round_cache
            # update reranker_data[doc][score] using current score
            if round_number == 1:
                cache_score = 0.0
            for single in reranker_data:
                qid = single["q_id"]
                for doc in single["top_docs"]:
                    d_id = doc["d_id"]
                    # 🌟 修复2：关键判定！只有在本轮被打分的文档才更新分数
                    if d_id in current_score.get(qid, {}):
                        new_val = current_score[qid][d_id]
                        # 核心公式：历史分加权 + 晋级奖励
                        doc["score"] = ((new_val + (round_number - 1) * doc["score"]) / round_number) + 5
                        doc["score"] = max(doc["score"], cache_score)
                    else:
                        pass

            for single_re in reranker_data:
                # 1、对 rank_start 到 rank_end 范围内的元素进行排序（容易局部最优）
                # sub_docs = single_re["top_docs"][rank_start:rank_end]
                # sub_docs.sort(key=lambda d: d["score"], reverse=True)
                # single_re["top_docs"][rank_start:rank_end] = sub_docs
                # 2、对全局进行排序，这里选择了对全局进行排序
                single_re["top_docs"].sort(key=lambda d: d["score"], reverse=True)
            
            mean_num_above = (rank_end - rank_start) / 2
            if round_number == 1:
                mean_num_above += 10
            rank_end = min(((int(mean_num_above) + step - 1) // step) * step, rank_end)
            
            # Shuffle Document for Next Round
            for single_re in reranker_data:
                # 🌟 修复3：Shuffle 也要防止越界
                current_active_len = min(rank_end, len(single_re["top_docs"]))
                if rank_start < current_active_len:
                    docs = single_re["top_docs"][rank_start: current_active_len]
                    random.shuffle(docs)
                    single_re["top_docs"][rank_start: current_active_len] = docs
            round_number += 1  
        return reranker_data, reranker_cache, all_run_time, all_completion_tokens

   