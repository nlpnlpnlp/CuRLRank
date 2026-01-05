import os
import json
import logging
from typing import Dict, List, Optional, Union

from FlagEmbedding.abc.evaluation import AbsEvaluator
from FlagEmbedding.abc.evaluation.utils import evaluate_metrics, evaluate_mrr
from FlagEmbedding.abc.evaluation.searcher import EvalRetriever, EvalReranker

from custom_searcher import CustomEvalRetriever

logger = logging.getLogger(__name__)


class BrightEvaluator(AbsEvaluator):
    """
    Evaluator class of Bright 
    """
    
    @staticmethod
    def compute_metrics(
        qrels: Dict[str, Dict[str, int]],
        search_results: Dict[str, Dict[str, float]],
        k_values: List[int],
    ):
        # remove ex_docids from search_results
        for qid in qrels:
            for docid, score in qrels[qid].items():
                if score == 0:
                    search_results[qid].pop(docid, None)
        
        ndcg, _map, recall, precision = evaluate_metrics(
            qrels=qrels,
            results=search_results,
            k_values=k_values,
        )
        mrr = evaluate_mrr(
            qrels=qrels,
            results=search_results,
            k_values=k_values,
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }
        return scores
    
    @staticmethod
    def save_search_results(
        eval_name: str,
        model_name: str,
        reranker_name: str,
        search_results: Dict[str, Dict[str, float]],
        output_path: str,
        split: str,
        dataset_name: Optional[str] = None,
        completion_tokens: Optional[float] = None,
        running_time: Optional[float] = None,
    ):
        data = {
            "eval_name": eval_name,
            "model_name": model_name,
            "reranker_name": reranker_name,
            "split": split,
            "dataset_name": dataset_name,
            "running_time": running_time,
            "completion_tokens": completion_tokens,
            "search_results": search_results,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    
    def __call__(
        self,
        splits: Union[str, List[str]],
        retriever_result_path: str,
        reranker_result_path: str, 
        retriever: EvalRetriever,
        reranker: Optional[EvalReranker] = None,
        ignore_identical_ids: bool = False,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000],
        dataset_name: Optional[str] = None,
        **kwargs,
    ):
        # Check Splits
        checked_splits = self.data_loader.check_splits(splits, dataset_name=dataset_name)
        if len(checked_splits) == 0:
            logger.warning(f"{splits} not found in the dataset. Skipping evaluation.")
            return
        splits = checked_splits

        if dataset_name is not None:
            save_name = f"{dataset_name}-" + "{split}.json"
        else:
            save_name = "{split}.json"
        # Retrieval Stage
        no_reranker_search_results_dict = {}
        if len(splits) != 1:
            logger.warning(f"{splits} only examples")
            return
        for split in splits:
            if isinstance(retriever, CustomEvalRetriever):
                # load retrieval results
                retriever_result_path = os.path.join(retriever_result_path, save_name.format(split=split))
                data_info, search_results = self.load_search_results(retriever_result_path)
            else:
                raise NotImplementedError("Only CustomEvalRetriever is supported in the current version.")

            self.check_data_info(
                data_info=data_info,
                model_name=str(retriever),
                reranker_name="NoReranker",
                split=split,
                dataset_name=dataset_name,
            )
            no_reranker_search_results_dict[split] = search_results

        reranker_search_results_save_dir_dict = {}
        for split in splits:
            reranker_search_results_save_dir_dict[split] = reranker_result_path
            os.makedirs(reranker_search_results_save_dir_dict[split], exist_ok=True)

        corpus = self.data_loader.load_corpus(dataset_name=dataset_name)

        queries_dict = {
            split: self.data_loader.load_queries(dataset_name=dataset_name, split=split)
            for split in splits
        }

        flag = False
        for split in splits:
            rerank_search_results_save_path = os.path.join(
                reranker_search_results_save_dir_dict[split], save_name.format(split=split)
            )

            if os.path.exists(rerank_search_results_save_path) and not self.overwrite:
                continue

            flag = True
            # reranker
            kwargs['dataset_name'] = dataset_name
            rerank_search_results, reranker_cache, running_time, completion_tokens = reranker(
                corpus=corpus,
                queries=queries_dict[split],
                search_results=no_reranker_search_results_dict[split],
                ignore_identical_ids=ignore_identical_ids,
                **kwargs,
            )

            # save_search_results
            self.save_search_results(
                eval_name=self.eval_name,
                model_name=str(retriever),
                reranker_name=str(reranker),
                search_results=rerank_search_results,
                output_path=rerank_search_results_save_path,
                split=split,
                dataset_name=dataset_name,
                running_time=running_time,
                completion_tokens=completion_tokens
            )
        ##################### save metrics and reranker_cache #####################
        cache_results_save_path = os.path.join(reranker_search_results_save_dir_dict[split], 'cache', save_name.format(split=split))
        if not os.path.exists(cache_results_save_path):
            self.output_eval_results_to_json(reranker_cache, cache_results_save_path)
            
        eval_results_save_path = os.path.join(reranker_search_results_save_dir_dict[split], 'EVAL', 'eval_results.json')
        if not os.path.exists(eval_results_save_path) or self.overwrite or flag:
            reranker_eval_results = self.evaluate_results(reranker_search_results_save_dir_dict[split], k_values=k_values)
            self.output_eval_results_to_json(reranker_eval_results, eval_results_save_path)
