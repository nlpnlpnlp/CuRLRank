
import logging
from typing import List
from FlagEmbedding.abc.evaluation import AbsEvalDataLoader
logger = logging.getLogger(__name__)
EXTENDED_SPLITS = [
    # w/ reasoning splits
]

class BrightShortEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        return [
            # StackExchange
            "biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living",
            # Coding
            "leetcode", "pony",
            # Theorem-based
            "aops", "theoremqa_questions", "theoremqa_theorems"
        ]

    def available_splits(self, dataset_name: str) -> List[str]:
        return [
            # normal splits
            "examples",
            # w/ reasoning splits
            "Gemini-1.0_reason", "claude-3-opus_reason", "gpt4_reason", "grit_reason", "llama3-70b_reason",
        ] + EXTENDED_SPLITS
 
class R2medEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        return [
            "r2med_Biology", "r2med_Bioinformatics", "r2med_Medical-Sciences", 
            "r2med_MedXpertQA-Exam", "r2med_MedQA-Diag", "r2med_PMC-Treatment",
            "r2med_PMC-Clinical", "r2med_IIYi-Clinical",
        ]

    def available_splits(self, dataset_name: str) -> List[str]:
        return [
            # normal splits
            "examples",
            # w/ reasoning splits
            "Gemini-1.0_reason", "claude-3-opus_reason", "gpt4_reason", "grit_reason", "llama3-70b_reason",
        ] + EXTENDED_SPLITS

class BeirEvalDataLoader(AbsEvalDataLoader):
    def available_dataset_names(self) -> List[str]:
        return [
            "trec-covid", "dbpedia-entity", "scifact",
            "nfcorpus", "signal", "robust04", "news",
        ]

    def available_splits(self, dataset_name: str) -> List[str]:
        return [
            # normal splits
            "examples",
            # w/ reasoning splits
            "Gemini-1.0_reason", "claude-3-opus_reason", "gpt4_reason", "grit_reason", "llama3-70b_reason",
        ] + EXTENDED_SPLITS
