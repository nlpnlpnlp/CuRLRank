import re
from ftfy import fix_text
from typing import List
from beir.retrieval.evaluation import EvaluateRetrieval

def get_prompt_template(
    relevance_definition: str, 
    query_text: str, 
    document_text: str, 
    doc_num: int
) -> str:
        prompt =f"""
You are asked to evaluate the relevance between a **query** and **{doc_num} documents** based on the following **relevance definition**:  
{relevance_definition}
Please follow the steps below carefully. Your reasoning should be clear, structured, and concise.
----------------------------------------
### 1. Query Analysis
Analyze the query and describe what information would be most helpful for answering it.  
- Focus on the key information needs implied by the query.  
- Consider context, intent, and any specific constraints.  
- **Token limit:** Maximum 512 tokens.  
**Output format:**  
[query_analysis]
----------------------------------------
### 2. Document Analysis
Analyze each document individually. For each document:  
- Discuss whether the document provides the key information required to answer the query.  
- Identify any missing, incomplete, or irrelevant information.  
- Provide structured reasoning (bullet points recommended).  
- **Token limit:** Maximum 256 tokens per document.  
**Output format:**  
[1]: [doc_1_analysis]
[2]: [doc_2_analysis]
[3]: [doc_3_analysis]
...(repeat for all documents)
----------------------------------------
### 3. Relevance Annotation
For each document:  
1. Assign its **contribution to the query**: complete, partial, or minimal.  
- **Token limit:** Maximum 128 tokens per document.  
2. Evaluate each document independently to determine its **absolute relevance** to the query. 
3. Assign an **integer relevance score (0–100)** according to the guide:  
- 81–100 (Highly Relevant): Fully and directly addresses the query; core, authoritative content.
- 61–80 (Relevant): Mostly addresses the query; key info included, minor details may be missing.
- 41–60 (Moderately Relevant): Partially addresses the query; on-topic but not comprehensive.
- 21–40 (Slightly Relevant): Mentions query keywords; main topic is different, limited value.
- 0–20 (Irrelevant): Does not address the query; off-topic.
**Output format:**  
<score> [doc1_score] [doc2_score] [doc3_score] ... </score>
- Each score should be enclosed in brackets [ ] and separated by a single space.
----------------------------------------
### 4. Data Difficulty Evaluation
Rate the difficulty of assessing relevance on a 1–5 scale:
1 (Very Easy): Documents are clear and directly match the query.  
2 (Easy): Most key info is present; minor reasoning needed.  
3 (Medium): Partial info or multiple points require integration.  
4 (Hard): Info is scattered, partially irrelevant, or ambiguous.  
5 (Very Hard): Documents are complex, ambiguous, or noisy; deep reasoning required.
**Output format:**  
<Difficulty>[1-5]</Difficulty>
<Justification>[1–2 sentence explanation]</Justification>
----------------------------------------
### Notes
- Keep reasoning clear, structured, and concise.  
- Use bullet points when analyzing documents.  
- Stick to token limits; do not exceed them.  
- Ensure document numbering matches the score order.
----------------------------------------
**Input format:**  
Query:
{query_text}
Documents:
{document_text}
"""
        return prompt

def truncate_texts(
    tokenizer,
    texts: List[str],
    max_length: int,
    front_ratio: float = 0.7,
) -> List[str]:
    processed_texts = []
    estimated_char_limit = max_length * 6
    
    for text in texts:
        text = text.strip()
        text = fix_text(text)
        if len(text) > estimated_char_limit:
            half_limit = estimated_char_limit // 2
            text = text[:half_limit] + " ... " + text[-half_limit:]
        processed_texts.append(text)

    batch_encodings = tokenizer(processed_texts, add_special_tokens=False)
    input_ids_list = batch_encodings["input_ids"]

    ids_to_decode = []
    front_len = int(max_length * front_ratio)
    back_len = max_length - front_len

    for ids in input_ids_list:
        if len(ids) > max_length:
            new_ids = ids[:front_len] + ids[-back_len:]
            ids_to_decode.append(new_ids)
        else:
            ids_to_decode.append(ids)

    truncated_texts = tokenizer.batch_decode(ids_to_decode, skip_special_tokens=True)

    return [replace_number(t) for t in truncated_texts]

def replace_number(s: str):
    return re.sub(r"\[(\d+)\]", r"(\1)", s)

def compute_beir_metrics(qrels, results, k_values=[1, 10, 100]):
    qrels = {str(qid): {str(docid): rel for docid, rel in qrel.items()} 
              for qid, qrel in qrels.items()}

    results = {str(qid): {str(docid): score for docid, score in run.items()} 
                for qid, run in results.items()}
    retriever = EvaluateRetrieval()
    metrics = retriever.evaluate(qrels, results, k_values)
    return metrics

def extract_scores(response: str, output_passages_num: int):
    match = re.search(r"<score>(.*?)</score>", response, re.DOTALL)
    if not match:
        match = re.search(r"<score>\s*(.*)", response, re.DOTALL)
    if match:
        content = match.group(1)
        # Extract numbers, with or without brackets
        numbers = re.findall(r"\d+", content)
        numbers = list(map(int, numbers))[:output_passages_num]
        if len(numbers) == output_passages_num:
            if all(0 <= x <= 100 for x in numbers):
                return True, numbers
            else:
                # print("Some scores are invalid (not within the range 0–100)!!!!!")
                fixed_numbers = [min(100, max(0, x)) for x in numbers]
                return True, fixed_numbers
        else:
            # print("Incorrect number of digits!!!!!!!!!!!")
            return False, []
    else:
        # print("No <score> tag found!!!!!!!!!!!")
        return False, []
