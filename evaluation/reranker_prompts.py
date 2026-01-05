RELEVANCE_DEFINITIONS = {
    # BRIGHT benchmark
    "theoremqa_questions": "Given a query (math problem) and a set of documents (math problem solutions), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query.",
    "theoremqa_theorems": "Given a query (math problem) and a set of documents (math-related passages), the document is relevant to the query if the theorem described in the document can provide helpful insights for solving the problem in the query.",
    "biology": "Given a query (biology post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "earth_science": "Given a query (earth science post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "economics": "Given a query (economics post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "psychology": "Given a query (psychology post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "robotics": "Given a query (robotics post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "stackoverflow": "Given a query (Stack Overflow post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "sustainable_living": "Given a query (sustainable living post) and a set of documents (passages), the document is relevant to the query if the critical concepts or theories discussed in the document can provide helpful insights for domain experts to answer the query.",
    "leetcode": "Given a query (LeetCode problem) and a set of documents (coding problem solutions), the document is relevant to the query if the underlying algorithms or data structures used in the document can provide helpful insights for solving the problem in the query.",
    "pony": "Given a query (Pony coding instruction) and a set of documents (Pony documentation passages), the document is relevant to the query if the Pony syntax described in the document is necessary for beginners with no prior knowledge of Pony to complete the coding instruction in the query.",
    "aops": "Given a query (math problem) and a set of documents (math problem solutions), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query.",
    "theoremqa_questions": "Given a query (math problem) and a set of documents (math problem solutions), the document is relevant to the query if the theorems used in the document can provide helpful insights for solving the problem in the query.",
    "theoremqa_theorems": "Given a query (math problem) and a set of documents (math-related passages), the document is relevant to the query if the theorem described in the document can provide helpful insights for solving the problem in the query.",
    
    # BEIR benchmark
    "trec-covid": "Given a COVID-19 related query and a set of documents (passages), the document is relevant to the query if it answers the query.",
    "dbpedia-entity": "Given a query and a set of documents (passages) describing an entity from DBpedia, the document is relevant to the query if the entity described in the document matches the query.",
    "scifact": "Given a scientific claim query and a set of documents (passages), the document is relevant to the query if it provides evidence supporting or refuting the scientific claim.",
    "nfcorpus": "Given a question and a set of documents (passages), the document is relevant to the query if the document can best answer the question.",
    "signal": "Given a news event or topic query and a set of documents (headlines or summaries), the document is relevant to the query if it reports on, summarizes, or directly relates to the same event or topic.",
    "robust04": "Given an information-need query and a set of documents, the document is relevant to the query if it contains information that satisfies the intent or topic described in the query, even if phrased differently.",
    "news": "Given a query about a contemporary news topic or event and a set of documents (news article from The Washington Post), the document is relevant to the query if it discusses, explains, or provides factual coverage of that event or topic.",
    # R2MED benchmark
    "r2med_Biology": "Given a query (biology question) and a set of documents (passages), the document is relevant to the query if it provides scientific explanations, biological mechanisms, or essential evidence that can help answer the query.",
    "r2med_Bioinformatics": "Given a query (bioinformatics question) and a set of documents (passages), the document is relevant to the query if it provides methods, databases, tools, or evidence that can help answer the query.",
    "r2med_Medical-Sciences": "Given a query (medical question) and a set of documents (passages), the document is relevant to the query if it provides scientific explanations, biological or clinical mechanisms, or evidence that can help answer the query.",
    "r2med_MedXpertQA-Exam": "Given a query (clinical diagnostic question) and a set of documents (passages), the document is relevant to the query if it provides clinical guidelines, evidence, reasoning, or explanations that can help answer the query.",
    "r2med_MedQA-Diag": "Given a query (clinical diagnostic question) and a set of documents (passages), the document is relevant if it provides evidence, clinical reasoning, or explanations that support the identification or confirmation of the correct diagnosis for the patient scenario described in the query.",
    "r2med_PMC-Treatment": "Given a query (clinical case and treatment question) and a set of documents (passages), the document is relevant if it provides evidence, guidelines, or explanations that support appropriate treatment strategies, drug choices, or therapeutic rationale for the patient case described in the query.",
    "r2med_PMC-Clinical": "Given a query (clinical case and diagnostic question) and a set of documents (passages), a document is considered relevant if it provides clinical evidence, case descriptions, or diagnostic reasoning that supports identifying the correct diagnosis for the patient described in the query.",
    "r2med_IIYi-Clinical": "Given a query (clinical case) and a set of documents (passages), the document is relevant if it provides evidence, case descriptions, or diagnostic reasoning that supports identifying the correct diagnosis or understanding the patient’s clinical presentation.",
}

QUERY_MAX_LEN = {
    # BRIGHT benchmark
    "biology": 512,
    "earth_science": 512,
    "economics": 768,
    "psychology": 512,
    "robotics": 768,
    "stackoverflow": 768,
    "sustainable_living": 768,
    "leetcode": 768,
    "pony": 256,
    "aops": 512,
    "theoremqa_questions": 512,
    "theoremqa_theorems": 512,
    # BEIR benchmark
    "trec-covid": 128,
    "dbpedia-entity": 128,
    "scifact": 128,
    "nfcorpus": 128,
    "signal": 128, 
    "robust04": 128, 
    "news": 128,
    # R2MED benchmark
    "r2med_Biology": 512,
    "r2med_Bioinformatics": 768,
    "r2med_Medical-Sciences": 768,
    "r2med_MedXpertQA-Exam": 768,
    "r2med_MedQA-Diag": 512,
    "r2med_PMC-Treatment": 768,
    "r2med_PMC-Clinical": 512,
    "r2med_IIYi-Clinical": 768
}

DOC_MAX_LEN = {
    # BRIGHT benchmark
    "biology": 1024, 
    "earth_science": 1024,
    "economics": 1024,
    "psychology": 1024,
    "robotics": 1024,
    "stackoverflow": 1024,
    "sustainable_living": 1024,
    "leetcode": 1024,
    "pony": 1024,
    "aops": 1024,
    "theoremqa_questions": 1024,
    "theoremqa_theorems": 1024,
    # BEIR benchmark
    "trec-covid": 1024,
    "dbpedia-entity": 1024,
    "scifact": 1024,
    "nfcorpus": 1024,
    "signal": 256, 
    "robust04": 1024, 
    "news": 1024,
    # R2MED benchmark
    "r2med_Biology": 1024,
    "r2med_Bioinformatics": 1024,
    "r2med_Medical-Sciences": 512,
    "r2med_MedXpertQA-Exam": 512,
    "r2med_MedQA-Diag": 512,
    "r2med_PMC-Treatment": 1024,
    "r2med_PMC-Clinical": 1024,
    "r2med_IIYi-Clinical": 1024
}

WINDOW_SIZE = {
    # BRIGHT benchmark
    "biology": 5, 
    "earth_science": 5,
    "economics": 5,
    "psychology": 5,
    "robotics": 5,
    "stackoverflow": 5,
    "sustainable_living": 5,
    "leetcode": 5,
    "pony": 5,
    "aops": 5,
    "theoremqa_questions": 5,
    "theoremqa_theorems": 5,
    # BEIR benchmark
    "trec-covid": 5,
    "dbpedia-entity": 5,
    "scifact": 5,
    "nfcorpus": 5,
    "signal": 10, 
    "robust04": 5, 
    "news": 5,
    # R2MED benchmark
    "r2med_Biology": 5,
    "r2med_Bioinformatics": 5,
    "r2med_Medical-Sciences": 10,
    "r2med_MedXpertQA-Exam": 10,
    "r2med_MedQA-Diag": 10,
    "r2med_PMC-Treatment": 5,
    "r2med_PMC-Clinical": 5,
    "r2med_IIYi-Clinical": 5
}