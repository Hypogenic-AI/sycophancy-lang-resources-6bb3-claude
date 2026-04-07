# Downloaded Datasets

This directory contains datasets for the research project. Large data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: MKQA (Multilingual Knowledge Questions and Answers)

### Overview
- **Source**: https://github.com/apple/ml-mkqa
- **Size**: 10,000 question-answer pairs x 26 languages = 260,000 total
- **Format**: JSONL (gzipped)
- **Task**: Open-domain question answering
- **Languages**: ar, da, de, en, es, fi, fr, he, hu, it, ja, ko, km, ms, nl, no, pl, pt, ru, sv, th, tr, vi, zh_cn, zh_hk, zh_tw
- **License**: CC BY-SA 3.0

### Download Instructions

**From cloned repo (already available):**
```bash
cp code/ml-mkqa/dataset/mkqa.jsonl.gz datasets/
```

**Alternative (direct download):**
```bash
curl -sL https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz -o datasets/mkqa.jsonl.gz
```

### Loading the Dataset

```python
import gzip
import json

with gzip.open("datasets/mkqa.jsonl.gz", "rt") as f:
    data = [json.loads(line) for line in f]

# Each entry has: query, queries (dict by language), answers (dict by language), example_id
# Access specific language:
for item in data[:5]:
    print(item["queries"]["en"], "->", item["answers"]["en"])
```

### Sample Data
See `datasets/mkqa_sample.json` for 3 example entries with all 26 languages.

### Notes
- Questions are parallel (same content across all languages via human translation)
- Answers are language-independent (linked to Wikidata entity IDs)
- Answer types: entity (4221), long_answer (1815), unanswerable (1427), date (1174), number (485), number_with_unit (394), short_phrase (346), binary (138)
- Includes low-resource languages (Khmer, Malay) useful for our resource-level analysis

---

## Dataset 2: BullshitBench v2 Questions

### Overview
- **Source**: https://github.com/petergpt/bullshit-benchmark
- **Size**: 100 nonsense questions
- **Format**: JSON
- **Task**: Nonsense detection / bullshit pushback
- **Domains**: software (40), finance (15), legal (15), medical (15), physics (15)
- **Techniques**: 13 nonsense techniques (e.g., plausible_nonexistent_framework, misapplied_mechanism)
- **License**: MIT

### Download Instructions

**From cloned repo (already available):**
```bash
python3 -c "
import json
with open('code/bullshit-benchmark/questions.v2.json') as f:
    d = json.load(f)
questions = []
for tech in d.get('techniques', []):
    for q in tech.get('questions', []):
        questions.append({
            'id': q['id'],
            'question': q['question'],
            'nonsensical_element': q.get('nonsensical_element', ''),
            'domain': q.get('domain', ''),
            'technique': tech['technique']
        })
with open('datasets/bullshitbench_v2_questions.json', 'w') as f:
    json.dump(questions, f, indent=2)
"
```

### Loading the Dataset

```python
import json
with open("datasets/bullshitbench_v2_questions.json") as f:
    questions = json.load(f)
# Each entry: id, question, nonsensical_element, domain, technique
```

### Notes
- Questions are English-only; experiment requires translating to target languages
- Each question contains a fabricated/nonsensical element that models should detect
- Scoring: 0 = accepted nonsense, 1 = partial recognition, 2 = clear pushback
- BullshitBench uses a 3-judge panel (Claude, GPT, Gemini) for evaluation
- For our experiment: translate questions, query LLMs in each language, compare pushback rates
