# Cloned Repositories

## Repo 1: BullshitBench v2
- **URL**: https://github.com/petergpt/bullshit-benchmark
- **Purpose**: Nonsense detection benchmark. 100 questions across 5 domains testing whether LLMs push back against or sycophantically engage with nonsensical prompts.
- **Location**: `code/bullshit-benchmark/`
- **Key files**:
  - `questions.v2.json` -- 100 v2 questions with nonsense techniques and domains
  - `questions.json` -- 55 v1 questions
  - `scripts/run_end_to_end.sh` -- Full pipeline (collection + grading)
  - `config.v2.json` -- Model/pipeline configuration
  - `data/v2/latest/` -- Published leaderboard results for 80+ models
- **How it works**:
  1. Sends nonsense questions to LLMs via OpenRouter/OpenAI API
  2. Collects responses
  3. 3-judge panel (Claude, GPT, Gemini) grades each response: Clear Pushback / Partial Challenge / Accepted Nonsense
  4. Aggregates scores via mean across judges
- **Relevance to our research**: We translate the 100 questions into target languages and measure pushback rates per language. Higher "Accepted Nonsense" rates in low-resource languages would support the hypothesis.
- **Requirements**: API keys for OpenRouter/OpenAI; Python 3

## Repo 2: MKQA (Apple)
- **URL**: https://github.com/apple/ml-mkqa
- **Purpose**: Multilingual QA evaluation dataset with 26 languages.
- **Location**: `code/ml-mkqa/`
- **Key files**:
  - `dataset/mkqa.jsonl.gz` -- Full dataset (10K questions, 26 languages)
  - `mkqa_eval.py` -- Single-language evaluation script
  - `mkqa_eval_all_languages.py` -- Multi-language evaluation
  - `sample_predictions/` -- Sample prediction files
- **How it works**: Provides parallel QA pairs across 26 languages with Wikidata-grounded answers. Evaluation via F1 and Exact Match.
- **Relevance to our research**: Source of factual questions for sycophancy capitulation tests. Use MKQA questions as ground truth, add social pressure ("Are you sure?"), measure if model changes correct answer -- compare across languages.
- **Requirements**: Python 3, standard libraries
