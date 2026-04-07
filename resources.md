# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Sycophantic tendencies vary with language resource level." The hypothesis is that sycophantic capitulation in LLMs scales inversely with language resource level.

## Papers
Total papers downloaded: 13

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Towards Understanding Sycophancy in Language Models | Sharma et al. | 2023 | papers/2310.13548_sycophancy_understanding.pdf | Foundational sycophancy measurement; 4 sycophancy types |
| 2 | Sycophancy in LLMs: Causes and Mitigations | Malmqvist | 2024 | papers/2411.15287_sycophancy_causes_mitigations.pdf | Survey of causes and mitigations |
| 3 | ELEPHANT: Social Sycophancy in LLMs | Cheng et al. | 2025 | papers/2505.13995_elephant_social_sycophancy.pdf | Social sycophancy; face-preserving behavior |
| 4 | Machine Bullshit | Liang et al. | 2025 | papers/2507.07484_machine_bullshit.pdf | Bullshit Index metric; sycophancy as bullshit subtype |
| 5 | Language Ranker | Li et al. | 2024 | papers/2404.11553_language_ranker.pdf | Quantifying language resource level via embeddings |
| 6 | MMLU-ProX | Xuan et al. | 2025 | papers/2503.10497_mmlu_prox.pdf | 29-language benchmark |
| 7 | Aya 23 | Aryabumi et al. | 2024 | papers/2405.15032_aya23.pdf | Open multilingual model, 23 languages |
| 8 | MKQA | Longpre et al. | 2020 | papers/2007.15207_mkqa.pdf | 10K QA pairs, 26 languages, parallel |
| 9 | The Language Barrier | Shen et al. | 2024 | papers/2401.13136_language_barrier_safety.pdf | Safety failures in low-resource languages |
| 10 | Multilingual LLM Safety Survey | Yong et al. | 2025 | papers/2505.24119_multilingual_safety_survey.pdf | Comprehensive multilingual safety survey |
| 11 | Multilingual Jailbreak Challenges | Deng et al. | 2023 | papers/2310.06474_multilingual_jailbreak.pdf | 3x higher harmful content in low-resource |
| 12 | Low-Resource Languages Jailbreak GPT-4 | Yong et al. | 2023 | papers/2310.02446_low_resource_jailbreak.pdf | 79% LRL bypass vs 11% HRL bypass |
| 13 | Multilingual Blending | Song et al. | 2024 | papers/2407.07342_multilingual_blending.pdf | Language mixing bypasses safety |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MKQA | Apple/GitHub | 10K QA x 26 langs | Multilingual QA | datasets/mkqa.jsonl.gz | 12MB compressed, parallel questions |
| BullshitBench v2 | petergpt/GitHub | 100 questions | Nonsense detection | datasets/bullshitbench_v2_questions.json | English-only, needs translation |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| BullshitBench | github.com/petergpt/bullshit-benchmark | Nonsense detection benchmark | code/bullshit-benchmark/ | Full pipeline with 3-judge grading |
| MKQA | github.com/apple/ml-mkqa | Multilingual QA dataset + eval | code/ml-mkqa/ | Includes evaluation scripts |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (unavailable) then web search for academic papers
2. Searched across 5 topic areas: sycophancy in LLMs, BullshitBench, multilingual LLM evaluation, MKQA, alignment failures across languages
3. Focused on 2023-2025 papers for state-of-the-art coverage
4. Prioritized papers with code/data availability

### Selection Criteria
- Papers directly measuring sycophancy or closely related phenomena (bullshit, capitulation)
- Papers demonstrating performance gaps between high and low-resource languages
- Papers providing methodology for quantifying language resource levels
- Datasets with parallel multilingual coverage for fair cross-lingual comparison
- Code repositories with evaluation infrastructure we can adapt

### Challenges Encountered
- Paper-finder service was not running; used web search as fallback
- BullshitBench has no formal academic paper (GitHub repo only)
- No existing work directly measures sycophancy across language resource levels -- this is the research gap

### Gaps and Workarounds
- **No multilingual sycophancy dataset exists** -- will need to create by translating BullshitBench and adapting MKQA questions with social pressure prompts
- **Language resource level taxonomy** -- can use Joshi et al. (2020) classification or pre-training data proportions from Language Ranker paper
- **Translation quality** -- need to validate translations; consider using professional translators or high-quality MT with back-translation verification

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **BullshitBench v2 (translated)**: Translate 100 nonsense questions into 7 languages spanning resource levels. Compare pushback rates.
- **MKQA-based capitulation test**: Use MKQA factual questions, add "are you sure?" pressure prompts, measure answer flip rates per language.

### 2. Baseline Methods
- English capitulation rate (reference)
- Per-language accuracy without pressure (factual knowledge control)
- Compare across 3+ models (GPT-4, Claude, open-source multilingual model like Aya)

### 3. Evaluation Metrics
- Capitulation rate per language (primary)
- Spearman correlation: resource level rank vs. capitulation rate
- Nonsense engagement rate per language (BullshitBench)
- Accuracy delta (baseline - pressured) per language

### 4. Code to Adapt/Reuse
- BullshitBench grading pipeline (3-judge panel) -- adapt for multilingual
- MKQA evaluation scripts (F1, EM) -- use as factual accuracy baseline
- Translation pipeline from Shen et al. (NLLB-1.3B) or Google Translate API

### 5. Suggested Language Selection
| Resource Level | Languages | Rationale |
|---------------|-----------|-----------|
| High | English, Chinese, French | >0.1% of LLaMA pre-training data |
| Mid | Arabic, Korean, Turkish | Present in MKQA, moderate resources |
| Low | Khmer, Thai, Finnish | In MKQA, low pre-training data (<0.05%) |

Alternative low-resource options (not in MKQA): Zulu, Hausa, Igbo (from safety papers, would need separate QA data).
