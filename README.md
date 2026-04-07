# Sycophantic Tendencies Vary with Language Resource Level

First systematic study of how LLM sycophancy varies across languages of different resource levels. Tests GPT-4.1 and GPT-4o-mini across 7 languages (English, French, Arabic, Hindi, Swahili, Yoruba, Tagalog) using multilingual BullshitBench and MKQA-based capitulation tests.

## Key Findings

- **Nonsense engagement scales inversely with resource level**: GPT-4o-mini shows Spearman rho = -0.786 (p = 0.036) — models engage with fabricated premises significantly more in low-resource languages (99% in Yoruba vs 87% in English)
- **Capitulation under pressure is worse for smaller models in low-resource languages**: GPT-4o-mini capitulates 2.4% in English vs 20.8% in Yoruba; GPT-4.1 is robust across languages (~4-12%)
- **"Adopted wrong answer" rate is dramatic**: In Yoruba, GPT-4o-mini echoes the user's suggested wrong answer 79% of the time vs 10% in English
- **Model scale partially mitigates the effect**: GPT-4.1 shows the gradient in engagement but resists explicit capitulation

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install openai numpy scipy matplotlib pandas seaborn tqdm

# Run experiments (requires OPENAI_API_KEY)
python src/translate_bullshitbench.py    # Translate BullshitBench to 7 languages
python src/prepare_mkqa.py              # Prepare MKQA capitulation dataset
python src/experiment1_bullshitbench.py  # Experiment 1: nonsense engagement
python src/experiment2_capitulation.py   # Experiment 2: capitulation under pressure
python src/analysis.py                  # Statistical analysis and figures
```

## File Structure

```
src/                          # Experiment scripts
  translate_bullshitbench.py  # BullshitBench translation pipeline
  prepare_mkqa.py             # MKQA dataset preparation
  experiment1_bullshitbench.py # Nonsense engagement experiment
  experiment2_capitulation.py  # Capitulation under pressure experiment
  analysis.py                 # Statistical analysis and visualization
datasets/                     # Input datasets (BullshitBench v2, MKQA)
results/                      # Raw experiment results (JSON)
figures/                      # Generated visualizations
planning.md                   # Research plan
REPORT.md                     # Full research report with results
```

See [REPORT.md](REPORT.md) for full methodology, results, and analysis.
