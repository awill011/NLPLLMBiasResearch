# LLM Bias Research Pipeline

**Measuring demographic bias in large language model outputs through controlled prompt experiments**

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![NLP](https://img.shields.io/badge/NLP-LLM%20Evaluation-purple) ![AI Fairness](https://img.shields.io/badge/Topic-AI%20Fairness-orange)

---

## Research questions

- Do LLMs respond differently when demographic information (race, gender, age) is included in prompts?
- Can prompting strategies like Chain-of-Thought reduce measurable bias in outputs?
- How do OpenAI and HuggingFace models compare in bias patterns across the same prompt conditions?

---

## How it works

| File | Purpose |
|---|---|
| `pipeline1.py` | Generates controlled prompts with varied demographic conditions and queries LLM APIs |
| `pipeline2.py` | Processes raw responses and computes linguistic bias metrics |
| `results.csv` | Structured output of all model responses and computed metrics |


---

## Key metrics computed

- **Response length** — does the model write more or less for different demographic groups?
- **Lexical variation** — how much does word choice shift across conditions?
- **Sentiment consistency** — does tone change based on demographic framing?
- **Cross-condition agreement** — overall behavioral consistency across prompt variants

---

## Models evaluated

- OpenAI GPT-3.5 / GPT-4
- HuggingFace open-source models (varied by condition)

---

*Undergraduate research project · UC Merced · [@awill011](https://github.com/awill011)*
