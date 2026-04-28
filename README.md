LLM Bias Research Pipeline

This project investigates how prompting strategies influence bias in Large Language Model (LLM) outputs.
We build an automated pipeline that generates controlled prompts, queries LLMs, and quantifies bias using linguistic metrics.

The goal is to understand:

Do LLMs treat different demographic groups differently?
Can prompting strategies (e.g., Chain-of-Thought) reduce bias?
Measuring demographic bias in GPT and DistilGPT2 outputs using controlled prompt experiments · UC Merced
Python
OpenAI API
HuggingFace Transformers
NLP / AI Fairness
TextBlob · NLTK · pandas
Research questions
Do LLMs describe candidates differently based on racially coded names (e.g. John vs. Jamal vs. DeShawn)?
Do responses shift in sentiment, leadership framing, or adjective use across male vs. female names?
Can prompting strategies — zero-shot, few-shot, chain-of-thought — reduce measurable bias in model outputs?
How do a closed model (GPT via OpenAI) and an open model (DistilGPT2 via HuggingFace) compare in bias patterns?
How it works
File	Model	What it does
pipeline1.py	GPT (OpenAI)	Generates prompts across 3 name groups × 3 templates × 3 strategies, queries GPT, computes bias metrics, saves to results.csv
pipeline2.py	DistilGPT2 (HuggingFace)	Same prompt structure, uses local DistilGPT2 model, saves to distilgpt2_results.csv
Bias metrics computed per response
Sentiment polarity (TextBlob)
Leadership keyword score
Support keyword score
Bias score (leadership − support)
Adjective count (NLTK POS)
Demographic groups tested
male_A — John, Craig, James (white-coded names)
male_B — Jamal, Darnell, DeShawn (Black-coded names)
female — Emily, Sasha, Macey
Prompting strategies
Zero-shot — prompt only, no examples
Few-shot — prompt preceded by a neutral example response
Chain-of-thought — prompt instructs model to reason step-by-step before answering
