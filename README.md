# Natural Language Processing with Hugging Face Transformers

**Name:** Vinsensius Fendy Kurniawan

Institut Teknologi Batam

---

## 📌 Project Overview

This project introduces key Natural Language Processing (NLP) tasks using Hugging Face Transformers. The `pipeline()` function makes it easy to apply pre-trained transformer models to real-world scenarios such as sentiment analysis, topic classification, text generation, masked language modeling, named entity recognition (NER), question answering, text summarization, and translation.

---

## ⚙️ Required Libraries and Setup

Before running the code examples, make sure the required libraries are installed:

```bash
pip install transformers datasets evaluate [sentencepiece]
```

Optional:

```bash
pip install torch 
```

These libraries enable access to Hugging Face models (`transformers`), text datasets (`datasets`), and additional model evaluation tools.

### Python Imports

```python
# Recommended imports
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline
```

These imports allow us to use different task-specific pipelines provided by Hugging Face with pre-trained models.

---

## 🧪 Exercises and Results

### 1. Sentiment Analysis

```python
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I can't believe I wasted my time on this. Totally disappointing.")
```

**Result:** Negative (0.9998)

**Analysis:**
This model detects emotions in text. In this example, the sentence expresses frustration, which the model identifies with high accuracy as negative sentiment. It's useful for analyzing reviews, feedback, or social media posts.

---

### 2. Topic Classification

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "This module focuses on understanding the distribution of data, detecting outliers, and visualizing relationships between variables to prepare for further statistical modeling.",
    candidate_labels=["art", "natural science", "data analysis"]
)
```

**Result:** Data analysis (0.9429)

**Analysis:**
Using zero-shot learning, the model correctly associates the paragraph with the "data analysis" topic. Even without prior training on this specific sentence, it can still make accurate predictions—very handy when labeled data is limited.

---

### 3. Text Generation (Causal LM)

```python
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "This course will burn your brain",
    max_length=32,
    num_return_sequences=3
)
```

**Result:** Three unique sentence continuations.

**Analysis:**
This generator expands the input prompt into plausible next sentences. It's helpful for creative writing, chatbot responses, and generating content ideas.

---

### 3.5. Masked Language Modeling

```python
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("This magical <mask> can only be activated by those with a pure heart.", top_k=4)
```

**Result:** Top predictions: ability, power, energy, aura

**Analysis:**
Masked language modeling predicts missing words. It’s useful for tasks like text correction, auto-completion, or training domain-specific language models.

---

### 4. Named Entity Recognition (NER)

```python
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("Vinsen defeated the Dragon Lord in the Great War of Armageddon.")
```

**Result:** Entities: Vinsen (PER), Dragon Lord (PER), Great War of Armageddon (MISC)

**Analysis:**
NER identifies named entities like persons, organizations, and events. This is key for information extraction in documents, news articles, and historical data.

---

### 5. Question Answering

```python
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Who is the main character of the story?"
context = "Vinsen, a brilliant but misunderstood student, was transported to another world where he awakened as the Demon Lord."
qa_model(question=question, context=context)
```

**Result:** "Vinsen"

**Analysis:**
QA models extract direct answers from a given context. They’re useful in search systems, chatbots, or any task where users ask natural-language questions.

---

### 6. Text Summarization

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
In the forgotten realm of Azareth, a land once blessed by ancient dragons and celestial magic, darkness began to rise again after a thousand years of peace. Vinsen, a mere librarian from the Kingdom of Elaria, stumbled upon a forbidden tome that revealed his destiny as the chosen heir of the Demon Lord. With nothing but a rusted sword and a mysterious crest glowing on his hand, he set out on a perilous journey through cursed forests, floating islands, and ruined cities, seeking allies and unlocking sealed powers within himself. As the shadow of the resurrected Dragon King loomed over the world, Vinsen would have to decide whether to embrace the chaos of his lineage or forge a new path using his own will and the ancient magic he was never meant to wield.
"""
)
```

**Result:** Vinsen, a librarian from Elaria, discovers he is the heir of the Demon Lord and sets out on a dangerous journey.

**Analysis:**
The summarizer condenses a long fantasy narrative into a shorter version without losing the main story arc. Great for processing news, articles, and stories quickly.

---

### 7. Translation

```python
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Sekarang adalah hari Minggu.")
```

**Result:** "C'est dimanche."

**Analysis:**
This model handles translation from Indonesian to French accurately and fluently. It’s useful for multilingual apps, global content, or travel tech.

---

## 🧠 Project Summary

This guided project gives a hands-on introduction to NLP using Hugging Face’s transformers. It demonstrates how to apply powerful models with just a few lines of code using pipelines. Each task shows how these models can solve real-world problems like understanding emotions, answering questions, generating content, and translating text.

**Technologies Used:**

* Hugging Face Transformers
* Pretrained models (BERT, GPT-2, DistilBART, RoBERTa, T5)
* Python

---

✅ **Completed by:** Vinsensius Fendy K
🗓️ **Tools:** Hugging Face Transformers, Cognitive Class Platform
