
# Arabic Text Preprocessing, Named Entity Recognition (NER), and Embedding

This Python script performs **text preprocessing**, **Named Entity Recognition (NER)**, and **embedding generation** for Arabic text. It uses libraries like `nltk`, `transformers`, and `gensim` to process input text and extract meaningful entities for further analysis.

## Table of Contents
- [Requirements](#requirements)
- [Features](#features)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

---

## Requirements
To run the code, ensure you have the following dependencies installed:
- Python 3.x
- `transformers` (for NER using Hugging Face models)
- `nltk` (for stopwords and tokenization)
- `gensim` (for word embeddings)
- `scikit-learn` (for PCA)
- `scipy` (for cosine similarity)

Install the required packages:
```bash
pip install nltk transformers gensim scikit-learn scipy
```

Additionally, download the Arabic stopwords for `nltk`:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Features
1. Text Preprocessing:
   - Removes Arabic stopwords.
   - Cleans Arabic diacritics.
   - Normalizes Arabic text (e.g., replacing "أ", "إ", and "آ" with "ا").
   - Removes punctuation and special characters.

2. Named Entity Recognition (NER):
   - Utilizes the pre-trained Hugging Face model `hatmimoha/arabic-ner`.
   - Extracts entities and their types from Arabic text.
   - Cleans tokenized results for better readability.

3. Embedding Creation:
   - Converts recognized entities into word embeddings using `gensim`'s Word2Vec.
   - Performs dimensionality reduction using Principal Component Analysis (PCA).

---

## Code Overview

### 1. Text Preprocessing (`text_preprocessing_arabic`)
This function removes stopwords, diacritics, punctuation, and performs Arabic text normalization. It prepares the input text for downstream NER tasks.

```python
def text_preprocessing_arabic(text):
    ...
    return cleaned_text
```

### 2. Named Entity Recognition (`NER`)
This function uses Hugging Face's `pipeline` with a pre-trained Arabic NER model to extract entities and their categories.

```python
def NER(text):
    ...
    return my_entities
```

### 3. Full NER Pipeline (`fullNER`)
This function integrates text preprocessing and NER, returning cleaned and labeled entities.

```python
def fullNER(text):
    ...
    return text
```

### 4. Embedding Generation (`Embedding`)
This function:
- Extracts entities from the NER output.
- Converts entities into word embeddings using `Word2Vec`.
- Reduces the dimensionality of embeddings using PCA.

```python
def Embedding(postNER):
    ...
```

---

## Usage
### Example Workflow
1. Preprocess Arabic text.
2. Perform Named Entity Recognition.
3. Generate embeddings for recognized entities.

```python
# Input Arabic text
text = "الرئيس اللبناني يعقد اجتماعاً لمناقشة التحديات الاقتصادية."

# Full NER pipeline
postNER = fullNER(text)

# Generate embeddings for recognized entities
Embedding(postNER)
```

### Output Example
The script will output:
- A list of extracted entities with their categories.
- Word vectors for the entities.
- Reduced vectors after applying PCA.

---

## Customization
1. Adding/Removing Stopwords:
   Modify the `arabic_stopwords` set to include or exclude additional stopwords:
   ```python
   arabic_stopwords = set(stopwords.words('arabic')) | {'غيرها', 'هي', 'هو'}
   ```

2. Changing the NER Model:
   Replace `"hatmimoha/arabic-ner"` with another pre-trained model if needed.

3. Adjusting Word2Vec Parameters:
   Modify parameters like `vector_size`, `window`, or `min_count` in the `Word2Vec` function for different embeddings:
   ```python
   model = Word2Vec([entities], vector_size=200, window=3, min_count=2, workers=4)
   ```

---

## License
This project is open-source and can be used for educational or research purposes.
