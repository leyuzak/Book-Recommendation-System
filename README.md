# 📚 Book Recommendation System

An end-to-end **Book Recommendation System** built with **Python**, **Sentence Transformers**, and **Hugging Face**, using semantic similarity to recommend books based on their descriptions.

This project was developed as part of a **Data Science specialization** and demonstrates how transformer-based embeddings can be used for real-world recommendation systems.

---

## 🚀 Live Demos

* 🔗 **Hugging Face Space**: [https://huggingface.co/spaces/leyuzak/Book-Recommendation-System](https://huggingface.co/spaces/leyuzak/Book-Recommendation-System)
* 📓 **Kaggle Notebook**: [https://www.kaggle.com/code/leyuzakoksoken/book-recommendation-system](https://www.kaggle.com/code/leyuzakoksoken/book-recommendation-system)

---

## 🧠 How It Works

1. Book descriptions are converted into dense vector embeddings using a **Sentence-BERT** model.
2. User input (book title or description) is embedded using the same model.
3. **Cosine similarity** is calculated between the user query and all books.
4. The most semantically similar books are returned as recommendations.

This approach allows recommendations based on **meaning**, not just keywords.

---

## 🗂️ Project Structure

```
Book-Recommendation-System/
│
├── app.py                          # Main application script (inference & interface)
├── books.csv                       # Dataset of books
├── book-recommendation-system.ipynb# Model development & experimentation
├── requirements.txt                # Project dependencies
│
├── model.safetensors               # Trained Sentence-BERT model weights
├── config.json                     # Model configuration
├── sentence_bert_config.json       # Sentence-BERT specific configuration
├── config_sentence_transformers.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
└── modules.json
```

---

## 📊 Dataset

* Source: Custom / Kaggle-based book dataset
* Format: CSV
* Features include:

  * Book title
  * Author
  * Description

The dataset is stored as **`books.csv`** and is used to generate embeddings for recommendation.

---

## 🧪 Model

* **Model Type**: Sentence-BERT (via `sentence-transformers`)
* **Embedding Strategy**: Semantic text embeddings
* **Similarity Metric**: Cosine Similarity

The trained model and tokenizer files are included to allow **offline inference without retraining**.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Book-Recommendation-System.git
cd Book-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application locally:

```bash
python app.py
```

---

## 💡 Example Use Case

* Discover books similar to your favorite novel
* Recommendation based on themes, genre, and writing style
* Keyword-free, meaning-aware search

---

## 📌 Key Features

* Transformer-based semantic recommendations
* Lightweight and fast inference
* Deployed on Hugging Face Spaces
* Reproducible and beginner-friendly

---

## 🔮 Future Improvements

* User-based collaborative filtering
* Genre and rating-aware recommendations
* Advanced UI with filters and sorting
* Larger and more diverse dataset

---

## 👩‍💻 Author

**Leyuza Köksöken**
Data Science & Machine Learning Enthusiast

* Kaggle: [https://www.kaggle.com/leyuzakoksoken](https://www.kaggle.com/leyuzakoksoken)
* Hugging Face: [https://huggingface.co/leyuzak](https://huggingface.co/leyuzak)

