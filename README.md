# -SMS-Spam-Detector-Naive-Bayes-TF-IDF
NLP | SMS spam detection using TF-IDF vectorization + Multinomial Naive Bayes | sklearn | Python
##  What This Project Does

Spam is annoying. This model isn't.

Given any SMS text, the pipeline predicts in milliseconds whether it's a legitimate message or unsolicited junk — with a clean interactive classifier function 

---

##  How It Works

```
Raw SMS Text
     │
     ▼
TF-IDF Vectorization (removes stop words, extracts term weights)
     │
     ▼
Multinomial Naive Bayes Classifier
     │
     ▼
"Ham ✅" or "Spam 🚫"
```

**Why Naive Bayes?**  
It's probabilistic, fast, and surprisingly powerful for text classification — especially on short messages like SMS where word frequency patterns are strong signals.

**Why TF-IDF over Bag of Words?**  
Raw word counts reward common words. TF-IDF penalizes them, so terms like "FREE", "WIN", "CLAIM" get proportionally more weight — exactly what a spam filter needs.

---

##  Known Issue: Class Imbalance

The dataset is imbalanced (~87% Ham, ~13% Spam). This was identified during EDA and directly impacts **recall on the Spam class** — the model may miss some spam messages. This is explicitly noted in the notebook. Potential fixes (SMOTE, class weighting, threshold tuning) are left as extensions.

---

##  EDA Highlights

- Checked null values, data types, and class distribution
- Engineered a `length` feature to analyze message length patterns
- Found that spam messages tend to be **longer** than ham (visible in boxplot)
- Identified the shortest and longest messages in the dataset

---

##  Project Structure

```
 spam-ham-classifier
 ┣  Spam_Ham.ipynb       # Full pipeline: EDA → Model → Prediction
 ┣  README.md
```

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading & manipulation |
| `matplotlib` / `seaborn` | Exploratory visualization |
| `TfidfVectorizer` | Text → numeric feature matrix |
| `MultinomialNB` | Classification model |
| `sklearn.metrics` | Accuracy, confusion matrix, classification report |

---

At the end of the notebook, there's a live prediction function:

```python
predict_message()
# Enter a message to classify spam/ham: Congratulations! You've won a $1000 gift card. Click now.
# → The message is Spam.
```

```python
predict_message()
# Enter a message to classify spam/ham: Hey, are we still meeting at 6?
# → The message is Ham.
```

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 98.23% |
| Testing Accuracy | 97.4% |

*(Check the notebook for the full classification report and confusion matrix.)*

---
## 👩‍💻 Author

Built with curiosity and mild frustration at promotional texts.  
Feel free to fork, improve, or just use it to finally feel in control of your inbox.

---

## 📄 License

MIT — use it, break it, improve it.
