# Mental Health Crisis Detection from Social Media

**Award:** 🏆 Best Paper — International Conference on AI in Digital Growth (ADG-2026), IEEE Computational Intelligence Society, April 2026

**Author:** Saanvi Gupta  
**Affiliation:** Department of Computer Science and Engineering (AI Specialization), Babu Banarasi Das University, Lucknow, India  
**Contact:** gsaanvi08@gmail.com

---

## Overview

This repository documents the methodology and findings of a machine learning framework for detecting mental health crisis triggers from Reddit social media posts. The work proposes a **dual-signal feature extraction approach** combining linguistic and behavioral patterns, evaluated across four classical ML classifiers.

The full paper was presented at ADG-2026 (IEEE Computational Intelligence Society) and awarded **Best Paper**.

---

## Problem Statement

Mental health disorders affect approximately one billion people globally. In countries like India, where the psychiatrist-to-population ratio is approximately 0.3 per 100,000 people, scalable automated screening tools are critical. Traditional approaches rely on self-reporting and clinical interviews, both of which suffer from significant underreporting and access limitations.

This work investigates whether behavioral and linguistic patterns extracted from Reddit posts can reliably detect early mental health crisis triggers — without relying on computationally expensive deep learning architectures.

---

## Dataset

**Source:** Reddit Mental Health Dataset (RMHD)  
**Reference:** Saima R, "Reddit Mental Health Dataset (RMHD)," Kaggle, 2019-2022. [[Link]](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)

| Property | Value |
|---|---|
| Total posts | 27,977 |
| Positive class (mental health subreddits) | ~58% |
| Negative class (general subreddits) | ~42% |
| Subreddits included | r/depression, r/Anxiety, r/SuicideWatch, r/mentalhealth, r/offmychest |
| Post metadata available | timestamp, author, subreddit, score, text |

**Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique) applied during preprocessing to prevent classifier bias toward the majority class.

> **Note:** The exact dataset subset used in the paper is not publicly redistributable. The RMHD source dataset linked above contains the raw data from which this subset was compiled.

---

## Methodology

### Dual-Signal Feature Framework

Features are extracted along two independent dimensions. The key motivation: behavioral patterns carry predictive information **orthogonal** to text content — a user posting exclusively at 3 AM with tripled posting frequency may be at elevated risk regardless of whether their posts explicitly discuss distress.

---

### A. Linguistic Features (4 features)

Extracted from post text after preprocessing (tokenization, lowercasing, URL removal, stop-word removal via NLTK):

| Feature | Description | Motivation |
|---|---|---|
| **First-Person Pronoun Ratio (FPR)** | Proportion of tokens that are first-person singular pronouns (I, me, my, myself, mine) | Increased self-focused language correlates with depressive states |
| **Absolutist Word Density (AWD)** | Proportion of tokens from absolutist vocabulary (always, never, nothing, completely, everyone, no one) | Strongest single linguistic predictor of mental health disorders (Al-Mosaiwi & Johnstone, 2018) |
| **Negative Affect Word Ratio (NAWR)** | Proportion of tokens from LIWC negative affect lexicon | Captures emotional valence of language |
| **Type-Token Ratio (TTR)** | Unique word types / total tokens | Linguistic diversity decreases during depressive episodes |

---

### B. Behavioral Features (4 features)

Derived from post metadata, aggregated over a rolling 30-day window per user:

| Feature | Description | Motivation |
|---|---|---|
| **Late-Night Posting Rate (LNPR)** | Proportion of posts published between 00:00–05:00 local time | Disrupted sleep patterns are a documented symptom of mood disorders |
| **Posting Frequency Variance (PFV)** | Coefficient of variation in daily posting count over 30-day window | Sudden spikes/drops in posting activity precede crisis events |
| **Subreddit Entropy (SE)** | Shannon entropy over distribution of subreddits a user engages with | Lower entropy (fewer communities) may indicate social withdrawal |

---

### C. Preprocessing Pipeline

```
1. Raw text cleaning and normalization
2. Linguistic feature extraction (NLTK + LIWC lexicons)
3. Behavioral feature computation from post metadata
4. Feature concatenation → unified feature vector per post
5. SMOTE for class rebalancing
6. Min-Max normalization across all features
7. Train-test split: 80:20 with stratification
```

---

### D. Classifiers

Four classifiers selected to span a range of model complexities:

| Model | Rationale |
|---|---|
| **Logistic Regression** | Linear baseline; interpretable coefficient weights for feature analysis |
| **Random Forest** | Ensemble; handles non-linear feature interactions; Gini-based feature importance |
| **SVM (RBF kernel)** | Margin-based; suited to moderately high-dimensional feature spaces |
| **XGBoost** | Gradient boosted trees; consistently strong on structured tabular data |

**Hyperparameter tuning:** 5-fold cross-validation with grid search  
**Primary metric:** F1-score (chosen due to class imbalance context)

---

## Results

### Classifier Performance (Held-Out Test Set)

| Classifier | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.781 | 0.779 | 0.780 | 0.782 | 0.841 |
| SVM (RBF) | 0.812 | 0.808 | 0.811 | 0.809 | 0.874 |
| Random Forest | 0.834 | 0.829 | 0.833 | 0.831 | 0.891 |
| **XGBoost** | **0.857** | **0.849** | **0.846** | **0.847** | **0.903** |

**XGBoost achieved the best F1-score of 0.847**, outperforming all baseline classifiers.

---

### Feature Importance (XGBoost — Top 5)

| Rank | Feature | Type | Importance |
|---|---|---|---|
| 1 | Absolutist Word Density (AWD) | Linguistic | 18.4% |
| 2 | Late-Night Posting Rate (LNPR) | Behavioral | 16.1% |
| 3 | Negative Affect Word Ratio (NAWR) | Linguistic | 14.7% |
| 4 | Posting Frequency Variance (PFV) | Behavioral | 13.2% |
| 5 | First-Person Pronoun Ratio (FPR) | Linguistic | 11.9% |

The interleaving of linguistic and behavioral features in the top-5 validates the dual-signal design. A model trained on linguistic features alone achieved F1=0.811; behavioral features alone achieved F1=0.774. The combined model (F1=0.847) demonstrates meaningful complementary contribution from both signal types.

---

### Ablation Study

| Feature Set | F1-Score |
|---|---|
| Linguistic only | 0.811 |
| Behavioral only | 0.774 |
| **Combined (full model)** | **0.847** |

---

## Key Findings

- XGBoost outperforms all baselines, confirming that non-linear ensemble methods better capture cross-feature interactions in mental health signals
- **Absolutist Word Density** is the strongest linguistic predictor — consistent with Al-Mosaiwi & Johnstone (2018), who found absolutist language distinguishes mental health communities from control groups with >90% accuracy
- **Late-Night Posting Rate** is the strongest behavioral predictor — consistent with Mansoor & Ansari (2024), who identified late-night posting as among the strongest behavioral markers preceding crisis events
- The performance gap between SVM and Random Forest is attributed to ensemble models' ability to capture non-linear threshold effects in behavioral features (e.g., extreme posting frequency variance in either direction is predictive, not moderate variance)

---

## Ethical Considerations

| Concern | Position |
|---|---|
| **Privacy** | Dataset contains no PII; no re-identification attempted |
| **False Positives** | System is a first-stage screening tool only — not a diagnostic instrument. No automated action should be taken on model output alone |
| **Algorithmic Bias** | Dataset is predominantly English-language Western content; model may not generalize across cultural/linguistic contexts |
| **Human Oversight** | Any real-world deployment must maintain human-in-the-loop architecture; clinicians retain final decision-making authority |

---

## Limitations

- Dataset confined to Reddit — demographic profile may not be representative of broader population
- Behavioral features rely on post-level metadata rather than true longitudinal user-level tracking
- Binary classification (mental health vs. non-mental health) is a simplification of clinical reality where risk exists on a spectrum

---

## Future Work

- Multi-class models distinguishing depression, anxiety, and suicidal ideation
- Longitudinal user-level tracking for improved behavioral feature precision
- Multilingual and cross-platform dataset construction
- SHAP (SHapley Additive exPlanations) integration for clinical interpretability
- Comparison with transformer-based approaches (BERT, RoBERTa)

---

## Tech Stack

```
Language:     Python 3.x
ML:           Scikit-learn, XGBoost
NLP:          NLTK
Data:         NumPy, Pandas
Resampling:   imbalanced-learn (SMOTE)
```

---

## Citation

If you find this work useful, please cite:

```
Gupta, S. (2026). Detecting Mental Health Crisis Triggers from Social Media Using 
Behavioral and Linguistic Machine Learning Features. International Conference on 
AI in Digital Growth (ADG-2026), IEEE Computational Intelligence Society. 
Best Paper Award.
```

---

## References

1. World Health Organization, "Mental Health," WHO, Geneva, 2022.
2. R. Garg, "Mental Health in India: Issues, Challenges, and Opportunities," Indian Journal of Psychiatry, 2020.
3. M. De Choudhury et al., "Social Media as a Measurement Tool of Depression in Populations," ACM Web Science, 2013.
4. M. A. Mansoor and K. Ansari, "Early Detection of Mental Health Crises through AI-Powered Social Media Analysis," J. Personalized Medicine, 2024.
5. A. Bokolo and G. Liu, "Comparative Study of ML and Transformer Models for Depression Detection," IEEE Access, 2024.
6. N. Ghoshal, "Reddit Mental Health Data," Kaggle Dataset, 2022.
7. M. Al-Mosaiwi and T. Johnstone, "In an Absolute State: Elevated Use of Absolutist Words is a Marker Specific to Anxiety, Depression, and Suicidal Ideation," Clinical Psychological Science, 2018.
