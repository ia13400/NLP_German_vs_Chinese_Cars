
# Car Reviews Aspect Classification using Custom NER

## Overview

This project presents a complete Natural Language Processing (NLP) pipeline for automatically analyzing vehicle reviews collected from the CarWale website. The system combines web scraping, Named Entity Recognition (NER), and machine learning to identify the main aspect discussed in each review.

Instead of relying only on traditional text vectorization methods, this project introduces a custom NER model to extract domain-specific automotive entities. These extracted entities are then transformed into numerical features and used for aspect classification.

The project was developed as part of the Master's course in Natural Language Processing.

---

## Project Workflow

```
CarWale Reviews
        │
        ▼
Web Scraping
        │
        ▼
Manual Annotation
        │
        ▼
Custom spaCy NER Training
        │
        ▼
Entity Extraction
        │
        ▼
Feature Generation
(Entity Counts)
        │
        ▼
Machine Learning Classification
        │
        ▼
Prediction on New Reviews
        │
        ▼
Visualization & Analysis
```

---

## Features

- Web scraping of vehicle reviews from CarWale
- Automatic extraction of:
  - Vehicle brand
  - Vehicle model
  - Review title
  - Review text
  - Rating
- Custom Named Entity Recognition (spaCy)
- Domain-specific entity labels
- Feature extraction based on detected entities
- Aspect classification using multiple ML algorithms
- Visualization of review statistics
- Prediction on unseen reviews

---

## Project Structure

```
project/
│
├── data/
│   ├── cars_reviews.json
│   ├── cars_reviews.txt
│   ├── cars_reviews_ner_inline.txt
│
├── models/
│   ├── my_ner_model/
│   └── car_reviews_classifier.pkl
│
├── notebooks/
│   ├── 01_Download_Data.ipynb
│   ├── 02_NER_Train_Model.ipynb
│   ├── 03_Classification.ipynb
│   └── 04_Test_New_Data.ipynb
│
└── README.md
```

---

## Notebook Description

### 01_Download_Data.ipynb

Downloads vehicle reviews from CarWale.

Functions:

- Crawl vehicle brands
- Extract review pages
- Download reviews
- Save dataset as JSON

---

### 02_NER_Train_Model.ipynb

Creates and trains a custom Named Entity Recognition model.

Steps:

- Load annotated reviews
- Clean text
- Convert inline annotations
- Train spaCy NER model
- Save trained model

---

### 03_Classification.ipynb

Builds the aspect classification model.

Pipeline:

- Load NER model
- Extract entities
- Generate entity-count features
- Train classifiers
- Evaluate performance
- Save best model

Models evaluated:

- Logistic Regression
- Random Forest
- Support Vector Classifier (SVC)

Evaluation metric:

- 5-Fold Cross Validation
- Macro F1-Score

Results:

| Model | Macro F1 |
|--------|---------:|
| Logistic Regression | **0.610** |
| Random Forest | 0.581 |
| SVC | 0.443 |

The Logistic Regression model achieved the highest Macro F1-score and was selected as the final classifier.

---

### 04_Test_New_Data.ipynb

Tests the trained models on previously unseen vehicle reviews.

Workflow:

- Download new reviews
- Apply custom NER
- Generate features
- Predict review aspect
- Visualize results

---

## Named Entity Labels

The custom NER model recognizes automotive-specific entities, including:

- ENGINE
- PERFORMANCE
- COMFORT
- HANDLING
- DESIGN
- BUILD_QUALITY
- PRICE
- SERVICE
- SAFETY
- BATTERY
- RANGE
- PROBLEM

---

## Aspect Classes

Each review is classified into one of the predefined aspect categories.

Example categories include:

- Performance
- Comfort
- Design
- Safety
- Price
- Service
- Battery
- Build Quality

---

## Technologies

- Python
- spaCy
- BeautifulSoup
- Requests
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib


---

## How to Run

Run the notebooks in the following order:

1. 01_Download_Data.ipynb
2. 02_NER_Train_Model.ipynb
3. 03_Classification.ipynb
4. 04_Test_New_Data.ipynb

---

## Example Output

Input review:

> "The engine is powerful, but the seats are uncomfortable."

Detected entities:

```
ENGINE
COMFORT
```

Predicted aspect:

```
Performance
```

---

