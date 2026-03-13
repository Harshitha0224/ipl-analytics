# 🏏 IPL Analytics & Win Predictor

> An end-to-end Data Science project analyzing 13 seasons of IPL cricket data and predicting match winners using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Scikit-learn](https://img.shields.io/badge/ML-RandomForest-orange)
![Pandas](https://img.shields.io/badge/Pandas-DataAnalysis-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey)

---

## 🌐 Live Demo
👉 **[Click here to open the dashboard](YOUR_STREAMLIT_LINK_HERE)**

---

## 📌 Project Overview

This project covers the complete Data Science lifecycle — from raw data to a deployed interactive web application — using real IPL ball-by-ball match data spanning 2008 to 2024.

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib & Seaborn | Data visualizations |
| Scikit-learn | Machine learning |
| SQLite | Storing predictions |
| Streamlit | Interactive web dashboard |

---

## 📁 Project Structure
```
ipl-analytics/
│
├── data/
│   ├── raw/                  
│   └── processed/            
│
├── notebooks/
│   ├── 01_data_exploration   
│   ├── 02_data_cleaning      
│   ├── 03_visualizations     
│   └── 04_ml_model           
│
├── src/
│   ├── database.py           
│   └── predictor.py          
│
├── dashboard/
│   └── app.py                
│
├── requirements.txt          
└── README.md                 
```

---

## 📊 Key Insights

| Insight | Finding |
|---|---|
| Most successful team | Mumbai Indians (144 wins) |
| Toss advantage | Only 51% — barely matters! |
| Average 1st innings score | 165.5 runs |
| Teams prefer after toss | Field (60%) over Bat (40%) |
| Most important predictor | Target score |
| Least important predictor | Toss decision |

---

## 🤖 Machine Learning

### Problem
Binary Classification — predict if chasing team wins (1) or loses (0)

### Models Compared

| Model | Train Accuracy | Val Accuracy | Status |
|---|---|---|---|
| Logistic Regression | 59.33% | 56.42% | Baseline |
| Random Forest (overfit) | 98% | 61% | Overfit |
| Random Forest (tuned) | 65% | 59.63% | Best ✅ |

### Best Model
- Algorithm: Random Forest Classifier
- Trees: 200
- Max Depth: 6
- Validation Accuracy: 59.63%
- Improvement over random: +9.63%

### Features Used
- Teams (batting first and chasing)
- Venue
- Toss winner and decision
- Target score

---

## 📈 Dashboard Pages

| Page | Description |
|---|---|
| 📊 Analytics | Team wins, top batsmen, runs per season, over heatmap |
| 🤖 Win Predictor | Enter match details and get winner prediction with probability |
| 📋 Prediction History | All past predictions stored in SQLite database |
| 📈 Team Stats | Win record, season trends and head to head for any team |

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ipl-analytics.git
cd ipl-analytics
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download dataset**

Download from 👉 [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

Place `matches.csv` and `deliveries.csv` inside `data/raw/`

**4. Run notebooks in order**
```
01_data_exploration → 02_data_cleaning → 03_visualizations → 04_ml_model
```

**5. Launch dashboard**
```bash
streamlit run dashboard/app.py
```

---

## 🎯 Skills Demonstrated

- Exploratory Data Analysis on 260,000+ records
- Data cleaning and preprocessing pipelines
- Feature engineering from raw cricket data
- Overfitting and underfitting detection and resolution
- Cross validation and hyperparameter tuning
- SQL database design and integration
- End to end ML pipeline from raw data to prediction
- Streamlit web app development and cloud deployment

---