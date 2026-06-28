# 🎓 Student Score Prediction

A Machine Learning web application that predicts a student's mathematics score based on demographic, social, and academic factors. The project demonstrates an end-to-end ML workflow, including data preprocessing, model training, evaluation, and deployment using Flask.

---

## 📌 Project Overview

Student academic performance is influenced by several factors such as parental education, test preparation, lunch type, gender, and previous reading and writing scores.

This project uses supervised machine learning to predict a student's **Math Score** from these input features.

---

## 🚀 Features

- End-to-end Machine Learning pipeline
- Data preprocessing using Scikit-learn Pipelines
- Automatic handling of categorical and numerical features
- Multiple regression models trained and compared
- Best-performing model selected automatically
- Flask web interface for real-time predictions
- Modular and production-ready project structure
- Exception handling and logging

---

## 📂 Project Structure

```text
Student-Score-Prediction/
│
├── artifacts/                 # Saved models and preprocessing objects
├── notebook/                  # Jupyter notebooks for EDA and experiments
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── ...
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── home.html
│   └── index.html
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📊 Dataset

The dataset contains information related to student demographics and academic background.

### Input Features

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

### Target Variable

- Mathematics Score

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- Bootstrap
- Git

---

## 🤖 Machine Learning Workflow

1. Data Ingestion
2. Data Cleaning
3. Feature Engineering
4. Data Transformation
5. Model Training
6. Model Evaluation
7. Model Selection
8. Prediction Pipeline
9. Flask Deployment

---

## 📈 Models Evaluated

Several regression algorithms were trained and compared, including:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- XGBoost Regressor
- CatBoost Regressor

The model with the highest performance (R² Score) was selected for deployment.

---

## ⚙ Installation

Clone the repository.

```bash
git clone https://github.com/your-username/student-score-prediction.git
```

Navigate to the project directory.

```bash
cd student-score-prediction
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

Start the Flask server.

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📋 Sample Prediction

Enter:

- Gender
- Race/Ethnicity
- Parental Education
- Lunch Type
- Test Preparation
- Reading Score
- Writing Score

The application predicts the student's expected **Mathematics Score**.

---

## 📷 Application Preview

You can add screenshots here.

```
images/home.png
images/result.png
```

---

## 📦 Future Improvements

- Docker deployment
- Cloud deployment (AWS/Azure)
- Model monitoring
- REST API support
- User authentication
- Performance dashboard


## 📚 Learning Outcomes

This project demonstrates:

- End-to-end Machine Learning development
- Data preprocessing pipelines
- Model evaluation and selection
- Production-ready code organization
- Flask integration
- Model serialization using Pickle

