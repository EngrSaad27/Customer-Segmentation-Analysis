#  📚 E-Commerce Customer Segmentation and Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![VSCode](https://img.shields.io/badge/VSCode%20-Enabled-76B900.svg)](https://code.visualstudio.com/download)

<p align="center">
  <img src="app front.PNG" alt="Project Banner" width="800"/>
</p>

## 🌟 Project Description

This project aims to enhance marketing strategies and customer retention for an e-commerce company by gaining a deeper understanding of their customer base based on their purchasing pattern.The goal is to develop a robust customer segmentation model and a predictive classifier to categorize customers based on their purchasing patterns, enabling the company to tailor marketing strategies, improve customer retention, and optimize inventory management.

## 🎯 Features
- 📊 Detail (Recency Frequency Monetary) RFM Analysis
- ⚙️ Feature Engineering
- 📈 Analyzing Different Models

## 🛠️ Technical Architecture

### Component Stack
```mermaid
graph TD
    A[CSV File] --> B[Data Loading]
    B --> C[Exploratory Data Analysis]
    C --> D[Data Preprocessing]
    D --> E[RFM Analysis]
    E --> F[Feature Scaling]
    F --> G[Dimentionality Reduction]
    G --> H[Clustering Models Training]
    H --> I[Evaluating Silhoutte Score]
    I --> J[Customer Segmentation Visualization]
```

## Installation & Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# For Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage Guide

### Application Launch
```bash
streamlit run app.py
```


2. Navigate to the project directory: `cd project-name`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the project: `python main.py`

## Technologies Used
- Python
- Pandas
- Scikit-learn

## Future Enhancements
- Add feature X
- Improve performance

## Contact
Feel free to reach out: [Email](mailto:your_email@example.com) | [LinkedIn](your_linkedin_url)
