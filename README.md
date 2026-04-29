 Aura Gastronomica

![Python](https://img.shields.io/badge/Python-3.x-171717?style=flat-square&logo=python&logoColor=38B2AC)
![Flask](https://img.shields.io/badge/Flask-Backend-171717?style=flat-square&logo=flask&logoColor=38B2AC)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-ML-171717?style=flat-square&logo=scikit-learn&logoColor=38B2AC)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-UI/UX-171717?style=flat-square&logo=tailwind-css&logoColor=38B2AC)

**Aura Gastronomica** is an intelligent, web-based recommendation engine designed to curate personalized dining experiences in Bangalore. Moving beyond generic "top-10" lists, this project leverages **Natural Language Processing (NLP)** to analyze unstructured customer reviews and quantitative metrics (cuisine, cost, ratings).

By processing these features through a custom TF-IDF vectorization engine, Aura Gastronomica provides highly accurate, taste-matched restaurant recommendations instantly.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Real-Time Engine** | Optimized Flask backend computes cosine similarity across thousands of data points in milliseconds. |
| **Smart Auto-Complete** | Features a dynamic, AJAX-powered search bar with full keyboard navigation (Up/Down/Enter) for seamless data entry. |
| **Advanced NLP Model** | Powered by Scikit-Learn's `TfidfVectorizer` to extract semantic meaning from reviews, utilizing `NLTK` for precise stop-word filtering. |
| **Premium UI/UX** | A sleek, modern "Light Mode" interface built with **Tailwind CSS**, featuring glassmorphism and custom-styled data dashboards. |
| **Pre-Trained Efficiency** | The heavy mathematical matrix is pre-compiled into a `.pkl` file, ensuring zero-lag user queries in production. |

---

## 📂 Project Structure

This architecture strictly adheres to the SmartInternz workspace requirements for model training and deployment:

```text
Restaurant_Recommendation_System/
│
├── Dataset/
│   └── Dataset.txt                         # Compressed raw dataset
│
├── Document/
│   └── RESTAURANT_RECOMMENDATION_SYSTEM.docx # Final Project Report
│
├── Model/
│   └── Restaurant_Recommendation_System.ipynb # Backup of analysis notebook
│
├── Flask/
│   ├── templates/
│   │   ├── index.html                      # Premium landing page
│   │   ├── web.html                        # Auto-complete search interface
│   │   └── result.html                     # Formatted data output dashboard
│   │
│   ├── app1.py                             # Main Flask Web Server
│   ├── Restaurant_Recommendation_System.ipynb # Primary Training Script (EDA & Model)
│   │
│   │   
│   ├── zomato.csv                          # Extracted raw dataset
│   ├── restaurant.pkl                      # Trained Model Matrix (Generated)
│   └── restaurant1.csv                     # Cleaned Dataset (Generated)
│
├── requirements.txt               
└── README.md  
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python installed on your system. Install all required dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Jupyter Notebook)
> **Note:** Before booting the web server, you must clean the raw dataset and generate the trained machine learning matrix.

1.  Navigate to the `Flask/` directory and ensure your extracted `zomato.csv` file is present.
2.  Open `Restaurant_Recommendation_System.ipynb` using Jupyter or VS Code.
3.  Execute all cells sequentially to perform **Exploratory Data Analysis (EDA)** and train the model.
4.  Verify that `restaurant.pkl` and `restaurant1.csv` have been generated in your folder.

### 3. Run the Application
Start the local server by executing the main Flask application:
```bash
python app1.py
```

### 4. Access the Interface
Open your web browser and navigate to:
`http://127.0.0.1:5000/`

---

## 🧠 Architectural Workflow

1.  **Dynamic Input:** As the user types, an AJAX request queries the backend to suggest matching restaurant names, ensuring accurate data entry.
2.  **Vectorization & Inference:** Upon submission, the application loads the `restaurant.pkl` file, containing a pre-computed matrix of **TF-IDF** (Term Frequency-Inverse Document Frequency) scores.
3.  **Similarity Scoring:** The system identifies the target restaurant's index and calculates the top 10 matches based on **Cosine Similarity** scores.
4.  **Data Presentation:** The Pandas DataFrame is formatted into a clean, Tailwind-styled HTML dashboard for the user.

---

## 📝 License & Credits

This project was developed for educational purposes as part of the **SmartInternz Applied Data Science Internship Program**.

Developed by 
Soundarya Udagatti