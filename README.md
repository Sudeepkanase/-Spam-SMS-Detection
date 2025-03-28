# -Spam-SMS-Detection
📝 Project Overview
This project implements an SMS Spam Detection system using Natural Language Processing (NLP) and Machine Learning. It classifies messages as Spam or Not Spam (Ham) using a Logistic Regression model trained on a labeled dataset of SMS messages.

📂 Dataset
The dataset used is the SMS Spam Collection Dataset, which contains labeled messages categorized as:

Spam (1): Unwanted promotional or fraudulent messages.

Ham (0): Legitimate messages.

🔍 Features and Methodology
✔ Data Preprocessing:

Convert text to lowercase.

Remove special characters and extra spaces.

Remove stopwords and apply lemmatization.

✔ Feature Extraction:

Convert text to numerical form using TF-IDF (Term Frequency-Inverse Document Frequency).

✔ Model Training & Evaluation:

Train a Logistic Regression model for classification.

Evaluate using accuracy, precision, recall, and F1-score.

✔ User Input Testing:

Users can enter an SMS to check if it is spam or not.

✔ Model Persistence:

Save and load the trained model using Pickle.

🚀 How to Run the Project
1️⃣ Install Dependencies
Ensure you have Python installed, then install the required libraries:

bash
Copy
Edit
pip install pandas numpy nltk scikit-learn pickle-mixin
2️⃣ Download the Dataset
Place the spam.csv dataset in the project folder.

You can download it from this Kaggle link.

3️⃣ Run the Jupyter Notebook or Python Script
Using Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open the SMS_Spam_Detection.ipynb and run all cells.

Using Python Script:

bash
Copy
Edit
python sms_spam_detector.py
🖥️ Example Usage
bash
Copy
Edit
Enter an SMS to check (or type 'exit' to stop): 
"Congratulations! You've won a free vacation. Call now to claim."
Prediction: Spam
bash
Copy
Edit
Enter an SMS to check (or type 'exit' to stop): 
"Hey, are we still meeting for coffee today?"
Prediction: Not Spam
📊 Model Performance
The trained Logistic Regression model achieves:
✅ Accuracy: ~98%
✅ Precision, Recall, and F1-score: High on both Spam & Ham classes

📦 Project Structure
bash
Copy
Edit
📂 SMS_Spam_Detection
│── 📄 README.md  # Project documentation

│── 📄 spam.csv  # Dataset file

│── 📄 sms_spam_detector.py  # Main Python script

│── 📄 SMS_Spam_Detection.ipynb  # Jupyter Notebook with detailed steps

│── 📄 spam_classifier.pkl  # Saved ML model

│── 📄 tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer

🔗 References
SMS Spam Dataset: Kaggle Dataset

Scikit-learn: Documentation

NLTK: Documentation
❗ Limitations
While this spam detection model performs well, it has some limitations:
🔴 Limited Generalization: The model is trained on a specific dataset and may not generalize well to other types of spam messages (e.g., WhatsApp, emails, or social media).

📌 Future Improvements
✅ Try more models (Random Forest, SVM, Deep Learning)
✅ Enhance preprocessing (handling emojis, slang words, etc.)
✅ Deploy as a web app using Flask or Streamlit
