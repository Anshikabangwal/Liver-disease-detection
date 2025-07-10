# Liver-disease-detection

🧪 Liver Disease Detection System
This project predicts the likelihood of liver disease using a Random Forest Classifier trained on the Indian Liver Patient Dataset. It also generates a detailed, personalized medical report in PDF format based on user inputs.

📁 Project Structure
├── main.py                   # Main script for training, prediction, and report generation
├── requirements.txt          # List of Python dependencies
├── indian_liver_patient.csv  # Dataset used for model training
└── user-reports/             # Folder where generated reports will be stored

🚀 Features
🔬 Machine Learning Model: Random Forest Classifier for binary classification (Liver Disease / No Liver Disease).

📈 Model Training: Includes preprocessing (label encoding, scaling) and evaluation (confusion matrix, accuracy).

🧾 Prediction: Takes real-time user inputs for liver-related test parameters and gives an instant diagnosis.

📄 PDF Report: Automatically generates a personalized report including:

📦 Setup Instructions
1️⃣ Clone or Download the Project
Make sure all files (main.py, requirements.txt, indian_liver_patient.csv) are in the same folder.

2️⃣ Create Output Folder
    mkdir -p user-reports
This folder will store the generated PDF reports.

3️⃣ Install Required Libraries
Make sure Python is installed, then run:
pip install -r requirements.txt

🏃 How to Run
Run the project using:
python main.py

You'll see a menu like this:

--- Liver Disease Detection Menu ---
1. Train and Evaluate Model
2. Predict Liver Disease
3. Exit
Choose Option 1 to train the model (required before prediction).

Choose Option 2 to enter user details and predict liver condition.

A PDF report will be saved in the user-reports/ folder.

📄 Sample Output (PDF Report)
The PDF includes:

👤 Name, Age, Gender

📊 Entered medical values

✅ Health status and suggestions per parameter

📌 Final prediction

🕒 Timestamp and disclaimer
