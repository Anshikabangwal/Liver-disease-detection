import pandas as pd
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime

# Load and preprocess dataset
def load_and_prepare_data():
    data = pd.read_csv("indian_liver_patient.csv")
    data = data.dropna()
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    X = data.drop(['Dataset'], axis=1)
    y = data['Dataset'].apply(lambda x: 1 if x == 1 else 0)
    return X, y

# Train model
def train_model(X, y):
    global scaler, model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Model Trained Successfully!\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save report to PDF (without emojis)


def save_report_to_pdf(user_name, input_data, health_status, prediction):
    from fpdf import FPDF
    from datetime import datetime
    import os

    os.makedirs("user-reports", exist_ok=True)
    report_filename = f"user-reports/{user_name}_Liver_Report.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(15, 15, 15)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "Liver Disease Report", ln=True, align="C")
    pdf.ln(6)

    # Personal Details
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Personal Details:", ln=True)

    pdf.set_font("Arial", "", 12)
    name = f"Name    : {user_name}"
    age = f"Age     : {input_data.get('Age', '')}"
    gender = f"Gender  : {'Male' if int(input_data.get('Gender', 1)) == 1 else 'Female'}"
    pdf.cell(0, 8, name, ln=True)
    pdf.cell(0, 8, age, ln=True)
    pdf.cell(0, 8, gender, ln=True)

    # Test Results
    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Test Results:", ln=True)

    pdf.set_font("Arial", "", 12)
    for param, value in input_data.items():
        if param in ["Age", "Gender"]:
            continue
        pdf.cell(0, 8, f"{param}: {value}", ln=True)

    # Prediction
    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Prediction:", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, prediction)
    pdf.ln(2)

    # Table Header
    pdf.set_font("Arial", "B", 11)
    pdf.cell(50, 10, "Parameter", 1, 0, 'C')
    pdf.cell(40, 10, "Status", 1, 0, 'C')
    pdf.cell(100, 10, "Suggestion", 1, 1, 'C')

    # Table Content
    pdf.set_font("Arial", "", 10)
    for param, message in health_status.items():
        if param in ["Age", "Gender"]:
            continue

        parts = message.split("\n Suggestion: ")
        status = parts[0].split(": ", 1)[1] if ": " in parts[0] else "Normal"
        suggestion = parts[1] if len(parts) > 1 else "Normal"

        if len(suggestion) > 250:
            suggestion = suggestion[:250] + "..."

        x = pdf.get_x()
        y = pdf.get_y()
        row_height = 8

        pdf.multi_cell(50, row_height, param, border=1)
        pdf.set_xy(x + 50, y)
        pdf.multi_cell(40, row_height, status, border=1)
        pdf.set_xy(x + 90, y)
        pdf.multi_cell(100, row_height, suggestion, border=1)
        pdf.set_y(y + row_height)

    # Footer Note
    bottom_y = 275
    if pdf.get_y() < bottom_y:
        pdf.set_y(bottom_y)

    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6, "Note: This is a digitally generated report and not a substitute for professional medical advice. No signature needed.")

    # Date in Footer
    pdf.set_y(-15)
    pdf.set_x(-60)
    now = datetime.now().strftime("%d-%m-%Y %I:%M %p").lower()
    pdf.set_font("Arial", "", 9)
    pdf.cell(0, 10, now, 0, 0, 'R')

    pdf.output(report_filename, "F")




# Predict function
def predict_liver_disease():
    print("\nEnter the following details:")
    try:
        user_name = input("Your Name: ")

        sample_input = []
        sample_input.append(float(input("Age: ")))
        sample_input.append(int(input("Gender (1 for Male, 0 for Female): ")))
        sample_input.append(float(input("Total Bilirubin (TB): ")))
        sample_input.append(float(input("Direct Bilirubin (DB): ")))
        sample_input.append(float(input("Alkaline Phosphotase (Alkphos): ")))
        sample_input.append(float(input("SGPT (Sgpt): ")))
        sample_input.append(float(input("SGOT (Sgot): ")))
        sample_input.append(float(input("Total Proteins (TP): ")))
        sample_input.append(float(input("Albumin (ALB): ")))
        sample_input.append(float(input("A/G Ratio: ")))

        sample_df = pd.DataFrame([sample_input], columns=X.columns)
        sample_scaled = scaler.transform(sample_df)
        prediction = model.predict(sample_scaled)[0]

        # Output prediction to terminal with emojis
        print("\n Prediction:", " Liver Disease Detected" if prediction == 1 else " No Liver Disease")

        # Normal ranges and suggestions
        health_data = {
            'Age': ((18, 65), "Maintain healthy lifestyle."),
            'Gender': ((0, 1), ""),  # Not health-dependent
            'TB': ((0.1, 1.2), "Avoid alcohol, fatty foods; monitor bilirubin levels."),
            'DB': ((0.0, 0.3), "Check liver function tests regularly for safety."),
            'Alkphos': ((44, 147), "Possible liver or bone issue; consult doctor."),
            'Sgpt': ((7, 56), "Avoid processed foods; support liver health."),
            'Sgot': ((5, 40), "Stay hydrated and avoid alcohol consumption."),
            'TP': ((6.0, 8.3), "Eat protein-rich foods like eggs and milk."),
            'ALB': ((3.4, 5.4), "Increase protein intake for better liver function."),
            'A/G Ratio': ((1.0, 2.5), "Monitor liver and kidney health regularly.")
        }

        print("\n Parameter Health Report with Suggestions:")
        health_status = {}  # To save health status for PDF
        for value, (col, ((low, high), tip)) in zip(sample_input, health_data.items()):
            if col == 'Gender':
                print(f"{col}: {value}\n ")
                health_status[col] = f"{col}: {value} (Not analyzed)"
                continue
            if value < low:
                status = f" Low ({value} < {low})"
                print(f"{col}: {status}\n Suggestion: {tip}\n")
                health_status[col] = f"{col}: {status}\n Suggestion: {tip}"
            elif value > high:
                status = f" High ({value} > {high})"
                print(f"{col}: {status}\n Suggestion: {tip}\n")
                health_status[col] = f"{col}: {status}\n Suggestion: {tip}"
            else:
                print(f"{col}: {value} : Normal\n")
                health_status[col] = f"{col}: {value} : Normal"

        # Save report to PDF without emojis (replacing → with simple 'Normal' text)
        save_report_to_pdf(user_name, dict(zip(['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio'], sample_input)), health_status, "Liver Disease Detected" if prediction == 1 else "No Liver Disease")

    except Exception as e:
        print(" Error in input. Please enter numeric values correctly.")
        print(str(e))

# Menu
def menu():
    while True:
        print("\n--- Liver Disease Detection Menu ---")
        print("1. Train and Evaluate Model")
        print("2. Predict Liver Disease")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            train_model(X, y)
        elif choice == '2':
            if model is None or scaler is None:
                print("Please train the model first (Option 1).")
            else:
                predict_liver_disease()
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

# Main Execution
if __name__ == "__main__":
    model = None
    scaler = None
    X, y = load_and_prepare_data()
    menu()
