# app.py

# pip install scikit-learn==1.3.2
# pip install numpy
# pip install flask

# Load packages==============================================================
from flask import Flask, render_template, request, redirect, flash
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a strong secret key in production

# Load the scaler, label encoder, model, and class names=====================
scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

# Recommendations ===========================================================
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job == 'true' else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities == 'true' else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score, total_score,
                               average_score]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top three predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    top_classes_names_probs = [(class_names[idx], round(probabilities[0][idx] * 100, 2)) for idx in top_classes_idx]

    return top_classes_names_probs

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST'])
def pred():
    try:
        # Extract form data
        gender = request.form.get('gender')
        part_time_job = request.form.get('part_time_job')
        absence_days = request.form.get('absence_days')
        extracurricular_activities = request.form.get('extracurricular_activities')
        weekly_self_study_hours = request.form.get('weekly_self_study_hours')
        math_score = request.form.get('math_score')
        history_score = request.form.get('history_score')
        physics_score = request.form.get('physics_score')
        chemistry_score = request.form.get('chemistry_score')
        biology_score = request.form.get('biology_score')
        english_score = request.form.get('english_score')
        geography_score = request.form.get('geography_score')
        total_score = request.form.get('total_score')
        average_score = request.form.get('average_score')

        # Check for missing fields
        required_fields = [gender, part_time_job, absence_days, extracurricular_activities,
                           weekly_self_study_hours, math_score, history_score, physics_score,
                           chemistry_score, biology_score, english_score, geography_score,
                           total_score, average_score]
        if not all(field is not None and field != '' for field in required_fields):
            flash("All fields are required.", "danger")
            return redirect('/recommend')

        # Convert and validate numerical fields
        try:
            absence_days = int(absence_days)
            if absence_days < 0 or absence_days > 235:
                flash("Please enter a valid number between 0 to 235 for Absence Days.", "danger")
                return redirect('/recommend')
        except ValueError:
            flash("Absence Days must be an integer.", "danger")
            return redirect('/recommend')

        try:
            weekly_self_study_hours = int(weekly_self_study_hours)
            if weekly_self_study_hours < 0 or weekly_self_study_hours > 168:
                flash("Weekly Self-Study Hours must be between 0 and 168.", "danger")
                return redirect('/recommend')
        except ValueError:
            flash("Weekly Self-Study Hours must be an integer.", "danger")
            return redirect('/recommend')

        # List of score fields
        score_fields = {
            'Math Score': math_score,
            'History Score': history_score,
            'Physics Score': physics_score,
            'Chemistry Score': chemistry_score,
            'Biology Score': biology_score,
            'English Score': english_score,
            'Geography Score': geography_score
        }

        # Convert and validate each score
        for field_name, value in score_fields.items():
            try:
                score = int(value)
                if score < 0 or score > 100:
                    flash(f"{field_name} must be between 0 and 100.", "danger")
                    return redirect('/recommend')
            except ValueError:
                flash(f"{field_name} must be an integer.", "danger")
                return redirect('/recommend')

        # Convert total_score and average_score
        try:
            total_score = float(total_score)
            average_score = float(average_score)
        except ValueError:
            flash("Total Score and Average Score must be numbers.", "danger")
            return redirect('/recommend')

        # Proceed with recommendations
        recommendations = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                          weekly_self_study_hours, int(math_score), int(history_score),
                                          int(physics_score), int(chemistry_score), int(biology_score),
                                          int(english_score), int(geography_score),
                                          total_score, average_score)

        return render_template('results.html', recommendations=recommendations)

    except Exception as e:
        # Optionally log the exception e for debugging
        # For example: app.logger.error(f"An error occurred: {e}")
        flash("An error occurred while processing your request. Please try again.", "danger")
        return redirect('/recommend')

if __name__ == '__main__':
    app.run(debug=True)
