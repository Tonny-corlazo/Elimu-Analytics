🎓 Elimu Analytics
An automated academic performance dashboard and Machine Learning early-warning system for Kenyan schools.

Elimu Analytics transforms raw exam data into actionable insights. Built to handle the unique structure of the Kenyan school system (3 Terms, Opener & End Term exams), this tool helps teachers and principals identify struggling students before they fail.

✨ Features
📊 Automated KPI Dashboard: Instant view of school mean, total students, exams taken, and overall at-risk percentage.

📈 Trend Analysis: Visualizes school progression across all 6 exams in a year to see if performance is actually improving.

🚨 Machine Learning Risk Engine: Uses a trained Random Forest Classifier to identify patterns of failure based on class, gender, subject, and exam type.

🔮 Interactive Predictor: A sandbox tool for teachers to simulate a student's score and see if the ML model flags them as high-risk.

📥 Exportable Reports: Instantly download lists of at-risk students or top performers as CSVs for staff meetings.

🛠️ Built With
Frontend/App: Streamlit

Data Manipulation: Pandas

Machine Learning: Scikit-Learn (Random Forest)

Visualizations: Matplotlib & Seaborn

🚀 How to Run Locally
Clone the repository:

bash
git clone https://github.com/your-username/elimu-analytics.git
cd elimu-analytics
Install the required dependencies:

bash
pip install streamlit pandas matplotlib seaborn scikit-learn
Launch the app:

bash
streamlit run app.py
Upload your data:
Use the sidebar to upload a CSV file. For testing, you can use the provided elimu_analytics_2026.csv (sample data).

📁 Expected CSV Format
To ensure the machine learning model and visualizations work perfectly, your CSV must contain the following columns:
student_id, name, class, gender, term, exam_type, exam_date, subject, marks, out_of, percentage, grade, exam_num

(Note: The app will automatically recalculate percentage and exam_num if they are missing or incorrect).

🧠 How the ML Works
The Random Forest model looks at historical data (Class, Gender, Subject, and Exam Type) and predicts the probability of a student scoring below 50%. By analyzing feature importance, the dashboard also reveals which subjects or which terms have the highest risk patterns.
