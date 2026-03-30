Project: Annual Medical Insurance Charges Predictor

Tech Stack: Python, Random Forest (Scikit-Learn), Pandas,numpy, Streamlit

1. What Problem Does This Solve?
The healthcare industry faces a massive challenge in actuarial risk assessment. Traditional methods of pricing insurance premiums can be rigid and fail to account for the complex, non-linear interactions between a person's lifestyle and their health risks.

This project solves the "Pricing Accuracy" problem. By using machine learning, we can automate the estimation of medical charges, ensuring that:

Insurers can set premiums that accurately reflect risk, maintaining profitability.

Customers receive fair, data-driven pricing based on their specific health profile.

2. What Data Was Used?
The model was trained on a comprehensive dataset of medical insurance claimants. The key features (the "drivers" of cost) include:

Demographics: Age, Sex, and Region.

Health Metrics: BMI (Body Mass Index) – a critical indicator of potential chronic issues.

Lifestyle & Family: Smoking status (often the highest correlation to cost) and the Number of Children (dependents).

3. What Models Were Used?
The core engine of this project is a Random Forest Regressor.

Why Random Forest? I chose this ensemble learning method because it excels at handling both categorical and numerical data while capturing complex relationships (like how the impact of BMI might change significantly if the person is also a smoker).

Performance Tuning: As seen in my terminal outputs, I utilized RandomizedSearchCV for hyperparameter tuning—optimizing n_estimators, max_depth, and min_samples_split to ensure the model generalizes well to new, unseen data.

4. What Are The Results?
The model demonstrates strong predictive power and high reliability:

R² Score: My model achieved an R 
2
  of ~0.87 (87%) on the test set. This means the model explains 87% of the variance in medical charges.

Generalization: The gap between training and testing performance is minimal, indicating the model is robust and not "overfit" to the training data.

baseline model :r2=0.82
tuned model  : r2= 0.88 showing an improvement by ~6%

Deployment: I built a Streamlit Web App to demonstrate how this model can be integrated into a real-world business workflow, allowing a user to get an instant prediction via a manual form or a bulk CSV upload.

5. Why Should You Care? (The Value Add)
For a recruiter, this project demonstrates more than just "coding skills"—it shows end-to-end product thinking:

Data Engineering: I built a structured pipeline (as seen in my src folder) for data ingestion and preprocessing.

Statistical Rigor: I didn't just pick a model; I tuned it and validated its performance with industry-standard metrics.

Business Integration: By creating a UI, I've shown I can bridge the gap between "Black Box" AI and the end-user (e.g., an insurance agent or a customer).

Key Technical Highlight from the Screenshots:
Modular Architecture: My project structure follows professional software engineering patterns, separating the Data Ingestion, Feature Engineering, and Model Pipeline into distinct modules for maintainability and scalability.
