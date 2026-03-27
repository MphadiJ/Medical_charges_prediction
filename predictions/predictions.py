from inference.Prediction import Predictor
import pandas as pd

predictor = Predictor(artifacts_path="models/artifacts.pkl", debug=True)

data = pd.DataFrame([
    {
        "age": 25,
        "sex": "male",
        "bmi": 29,
        "children": 1,
        "smoker": "no",
        "region": "northwest"
    }
])

result = predictor.predict(data)

print(result)
