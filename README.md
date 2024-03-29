# Heart Failure Prediction

This repository focuses on predicting heart failure based on the Hungarian dataset, which can be found [here](https://archive.ics.uci.edu/dataset/45/heart+disease). The dataset contains 76 attributes, but this experiment concentrates on 14 key attributes: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, and target.

## Dataset Information

- **age**: Age of the patient
- **sex**: Gender of the patient (1 = male, 0 = female)
- **cp**: Chest pain type (Value 1-4)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (values 0,1,2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
- **target**: Presence of heart disease (1 = yes; 0 = no)

## Screenshot

![Home](Screenshot/home.png)

## Deployment

The prediction model is deployed using Streamlit, and you can interact with it [here](https://heart-failure.streamlit.app/).

## Usage

To replicate the experiment or use the prediction model locally:

1. Clone this repository.
   ```bash
   git clone https://github.com/bimarakajati/Heart-Failure-Prediction.git
   cd heart-failure-prediction
   ```

2. Install the required packages.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app.
   ```bash
   streamlit run app.py
   ```

Now, you can access the prediction model locally through your web browser.

Feel free to explore and contribute to this project!