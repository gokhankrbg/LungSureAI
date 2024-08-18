import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import base64

# Function to convert image to base64 format
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert background image to base64 format
background_image_path = "background.jpg"
base64_background = get_base64_of_image(background_image_path)

# Add HTML and CSS code
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), url("data:image/jpg;base64,{base64_background}");
        background-size: cover;
    }}
    .stText, .stMarkdown, .stHeader, .stSubheader, .stTitle, .stCaption {{
        color: #2E4053;
        font-family: Arial, Helvetica, sans-serif;
        font-weight: bold;
    }}
    .stHeader {{
        color: #2E4053;
        font-size: 24px;
    }}
    .stSubheader {{
        color: #2E4053;
        font-size: 20px;
    }}
    .stTitle {{
        color: #2E4053;
        font-size: 28px;
    }}
    .stCaption {{
        color: #2E4053;
        font-size: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
model = joblib.load("lungsureai.pkl")

# Get the feature names expected by the model
expected_columns = model.estimators_[0].feature_names_

# Function for user inputs
def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 50)
    gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    neu = st.sidebar.number_input('NEU% (0.0 - 100.0)', min_value=0.0, max_value=100.0, value=50.0)
    lym = st.sidebar.number_input('LYM% (0.0 - 100.0)', min_value=0.0, max_value=100.0, value=50.0)
    mon = st.sidebar.number_input('MON% (0.0 - 100.0)', min_value=0.0, max_value=100.0, value=50.0)
    plt = st.sidebar.number_input('PLT (x10^9/L) (0 - 1000)', min_value=0, max_value=1000, value=250)
    crp = st.sidebar.number_input('CRP (mg/L) (0.0 - 100.0)', min_value=0.0, max_value=100.0, value=1.0)
    hgb = st.sidebar.number_input('HGB (g/dL) (11.0 - 40.0)', min_value=11.0, max_value=40.0, value=12.0)
    bun = st.sidebar.number_input('BUN (mg/dL) (0.0 - 100.0)', min_value=0.0, max_value=100.0, value=15.0)
    kreatinin = st.sidebar.number_input('KREATININ (mg/dL) (0.0 - 10.0)', min_value=0.0, max_value=10.0, value=0.5)
    wbc = st.sidebar.number_input('WBC (x10^9/L) (0.0 - 30.0)', min_value=0.0, max_value=30.0, value=7.0)
    klor = st.sidebar.number_input('KLOR (mmol/L) (10.0 - 200.0)', min_value=10.0, max_value=200.0, value=100.0)
    sodyum = st.sidebar.number_input('SODYUM (mmol/L) (10.0 - 300.0)', min_value=10.0, max_value=300.0, value=140.0)
    kalsiyum = st.sidebar.number_input('KALSIYUM (mg/L) (1.0 - 20.0)', min_value=1.0, max_value=20.0, value=9.5)
    so2 = st.sidebar.number_input('SO2% (0.0 - 100.0)', min_value=0.0, max_value=100.0, value=98.0)

    # Convert gender to numeric
    gender = 0 if gender == 'Female' else 1

    data = {
        'AGE': age,
        'GENDER': gender,
        'NEU': neu,
        'LYM': lym,
        'MON': mon,
        'PLT': plt,
        'CRP': crp,
        'HGB': hgb,
        'BUN': bun,
        'KREATININ': kreatinin,
        'WBC': wbc,
        'KLOR': klor,
        'SODYUM': sodyum,
        'KALSIYUM': kalsiyum,
        'SO2': so2
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit title and tabs
st.title("LungSureAI Pneumonia Prediction Application")

tabs = st.tabs(["About the Project", "Values and Results", "Model Information and Important Features"])

shap_values = None
input_df = None

with tabs[0]:
    st.header("About the Project")
    st.image("lungsureai.jpeg", use_column_width=True)
    st.write("""
    The LungSureAI project is an AI project in healthcare that predicts the probability and detection of pneumonia based on laboratory results.
    With this project, the aim is to prevent unnecessary medical imaging and exposure to radiation such as MRI, CT, and X-Ray by identifying patients who do not actually need imaging, based on their lab results.
    """)

    st.header("About the Variables")
    st.write("""
    **AGE:** The age of the patient.  
    **GENDER:** The gender of the patient.  
    **NEU:** Neutrophil count (%)  
    **LYM:** Lymphocyte count (%)  
    **MON:** Monocyte count (%)  
    **PLT:** Platelet count (x10^9/L)  
    **CRP:** C-reactive protein level (mg/L)  
    **HGB:** Hemoglobin level (g/dL)  
    **BUN:** Blood urea nitrogen level (mg/dL)  
    **KREATININ:** Creatinine level (mg/dL)  
    **SO2:** Oxygen saturation (%)  
    **WBC:** White blood cell count (x10^9/L)  
    **KLOR:** Chloride level (mmol/L)  
    **SODYUM:** Sodium level (mmol/dL)  
    **KALSIYUM:** Calcium level (mg/L)  
    """)

with tabs[1]:
    st.header("Values and Results")
    input_df = user_input_features()

    # Customize button style with CSS
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 16px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Information window
    with st.expander("How It Works?"):
        st.write("""
        1. Enter the patient's gender, age, and relevant lab results in the sidebar.
        2. Press the 'Predict' button.
        3. Review the results and probabilities.
        """)

    # Information text
    st.write("Please press the 'Predict' button to perform the prediction. This may take a few seconds.")

    # Predict button
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            input_df['GENDER'] = input_df['GENDER'].apply(lambda x: 1 if x == 'Female' else 0)

            # Fill missing values with the mean
            input_df.fillna(input_df.mean(), inplace=True)

            # Save raw data
            raw_data = input_df.copy()

            # Create new features
            input_df['BUN_KREATININ'] = input_df['BUN'] / (input_df['KREATININ'] + 1e-6)
            input_df['LOG_CR'] = np.log(input_df['CRP'] + 1e-6)
            input_df['SQRT_KREATININ'] = np.sqrt(input_df['KREATININ'] + 1e-6)
            input_df['GENDER_AGE'] = input_df['GENDER'] * input_df['AGE']
            input_df['WBC_PLT_RATIO'] = input_df['WBC'] / (input_df['PLT'] + 1e-6)
            input_df['LYM_NEU_RATIO'] = input_df['LYM'] / (input_df['NEU'] + 1e-6)
            input_df['WBC_LOG'] = np.log(input_df['WBC'] + 1e-6)
            input_df['NLR'] = input_df['NEU'] / input_df['LYM']
            input_df['MLR'] = input_df['MON'] / input_df['LYM']
            input_df['PLR'] = input_df['PLT'] / input_df['LYM']
            # Create age categories
            input_df['NEW_AGE_CAT'] = pd.cut(input_df['AGE'], bins=[0, 35, 55, 100],
                                             labels=['CHILD_YOUNG', 'MIDDLEAGE', 'OLD'])
            input_df = pd.get_dummies(input_df, columns=['NEW_AGE_CAT'], drop_first=True)

            # Add disease columns and fill them with 0 initially
            for col in ['DISEASES_1', 'DISEASES_2', 'DISEASES_3', 'DISEASES_4', 'DISEASES_6']:
                input_df[col] = 0

            # Add gender column and fill it with 0 initially
            if 'GENDER_FEMALE_1' not in input_df.columns:
                input_df['GENDER_FEMALE_1'] = 0

            # Ensure we have all the columns expected by the model
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder and format the data as expected by the model
            input_df = input_df[expected_columns]

            # Prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.success('Prediction completed!')

            st.subheader('Entered Data:')
            st.write(raw_data)

            st.subheader('Prediction Result:')
            pneumonia_risk = 'Pneumonia Risk Present' if prediction[0] == 1 else 'No Pneumonia Risk'
            st.write(pneumonia_risk)

            st.subheader('Prediction Probabilities:')
            st.write(f"No Pneumonia Risk: %{prediction_proba[0][0] * 100:.2f}")
            st.write(f"Pneumonia Risk Present: %{prediction_proba[0][1] * 100:.2f}")

            # Visualize prediction probabilities
            st.subheader('Prediction Probabilities Visualization:')
            fig, ax = plt.subplots()
            probabilities = [prediction_proba[0][0], prediction_proba[0][1]]
            labels = ['No Pneumonia Risk', 'Pneumonia Risk Present']
            ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
            ax.axis('equal')
            st.pyplot(fig)

            # Feedback and comments section
        st.subheader("Feedback and Comments")
        feedback = st.text_area("Write your feedback or comments about the result here")
        if st.button("Submit Feedback"):
            st.write("Thank you for your feedback!")
    with tabs[2]:
        st.header("Model Information and Important Features")
        st.write("""
        This section provides information about the model and displays the most important features with graphs.
        """)
        st.image("model_graph.jpeg", caption="Model Performance", use_column_width=True)
        st.image("voting_importance.png", caption="Feature Importance", use_column_width=True)
        st.image("CAT_shap.jpeg", caption="CatBoost SHAP", use_column_width=True)
        st.image("LGBM_shap.jpeg", caption="LGBM SHAP", use_column_width=True)
        st.image("XG_shap.jpeg", caption="XGBoost SHAP", use_column_width=True)
        st.image("RF_shap.jpeg", caption="RF SHAP", use_column_width=True)
