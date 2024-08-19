# LungSureAI 🫁
**LungSureAI: AI-Driven Pneumonia Prediction**

LungSureAI is a pioneering healthcare AI project designed to predict the likelihood of pneumonia using clinical laboratory results. By harnessing the power of machine learning, LungSureAI aims to support healthcare professionals in making faster, more accurate diagnoses while minimizing the need for unnecessary medical imaging. This project is particularly focused on reducing patients’ exposure to radiation from imaging techniques like MRI, CT, and X-Ray by identifying those who may not require such procedures based on their lab results.

## Project Goals

The core goal of LungSureAI is to develop and deploy machine learning models that analyze clinical lab results and predict the probability of pneumonia in patients. By providing real-time insights, LungSureAI assists healthcare providers in decision-making, potentially improving patient outcomes and optimizing the use of medical resources.

## Data Overview

The dataset used in LungSureAI contains the following key columns:

- **AGE**: Represents the patient's age.
- **GENDER**: Indicates the patient's gender.
- **DIAGNOSIS**: Refers to the diagnosis given by the healthcare provider.
- **DIAGNOSISCODE**: The international ICD code associated with the diagnosis.
- **LABRESULTS**: Contains the patient's laboratory results, which serve as the primary data for predicting pneumonia.

These data points are critical in training and evaluating the machine learning models, ensuring they are robust and capable of making accurate predictions.

## Workflow and Methodology

1. **Data Collection**: The project utilizes clinical laboratory results as the primary data source for its predictive analytics.
2. **Model Training**:
   - **Algorithms**: The project employs advanced machine learning algorithms, including CatBoost and LightGBM, to build robust predictive models.
   - **Interpretability**: SHAP (SHapley Additive exPlanations) is utilized to interpret the model outputs, helping to understand the impact of different features on the prediction of pneumonia.
3. **Data Analysis**: Through comprehensive data analysis, the project extracts significant insights and identifies patterns that could be indicative of pneumonia.
4. **Prediction Pipeline**: The trained models are integrated into a prediction pipeline, providing real-time predictions for new data inputs.

## Technologies and Tools

- **Programming Language**: Python
- **Machine Learning Models**: Random Forest, XGBoost, LightGBM, CatBoost
- **Model Interpretability**: SHAP
- **Data Analysis and Visualization**: Pandas, NumPy for data manipulation; Matplotlib, Seaborn for visualizations

## Future Developments

- **Model Enhancement**: Integrate more advanced algorithms and techniques to further improve the accuracy of predictions.
- **Dataset Expansion**: Expand the dataset to include a wider variety of clinical results, enhancing the model’s robustness.
- **Healthcare Collaboration**: Work closely with healthcare professionals to validate and refine the predictive models, ensuring they meet clinical standards.

## Streamlit Link

Upon deployment, the LungSureAI application will be accessible at:
https://lungsureai.streamlit.app

## Contact Information

For further information or suggestions, please feel free to reach out to the project team via LinkedIn:

- [Nihal Özdemir Köse](https://www.linkedin.com/in/nihal-%C3%B6zdemir-k%C3%B6se-a5481463/) 
- [Serdar Demir](https://www.linkedin.com/in/serdar-demir-b299161/) 
- [Gökhan Karabağ](https://www.linkedin.com/in/gokhankarabag/) 

## Deployment Instructions

To deploy this project on GitHub and Streamlit, follow these steps:

1. **Set Up a GitHub Repository**:
   - Create a new repository on GitHub and upload all relevant project files (scripts, images, data, models).
   - Include the `requirements.txt` file to ensure all dependencies are easily installed.
2. **Prepare the Streamlit App**:
   - Ensure that your Streamlit app script (e.g., `app.py`) is properly configured to load the models and data files.
   - Test the app locally using `streamlit run app.py` to confirm that it runs without issues.
3. **Deploy on Streamlit**:
   - Once the app is running smoothly, deploy it on Streamlit Cloud by connecting your GitHub repository to Streamlit.
   - Follow the on-screen instructions to complete the deployment.
4. **Update the README**:
   - After deployment, update the README with the Streamlit app link.


This README content provides a clear and detailed overview of the LungSureAI project, highlighting its objectives, methodologies, and future directions, while also offering step-by-step instructions for deployment. Let me know if you need any more adjustments or further information!
