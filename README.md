# Breast_Cancer Predictor

This project is an interactive web application built using Streamlit that predicts whether a breast tumor is malignant or benign based on user-provided input. The app uses a trained Artificial Neural Network (ANN) model on the Breast Cancer dataset to provide predictions with high accuracy.

This project include:

    - Reliable Dataset: the Breast Cancer Data Set dataset used is download from sklearn.datasets
    - Visualization of selected input data for transparency.
    - Feature Selection: implementing SelectKBest from sklearn.feature_selection.
    - Grid Search CV for Model Tuning: to find the best estimator and best hyperparameters.
    - Robust machine learning model: The MLPClassifier (Multi-Layer Perceptron Classifier) from sklearn.neural_network model was used for ANN implementation which performs well on complex datasets providing accurate predictions.
    - Real-time predictions: the predictions are displayed after to imput the values with confidence scores.
    - Interactive: The web application allows to user interact and input feature values.


Why the Project is Useful
    - Useful tool for healthcare: Enables healthcare professionals or researchers to perform preliminary analyses of tumor data efficiently.
    - User-Friendly Interface: Provides a no-code, intuitive tool for predicting cancer types.
    - Academic Value: It demonstrates the integration of machine learning and web application frameworks for academic purpose.
    - Reproducibility: Offers a complete workflow for feature selection, scaling, and deploying a model, making it a great resource for students and developers.

How Users Can Get Started with the Project
    1. Clone the Repository: git clone https://github.com/nparra75/Breast_Cancer.git
    2. Navigate to the Project Directory: cd Breast_Cancer 
    3. Install Dependencies: pip install -r requirements.txt
    4. Prepare the Files:
        - model/: Contains the pre-trained model, scaler, and selected feature files (This feature selected file is not used in this version, but could be used in future implementations to comparing with other feature selection technique).
        - UI/: Contains the app.py Streamlit application.

Run the App
    Navigate to the UI folder and run the Streamlit app: 
        - cd UI
        - streamlit run app.py

Access the APP
     Open the provided local URL (e.g., http://localhost:8501) or https://nparra75-breast-cancer-uiapp-4gsg3p.streamlit.app/ in any web browser to interact with the app.
