# Concrete Strength Prediction

## Project Overview
This project predicts the compressive strength of concrete using a trained neural network model. The model was trained using a dataset of concrete samples with various material compositions and curing ages.

## Dataset Features
Each data point consists of the following features:

- **Cement (kg/m³):** Primary binding material that contributes to the concrete's strength.  
- **Blast Furnace Slag (kg/m³):** By-product from steel production, used as a partial cement replacement to improve durability.  
- **Fly Ash (kg/m³):** A fine powder from coal combustion, enhances strength and workability.  
- **Water (kg/m³):** Reacts with cement for hydration, essential for concrete setting and strength.  
- **Superplasticizer (kg/m³):** Chemical additive that improves workability without increasing water content.  
- **Coarse Aggregate (kg/m³):** Large particles (gravel/crushed stone) providing bulk and structural integrity.  
- **Fine Aggregate (kg/m³):** Smaller particles (sand) filling voids and improving compactness.  
- **Age (days):** Time elapsed since concrete was mixed, affecting final strength development.  

## Model Performance Metrics
The trained model was evaluated using the following metrics:

- **R-squared (R²):** 87.83% - Measures how well the model explains the variance in concrete strength. Higher values indicate better predictions.
- **Mean Absolute Error (MAE):** 3.83 - Represents the average difference between predicted and actual values, lower values indicate better accuracy.

## Running the Model
To use this model:

1. Load the trained neural network model (`concrete_strength.h5`).
2. Apply the pre-trained scaler (`scaler.pkl`) to standardize input data.
3. Input values for concrete composition and age.
4. Obtain the predicted strength (MPa) using the neural network.

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **Scikit-learn**
- **Streamlit (for frontend UI)**

This project provides a user-friendly interface to predict concrete strength efficiently. 🚀

