import streamlit as st 
import pandas as pd
from joblib import load
from time import sleep
import math


@st.cache_resource(show_spinner="Loading model....")
def load_model(path):
    """
    Load the model 

    Args:
        path (String): File path of the model

    Returns:
        model: Decision Tree model
    """

    # sleep(2)
    model = load(path)

    return model


def main():
    
    # Load the model
    model = load_model('model/model_selected.joblib')

    # Main content
    st.title('🌠 Star Type Classifier')
    st.divider()

    st.header('Enter star details to classify the star:')
    
    # Only collect the 3 selected features
    rad = st.slider('Radius (R/Ro)', min_value=0.01, max_value=2000.0, step=0.1)
    mag = st.slider('Absolute magnitude (Mv)', min_value=-12.0, max_value=21.0, step=0.1)
    lum = st.slider('Luminosity (L/Lo)', min_value=0.0, max_value=850000.0, step=100.0)

    if st.button('Predict'):
        
        # --- PHYSICS VALIDATION ---
        # Real stars follow a strict math formula connecting Luminosity and Absolute Magnitude.
        # If the user enters random numbers that don't match, we show a warning.
        
        # Mv = 4.83 - 2.5 * log10(Luminosity)
        
        if lum > 0:
            expected_mag = 4.83 - 2.5 * math.log10(lum)
            
            # If their entered Magnitude is more than 2.0 away from expected, warn them
            if abs(expected_mag - mag) > 2.0:
                st.warning("⚠️ **Warning:** The entered values are physically inconsistent. Real stars don't have this combination of Luminosity and Magnitude! The prediction may be unreliable.")
        else:
            if mag < 10:
                st.warning("⚠️ **Warning:** Very bright stars (low magnitude) cannot have 0 Luminosity! The prediction may be unreliable.")

        # 1. Create a dictionary with only the selected features in the EXACT order 
        # the model was trained on: ['Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Luminosity(L/Lo)']
        input_data = {
            'Radius(R/Ro)': rad,
            'Absolute magnitude(Mv)': mag,
            'Luminosity(L/Lo)': lum
        }
            
        # 2. Convert to DataFrame and predict
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]

        # Map prediction number to proper text
        star_types = {
            0: 'Red Dwarf',
            1: 'Brown Dwarf',
            2: 'White Dwarf',
            3: 'Main Sequence',
            4: 'SuperGiants',
            5: 'HyperGiants'
        }
        
        predicted_type = star_types.get(prediction, f"Unknown ({prediction})")
        
        st.success(f"Predicted Star Type: **{predicted_type}**")
if __name__ == '__main__':
    main()