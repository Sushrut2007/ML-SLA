import streamlit as st 
import pandas as pd
from joblib import load
from time import sleep


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
    model = load("model/model.joblib")

    return model


def main():
    
    # Load the model
    model = load_model('model/model.joblib')

    # Extract the feature names the model used
    feature_names = list(model.feature_names_in_)


    # Main content
    st.title('🌠 Star Type Classifier')
    st.divider()

    st.header('Enter star details to classify the star: ')
    
    temp = st.slider('Temperature (K)', min_value=1900, max_value=40000, step=100)
    lum = st.slider('Luminosity (L/Lo)', min_value=0.0, max_value=850000.0, step=100.0)
    rad = st.slider('Radius (R/Ro)', min_value=0.0, max_value=2000.0, step=0.1)
    mag = st.slider('Absolute magnitude (Mv)', min_value=-12.0, max_value=21.0, step=0.1)

    color = st.selectbox('Star Color', options=['Blue', 'Blue ', 'Blue White', 'Blue white', 'Blue white ', 'Blue-White', 'Blue-white', 'Orange', 'Orange-Red', 'Pale yellow orange', 'Red', 'White', 'White-Yellow', 'Whitish', 'Yellowish', 'Yellowish White', 'white', 'yellow-white', 'yellowish'])
    spectral_class = st.selectbox('Spectral Class', options=['A', 'B', 'F', 'G', 'K', 'M', 'O'])

    if st.button('Predict'):
        # 1. Create a dictionary with all features initialized to 0
        input_data = {col: 0 for col in feature_names}
        
        # 2. Set the numerical features
        input_data['Temperature (K)'] = temp
        input_data['Luminosity(L/Lo)'] = lum
        input_data['Radius(R/Ro)'] = rad
        input_data['Absolute magnitude(Mv)'] = mag
        
        # 3. Set the chosen Star Color to 1 (if it's not the dropped 'drop_first' baseline category)
        color_col = f"Star color_{color}"
        if color_col in input_data:
            input_data[color_col] = 1
            
        # 4. Set the chosen Spectral Class to 1
        spectral_col = f"Spectral Class_{spectral_class}"
        if spectral_col in input_data:
            input_data[spectral_col] = 1
            
        # 5. Convert to DataFrame and predict
       
        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)
        
        st.success(f"Predicted Star Type: {prediction[0]}")

if __name__ == '__main__':
    main()