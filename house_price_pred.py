import joblib
import pandas as pd
import streamlit as st

# Update paths to absolute paths
model_path = r"C:\Users\User\house_price_prediction\notebook\house_price_pred.pkl"
encoder_path = r"C:\Users\User\house_price_prediction\notebook\onehot_encoder.pkl"
scaler_path = r"C:\Users\User\house_price_prediction\notebook\standard_scaler.pkl" 

# Load models and scaler
try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)  # Load the scaler
except Exception as e:
    st.error(f"Error loading model, encoder, or scaler: {e}")
    st.stop()

# Streamlit App
st.title("🏡 House Price Prediction App")
st.markdown("Enter the house details to predict its price.")

# User Inputs
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, step=1)
stories = st.number_input("Stories", min_value=1, max_value=4, step=1)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, step=1)

main_road = st.selectbox("Main Road Access", ["yes", "no"])
guest_room = st.selectbox("Guest Room Available", ["yes", "no"])
basement = st.selectbox("Basement Available", ["yes", "no"])
hot_water_heating = st.selectbox("Hot Water Heating", ["yes", "no"])
air_conditioning = st.selectbox("Air Conditioning", ["yes", "no"])
preferred_location = st.selectbox("Preferred Location", ["yes", "no"])
furnishing_status = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])

# Prediction
if st.button("Predict House Price"):
    # Create a DataFrame for the new house
    new_house = pd.DataFrame([{
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Stories": stories,
        "Parking_Spaces": parking,
        "Main_Road_Access": main_road,
        "Guest_Room_Available": guest_room,
        "Basement_Available": basement,
        "Hot_Water_Heating": hot_water_heating,
        "Air_Conditioning": air_conditioning,
        "Preferred_Location": preferred_location,
        "Furnishing_Status": furnishing_status
    }])

    # Define categorical and numerical columns
    categorical_columns = ["Main_Road_Access", "Guest_Room_Available", "Basement_Available", 
                        "Hot_Water_Heating", "Air_Conditioning", "Preferred_Location", "Furnishing_Status"]
    numerical_columns = ["Area_sqft", "Bedrooms", "Bathrooms", "Stories", "Parking_Spaces"]

    # Preprocess categorical features
    new_house_cat = encoder.transform(new_house[categorical_columns])
    new_house_cat_df = pd.DataFrame(
        new_house_cat, columns=encoder.get_feature_names_out(categorical_columns))

    # Combine numerical and encoded categorical data
    new_house_final = pd.concat(
        [new_house[numerical_columns], new_house_cat_df], axis=1)

    # Scale the numerical features using the same scaler
    new_house_final_scaled = scaler.transform(new_house_final)

    # Predict price
    predicted_price = model.predict(new_house_final_scaled)

    # Display result
    st.success(f"🏠 Estimated House Price: **${predicted_price[0]:,.2f}**")

st.write("This model is trained using a Machine Learning algorithm for house price prediction.")