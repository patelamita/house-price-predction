# House Price Prediction Project

## Overview
This project aims to predict house prices based on features such as'House_Price', 'Area_sqft', 'Bedrooms', 'Bathrooms', 'Stories' and other amenities. The dataset contains 545 rows and 13 columns.

## Dataset Description
- House_price: Target variable (house price)
- Area_sqft: Area of the house in square feet
- Bedrooms: Number of bedrooms
- Bathrooms: Number of bathrooms
- Stories: Number of stories
- Main_Road_Access: Whether the house is connected to the main road (yes/no)
- Guest_Room_Available: Whether the house has a guest room (yes/no)
- Basement_Available: Whether the house has a basement (yes/no)
- Hot_Water_Heating: Whether the house has a hot water heater (yes/no)
- Air_Conditioning: Whether the house has air conditioning (yes/no)
- Parking_Spaces: Number of parking spaces
- Preferred_Location: Whether the house is in a preferred area (yes/no)
- Furnishing_Status: Furnishing status of the house (furnished, semi-furnished, unfurnished)

## Steps
1. Data Preprocessing: Handle missing values, convert categorical variables to numerical, and scale features.
2. Model Evaluation: Evaluate the model using Mean Squared Error (MSE) and mean_absolute_error.
3. Model Training: Train a Linear Regression model to predict house prices.
4. Exploratory Data Analysis (EDA): Visualize distributions, correlations, and relationships between features. 

## Project Structure
```
house_price_predction/
│
├── dataset/
│   └── Housing.csv
│
├── notebook/            
│   └── price_prediction.ipynb
│ 
|__ .gitignore
|
|__README.md
|
└── requirement.txt
```

## Requirements
- Python 3.x
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd house_price_prediction
```
2. Create Virtual Environment file
 ```
python env -m virtualenv env
env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```
## Results
- Mean Squared Error: [Value]
- R-squared: [Value]