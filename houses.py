import pandas as pd

# Load the dataset
file_path = 'house-prices.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate numeric and categorical features
numeric_features = ['SqFt', 'Bedrooms', 'Bathrooms', 'Offers']
categorical_features = ['Brick', 'Neighborhood']

# Define the preprocessing steps for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define the preprocessing steps for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the transformations to the data
X = data.drop(['Price', 'Home'], axis=1)
y = data['Price']

X_preprocessed = preprocessor.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2

import streamlit as st
import pandas as pd

# Load the preprocessing pipeline and model

# Streamlit app
def main():
    # Add a sidebar for navigation and inputs
    st.sidebar.title("House Price Prediction")
    st.sidebar.write("## Enter house details:")

    # Collect user input for all necessary features in the sidebar
    sqft = st.sidebar.number_input('Square Feet', min_value=0, max_value=10000, value=1500)
    bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
    offers = st.sidebar.number_input('Number of Offers', min_value=0, max_value=20, value=1)
    brick = st.sidebar.selectbox('Brick', ('No', 'Yes'))
    neighborhood = st.sidebar.selectbox('Neighborhood', ('East', 'West', 'North', 'South'))

    # Prepare the input features
    input_features = pd.DataFrame({
        'SqFt': [sqft],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Offers': [offers],
        'Brick': [brick],
        'Neighborhood': [neighborhood]
    })

    # Display a header and some introductory text
    st.title("House Price Prediction Dashboard")
    st.markdown("""
    ### Welcome to the House Price Prediction App
    Enter the details of the house in the sidebar and click on 'Predict' to get the estimated price.
    """)

    # Use columns to organize the layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input Features")
        st.write(input_features)

    with col2:
        st.markdown("### Predicted House Price")
        # Preprocess the input features
        input_features_preprocessed = preprocessor.transform(input_features)

        if st.sidebar.button('Predict'):
            prediction = model.predict(input_features_preprocessed)
            st.success(f"${prediction[0]:,.2f}")
        else:
            st.write("Click 'Predict' to see the price.")

    # Add an image or chart (optional)
    st.image("house.jpg", use_column_width=True)

    # Add some informative text or a table
    st.markdown("""
    ### Additional Information
    - Use this app to get an estimated price for your house based on its features.
    - The prediction model is trained on historical data and may not reflect the actual market value.
    """)

    # Example table (optional)
    st.table(data.head())


if __name__ == '__main__':
    main()
