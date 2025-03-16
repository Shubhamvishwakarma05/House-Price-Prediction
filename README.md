🏠 House Price Prediction App

This is a Machine Learning web application built using Streamlit that predicts house prices based on user-inputted features like square footage, number of bedrooms, bathrooms, type of brick, neighborhood, and number of offers.


![Screenshot 2025-03-16 232508](https://github.com/user-attachments/assets/1a7cecdd-d158-4d38-ae42-ece6bf3f3dba)


📊 Features

User-Friendly UI: Simple and interactive interface using Streamlit.
Dynamic Input: Users can enter custom house details.
ML Pipeline: Preprocessing with scaling, encoding, and imputation.
Real-Time Prediction: Instant house price prediction based on inputs.
Informative Layout: Displays input features, predicted price, and additional info.
Visual Content: Displays an image related to housing for better presentation.


![Screenshot 2025-03-16 232553](https://github.com/user-attachments/assets/918df9ca-ef4b-4469-a3ca-7d04bd7fd842)


🧩 How It Works

Input Details:


Enter house details like square footage, bedrooms, bathrooms, and more in the sidebar.
Click 'Predict':


The model processes the input using a preprocessing pipeline and makes a prediction.
View Prediction:


The predicted house price is displayed on the dashboard.


📚 Tech Stack

Python - Main programming language.
Pandas & Scikit-learn - For data preprocessing and model building.
Streamlit - For building the web app interface.


✅ Dataset Information

Dataset Name: house-prices.csv
Features Used:
SqFt - Square Footage
Bedrooms - Number of Bedrooms
Bathrooms - Number of Bathrooms
Offers - Number of Offers
Brick - Whether the house is built with bricks (Yes/No)
Neighborhood - The neighborhood category
Target Variable: Price


⚠️ Notes

The model is trained on historical data and predictions are for reference purposes only.
For a production-level model, consider using more advanced techniques and datasets.


🤝 Contribution

Feel free to fork this project, make improvements, and raise pull requests. Contributions are welcome!


💡 Future Improvements

Use advanced regression models for better accuracy.
Deploy the app to a cloud platform for public access.
Enhance the UI with additional visualizations.
