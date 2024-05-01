import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the dataset
data = pd.read_csv("Movie_classification.csv")

# Check for missing values in the DataFrame
# missing_values = data.isna().sum()
# print("Missing values in the dataset:")
# print(missing_values)

# Since there is a missing value in the column Time_taken, we should clean that one by dropping the rows with missing values
clean_data = data.copy()
clean_data = clean_data.dropna()
clean_data.reset_index(inplace=True,drop=True)

clean_data.isna().sum()

# Set the target label since there is no missing values
target = clean_data["Start_Tech_Oscar"]


# Handle the non-numeric values
clean_data = pd.get_dummies(clean_data)

# See the new data
# print(clean_data.head())

# Lets create a new feature for the rating since there is a correlation, we'll name it Ratings
clean_data["Ratings"] = clean_data["Lead_ Actor_Rating"] + clean_data["Lead_Actress_rating"] + clean_data["Director_rating"] + clean_data["Producer_rating"]

# Let us also add a new Expense which is the total of marketing and production expense
clean_data["Expense"] = clean_data["Marketing expense"] + clean_data["Production expense"]

# Since we have a new feature, lets drop the features that have been converted
clean_data = clean_data.drop(["Marketing expense","Production expense","Lead_ Actor_Rating","Lead_Actress_rating","Director_rating","Producer_rating"],axis = 1)

# Lets see our new dataset now
# print(clean_data.head())

# Lets fix only the Time_taken column

clean_data.drop(clean_data[clean_data['Time_taken'] < 100].index, inplace = True)
clean_data.reset_index(drop=True,inplace = True)

# Lets look again the graph
clean_data.hist(bins = 30, figsize=(20, 15), color = '#005b96')

# Lets check first the numeric values to be affected if it is relevant
# clean_data.info()

# We can see above that all numeric values are good to scale and will not affect the overall data
# Lets use a standard scaler to our copy of dataset
from sklearn.preprocessing import StandardScaler

# Lets create a copy of data to ensure that all values in the columns are good to proceed in training
clean_data_copy = clean_data.copy()

# Remove also the target column which is Start_Tech_Oscar since that is our output label
clean_data_copy = clean_data_copy.drop(columns=["Start_Tech_Oscar"])

sc = StandardScaler()

clean_data_copy[clean_data_copy.select_dtypes(np.number).columns] = sc.fit_transform(clean_data_copy[clean_data_copy.select_dtypes(np.number).columns])

# print(clean_data_copy.head())

# Lets set the new target from the original clean dataset
target = target.astype(int)
# print(target)

from sklearn.model_selection import train_test_split

# Update the target variable to match the new number of rows
target = target[clean_data_copy.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(clean_data_copy, target, test_size= 0.2, stratify= target, random_state= 42)
print(X_train.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Load the pretrained model
loaded_model = load('random_forest_model.pkl')

# Function to preprocess input data
def preprocess_input(clean_data_copy):
    # Scale numerical features
    clean_data_copy[clean_data_copy.select_dtypes(np.number).columns] = sc.transform(clean_data_copy[clean_data_copy.select_dtypes(np.number).columns])
    data = clean_data_copy
    return data

# Main function for Streamlit app
def main():
    st.title("Movie Classification")

    # Create input fields for user input
    multiplex_coverage = st.slider("Multiplex Coverage", 0.0, 1.0, 0.462)
    budget = st.number_input("Budget", value=36524.125)
    movie_length = st.number_input("Movie Length", value=138.7)
    critic_rating = st.number_input("Critic Rating", value=7.94)
    trailer_views = st.number_input("Trailer Views", value=527367)
    time_taken = st.number_input("Time Taken", value=109.6)
    twitter_hashtags = st.number_input("Twitter Hashtags", value=223.84)
    avg_age_actors = st.number_input("Average Age of Actors", value=23)
    num_multiplex = st.number_input("Number of Multiplex", value=494)
    collections = st.number_input("Number of Collections", value=48000)
    available_3d = st.radio("3D Availability", ["Not Available", "Available"])
    genre_options = ["Action", "Comedy", "Drama", "Thriller"]
    selected_genre = st.selectbox("Genre", genre_options)
    ratings = st.number_input("Ratings", value=31.825)
    expense = st.number_input("Expense", value=79.7464)

    # Set values for both "3D_available_NO" and "3D_available_YES" columns based on user input
    if available_3d == "Not Available":
        is_3d_available_no = True
        is_3d_available_yes = False
    else:
        is_3d_available_no = False
        is_3d_available_yes = True

    # Create binary columns for each genre based on user selection
    genre_columns = {
        "Action": False,
        "Comedy": False,
        "Drama": False,
        "Thriller": False
    }
    if selected_genre in genre_columns:
        genre_columns[selected_genre] = True



    # Create input DataFrame
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "Multiplex coverage": [multiplex_coverage],
            "Budget": [budget],
            "Movie_length": [movie_length],
            "Critic_rating": [critic_rating],
            "Trailer_views": [trailer_views],
            "Time_taken": [time_taken],
            "Twitter_hastags": [twitter_hashtags],
            "Avg_age_actors": [avg_age_actors],
            "Num_multiplex": [num_multiplex],
            "Collection": [collections],
            "3D_available_NO": [is_3d_available_no],
            "3D_available_YES": [is_3d_available_yes],
            "Genre_Action": [genre_columns["Action"]],
            "Genre_Comedy": [genre_columns["Comedy"]],
            "Genre_Drama": [genre_columns["Drama"]],
            "Genre_Thriller": [genre_columns["Thriller"]],
            "Ratings": [ratings],
            "Expense": [expense],
        })

        # Load scaler
        # scaler = joblib.load("your_scaler.pkl")

        processed_data = preprocess_input(input_data)

        # print(processed_data)
        # Make predictions
        prediction = loaded_model.predict(processed_data)
        result = "Oscar" if prediction == 1 else "Not"

        # Display prediction result
        st.write("Prediction:", result)
        st.write()

if __name__ == "__main__":
    main()
