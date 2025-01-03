import streamlit as st
from PIL import Image
import os
import pickle
import base64
import numpy as np

# Streamlit App - Music Recommendation
st.title("Music Recommendation System")

# Function to set the background image
def set_background(image_path):
    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        background_style = f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{img_base64}");
                background-size: cover;
            }}
            </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Background image not found at: {image_path}")

# Set the background image
set_background("cd.png")  # Replace with the actual path to your image

# Load data from pickle file
def load_data(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Pickle file '{file_path}' not found. Please check the path.")
        return None

data = load_data("track_data.pkl")

if data is not None:
    # Function to calculate Euclidean distance
    def euclidean_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # KNN prediction function
    def knn_predict(X_train, y_train, X_test, k=10):
        predictions = []
        for test_point in X_test:
            distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
            k_indices = np.argsort(distances)[:k]
            k_labels = [y_train[i] for i in k_indices]
            predictions.append(k_labels)
        return predictions

    # Find similar tracks based on features and genre
    def find_similar_tracks(data, track_name, k=10):
        feature_columns = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
        features = data[feature_columns]
        normalized_features = (features - features.min()) / (features.max() - features.min())

        try:
            track_index = data[data["track_name"].str.lower() == track_name.lower()].index[0]
        except IndexError:
            st.warning("Track not found. Please try another name.")
            return []

        track_features = normalized_features.iloc[track_index]
        X_train = normalized_features.drop(index=track_index).values
        X_test = track_features.values.reshape(1, -1)
        y_train = data.drop(index=track_index)[["track_name", "genre"]].values

        return knn_predict(X_train, y_train, X_test, k)[0]

    # Input for track name and number of recommendations
    track_name = st.text_input("Enter a track name:")
    k = st.slider("Number of similar tracks to recommend:", 1, 20, 10)

    if st.button("Find Similar Tracks") and track_name:
        recommendations = find_similar_tracks(data, track_name, k)
        if recommendations:
            st.subheader(f"Tracks similar to '{track_name}':")
            for idx, (track, genre) in enumerate(recommendations, start=1):
                st.write(f"{idx}. {track} (Genre: {genre})")

    # Genre exploration with images (display only once)
    genres = {
        "Hip-Hop": "hip pop.png",
        "Rock": "rock.png",
        "Classical": "claical.png",
        "R&B": "r&B.png",
        "Country": "country.png",
        "Electronic": "electronic.png",
        "Pop": "pop.png",
        "Jazz": "ja.png",
    }

    st.subheader("Explore Genres")
    cols = st.columns(4)

    # Display genre images and create a button to select the genre
    selected_genre = None
    for i, (genre, img_path) in enumerate(genres.items()):
        col = cols[i % 4]
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).resize((200, 200))
                col.image(image, caption=genre, use_container_width=True)  # Display genre image without key argument
                
                if col.button(f"{genre}"):  # Create button with genre name
                    selected_genre = genre  # Set the selected genre
            except Exception as e:
                col.error(f"Error loading image for {genre}: {e}")
        else:
            col.warning(f"Image for {genre} not found at path: {img_path}")

    # Fixed slider for the number of genre tracks to recommend (placed below the images)
    
    num_recommendations = st.slider("Number of tracks to recommend per genre:", min_value=1, max_value=20, value=5)

    # Display the top tracks below the slider for the selected genre
    if selected_genre:
        genre_tracks = data[data["genre"] == selected_genre]["track_name"].head(5).tolist()  # Get top tracks for selected genre
        st.write(f"Top {num_recommendations} tracks for {selected_genre}:")
        for idx, track in enumerate(genre_tracks[:num_recommendations], start=1):
            st.write(f"{idx}. {track}")
