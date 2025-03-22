from scripts.MovieRecommender import MovieRecommender
import streamlit as st
import pandas as pd

dataset_path = "data/refined_dataset.csv"
model_path = "knn_model.pkl"


dataset = pd.read_csv(dataset_path)

st.title("Movie Recommender System")

def check_user_exists(query_user_id):
    all_user_ids = dataset['user id'].unique().tolist()
    if query_user_id in all_user_ids:
        return True
    else:
        return False

# User Input
user_id = st.text_input("Enter User ID:")
n_similar_users = st.number_input("Number of similar users to consider:", min_value=1, max_value=10, value=3, step=1)
n_recommendations = st.number_input("Number of movies to recommend:", min_value=1, max_value=10, value=3, step=1)

if user_id:
    user_id = int(user_id)
    user_exists = check_user_exists(user_id)
    if user_exists:
        recommender = MovieRecommender(dataset_path, model_path)
        watched_movies = recommender.get_movies_seen_by_user(user_id)
        similar_users, distance_scores = recommender.get_similar_users(user_id, n_similar_users)
        recommended_movies = recommender.recommend_movies(user_id, n_similar_users, n_recommendations)
        
        st.subheader("Movies Watched by User (max 20 movies displayed)")
        if len(watched_movies) > 20:
            watched_movies = watched_movies[:20]
        
        watched_movies_df = pd.DataFrame({"Movie title": watched_movies})
        st.write(watched_movies_df if not watched_movies_df.empty else "No movies watched")
        
        st.subheader("Similar Users")
        similar_users_df = pd.DataFrame({"User ID": similar_users, "Distance Score": distance_scores})
        st.write(similar_users_df if not similar_users_df.empty else "No similar users found")
        
        st.subheader("Recommended Movies")
        recc_movies_df = pd.DataFrame({"Movie title": recommended_movies})
        st.write(recc_movies_df if not recc_movies_df.empty else "No recommendations available")
    else:
        st.error("User ID not found in dataset!")
