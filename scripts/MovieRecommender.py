import numpy as np
import pandas as pd
import pickle
import os
from pprint import pprint

class MovieRecommender:
    def __init__(self, refined_dataset_path, model_name):
        """
        Initialize the MovieRecommender with necessary data.

        Parameters:
        -----------
        refined_dataset_path : pandas.DataFrame
            Path to the dataset containing user-movie interactions with columns 'user id' and 'movie title'
        user_to_movie_df : pandas.DataFrame
            User-movie matrix where rows represent users and columns represent movies
        knn_model : sklearn.neighbors.NearestNeighbors
            Trained KNN model for finding similar users
        """
        self.refined_dataset = pd.read_csv(refined_dataset_path)
        self.user_to_movie_df = self.refined_dataset.pivot(index='user id',columns='movie title',values='rating').fillna(0)
        self.model_path = os.path.join("model", model_name)
        self.knn_model = pickle.load(open(self.model_path, 'rb')) 
        self.movies_list = self.user_to_movie_df.columns

    def get_movies_seen_by_user(self, user_id):
        """
        Get list of movies seen by a specific user.

        Parameters:
        -----------
        user_id : int
            ID of the user

        Returns:
        --------
        list
            List of movie titles seen by the user
        """
        return list(self.refined_dataset[self.refined_dataset['user id'] == user_id]['movie title'])

    def get_similar_users(self, user_id, n_similar_users=5):
        """
        Find users similar to the given user based on movie ratings.

        Parameters:
        -----------
        user_id : int
            ID of the user
        n_similar_users : int, optional
            Number of similar users to find (default is 5)

        Returns:
        --------
        tuple
            (similar_users_ids, distances)
        """
        knn_input = np.asarray([self.user_to_movie_df.values[user_id-1]])

        distances, indices = self.knn_model.kneighbors(knn_input, n_neighbors=n_similar_users+1)

        # print(f"Top {n_similar_users} users who are very much similar to the User-{user_id} are: ")
        # print(" ")

        # for i in range(1, len(distances[0])):
        #     print(f"{i}. User: {indices[0][i]+1}, separated by distance of {distances[0][i]}")
        # print("")

        return indices.flatten()[1:] + 1, distances.flatten()[1:]

    def calculate_weighted_ratings(self, similar_user_list, distance_list):
        """
        Calculate weighted ratings based on similar users and their distances.

        Parameters:
        -----------
        similar_user_list : numpy.ndarray
            List of similar user IDs
        distance_list : numpy.ndarray
            List of distances for the similar users

        Returns:
        --------
        numpy.ndarray
            Weighted mean rating list for all movies
        """
        # Normalize the distance scores
        weightage_list = distance_list / np.sum(distance_list)

        # Ratings given to movies by similar users
        mov_rtngs_sim_users = self.user_to_movie_df.values[similar_user_list-1]

        # Prepare weightage list for broadcasting
        weightage_list = weightage_list[:, np.newaxis] + np.zeros(len(self.movies_list))

        # Finding weighted rating - product of ratings given by similar users
        new_rating_matrix = weightage_list * mov_rtngs_sim_users

        # Find the sum of each movie
        mean_rating_list = new_rating_matrix.sum(axis=0)

        return mean_rating_list

    def get_filtered_movie_recommendations(self, user_id, mean_rating_list, n_movies=10):
        """
        Filter movie recommendations based on weighted ratings and user history.

        Parameters:
        -----------
        user_id : int
            ID of the user
        mean_rating_list : numpy.ndarray
            Weighted mean ratings for all movies
        n_movies : int, optional
            Number of movie recommendations to return (default is 10)

        Returns:
        --------
        list
            List of recommended movie titles
        """
        # Find the first index where 0 occurs in the mean rating list
        zero_indices = np.where(mean_rating_list == 0)[0]
        if len(zero_indices) > 0:
            first_zero_index = zero_indices[-1]
        else:
            first_zero_index = len(mean_rating_list) - 1

        # Sort the ratings
        sortd_index = np.argsort(mean_rating_list)[::-1]
        sortd_index = sortd_index[:list(sortd_index).index(first_zero_index)]

        # Limit the number of recommendations
        n = min(len(sortd_index), n_movies)

        # Get movies watched by current user
        movies_watched = self.get_movies_seen_by_user(user_id)

        # Filter movies not yet watched by the user
        filtered_movie_list = list(self.movies_list[sortd_index])
        count = 0
        final_movie_list = []

        for i in filtered_movie_list:
            if i not in movies_watched:
                count += 1
                final_movie_list.append(i)
            if count == n:
                break

        return final_movie_list

    def recommend_movies(self, user_id, n_similar_users=5, n_movies=10):
        """
        Generate movie recommendations for a user.

        Parameters:
        -----------
        user_id : int
            ID of the user
        n_similar_users : int, optional
            Number of similar users to consider (default is 5)
        n_movies : int, optional
            Number of movie recommendations to return (default is 10)

        Returns:
        --------
        list
            List of recommended movie titles
        """
        # Display movies seen by the user
        movies_seen = self.get_movies_seen_by_user(user_id)
        #print("Movies seen by the User:")
        #pprint(movies_seen)
        #print("")

        # Find similar users
        similar_user_list, distance_list = self.get_similar_users(user_id, n_similar_users)

        # Calculate weighted ratings
        mean_rating_list = self.calculate_weighted_ratings(similar_user_list, distance_list)

        # Get filtered recommendations
        recommendations = self.get_filtered_movie_recommendations(user_id, mean_rating_list, n_movies)

        # print("")
        # print("Movies recommended based on similar users are:")
        # print("")

        return recommendations