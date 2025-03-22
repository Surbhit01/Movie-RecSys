import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle, os

class ModelBuilder:
    def __new__(cls, prepared_data_path):
        instance = super(ModelBuilder, cls).__new__(cls)
        dataset_path = os.path.join("data", prepared_data_path)
        dataset = pd.read_csv(dataset_path)
        user_to_movie_df = dataset.pivot(
            index='user id', 
            columns='movie title', 
            values='rating'
        ).fillna(0)
        
        model_name = "knn_model.pkl"
        model_folder = "model"
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, model_name)

        user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(user_to_movie_sparse_df)

        pickle.dump(model, open(model_path, 'wb'))
        print(f"Model saved successfully to {model_path}")

        return model