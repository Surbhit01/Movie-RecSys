import io, os, requests, zipfile
import pandas as pd

class DatasetPreparation:
  def __init__(self, dataset_url):
    files = self.download_and_extract_zip(dataset_url)
    
    overall_stats = pd.read_csv('data/ml-100k/u.info', header=None)
    print("Movie lens dataset -: ",list(overall_stats[0]))
    
    # item id is same as movie id, item id column is renamed as movie id
    column_names1 = ['user id','movie id','rating','timestamp']
    dataset = pd.read_csv('data/ml-100k/u.data', sep='\t',header=None,names=column_names1)
    
    cols = 'movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western'
    column_names2 = cols.split(' | ')

    items_dataset = pd.read_csv('data/ml-100k/u.item', sep='|',header=None,names=column_names2,encoding='latin-1')

    print(f"Movie id range : {min(dataset['movie id'])} to {max(dataset['movie id'])}")
    print(f"Total number of movies {items_dataset['movie id'].nunique()}")

    movie_dataset = items_dataset[['movie id','movie title']]

    print(f"Length of movie datset : {len(movie_dataset)}")

    merged_dataset = pd.merge(dataset, movie_dataset, how='inner', on='movie id')
    duplicates = items_dataset.groupby("movie title")["movie id"].nunique()

    movies_with_duplicates_ids = duplicates[duplicates > 1].index

    refined_dataset = merged_dataset.groupby(by=['user id','movie title'], as_index=False).agg({"rating":"mean"})
    
    # Store the refined dataset for later use
    self.refined_dataset = refined_dataset
    refined_dataset.to_csv("data//refined_dataset.csv", index=False)
    print(f"Dataset stored successfully to data/")

  # Fixed method definition - added self parameter
  def download_and_extract_zip(self, url, extract_to='data', chunk_size=8192):
      try:
          # Create the extraction directory if it doesn't exist
          os.makedirs(extract_to, exist_ok=True)

          # Download the file
          print(f"Downloading zip file from {url}...")
          response = requests.get(url, stream=True)
          response.raise_for_status()  # Raise an exception for HTTP errors

          # Download and store in memory
          zip_content = io.BytesIO()
          for chunk in response.iter_content(chunk_size=chunk_size):
              if chunk:
                  zip_content.write(chunk)

          # Reset the pointer to the beginning of the BytesIO object
          zip_content.seek(0)

          # Extract the zip file
          with zipfile.ZipFile(zip_content) as z:
              z.extractall(extract_to)
              extracted_files = z.namelist()

          return extracted_files

      except requests.exceptions.RequestException as e:
          print(f"Error downloading the data: {e}")
          raise
      except zipfile.BadZipFile:
          print("The downloaded file is not a valid zip file.")
          raise
      except Exception as e:
          print(f"An unexpected error occurred: {e}")
          raise