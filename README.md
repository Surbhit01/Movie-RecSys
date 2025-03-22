# Movie Recommendation System

### Movie recommendation system on MovieLens-100k dataset

The dataset contains 100000 ratings on a scale of 1 to 5 given by 943 users from 1682 movies.

The goal of the system is to build a recommendation system that will find users similar to a given user and suggest movies which the user might like based on the ratings given by the users for different movies.

#### Approach

1. Based on the given user id, find the list of movies watched by that user.
2. Find users similiar to the given user by using k nearest neighbours method, also find the distance for each user.
3. Normalize the above distances by dividing each score by the total score.
4. Find the feature matrix for the given user - this matrix represents the rating given by the user to all the movies watched. If a movie has not been watched, the score is considered to be 0.
5. Based on the above feature matrix, find similar users and the distance between them.
6. Find the feature representation for all the similar users and multiply the matrix with the corresponding user's weighted score. A more similar user will have a higher weightage.
7. Find the sum of the above matrix for all the movies. 
8. Sort the above scores and find the indexes of top scores.
9. Use these idx to find the names of the recommended movies. From this list, if a movie has not been watched by the user, recommend it.
