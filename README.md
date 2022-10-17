# Movie Recommendation System

**Authors:** Elena Burlando

## Overview

For this project we will be analyzing MovieLens data in order to develop a better performing recommendation system that will generate Top-N recommended movies to a client.

The objective:

* to develop a Top-N movie recommendations for existing clients with an improved Cumulative Hit Rate (to ~ 0.05) for ratings >= 4.0.

Stakeholder:

* Engineering team


## Business Problem

A movie streaming company is looking to fine-tune its recommendation system. After a recent survey and a followed up research it is finding that its current recommendation system is too overwhelming to clients and presents too many options (Top-N = 100) with some that are clearly irrelevant. Low engagement is also indicated by a low Hit Rate (0.01) of top recommenders.

One of the main pain points for this project is a lack of active environment (a website) to perform A/B testing which would be the ultimate test for a new system. The second pain point is the lack of implicit data such as minutes watched, clicks, browsing history, etc.


## Data Understanding and Methods

* [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/)

The dataset 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018. 

**Data is broken down into four datasets**: 
 * links: 'movieId', 'imdbId', 'tmdbId'
 * movies: 'movieId', 'title', 'genres' 
 * ratings: 'userId',  'movieId', 'rating', 'timestamp' 
 * tags: 'userId', 'movieId', 'tag', 'timestamp' 
 
**Target and methods**:  
* The target variable is rating.  
* The main metric is Cumulative Hit Rate. Other metrics that wil be used are listed below.  
* We intend to use Surprise library to test and develop a new Recommendation system.  
* For hyper parameter tuning we will use GridSearchCV and RandomizedSearchCV. 
* We will be following CRISP-DM process for this project. 


**Metrics**:

| **Metric** | **Description** | **Interpretation** |
|---|---|---|
| **RMSE** | Root Mean Squared Error. | Lower values mean better accuracy. |
| **MAE** | Mean Absolute Error. | Lower values mean better accuracy. |
| **HR** | Hit Rate; how often we are able to recommend a left-out rating. | Higher is better. |
| **rHR** | Rated Hit Rate; hit rate broken down by rating scale. | Higher is better. |
| **cHR** | Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. | Higher is better. |
| **ARHR** | Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. | Higher is better. |
| **Coverage** | Ratio of users for whom recommendations above a certain threshold exist. | Higher is better. |
| **Diversity** | 1-S, where S is the average similarity score between every possible pair of recommendations for a given user. | Higher means more diverse. |
| **Novelty** | Average popularity rank of recommended items. | Higher means more novel. |

Above metrics and coding associated with it came from Frank Kane's ['Building Recommender Systems with Machine Learning and AI' course'](https://www.linkedin.com/learning/building-recommender-systems-with-machine-learning-and-ai/)


## Rating Distribution:

* Rating scale is 0-5. 
* Rating distribution is skewed left indicating that customers tend to leave positive ratings. 
* The largest # of rating given is 4.0 followed by 3.0, 3.5 and 5. 

![alt text](https://github.com/rusalka013/recommendation_project/blob/main/Visuals/Rating_Distribution.png)


## Ratings Distribution by User

* Ratings distribution by user is skewed right indicating that majority of users have left 20+ reviews. 
* Few users left 100+ reviews. 

![alt text](https://github.com/rusalka013/recommendation_project/blob/main/Visuals/Ratings_dist_by_user.png)


## Top 20 Recommendations Based on Popularity


![alt text](https://github.com/rusalka013/recommendation_project/blob/main/Visuals/Ratings_dist_by_movie.png)
 

## Untuned SVD Model

Metrics: 

![alt text](https://github.com/rusalka013/recommendation_project/blob/main/Visuals/SVD_metrics.png)

Top 10 Recommendations: 

![alt text](https://github.com/rusalka013/recommendation_project/blob/main/Visuals/SVD_recs.png)
 

## Untuned SVD++ MOdel 

Metrics:

![alt text](https://github.com/rusalka013/recommendation_project/blob/main/Visuals/SVDpp_metrics.png)

Top 10 Recommendations: 

![alt text](https://github.com/rusalka013/recommendation_project/blob/main/Visuals/SVDpp_recs.png)


## Conclusions

Business Recommendations: 

* Implement Popularity-based recommendation system for new users.
* A/B test SVD and SVD++ recommendation models on website. 

Next Steps: 
* Further fine tune SVD and SVD++ models
* Try integrating other features into content-based recommendation system such as year of release, movie details, tags.
* Potentially develop a hybrid model to incorporate different rec systems. 
* Allow for input of reviews for new users. 
* Try neural networks for developing recommendation systems. 




