# Neural Networks Recommendation System on Amazon Reviews Dataset using TensorFlow Recommenders. 

**Authors:** Elena Burlando

## Overview
For this project we will be exploring Neural Networks to build recommendation systems for Amazon Outdoor product reviews dataset. TensorFlow Recommenders will be used to develop Retrieval, Ranking, and Sequential algorithms. BruteForce and ScaNN will be tested for serving models. 

The objective:

* Explore Neural Networks by using TensorFlow Recommenders to address Cold Start problem. 

Stakeholder:

* Marketing team at Amazon


## Business Problem

According to Statista, the Sports and Outdoor market segment is projected to reach 20.75bn US dollars in 2022 with an annual growth rate of 13.29% (link to source). User penetration is expected to be 20% in 2022 and 22.2% by 2025. However, with a projected market volume of 23,890.00m US dollars in 2022, most revenue will be coming from China. The average revenue per customer is set to be 310.00 US dollars.

With the above statistics in mind, Amazon has been looking into further increasing user penetration locally in US thus driving increase in sales and revenue. One of the strategies is to build a rec system to deal with a cold-start problem in order to acquire new users. For this problem we will explore neural networks by using TensorFlow Recommenders.


## Data Understanding and Methods

* [Amazon Reviews: Outdoors dataset](https://www.tensorflow.org/datasets/catalog/amazon_us_reviews)

Amazon Review datasets has an expansive records of reviews spanning for over 20 years from 1995 through 2015. I picked 'Outdoor' category due to my passion for nature and assumption that to some extend most people have at least one outdoor activity that they like. The last point is to find a few people to test a final model in real life. 
Outdoor dataset has over 2.3M review entries. I will only use 5% of data that corresponds to 115120 entries to develop and test models. 

A note on dataset (from Amazon): 
* Dataset is in TCV file format. 
*  Each line in the data files corresponds to an individual review (tab delimited, with no quote and escape characters).
* 'product_pareent' - random ideentifier that can be used to aggregate reviews for the same product. 
* 'total_votes' - number of total votes the review received.
* 'helpful votes' - number of helpful votes.
* 'vine' - review has been written as a part of the Vine program.
* Rating Scale is 1-5. 
 
**Target & Metrics**:  
* The target variable is rating.
* The main metric is Accuracy Rate and RMSE for Ranking models. 
* The second important metric is Top-n recommendations. The final algorithms will be tested on multiple users to determine model subjective performance. 
* We intend to use Neural Network TensorFlow Recommenders to test and develop a new Recommendation system.  
* We will be following CRISP-DM process for this project.



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




