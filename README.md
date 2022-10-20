# Neural Networks Recommendation System on Amazon Reviews Dataset using TensorFlow Recommenders. 

**Authors:** Elena Burlando

## How to Use This Repo
This project was developed in Google Colab. With respect to Colab's runtime restrictions, the decision was made to break down the project into multiple notebooks vs to keep everything in one notebook. 

EDA notebook is the first notebook that is covering data, business problem, data preeprocessing, and exploratory analysis. 

Model notebooks are located in the same GitHub repo under [models](https://github.com/rusalka013/Amazon_Rec_Systems/tree/main/models) folder. They are listed iin a numerical order with a short description of task in its name. Our best performing model is [model_5](https://github.com/rusalka013/Amazon_Rec_Systems/blob/main/models/model_5_Retrieval_item_item_fine_tuned_full_dataset.ipynb) (Retrieval item-to-item fine-tuned model). 


## Overview
For this project we will be exploring Neural Networks to build recommendation systems for Amazon Outdoor product reviews dataset. TensorFlow Recommenders will be used to develop Retrieval, Ranking, and Sequential algorithms. BruteForce and ScaNN will be tested for serving models. 

The objective:

* Explore Neural Networks by using TensorFlow Recommenders to address Cold Start problem. 

Projected Outcome: 
* Test Retrieval, Ranking, and Sequential models. 

Stakeholder:

* Marketing team at Amazon
* 

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
* The largest # of ratings given is 5 star followed by 4 star. 

![alt text](https://github.com/rusalka013/Amazon_Rec_Systems/blob/main/Visuals/Rating_Distribution.png)


## Historic Ratings Distribution

* Since 2008 the number of reviews has been progressively increasing each year with more than doubling in 2013.

![alt text](https://github.com/rusalka013/Amazon_Rec_Systems/blob/main/Visuals/Historic_Rating_Distribution.png)


## Top 5 Recommendations Based on Popularity


![alt text](https://github.com/rusalka013/Amazon_Rec_Systems/blob/main/Visuals/Most_popular_products.png)
 

## Item-to-item Retrieval Model
* Accuracy rate of  Top-10 recommendations is 63.26% on test data. 
* Roughly over half rcommendations are relevant to out test user whose previous purchase history indicated his/her interest in mountain biking. 


Top 10 Recommendations: 

![alt text](https://github.com/rusalka013/Amazon_Rec_Systems/blob/main/Visuals/Top-10_recs.png)
 


## Conclusions

Business Recommendations: 

* A/B test item-to-item Retrieval recommendation model on the website for Cold start problem. 
* Implement ScaNN for efficient and fast serving.

Next Steps: 
* Scrub current data from Amazon.com to run the models on the up-to-date dataset. 
* Further explore TensorFlow Recommenders to build advanced machine learning models. 
* Incorporate time series data to fine tune Sequential models (RNN: Recurrent Neural Networks).
* Run Convolutional Neural Networks (CNN) on image and video data. 





