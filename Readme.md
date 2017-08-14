# Adaptive content replication using predictive analysis

### Adaptive Content Replication(ACR)

  - ACR, an easy to depoly adaptive replication sheme that provides highly available videos by autoscaling infrastructure to handle future requests.
  - ACR helps to predict popularity of video before it even becomes trending & provides better user expectations from CDN by promising higher QoS.

#### How it works
 1. *Datasets* : Used popularity growth curve from Youtube historical report API from numerous popular videos.
 2. *Learning Model*: Used Recurrent Neural Networks(RNN) to analyse patten over time series data and provides forecast on videos at real time.
 3. *Autoscaling*:  Infrastructure from servers to Kafka brokers are autoscaled based on forecast result, there by foreseeing possible issues or handling future requests at scale.

##### Architecture & Demo
![Architecture](https://github.com/arpith-kp/ACR/blob/master/LearnignPredicting.png)
![Demo](https://github.com/arpith-kp/ACR/blob/master/Demo.gif)

##### Tech stack
 - Python
 - Pandas
 - Scipy
 - Kafka
 - Tensorflow
 - Docker
 
##### Detailed architecture description

Using Youtube reporting metrics API fetch statistics for highly viewed videos globally. Once you have dataset ready, parse report metrics to get number of view counters by day. I'm using day & view count to train model, for better prediction I could also other metrics which helps to provide better result.

To train model, I'm using time series prediction using recurrent neural networks a commonly know **Long short-term memory network**, that is trained through backpropogation in time and overcomes the vanishing gradient problem. Unlike regression predictive modeling, time series adds complexity of sequence dependency among input variables so I've to be careful at each step of iterations while modeling. 

Now the problem I'm trying to solve narrows down given higly viewed videos over a period of time, what is possibility of video currently being viewed becomes viral.

The network has visible layer of 1 input, 100 hidden layer and 1 output. Network is trained for 1000 epoch, with mean squared error and Adam optimiser. Once we have train data, we can use this to analyse any realtime video metrics. I've used **Spearman's rank correlation** to compare result and get probability of new videos.

Video metrics are streamed in real time to Kafka from various producer and consumer batches these stream and passes it into Prediction model to get possibility of a video going to be trending. If so, since Kafka runs on Docker instance I can easily spin up & down Kafka nodes by dynamically modifying BrokerId and BrokerListener and updating it. 

###### References:

[Kafka Docker](https://github.com/spiside/kafka-cluster)
[Predictive & Adaptive replication](http://ieeexplore.ieee.org/abstract/document/6808201/)
[Time Series Prediction using TensorFlow](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

