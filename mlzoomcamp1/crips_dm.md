# Metholody for organizing machine learning projects


## Business Understanding

***Identify the business problem and how can we solve it?***
***Question do we actually need machine learning?***

1. Our users complain about spam
2. Analyze to what extent it's a problem
3. Will machine learning help?
4. If not; propose an alternative solution(Rule based system).

***Define the goal***
_Reduce the amount of span messages or_
_Reduce the amount of complaints about spam_

***The goal has to be measurable**
_Reduce the spam message by 50%_

## Data Understanding

***Analyze available data sources, decide if we need to get more data***

***Identify the data sources***

1. We have a report spam button
2. Is the data behind this button good enough?
3. Is it reliable?
4. Do we track it correctly?
5. Is the dataset large enough?
6. Do we need to get more data?


***Identify the data sources***
1. It may influence the goal
2. We may go back to the previous step and adjust it.


## Data Preparation

***Transform the data so it can be put into a ML algorithm***


1. Clean the data, remove noises
_Raw data -> remove noises -> Convert it to a good form_
2. Build the pipelines
3. Convert into tabular form

## Modelling

***Train the machine learning model***

1. Try different model
2. Select the best model

#### Example of models

1. Logistic Regression
2. Decision Tree
3. Neural Network
4. and many more

#### Sometimes we may go back to data preparation

1. Add new features
2. Fix the issues

## Evaluation -> Business Understanding

***Measure how well the model solves the business problem***

_Check the actual goal and ask yourself " Have you reach the goal?_

1. Is the model good enough?
2. Have we reached our goal?

#### Do a retrospective(backward looking)

1. Was the goal achievable?
2. Did we solve/measure the right thing?

#### After that , we may decide to

1. Go back and adjust the model
2. Roll the model to more users/all users
3. stop working to the project

## Deployment

_evaluation + deployment happens together_
_Deploy the model to production_

1. Online evaluation: evaluation of live users
2. It means: deploy the model, evaluate it

#### After some time
1. Do proper monitoring
2. Ensuring the quality and maintainability


## Iterate

_ML projects require many iterations_
_how can we actually improve a model? or should we improve it?_

## Tips

1. Start simple (from simple model if it works)
2. Learn from feedback
3. Improve
