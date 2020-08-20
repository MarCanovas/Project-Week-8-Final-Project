<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Automatic Misogyny Identification
*Mar Cánovas*

*Data Analysis Full Time, Barcelona, June 2020*

## Content
- [Project Description](#project-description)
- [Hypotheses / Questions](#hypotheses-questions)
- [Dataset](#dataset)
- [Cleaning](#cleaning)
- [Analysis](#analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Workflow](#workflow)
- [Organization](#organization)
- [Links](#links)

## Project Description
Online social networks allow powerless people to gain enormous amounts of control over particular people's lives and profit from the anonymity or social distance that the Internet provides in order to harass other people. 

One of the most frequently targeted groups comprise women, as misogyny is, unfortunately, a reality in our society. However, although great efforts have recently been made to identify misogyny, it is still difficult to distinguish as it can sometimes be very subtle and deep, signifying that the use of statistical approaches is not sufficient. Moreover, as Spanish is spoken worldwide, context and cultural differences can complicate this identification.

In this project we are going to analyze a custom collected and classified corpus with word-embedding Deep Learning techniques to generate a model that is able to identify misogynous tweets.

## Hypotheses / Questions
The main point of this project is the believing that eventhough sex oriented hate is considered in normal hate speech detectors, they do not entail all the non aggresive traits that misogyny entails, and that should also be considered as hate towards someone and should not be allowed on social networks.

Regarding this, it arised the question of the 

## Dataset
* Where did you get your data? If you downloaded a dataset (either public or private), describe where you downloaded it and include the command to load the dataset.
* Did you build your own datset? If so, did you use an API or a web scraper? PRovide the relevant scripts in your repo.
* For all types of datasets, provide a description of the size, complexity, and data types included in your dataset, as well as a schema of the tables if necessary.
* If the question cannot be answered with the available data, why not? What data would you need to answer it better?

## Cleaning
For the cleaning process each of the messages was set to lowercase, removed the special characters from the hashtags and usernames but keeping the actual word so we can set the relation that has with the text and every link or image was also completely removed from the message.

## Analysis
* Overview the general steps you went through to analyze your data in order to test your hypothesis.
* Document each step of your data exploration and analysis.
* Include charts to demonstrate the effect of your work.
* If you used Machine Learning in your final project, describe your feature selection process.

## Model Training and Evaluation
*Include this section only if you chose to include ML in your project.*
* Describe how you trained your model, the results you obtained, and how you evaluated those results.

## Conclusion
* Summarize your results. What do they mean?
* What can you say about your hypotheses?
* Interpret your findings in terms of the questions you try to answer.

## Future Work
Unfortunately, I'm sure that many more situations where women will be indiscriminately harassed online, so increasing the dataset range to cover every area possible could become a never ending process.

Also, there are many different Machine Learning and Deep Learning techniques available that could overpass the accuracy obtained with the selected models, so trying a variety of new algorithms is also a good approach to improve our actual results.

Furthermore, as an advance improvement, we could consider the option of classifying each message into the different sub-categories that non aggresive misogyny can be divided into.

## Workflow
The starting point of this project was the collection of data and defining which topics we wanted to cover with it. When this was defined, we created the databases with their queries to call the Twitter API. Then, once the dataset is defined, I preprocessed every message and set up the script that transformed it into a word-embedding vector and created a model. Once everything is set, I tested it all and analysed the results. 

The accuracy was measured as an average of 10 executions of a 10 fold cross-validation on each model created.

## Organization
The workflow was represented on the trello project linked below.

The repository is structured in directories that contain each type of documents that is used or generated in order to obtain the results presented in this project.

## Links

[Repository](https://github.com/MarCanovas/Project-Week-8-Final-Project/)  
[Slides](https://docs.google.com/presentation/d/1MwYAKmKscyIQbriB6rpH5VNpvduWsNNiS2oBdFeWvc8/edit?usp=sharing)  
[Trello](https://trello.com/b/8wJm3RjN/final-project)  
