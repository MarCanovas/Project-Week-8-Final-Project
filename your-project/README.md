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

## Dataset
The data was collected by various queries to the Twitter API automatized from a SQL database. Once the messages were collected on the database, I classified them through a custom API and return the messages classified to the database. 

There were a total of 8 different queries regarding 8 different topics surrounding misogynoys issues.

## Cleaning
For the cleaning process each of the messages was set to lowercase, removed the special characters from the hashtags and usernames but keeping the actual word so we can set the relation that has with the text and every link or image was also completely removed from the message.

## Analysis
After the preprocessing, the messages were transformed into three different word-embedding vectors (Glove, FastText and L-Model) and inputed into three different Deep Learning models (CNN, LSTM and Bi-LSTM).


## Conclusion
The results obtained by the trained models overpass the common baseline of accuracy in this kind of workshops which is 70%. 

Furthermore, misogyny is not an exact science and even different people could disagree to classify a message into misogynous or not, so we cannot expect a machine to do so. 

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
