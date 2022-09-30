## Description and Execution:

At first, the user would have to download the complete data set from the following link:
https://www.openicpsr.org/openicpsr/project/120321/version/V10/view;jsessionid=45163D0C8E64AC5F17A064ADD61FF2DB
The downloaded data set has to be named “COVID19_Twitter_Full_Dataset.csv”. After this, the user will run the “Tweet Reader” section of the notebook. This section will filter the data set on country and extract selected columns for specific year and month. For this example, we will be extracting data for April 2020. The most important column at this stage is tweets id column. Output from this section will be stored as “April_2020.csv”.
For the next step, the user would run “Tweet Sampler” section of the notebook. In this section, the user will sample 500K tweet ids randomly for each month and year. The output of this section will be stored as “April_2020_Sampled.csv”.
For the next step, the user would use the hydrator application. To use this application, the user will import the “April_2020_Sampled.csv” file into the hydrator application. The application will extract tweets from tweets IDs and the user should save the output as “April_2020_Hydrated.csv”. 
The user will input the output from the hydrator in the “Sentiment Analysis” section of our code. This section assigns sentiment to each tweet. The output from this section is stored as "April_2020_Sentiment_Analyzed.csv". 
Finally, the user will run the “Aggregation” section of our code. This section will aggregate the data in the required format, which will be later used to create our dashboard. This part of the section stores the aggregated data in three files which are saved as “April_2020_World_Map_Data.csv”, "April_2020_Hashtag_Summary.csv" and “"April_2020_Data_Top_Word.csv"

Due to the scale of our data set, our entire pipeline of our code, excluding hydrator section, was executed on the Google Cloud Platform. However, user can also use their local machine to run the code pipeline. 

## Installation:
We used the following the libraries for our project in Python:
* Pandas
* Numpy
* NLTK
* REGEX
* AST
* us

Hydrator application can be accessed from the following link: 
https://github.com/DocNow/hydrator/releases. 

Further instructions related to hydrator can be accessed from the following: link:
https://github.com/DocNow/hydrator/blob/main/README.md

Embeddings file can be downloaded from the following link:

https://nlp.stanford.edu/data/glove.twitter.27B.zip



