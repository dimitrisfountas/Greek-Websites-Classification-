# Table of Contents

1. [Project Scope](#1-project-scope)  
2. [Data](#2-data)  
   2.1 [Data Preprocessing](#21-data-preprocessing)  
   2.2 [Data Exploration](#22-data-exploration)  
   2.3 [Text Representation](#23-text-representation)  
   2.4 [Graph Representation](#24-graph-representation)  
3. [Modelling](#3-modelling)  
   3.1 [Parameter Optimization](#31-parameter-optimization)  
   3.2 [Benchmark](#32-benchmark)  
   3.3 [Final Model](#33-final-model)  
   3.4 [Model Trials](#34-model-trials)  
4. [Further Improvements](#4-further-improvements)




## 1. Project Scope 
  The goal of the project is to perform classification on a collection of data that represent web 
domains. Specifically, the purpose is to apply Machine Learning techniques in order to 
predict which class represents each domain in the most optimal way. This task can be 
classified in the problem categories of text classification combined with node classification, 
since the data not only contain textual representation of the websites but we also obtain a 
graph of the domains, based on the links between them.  


## 2. Data 
  The data provided for text classification consist of Greek domains. It consists of four main 
directories: 
• Domains.zip: This file contains the textual content of the domains that are included 
in our project.  
• Edgelist.txt: This file contains a large part of the Greek web graph stored as an edge
list. The nodes represent the domain names and edges represent the hyperlinks that 
can be found in the domain. 
• Train.txt: This file contains the domain names that will constitute the train set of the 
classification task. These domains are already classified in one of the 9 classes, which 
represent the theme of the domain.  
• Test.txt: This file contains the domain names that will constitute the test set of the 
classification task. Similarly, each of these domain names belongs to one of the 9 
possible classes. There are the domains which the models will be evaluated on, on 
Kaggle. 


## 2.1 Data Preprocessing

  In order to clean up the text, we perform preprocessing suitable for natural language processing (NLP) tasks. Specifically, we utilize the **Greek language model from spaCy**, which includes components necessary for tasks such as tokenization, part-of-speech tagging, and lemmatization specific to the Greek language. Using a pre-trained model ensures accurate and efficient processing of Greek text by leveraging spaCy's robust NLP capabilities.

The preprocessing steps are as follows:

1. **Truncation**  
   We truncate the textual representations of the domains to an upper limit for both memory efficiency and to discard noise in the data. The chosen upper limit is the **average text length of 10,095 tokens**.

2. **Removal of Line Breaks and URLs**  
   We remove newline characters and URLs, which are common in the texts due to the nature of the data.

3. **Normalization**  
   Accents are removed (due to the Greek language), and all characters are converted to lowercase. This step supports text normalization and aligns with the requirements of the embedding model used later.

4. **Lemmatization and Filtering**  
   Each token is:
   - Lemmatized to reduce it to its base form
   - Filtered to remove non-alphabetic tokens
   - Stripped of stop words and tokens with fewer than three characters
    Lemmatization helps unify different word forms and reduces noise, while filtering retains only meaningful words—important in Greek, where short words often carry  little semantic value. This results in more compact and computationally efficient text representations.

This preprocessing pipeline as described above effectively prepares Greek text for the 
following classification tasks, by ensuring the domain data is consistent, simplified, and 
focused on meaningful content. 
All the steps above are performed on the train and test domains, separately.  

## 2.2. Data Exploration 
  Following on our analysis, we perform Data exploration to better understand the available 
data. To do so, we combine the information from the textual context of the domains and the 
information from the graph and the connectivity of the nodes.  
Starting on our analysis, we study the class distribution of the train dataset. As we can see 
on the figure below, the classes are highly imbalanced. The Class 3 seems to have the 
highest percentage of the train samples, while the Classes 0, 6 and 7 have the lowest 
percentages.  



