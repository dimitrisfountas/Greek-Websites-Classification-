# Greek-Websites-Classification-
1. Project Scope
The goal of the project is to perform classification on a collection of data that represent web
domains. Specifically, the purpose is to apply Machine Learning techniques in order to
predict which class represents each domain in the most optimal way. This task can be
classified in the problem categories of text classification combined with node classification,
since the data not only contain textual representation of the websites but we also obtain a
graph of the domains, based on the links between them.
2. Data
The data provided for text classification consist of Greek domains. It consists of four main
directories:
• Domains.zip: This file contains the textual content of the domains that are included
in our project.
• Edgelist.txt: This file contains a large part of the Greek web graph stored as an edge-
list. The nodes represent the domain names and edges represent the hyperlinks that
can be found in the domain.
• Train.txt: This file contains the domain names that will constitute the train set of the
classification task. These domains are already classified in one of the 9 classes, which
represent the theme of the domain.
• Test.txt: This file contains the domain names that will constitute the test set of the
classification task. Similarly, each of these domain names belongs to one of the 9
possible classes. There are the domains which the models will be evaluated on, on
Kaggle.
