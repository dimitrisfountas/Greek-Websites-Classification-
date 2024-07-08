<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Greek Websites Classification Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1, h2 {
            color: #2c3e50;
        }
        p {
            margin-bottom: 1em;
        }
        ul {
            list-style-type: disc;
            margin-left: 20px;
        }
    </style>
</head>
<body>

    <h1>Greek Websites Classification Project</h1>

    <h2>1. Project Scope</h2>
    <p>The goal of the project is to perform classification on a collection of data that represent web domains. Specifically, the purpose is to apply Machine Learning techniques in order to predict which class represents each domain in the most optimal way. This task can be classified in the problem categories of text classification combined with node classification, since the data not only contain textual representation of the websites but we also obtain a graph of the domains, based on the links between them.</p>

    <h2>2. Data</h2>
    <p>The data provided for text classification consist of Greek domains. It consists of four main directories:</p>
    <ul>
        <li><strong>Domains.zip:</strong> This file contains the textual content of the domains that are included in our project.</li>
        <li><strong>Edgelist.txt:</strong> This file contains a large part of the Greek web graph stored as an edge-list. The nodes represent the domain names and edges represent the hyperlinks that can be found in the domain.</li>
        <li><strong>Train.txt:</strong> This file contains the domain names that will constitute the train set of the classification task. These domains are already classified in one of the 9 classes, which represent the theme of the domain.</li>
        <li><strong>Test.txt:</strong> This file contains the domain names that will constitute the test set of the classification task. Similarly, each of these domain names belongs to one of the 9 possible classes. These are the domains on which the models will be evaluated on Kaggle.</li>
    </ul>
    <p>Please read the file <strong>Report.pdf</strong> to understand the project and our proposed solution to it.</p>

</body>
</html>
