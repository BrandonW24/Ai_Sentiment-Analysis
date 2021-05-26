# assignment3 - Sentiment Analysis with Naive Bayes
- Brandon Withington
- CS331
- Dr. Rebecca Hutchinson
- 25 May 2021

## Running the code

The program looks for the following two files that were provided
in the program and homework description for this assignment :
trainingSet.txt and testSet.txt. Once the code is done running it
will produce a file called resulting_data.txt showing the results of
the program and the training / testing of naiive bayes. Each of the two
files previously stated must absolutely be in the same directory as the
main.py file otherwise it will not work.
To run the code you must first install several dependencies, spellchecker and
collections packages :
pip install pyspellchecker

once this is installed you can invoke the main.py file by typing :
python main.py


### Accuracy
:chart: In my tests, the sentiment analyzer achieved **79.7% accuracy**. Some ideas for improving accuracy are: 
1. Implementing a better way of spell-correction. This includes trimming down extra characters in otherwise correct words..
2. Increasing the amount of training data there is
