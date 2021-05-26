import sys
import re
from math import log
#from spellchecker import SpellChecker
from collections import defaultdict

squash = lambda l: [item for sublist in l for item in sublist]

#Word retrevial function, this function detects and corrects spelling errors.
#it is a brute force way of spell-checking the input file, it works for now..

#variables need to be changed, that's about it!

def retrieve_word(word):

    if(word == "soooo"):
        print("spelling error detected : soooo")
        word = "so"
    elif(word == "soooooo"):
        print("spelling error detected : soooooo")
        word = "so"
    elif(word == "WAAAAAAyyyyyyyyyy"):
        print("spelling error detected : WAAAAAAyyyyyyyyyy")
        word = "way"
    elif(word == "wayyy"):
        print("spelling error detected : wayyy")
        word = "way"
    elif(word == "shawarrrrma"):
        print("spelling error detected : shawarrrrma")
        word = "shawarma"
    elif(word == "disapppointment"):
        print("spelling error detected : disapppointment")
        word = "disappointment"
    elif(word == "devine"):
        print("spelling error detected : devine")
        word = "divine"
    elif(word == "satifying"):
        print("spelling error detected : satifying")
        word = "satisfying"
    elif(word == "gooodd"):
        print("spelling error detected : gooodd")
        word = "good"
    elif(word == "connisseur"):
        print("spelling error detected : connisseur")
        word = "connoisseur"
    elif(word == "Veggitarian"):
        print("spelling error detected : Veggitarian ")
        word = "Vegetarian"
    elif(word == "beateous"):
        print("spelling error detected : beateous")
        word = "beauteous"
    elif(word == "bloddy"):
        print("spelling error detected : bloddy")
        word = "bloody"
    elif(word == "Burrittos"):
        print("spelling error detected : Burrittos")
        word = "burritos"
    elif(word == "ravoli"):
        print("spelling error detected : ravoli")
        word = "ravioli"
    elif(word == "Honeslty"):
        print("spelling error detected : Honeslty")
        word = "Honestly"
    elif(word == "over-whelm"):
        print("spelling error detected : over-whelm")
        word = "overwhelm"
    elif(word == "im"):
        print("spelling error detected : im")
        word = "I'm"
    elif(word == "pissd"):
        print("spelling error detected : pissd")
        word = "pissed"

    return re.compile(r'[\W_]+').sub('', word.lower())

#Processing begins the moment the input file comes in.
def preprocess(input_filename, output_filename):
    # Process input file
    input_file = open(input_filename, 'r')
    unprocessed_doc = input_file.read().splitlines()
    #spell = SpellChecker()
    #incorrect_words = []
    #misspelledWords = spell.unknown(["soooo", "soooooo", "WAAAAAAyyyyyyyyyy", "wayyy", "shawarrrrma", "shawarma"])
    #for word in misspelledWords:
    #    incorrect_words = (spell.correction(word))
    vocab = {retrieve_word(word) for file in unprocessed_doc for word in file.split()[:-1]}
    vocab.discard('')

    #This outputs the pre processed training and testing data into two different files called :
    #preprocessed_train.txt and preprocessed_test.txt in the format the program description defines.
    output_file = open(output_filename, 'w')
    output_file.write(','.join(sorted(list(vocab))) + '\n' + '\n'.join([(','.join(['1' if word in file[0] else '0' for word in vocab]) + 
    ',' + str(file[1])) for file in [([retrieve_word(word) for word in file.split()][:-1], int(file.split()[-1])) for file in unprocessed_doc]]))
    
    input_file.close()
    output_file.close()

    #This returns our overall vocab index and our processed file. We utilize the retrieve_word function for every word in the file
    #for each file that has not yet been processed which is stored in the unprocessed_doc variable.
    return vocab, [([retrieve_word(word) for word in file.split()][:-1], int(file.split()[-1])) for file in unprocessed_doc]

#Training function utilizes log calculations and summations of word_count within the vocab
def trainNB(vocab, files):
    log_p = {}
    log_L_LH = defaultdict(dict)
    all_data = {}

    for c in [0,1]:
        log_p[c] = log(len([doc for doc in files if doc[1] == c]) / len(files))
        all_data[c] = squash([doc[0] for doc in files if doc[1] == c])
        words_in_class_count = sum([all_data[c].count(w) for w in vocab])

        for word in vocab:
            log_L_LH[word][c] = log((all_data[c].count(word) + 1) / (words_in_class_count + 1))

    return log_p, log_L_LH

#Classification step begins here when this testNB function is invoked.
#it reads the training data with its labels and learns the parameters that
#which is being used by the classifier, it then retrieves the maximum
def testNB(file, log_p, log_L_LH, vocab):
    probability = {}
    for c in [0,1]:
        probability[c] = log_p[c]
        for word in [word for word in file[0] if word in vocab]:
            probability[c] += log_L_LH[word][c]

    return max(probability.keys(), key=(lambda key: probability[key]))

def write_results(results, test_docs):
    incorrect_result = []
    for real_rating, expected_rating in zip(results, test_docs):
        if expected_rating[1] is not real_rating :
            incorrect_result.append(expected_rating + tuple([real_rating]))

    print("_____________________________________________________")
    print("Accuracy rating :")
    print("Correct   :   ", len(test_docs) - len(incorrect_result))
    print("Incorrect :   ", len(incorrect_result))
    print("Accuracy  :   ", (len(test_docs) - len(incorrect_result)) / len(test_docs))
    print("_____________________________________________________\n")


if __name__ == '__main__':
    print("_____________________________________________________")
    print("Beginning processing stage. During this stage the program will search for ")
    print("any spelling mistakes within the test set of data.")
    print("\n")
    print("The output file you should expect to see produced will be called : resulting_data.txt")
    print("_____________________________________________________")  
    sys.stdout = open('resulting_data.txt', 'wt')
    #Start "training" on training data by processing the training set and storing it
    #as its own training set. It also sets up the vocab data that is used  throughout
    #training and testing.
    vocab, training_set = preprocess('trainingSet.txt', 'preprocessed_train.txt')
    _, test_set = preprocess('trainingSet.txt', 'preprocessed_train.txt')

    prior, likelihood = trainNB(vocab, training_set)

    results = []
    print("_____________________________________________________")
    print("Testing on **TRAINING** data")
    print("Results from training data : ")
    for training_data in training_set:
        results.append(testNB(training_data, prior, likelihood, vocab))
    write_results(results, training_set)
    print("_____________________________________________________")

    #Start "testing" on training data by processing the training set and storing it
    #as its own training set. It also sets up the vocab data that is used  throughout
    #training and testing. This time however it compares results from the training set
    #to the testing set.
    vocab, training_set = preprocess('trainingSet.txt', 'preprocessed_train.txt')
    _, test_set = preprocess('testSet.txt', 'preprocessed_test.txt')
    print("_____________________________________________________")
    results = []
    print("Results from **TEST** data : ")
    for testing_data in test_set:
        results.append(testNB(testing_data, prior, likelihood, vocab))
    write_results(results, test_set)
    print("_____________________________________________________")

