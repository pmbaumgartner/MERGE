# MERGE
Implementation of NLP algorithm that I developed for my dissertation


Readme for MERGE algorithm
Alexander Wahl, 2017

This program implements an algorithm I developed as part of my dissertation project. It searches for multi-word expressions (MWEs) in corpora on the basis of word co-occurrence statistics. It operates by first extracting all word bigrams in a corpus, whereby word bigrams are defined as any co-occurrence of 2 words in some defined order and at some defined distance from each other (see below for gapsize parameter). The algorithm then calculates the association strength between the two words on the basis of a log likelihood statistic; a bigram whose words tends to occur together more often than they do with other words will receive a higher score than a bigram whose words occur with other words more often than they occur together. The bigram with the highest score is selected as the winner and its tokens are “merged” into single representations—that is, the algorithm now views the bigram as a single “word.” At the same time, the 2 positions occupied by the word elements of the bigram are retained as part of the representation via placeholder objects, so the actual order of words in the corpus does not change. The merged tokens cause there to be new possible bigrams in the corpus (that is, co-occurrences between the now-merged 2-word sequence and other, 3rd words) and other bigrams to not exist anymore. Thus, the list of bigrams and their log likelihoods are updated. The winner is chosen, and the process repeats iteratively until some number of iterations has elapsed.

Ultimately, the program generates a json file called merge_output.json that contains the winner at each iteration along with its merge order and log likelihood score. Because at later iterations one of the “words” in the bigram can actually be the output from a previous iteration, the algorithm is able to grow longer and longer word sequences.

The program also requires that you have a set of corpus files placed in their own directory. They should have the extension “.txt.” Each corpus sentence should be on its own line. The program will interpret any characters between whitespace that it sees as a word, so remove any extra formatting beyond the raw words in the corpus. Note that homonyms with different casing will be interpreted as different words (e.g., Earth and earth). Thus, you should, for example, change uppercase letters at beginnings of sentences to lowercase.

This algorithm bears similarities that the algorithm developed by Wible, et al. (2006).


USING THE PROGRAM:

The program requires that you have the following packages installed: numpy, pandas, nltk


1. To run the program, cd to the directory in which you’ve placed the program files, then start python.


2. Next, you must import everything from the main module:

from MERGE_Main import *


3. Instantiate the top-level class. Note that you must pass the directory path where the corpus files are located. Note that I have provided a test corpus to use entitled “combined corpus.” This corpus is a combination of the Santa Barbara Corpus of Spoken American English and the spoken component of the ICE Canada corpus (see below for references).

i = ModelRunner(“/Users/JohnDoe/Documents/Corpora/TestCorpus/”)


4. You must then set the free parameters. These include the max gap size, measured in words, that can intervene between 2 words forming a bigram, as well as the number of iterations that the program should run for.

i.set_params(gapsize=1, iteration_count=10000):


5. Finally, you can run the program. Output will print to screen. (note that on a low-powered laptop the program may take several hours or days to run. If you have access to a server cluster, I recommend using this instead. The program has been tested on corpora up to 10 million words; this was quite memory intensive and indeed necessitated use of a server cluster. A more computationally efficient implementation is in the works.)

i.run()





References:

Du Bois, John W., Wallace L. Chafe, Charles Meyers, Sandra A. Thompson, Nii Martey, and Robert Englebretson (2005). Santa Barbara corpus of spoken American English. Philadelphia: Linguistic Data Consortium.

Newman, John and Georgie Columbus (2010). The International Corpus of English – Canada. Edmonton, Alberta: University of Alberta.

Wible, David, Chin-Hwa Kuo, Meng-Chang Chen, Nai-Lung Tsao, and Tsung-Fu Hung (2006). A computational approach to the discovery and representation of lexical chunks. Paper presented at TALN 2006. Leuven, Belgium.
