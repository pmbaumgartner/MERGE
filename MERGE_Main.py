from __future__ import division
import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import itertools

import numpy as np
import pandas as pd
from nltk import FreqDist
import json

from Corpus_Initiator import *
from Bigram_Initiator import *
from Table_Initiator import *
from LL_Manager import *
from Bigram_Updater import *
from Frequency_Updater import *

NTword = namedtuple('NTword', 'wordstr position')             # (word,0)
NTlexeme = namedtuple('NTlexeme','lex token_index')


#corpus_dir_path = "/Users/alexanderwahl/Documents/Corpora/Combined_corpus/"
#corpus_dir_path = "/Users/alexanderwahl/Documents/Corpora/BNC_sample/"
#corpus_dir_path = "/Users/alexanderwahl/Documents/Corpora/BNC_cleaned/"

class ModelRunner(object):
    
    def __init__(self,corpus_dir_path="/"):
        
        self.corpus_dir_path = corpus_dir_path
        self.log_likelihood_manager = LogLikelihoodManager()
        self.bigram_updater = BigramUpdater()
        self.frequency_updater = FrequencyUpdater()
        self.merge_tracker = {}
        
    def set_params(self,gapsize=0,iteration_count=10001):
        
        self.gapsize = gapsize
        self.iteration_count = iteration_count
        
        self.corpus_initiator = InitCorpus()
        self.corpus_initiator.initialize(self.corpus_dir_path)
        self.all_lexemes = self.corpus_initiator.all_lexemes
        self.corpus_size = self.corpus_initiator.corpus_size
        
        self.bigram_initiator = InitBigramData()
        self.bigram_initiator.initialize(self.all_lexemes,self.gapsize)
        self.all_bigrams = self.bigram_initiator.bigrams
        
        self.candidate_table_initiator = InitCandidateTable()
        self.candidate_table_initiator.initialize(self.all_bigrams,self.all_lexemes)
        self.all_tables = self.candidate_table_initiator.all_tables
    
    def run(self):
        
        for iteration in range(1,self.iteration_count):
            self.curr_iteration = iteration
            self.log_likelihood_manager.calculate(self.all_tables,self.corpus_size)
            self.winner_info = self.log_likelihood_manager.get_winner()
            print self.curr_iteration,self.winner_info.winner
            self.merge_token = self.merge(self.winner_info.winner)
            self.merge_tracker[self.curr_iteration] = (self.winner_info.winner,self.winner_info.winner_ll)
            if iteration%100==0:
                with open("merge_output.json","w") as f:
                    json.dump(self.merge_tracker,f)
            self.bigram_updater.get_new_and_conflicting_bigrams(self.all_bigrams,
                                                                self.all_lexemes,
                                                                self.merge_token,
                                                                self.winner_info,
                                                                self.gapsize)
            self.new_bigrams = self.bigram_updater.new_bigrams
            self.conflicting_bigrams = self.bigram_updater.conflicting_bigrams
            self.merge_token_count = self.bigram_updater.merge_token_count
            self.all_lexemes = self.bigram_updater.all_lexemes
            self.corpus_size -= self.merge_token_count
            self.frequency_updater.set_new_freqs_for_elements_in_winner(self.all_lexemes,
                                                                        self.winner_info,
                                                                        self.merge_token_count)
            self.frequency_updater.process_bgrs_with_same_element_types_as_winner(self.gapsize,
                                                                              self.all_tables,
                                                                              self.all_bigrams)
            self.frequency_updater.add_new_bigrams(self.new_bigrams)
            self.frequency_updater.update_conflicting_bigram_freqs(self.conflicting_bigrams)
            self.frequency_updater.remove_winner()
            self.all_bigrams = self.frequency_updater.all_bigrams
            self.all_lexemes = self.frequency_updater.all_lexemes
            self.all_tables = self.frequency_updater.all_tables
        
    def merge(self, bigram):

        self.bigram = bigram
        self.el1_words = list(self.bigram.el1.lex)
        self.repositioned_el2_words = [NTword(wordstr=word, position=pos+self.bigram.gapsize+1) for word,pos in self.bigram.el2.lex]
        self.all_words = self.el1_words + self.repositioned_el2_words
        self.all_words.sort(key=lambda tup: tup[1])
        return NTlexeme(lex=tuple(self.all_words), token_index=0)
    
                
    

#if __name__=="__main__":
#    instance = ModelRunner("/vol/tensusers/awahl/BNC_cleaned/")
#    instance.set_params()
#    instance.run()
    
            

                    
                    
                    


