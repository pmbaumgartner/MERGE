from __future__ import division
import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import itertools

import numpy as np
import pandas as pd
from nltk import FreqDist

from Table_Initiator import ColumnLists
     
        
class FrequencyUpdater(object):


    def set_new_freqs_for_elements_in_winner(self, all_lexemes, winner_info, merge_token_count):
        
        self.all_lexemes = all_lexemes
        self.winner_info = winner_info
        self.merge_token_count = merge_token_count
    
        self.all_lexemes.deduct_freq(lexeme=self.winner_info.winner.el1, deduction=self.merge_token_count)
        self.new_el1_freq = self.all_lexemes.get_frequency(self.winner_info.winner.el1)
        self.all_lexemes.deduct_freq(lexeme=self.winner_info.winner.el2, deduction=self.merge_token_count)
        self.new_el2_freq = self.all_lexemes.get_frequency(self.winner_info.winner.el2)
        
    def process_bgrs_with_same_element_types_as_winner(self, max_gapsize, all_tables, all_bigrams):
        
        self.max_gapsize = max_gapsize
        self.all_tables = all_tables
        self.all_bigrams = all_bigrams
        
        self.bigrams_whose_el1_is_el1_of_winner = self.all_bigrams.get_bigrams_containing(lexeme=self.winner_info.winner.el1,
                                                                                          in_position=1,
                                                                                          max_gapsize=self.max_gapsize)
        self.update_element_freqs_in_tables(bigrams=self.bigrams_whose_el1_is_el1_of_winner,
                                            value=self.new_el1_freq,
                                            column="El1 Freq")
        
        
        self.bigrams_whose_el1_is_el2_of_winner = self.all_bigrams.get_bigrams_containing(lexeme=self.winner_info.winner.el1,
                                                                                          in_position=2,
                                                                                          max_gapsize=self.max_gapsize)
        self.update_element_freqs_in_tables(bigrams=self.bigrams_whose_el1_is_el2_of_winner,
                                            value=self.new_el2_freq,
                                            column="El1 Freq")
        
        
        self.bigrams_whose_el2_is_el1_of_winner = self.all_bigrams.get_bigrams_containing(lexeme=self.winner_info.winner.el2,
                                                                                          in_position=1,
                                                                                          max_gapsize=self.max_gapsize)
        self.update_element_freqs_in_tables(bigrams=self.bigrams_whose_el2_is_el1_of_winner,
                                            value=self.new_el1_freq,
                                            column="El2 Freq")
        
        
        self.bigrams_whose_el2_is_el2_of_winner = self.all_bigrams.get_bigrams_containing(lexeme=self.winner_info.winner.el2,
                                                                                          in_position=2,
                                                                                          max_gapsize=self.max_gapsize)
        self.update_element_freqs_in_tables(bigrams=self.bigrams_whose_el2_is_el2_of_winner,
                                            value=self.new_el2_freq,
                                            column="El2 Freq")
        
    def update_element_freqs_in_tables(self,bigrams=None,value=None,column=None):
        
        for bigram in bigrams:
            self.curr_bigram = bigram
            self.value = value
            self.all_tables.update_cell_value(bigram=self.curr_bigram, column=column, new_value=self.value)
            
    def update_conflicting_bigram_freqs(self, conflicting_bigrams):
        
        self.conflicting_bigrams = conflicting_bigrams
        self.all_bigrams.deduct_freqs(self.conflicting_bigrams)
        for bigram in self.conflicting_bigrams.get_bigrams():
            self.bigram = bigram
            self.all_tables.update_cell_value(bigram=self.bigram, column="Bigram Freq", new_value=self.all_bigrams.get_frequency(self.bigram))
    
    def remove_winner(self):
        
        self.all_bigrams.remove(self.winner_info.winner)
        self.all_tables.update_cell_value(bigram=self.winner_info.winner, column="Bigram Freq", new_value=0)

    def add_new_bigrams(self, new_bigrams):
        
        self.new_bigrams = new_bigrams
        self.all_tables.cleanup()
        self.all_bigrams.add(self.new_bigrams)
        self.new_data_for_tables = ColumnLists()
        for bigram in self.new_bigrams.get_bigrams():
            self.bigram = bigram
            self.bigram_freq = self.all_bigrams.get_frequency(self.bigram)
            self.el1_freq = self.all_lexemes.get_frequency(self.bigram.el1)
            self.el2_freq = self.all_lexemes.get_frequency(self.bigram.el2)
            self.new_data_for_tables.push_row(self.bigram,self.bigram_freq,self.el1_freq,self.el2_freq)
        self.all_tables.add(self.new_data_for_tables, self.new_bigrams.type_count)
        
        
        
        
