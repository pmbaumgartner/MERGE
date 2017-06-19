from __future__ import division
import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import itertools

import numpy as np
import pandas as pd


NTword = namedtuple('NTword', 'wordstr position')             # (word,0)
NTlexeme = namedtuple('NTlexeme','lex token_index')
NTlexemeWithPointer = namedtuple('NTlexemeWithPointer', 'lexeme pointer')





class AlreadySeenWords(object):
    
    def __init__(self):
        
        self._strings_to_lexemes = {}
        
    def add_word_with_lexeme(self,wordstring,lexeme):
        
        self._strings_to_lexemes[wordstring] = lexeme
        
    def get_lexeme(self,wordstring):
        
        try:
            return self._strings_to_lexemes[wordstring]
        except KeyError:
            print "Warning: word string has not already been seen."
            
    def is_present(self,wordstring):
        
        curr_bool=False
        if wordstring in self._strings_to_lexemes:
            curr_bool=True
        return curr_bool
  
  
  
    
class Lexemes(object):
    
    def __init__(self):
        
        self._lexemes_to_locations = defaultdict(set)
        self._locations_to_lexemes = defaultdict(dict)
        self._locations_to_locations = {}
        self.turn_lengths = Counter()
        self._lexemes_to_freqs = Counter()
        
    def add_location(self,lexeme,lineindex,wordindex):
        
        new_loc = (lineindex,wordindex)
        self._lexemes_to_locations[lexeme].add(new_loc)        
        self._locations_to_lexemes[lineindex][wordindex] = lexeme
        self._locations_to_locations[new_loc] = new_loc
        self.turn_lengths[lineindex]+=1
    
    def set_merge_token(self, merge_token):
        
        self.merge_token = merge_token
        self.satellite_lexemes = {0:self.merge_token}
        for wordstr,position in self.merge_token.lex:
            if position>0:
                self.curr_satellite_lex = NTlexeme(lex=self.merge_token.lex, token_index=position)
                self.satellite_lexemes[position] = self.curr_satellite_lex
                
    def set_merge_token_freq(self, merge_token_freq):
        
        self._lexemes_to_freqs[self.merge_token] = merge_token_freq
                
    def add_merge_token(self, turn_number,satellite_position,merge_token_leftanchor):           
        
        self.turn_number = turn_number
        self.satellite_position = satellite_position
        self.merge_token_leftanchor = merge_token_leftanchor
        self.absolute_position_in_turn = self.satellite_position+self.merge_token_leftanchor
        self.loc = self._locations_to_locations[(self.turn_number,self.satellite_position)]
        self._lexemes_to_locations[self.satellite_lexemes[self.satellite_position-self.merge_token_leftanchor]].add(self.loc)
        self._locations_to_lexemes[self.turn_number][self.satellite_position] = self.satellite_lexemes[self.satellite_position-self.merge_token_leftanchor]
        
        
    def count_frequencies(self):
        
        for key,value in self._lexemes_to_locations.iteritems():
            self._lexemes_to_freqs[key] = len(value)
        
    def get_frequency(self,lexeme):
        
        return self._lexemes_to_freqs[lexeme]
        
    def get_turn_lengths(self):
        
        for turnindex,turnlength in self.turn_lengths.iteritems():
            yield turnindex,turnlength
            
    def get_turn_length(self,turn):
        
        return self.turn_lengths[turn]
    
    def get_lexeme(self,turnindex,wordindex):       
        
        self.turnindex = turnindex
        self.wordindex = wordindex
        self.lexeme = self._locations_to_lexemes[self.turnindex][self.wordindex]
        self.leftanchor_pos_in_turn = self.wordindex-self.lexeme.token_index
        self.leftanchor_lexeme = self._locations_to_lexemes[self.turnindex][self.leftanchor_pos_in_turn]
        return self.leftanchor_lexeme,self.leftanchor_pos_in_turn
    
    def get_extant_loc_object(self,loc_tuple):
        
        if loc_tuple in self._locations_to_locations:
            return self._locations_to_locations[loc_tuple]
        else:
            return loc_tuple
        
    def deduct_freq(self,lexeme=None,deduction=None):
        
        self._lexemes_to_freqs[lexeme] -= deduction
        if self._lexemes_to_freqs[lexeme]<1:
            del self._lexemes_to_freqs[lexeme]

        




class InitCorpus(object):
    """Loads and processes corpus."""
    
    def initialize(self, corpus_dir_path, ext=".txt",
                 delimiter_set=set([" ",".",",","?",";",":","!","\r","\n"])):
        
        self.already_seen_words = AlreadySeenWords()
        self.all_lexemes = Lexemes()
        self.corpus_size = 0
        self.delimiter_set = delimiter_set
        self.corpus_dir_path = corpus_dir_path
        self.corpus_files = [fn for fn in os.listdir(self.corpus_dir_path)
                             if fn.lower().endswith(ext)]
        self.lineindex = 0
        
        for line in self._get_line_from_files():
            self.line = line
            for wordindex,word in enumerate(self._tokenize_line(self.line)):
                self.wordindex = wordindex
                self.word = word.lower()                                         
                if self.already_seen_words.is_present(self.word):
                    self.curr_lexeme = self.already_seen_words.get_lexeme(self.word)
                    self.all_lexemes.add_location(self.curr_lexeme,self.lineindex,self.wordindex)
                else:
                    self.new_lexeme = self._create_lexeme(self.word)
                    self.already_seen_words.add_word_with_lexeme(self.word,self.new_lexeme)
                    self.all_lexemes.add_location(self.new_lexeme,self.lineindex,self.wordindex)                        
                        
                self.corpus_size += 1
            self.lineindex += 1

        del self.already_seen_words
        
        self.all_lexemes.count_frequencies()
        
    def _get_line_from_files(self):
        
        for filename in self.corpus_files:                              
            file_path = "".join((self.corpus_dir_path,filename))
            with open(file_path) as f:
                for line in f:
                    yield line
                    
    def _create_lexeme(self,wordstring):
        
        new_lexeme = NTlexeme(lex=(NTword(wordstr=wordstring, position=0),),
                        token_index=0)
        
        return new_lexeme
                
    def _tokenize_line(self, line):
        for index,char in enumerate(line):
            if char not in self.delimiter_set:
                if 'left_edge' not in locals():
                    left_edge=index
            else:
                if 'left_edge' in locals():
                    yield line[left_edge:index]
                    del left_edge
        if 'left_edge' in locals():
            yield line[left_edge:len(line)]