from __future__ import division
import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import itertools

import numpy as np
import pandas as pd

from Bigram_Initiator import Bigrams

NTcontextpos = namedtuple("NTcontextpos", "context_pos satellite_pos gapsize merge_token_leftanchor")
NTlexeme = namedtuple('NTlexeme','lex token_index')
NTword = namedtuple('NTword', 'wordstr position')
NTbigram = namedtuple('NTbigram','el1 el2 gapsize') 




class BigramUpdater(object):

    def get_new_and_conflicting_bigrams(self,all_bigrams,all_lexemes,merge_token,winner_info,max_gapsize):
        
        #Attribute assignment
        self.all_bigrams = all_bigrams
        self.all_lexemes = all_lexemes
        self.merge_token = merge_token
        self.winner_info = winner_info
        self.max_gapsize = max_gapsize
        
        #Empty containers
        self.merged_satellite_positions = dict()
        self.merge_token_count = 0
        
        #Class instances
        self.context_pos_manager = ContextPositionManager(self.all_lexemes,self.max_gapsize)
        self.vicinity_lex_manager = VicinityLexemeManager(self.all_lexemes,self.merge_token)
        
        #Program
        self.winner_locations = self.all_bigrams.bigrams_to_locations[self.winner_info.winner]
        self.main_control_loop()
        self.update_all_lexemes_with_merge_tokens()
        
        #Output
        self.new_bigrams = self.vicinity_lex_manager.new_bigrams
        self.conflicting_bigrams = self.vicinity_lex_manager.conflicting_bigrams
                
    def main_control_loop(self):
        
        self.all_lexemes.set_merge_token(self.merge_token)
        for turn_number,merge_token_leftanchor in self.winner_locations:
            self.turn_number = turn_number
            self.merge_token_leftanchor = merge_token_leftanchor
            if not self.bigram_is_among_existing_conflicting_bigrams():
                self.merge_token_count += 1
                self.get_satellite_positions()
                self.context_positions = self.context_pos_manager.generate_positions_around_satellites(self.turn_number,
                                                                                                       self.satellite_positions,
                                                                                                       self.merge_token_leftanchor)
                self.vicinity_lex_manager.create_bigrams_with_lexemes_surrounding_satellites(self.turn_number,
                                                                                            self.context_positions,
                                                                                            self.satellite_positions,
                                                                                            self.merged_satellite_positions)
                self.store_satellite_positions()
                
    def update_all_lexemes_with_merge_tokens(self):
        
        self.all_lexemes.set_merge_token_freq(self.merge_token_count)
        for (turn_number,satellite_position),merge_token_leftanchor in self.merged_satellite_positions.iteritems():
            self.turn_number = turn_number
            self.satellite_position = satellite_position
            self.merge_token_leftanchor = merge_token_leftanchor
            self.all_lexemes.add_merge_token(self.turn_number,self.satellite_position,self.merge_token_leftanchor) 
            
            
    def bigram_is_among_existing_conflicting_bigrams(self):
        
        if (self.turn_number,self.merge_token_leftanchor) in self.vicinity_lex_manager.conflicting_bigrams.bigrams_to_locations[self.winner_info.winner]:
            return True
        else:
            return False
            
    def get_satellite_positions(self):
        
        self.satellite_positions = [satellite_position+self.merge_token_leftanchor for wordstr,satellite_position in self.merge_token.lex]
                    
    def store_satellite_positions(self):
        
        for satellite_position in self.satellite_positions:
            self.satellite_position = satellite_position
            self.merged_satellite_positions[(self.turn_number,self.satellite_position)] = self.merge_token_leftanchor





class ContextPositionManager(object):
    
    def __init__(self,all_lexemes,max_gapsize):
        
        self.all_lexemes = all_lexemes
        self.max_gapsize = max_gapsize
    
    def generate_positions_around_satellites(self,turn_number,satellite_positions,merge_token_leftanchor):
        
        self.turn_number = turn_number
        self.satellite_positions = satellite_positions
        self.merge_token_leftanchor = merge_token_leftanchor
        self.context_positions = []
        
        self.curr_turn_length = self.all_lexemes.get_turn_length(self.turn_number)
        for satellite_position in self.satellite_positions:
            self.satellite_position = satellite_position
            self.get_context_positions_for_satellite()
        return self.context_positions
    
    def get_context_positions_for_satellite(self):
        
        for curr_gapsize in range(self.max_gapsize+1):
            self.curr_gapsize = curr_gapsize
            self.get_left_context_position()
            self.get_right_context_position()
        
    def get_left_context_position(self):
    
        self.left_context_position = self.satellite_position-self.curr_gapsize-1
        if self.left_context_position>=0:
            self.curr_contextpos = NTcontextpos(context_pos=self.left_context_position,
                                                satellite_pos=self.satellite_position,
                                                gapsize=self.curr_gapsize,
                                                merge_token_leftanchor=self.merge_token_leftanchor)
            self.context_positions.append(self.curr_contextpos)
        
    def get_right_context_position(self):
        
        self.right_context_position = self.satellite_position+self.curr_gapsize+1
        if self.right_context_position<self.curr_turn_length:
            self.curr_contextpos = NTcontextpos(context_pos=self.right_context_position,
                                                satellite_pos=self.satellite_position,
                                                gapsize=self.curr_gapsize,
                                                merge_token_leftanchor=self.merge_token_leftanchor)
            self.context_positions.append(self.curr_contextpos)
            




            
class VicinityLexemeManager(object):
    
    def __init__(self,all_lexemes,merge_token):
        
        self.merge_token = merge_token
        self.all_lexemes = all_lexemes
        self.new_bigrams = Bigrams()
        self.conflicting_bigrams = Bigrams()
    
    def create_bigrams_with_lexemes_surrounding_satellites(self,turn_number,context_positions,satellite_positions,merged_satellite_positions):
        
        self.turn_number = turn_number
        self.context_positions = context_positions
        self.satellite_positions = satellite_positions
        self.merged_satellite_positions = merged_satellite_positions
        
        
        for context_position_info in self.context_positions:
            self.unpack_context_position_info_tuple(context_position_info)
            if self.context_position not in self.satellite_positions:   #the context 
                self.premerge_lexeme,self.premerge_leftanchor = self.all_lexemes.get_lexeme(self.turn_number,self.satellite_position)
                if (self.turn_number,self.context_position) in self.merged_satellite_positions:
                    self.context_lexeme = self.merge_token
                    self.context_leftanchor = self.merged_satellite_positions[(self.turn_number,self.context_position)]
                    #Don't need to create conflicting bigram since this confl same bigram would have already been created when the adjacent merge token was created at an earlier iteration
                else:
                    self.context_lexeme,self.context_leftanchor = self.all_lexemes.get_lexeme(self.turn_number,self.context_position)
                self.create_new_bigram()
                self.create_conflicting_bigram()
                
    def unpack_context_position_info_tuple(self, context_position_info):
        self.context_position = context_position_info.context_pos
        self.satellite_position = context_position_info.satellite_pos
        self.curr_gapsize = context_position_info.gapsize
        self.merge_token_leftanchor = context_position_info.merge_token_leftanchor
        
    def create_new_bigram(self):
        
        if self.context_lexeme_anchor_to_the_left_of_target_lexeme_anchor(target=self.merge_token_leftanchor):
            self.make_location_tuple(leftanchor=self.context_leftanchor)
            self.gap_between_leftanchors = self.merge_token_leftanchor-self.context_leftanchor-1
            self.new_bigrams.save_bigram_data(self.context_lexeme,self.merge_token,self.gap_between_leftanchors,self.location_tuple)
        else:
            self.make_location_tuple(leftanchor=self.merge_token_leftanchor)
            self.gap_between_leftanchors = self.context_leftanchor-self.merge_token_leftanchor-1
            self.new_bigrams.save_bigram_data(self.merge_token,self.context_lexeme,self.gap_between_leftanchors,self.location_tuple)
        
    def create_conflicting_bigram(self):
        
        if self.context_lexeme_anchor_to_the_left_of_target_lexeme_anchor(target=self.premerge_leftanchor):
            self.make_location_tuple(leftanchor=self.context_leftanchor)
            self.gap_between_leftanchors = self.premerge_leftanchor-self.context_leftanchor-1
            self.conflicting_bigrams.save_bigram_data(self.context_lexeme,self.premerge_lexeme,self.gap_between_leftanchors,self.location_tuple)
        else:
            self.make_location_tuple(leftanchor=self.premerge_leftanchor)
            self.gap_between_leftanchors = self.context_leftanchor-self.premerge_leftanchor-1
            self.conflicting_bigrams.save_bigram_data(self.premerge_lexeme,self.context_lexeme,self.gap_between_leftanchors,self.location_tuple)
        
    def context_lexeme_anchor_to_the_left_of_target_lexeme_anchor(self,target=None):
        
        if self.context_leftanchor<target:
            return True
        else:
            return False
        
    def make_location_tuple(self, leftanchor=None):
        
        self.location_tuple = self.all_lexemes.get_extant_loc_object((self.turn_number,leftanchor))
    
        