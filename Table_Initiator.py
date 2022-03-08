from __future__ import division
import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import itertools

import numpy as np
import pandas as pd


#def print_iterstatus(input_integer,message,interval=10000):
#    if input_integer%interval==0:
#        print(input_integer,message)


class ColumnLists(object):
    
    def __init__(self):
        
        self.initialize_columns()
        
    def initialize_columns(self):
        
        self.row_indices=[]
        self.bgr_freqs=[]
        self.el1_freqs=[]
        self.el2_freqs=[]
        
    def push_row(self,bgr,bgr_freq,el1_freq,el2_freq):
        
        self.row_indices.append(bgr)
        self.bgr_freqs.append(bgr_freq)
        self.el1_freqs.append(el1_freq)
        self.el2_freqs.append(el2_freq)




 
class Tables(object):
    
    def __init__(self, max_row_count):
        
        self.all_tables = {}
        self.bigrams_to_table = {}
        self.max_row_count = max_row_count
        self.table_sizes = {}
    
    def add_table(self,columnlists):
        
        self.columnlists = columnlists
        
        self.create_df()
        self.tablenum = len(self.all_tables)
        #print("adding table, tablenum:",self.tablenum)
        self.all_tables[self.tablenum] = self.df
        self.table_sizes[self.tablenum] = len(self.df)
        
        self.bgrs_to_tablenum = {bgr:self.tablenum for bgr in self.columnlists.row_indices}        #separate function to update bigrams_to_table?
        self.bigrams_to_table.update(self.bgrs_to_tablenum)
        
    def create_df(self):
        
        self.data = {"Bigram Freq":self.columnlists.bgr_freqs,
                     "El1 Freq":self.columnlists.el1_freqs,
                     "El2 Freq":self.columnlists.el2_freqs}
        
        self.df = pd.DataFrame(self.data,index=self.columnlists.row_indices)
        
    def update_cell_value(self, bigram=None, column=None, new_value=None):
        
        self.all_tables[self.bigrams_to_table[bigram]].set_value(bigram, column, new_value)
    
    def cleanup(self):
        
        self.updated_tables = {}
        for tablenum,table in self.all_tables.iteritems():
            self.updated_table = table.loc[table["Bigram Freq"]>0]
            self.updated_tables[tablenum] = self.updated_table
            self.table_sizes[tablenum] = len(self.updated_table)
            self.indices_for_deletion = table[table["Bigram Freq"]==0].index
            for bigram in self.indices_for_deletion:
                del self.bigrams_to_table[bigram]
        self.all_tables = self.updated_tables
        
    def add(self, columnlists, type_count):
        
        self.columnlists = columnlists
        self.type_count = type_count
        self.available_tables = []
        for tablenum,size in self.table_sizes.iteritems():
            if self.max_row_count-size>self.type_count:
                self.available_tables.append(tablenum)
        if self.available_tables:
            self.curr_table_num = self.available_tables[0]
            self.create_df()
            self.new_df = pd.concat((self.all_tables[self.curr_table_num],self.df))
            self.all_tables[self.curr_table_num] = self.new_df
            self.bgrs_to_tablenum = {bgr:self.curr_table_num for bgr in self.columnlists.row_indices}        
            self.bigrams_to_table.update(self.bgrs_to_tablenum)
            self.table_sizes[self.curr_table_num] = len(self.new_df)
        else:
            self.add_table(self.columnlists)
            
    def __iter__(self):
        
        for tablenum,table in self.all_tables.iteritems():
            yield tablenum,table
            




class InitCandidateTable(object):
    
    def initialize(self, all_bigrams, all_lexemes, max_row_count=50000):
        self.all_bigrams = all_bigrams
        self.column_lists = ColumnLists()
        self.all_tables = Tables(max_row_count)
        self.all_lexemes = all_lexemes
        self.max_row_count = max_row_count
        
        for index,bgr,freq in self.all_bigrams.iter_bgrs_and_freqs():
            self._el1_freq = self.all_lexemes.get_frequency(bgr.el1)
            self._el2_freq = self.all_lexemes.get_frequency(bgr.el2)
            self.column_lists.push_row(bgr,freq,self._el1_freq,self._el2_freq)
            if (index+1) % self.max_row_count==0 or index==self.all_bigrams.type_count-1:
                self.all_tables.add_table(self.column_lists)
                self.column_lists.initialize_columns()