from __future__ import division
import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import itertools

import numpy as np
import pandas as pd


NTbigram = namedtuple("NTbigram", "el1 el2 gapsize")
NTlexeme = namedtuple("NTlexeme", "lex token_index")
NTword = namedtuple("NTword", "wordstr position")


class Bigrams(object):
    def __init__(self):

        self.bigrams_to_freqs = Counter()
        self.bigrams_to_locations = defaultdict(set)
        self.left_lex_to_bigrams = defaultdict(set)
        self.right_lex_to_bigrams = defaultdict(set)
        self.type_count = 0

    def save_bigram_data(self, left_lex, right_lex, gapsize, location):

        if gapsize < 0:
            raise Exception("gapsize below zero")
        self.new_bgr = self.create_bigram(left_lex, right_lex, gapsize)
        if self.new_bgr not in self.bigrams_to_freqs:
            self.type_count += 1
            self.left_lex_to_bigrams[(left_lex, gapsize)].add(self.new_bgr)
            self.right_lex_to_bigrams[(right_lex, gapsize)].add(self.new_bgr)
        self.bigrams_to_freqs[self.new_bgr] += 1
        if location not in self.bigrams_to_locations[self.new_bgr]:
            self.bigrams_to_locations[self.new_bgr].add(location)

    def create_bigram(self, left_lex, right_lex, gapsize):

        return NTbigram(el1=left_lex, el2=right_lex, gapsize=gapsize)

    def iter_bgrs_and_freqs(self):

        for index, (bgr, freq) in enumerate(self.bigrams_to_freqs.items()):
            yield index, bgr, freq

    def get_bigrams_containing(self, lexeme=None, in_position=None, max_gapsize=None):

        self.hits = set()
        for gapsize in range(max_gapsize + 1):
            if in_position == 1 and (lexeme, gapsize) in self.left_lex_to_bigrams:
                self.hits = self.hits.union(self.left_lex_to_bigrams[(lexeme, gapsize)])
            if in_position == 2 and (lexeme, gapsize) in self.right_lex_to_bigrams:
                self.hits = self.hits.union(
                    self.right_lex_to_bigrams[(lexeme, gapsize)]
                )
        return self.hits

    def get_bigrams(self):

        for bigram in self.bigrams_to_freqs.keys():
            yield bigram

    def get_frequency(self, bigram):

        return self.bigrams_to_freqs[bigram]

    def deduct_freqs(self, other_bigrams_instance):

        self.other_bigrams_instance = other_bigrams_instance
        for index, bigram, freq in self.other_bigrams_instance.iter_bgrs_and_freqs():
            self.bigram = bigram
            self.freq = freq
            self.locations = self.other_bigrams_instance.bigrams_to_locations[
                self.bigram
            ]
            if self.bigram not in self.bigrams_to_freqs:
                raise Exception("bigram not in bigrams_to_freqs")
            self.bigrams_to_freqs[
                self.bigram
            ] -= (
                self.freq
            )  # IF BIGRAM IS NOT IN COUNTER, THIS SYNTAX WILL STICK IT BACK IN!!!!!!!
            for location in self.locations:
                self.location = location
                self.bigrams_to_locations[self.bigram].remove(self.location)

            if self.bigrams_to_freqs[self.bigram] < 1:
                del self.bigrams_to_freqs[self.bigram]
                del self.bigrams_to_locations[self.bigram]
                self.left_lex_to_bigrams[(self.bigram.el1, self.bigram.gapsize)].remove(
                    self.bigram
                )
                self.right_lex_to_bigrams[
                    (self.bigram.el2, self.bigram.gapsize)
                ].remove(self.bigram)
        self.type_count = len(self.bigrams_to_freqs)

    def remove(self, bigram):

        try:
            del self.bigrams_to_freqs[bigram]
            del self.bigrams_to_locations[bigram]
            self.left_lex_to_bigrams[(bigram.el1, bigram.gapsize)].remove(bigram)
            self.right_lex_to_bigrams[(bigram.el2, bigram.gapsize)].remove(bigram)
            self.type_count = len(self.bigrams_to_freqs)
        except:
            pass

    def add(self, other_bigrams_instance):

        self.other_bigrams_instance = other_bigrams_instance
        for index, bigram, freq in self.other_bigrams_instance.iter_bgrs_and_freqs():
            self.bigram = bigram
            self.gapsize = self.bigram.gapsize
            self.freq = freq
            self.bigrams_to_freqs[self.bigram] += self.freq
            self.curr_locs = self.bigrams_to_locations[self.bigram]
            self.bigrams_to_locations[self.bigram] = self.curr_locs.union(
                self.other_bigrams_instance.bigrams_to_locations[self.bigram]
            )

        for (
            el1,
            gapsize,
        ), bigrams in self.other_bigrams_instance.left_lex_to_bigrams.items():
            self.el1 = el1
            self.gapsize = gapsize
            self.bigrams = bigrams
            self.curr_left_lex_to_bigrams = self.left_lex_to_bigrams[
                (self.el1, self.gapsize)
            ]
            self.left_lex_to_bigrams[
                (self.el1, self.gapsize)
            ] = self.curr_left_lex_to_bigrams.union(self.bigrams)

        for (
            el2,
            gapsize,
        ), bigrams in self.other_bigrams_instance.right_lex_to_bigrams.items():
            self.el2 = el2
            self.gapsize = gapsize
            self.bigrams = bigrams
            self.curr_right_lex_to_bigrams = self.right_lex_to_bigrams[
                (self.el2, self.gapsize)
            ]
            self.right_lex_to_bigrams[
                (self.el2, self.gapsize)
            ] = self.curr_right_lex_to_bigrams.union(self.bigrams)

        self.type_count = len(self.bigrams_to_freqs)


class InitBigramData(object):
    def initialize(self, lexeme_data, max_gapsize):

        self.lexeme_data = lexeme_data
        self.bigrams = Bigrams()
        self.max_gapsize = max_gapsize
        self._counter = 0

        for turnindex, turnlength, curr_gapsize in self._turns_X_gapsizes():
            self.rightmost_leftedge = turnlength - curr_gapsize - 1
            self.turnindex = turnindex
            self.turnlength = turnlength
            self.curr_gapsize = curr_gapsize
            for left, right, location in self._get_elmnts_and_loc():
                self.bigrams.save_bigram_data(left, right, self.curr_gapsize, location)

    def _get_elmnts_and_loc(self):

        for lexindex in range(self.rightmost_leftedge):
            self.lexindex = lexindex
            self._left, self.left_anchor_pos = self.lexeme_data.get_lexeme(
                self.turnindex, self.lexindex
            )
            self._right, self.right_anchor_pos = self.lexeme_data.get_lexeme(
                self.turnindex, self.lexindex + self.curr_gapsize + 1
            )
            self._location = self.lexeme_data.get_extant_loc_object(
                (self.turnindex, self.lexindex)
            )

            yield self._left, self._right, self._location

    def _turns_X_gapsizes(self):

        self._lengths = self.lexeme_data.get_turn_lengths()
        self._gapsizes = range(self.max_gapsize + 1)
        self._cartesian_prdct = itertools.product(*(self._lengths, self._gapsizes))
        for (turnindex, turnlength), curr_gapsize in self._cartesian_prdct:

            yield turnindex, turnlength, curr_gapsize
