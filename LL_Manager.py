from __future__ import division
import os
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import itertools


import numpy as np
import pandas as pd

NTwinner = namedtuple("NTwinner", "winner winner_ll winner_tablenum")


class LogLikelihoodManager(object):
    def calculate(self, all_tables, corpus_size, number_of_cores=27):

        self.all_tables = all_tables
        self.corpus_size = corpus_size
        self.number_of_cores = number_of_cores
        self.ll_indexer = {}

        for tablenum, table in self.all_tables:
            # print("tablenum:",tablenum)
            self.tablenum = tablenum
            self.table = table
            self.calc_ll_for_single_table()

    def calc_ll_for_single_table(self):

        self.obsA = self.table["Bigram Freq"]
        self.obsB = self.table["El1 Freq"] - self.table["Bigram Freq"]
        self.obsC = self.table["El2 Freq"] - self.table["Bigram Freq"]
        self.obsD = self.corpus_size - (self.obsA + self.obsB + self.obsC)

        self.expA = self.table["El2 Freq"] / self.corpus_size * self.table["El1 Freq"]
        self.expB = (
            (self.corpus_size - self.table["El2 Freq"])
            / self.corpus_size
            * self.table["El1 Freq"]
        )
        self.expC = (
            self.table["El2 Freq"]
            / self.corpus_size
            * (self.corpus_size - self.table["El1 Freq"])
        )
        self.expD = (
            (self.corpus_size - self.table["El2 Freq"])
            / self.corpus_size
            * (self.corpus_size - self.table["El1 Freq"])
        )

        self.llA = self.obsA * np.log1p(self.obsA / self.expA)

        self.real_vals = np.where(self.obsB != 0)[0]
        self.llB = np.zeros(len(self.table))
        self.llB[self.real_vals] = self.obsB[self.real_vals] * np.log(
            self.obsB[self.real_vals] / self.expB[self.real_vals]
        )

        self.real_vals = np.where(self.obsC != 0)[0]
        self.llC = np.zeros(len(self.table))
        self.llC[self.real_vals] = self.obsC[self.real_vals] * np.log(
            self.obsC[self.real_vals] / self.expC[self.real_vals]
        )

        self.llD = self.obsD * np.log(self.obsD / self.expD)

        self.log_likelihood = 2 * (self.llA + self.llB + self.llC + self.llD)
        self.negs = np.where(self.llA < 0)[0]
        self.log_likelihood[self.negs] = self.log_likelihood[self.negs] * -1

        self.ll_indexer[self.tablenum] = self.log_likelihood

    def get_winner(self):

        self.winners_per_series = []
        for tablenum, ll_series in self.ll_indexer.items():
            self.tablenum = tablenum
            self.ll_series = ll_series
            self.winner = self.ll_series.idxmax()
            self.winner_ll = self.ll_series[self.winner]
            self.winner_triple = NTwinner(
                winner=self.winner,
                winner_ll=self.winner_ll,
                winner_tablenum=self.tablenum,
            )
            self.winners_per_series.append(self.winner_triple)
        self.winners_per_series.sort(key=lambda tup: tup[1], reverse=True)
        return self.winners_per_series[0]
