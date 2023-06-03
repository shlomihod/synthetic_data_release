"""A template file for writing a test for a new export generative model"""
from pathlib import Path
from unittest import TestCase
from warnings import filterwarnings

filterwarnings("ignore")

from os import path

cwd = path.dirname(__file__)


import json
from os import mkdir, path
from warnings import simplefilter

import numpy as np

from export.bayesian_network import BayesianNetDS, PrivBayesDS
from export.transformer import MyTransformer
from generative_models.data_synthesiser import BayesianNet, PrivBayes
from utils.datagen import load_local_data_as_df
from utils.logging import LOGGER
from utils.utils import set_random_seed

simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42
DATASET_SIZE = 1000
NUM_SYNTHETIC_DATASETS = 10
DEGREE = 1
EPSILON = 0.1
HISTOGRM_BINS = 25


class TestExport(TestCase):
    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, "../data/texas"))
        self.sizeS = len(self.raw)

        rawTrain = self.raw.query(
            "DISCHARGE in ['2013Q1', '2013Q2', '2013Q3', '2013Q4']"
        )

        rIdx = np.random.choice(
            list(rawTrain.index), size=DATASET_SIZE, replace=False
        ).tolist()
        self.rawTout = rawTrain.loc[rIdx]

    def test_orginal_export_implementation_on_fix_seed(self):

        gmList = [
            BayesianNet(self.metadata, histogram_bins=HISTOGRM_BINS, degree=DEGREE),
            BayesianNetDS(degree=DEGREE),
            PrivBayes(
                self.metadata,
                histogram_bins=HISTOGRM_BINS,
                degree=DEGREE,
                epsilon=EPSILON,
            ),
            PrivBayesDS(degree=DEGREE, epsilon=EPSILON),
        ]
        print(f"gmList = {gmList}")

        samplesList = []

        transformer = MyTransformer(self.metadata, histogram_bins=25)
        encoded_df = transformer.encode(self.rawTout)

        for GenModel in gmList:
            print(SEED)
            set_random_seed(SEED)
            LOGGER.info(f"Start: Evaluation for model {GenModel.__name__}...")
            if isinstance(GenModel, (BayesianNetDS, PrivBayesDS)):
                GenModel.fit(encoded_df, transformer.ranges)
                synTwithoutTarget = [
                    transformer.decode(GenModel.generate_samples(DATASET_SIZE))
                    for _ in range(NUM_SYNTHETIC_DATASETS)
                ]
                samplesList.append(synTwithoutTarget)
            else:
                GenModel.fit(self.rawTout)
                synTwithoutTarget = [
                    GenModel.generate_samples(DATASET_SIZE)
                    for _ in range(NUM_SYNTHETIC_DATASETS)
                ]
                samplesList.append(synTwithoutTarget)

        assert gmList[0].bayesian_network == gmList[1].bayesian_network
        assert gmList[2].bayesian_network == gmList[3].bayesian_network
        
        assert gmList[0].conditional_probabilities == gmList[1].conditional_probabilities
        assert gmList[2].conditional_probabilities == gmList[3].conditional_probabilities

        assert all(
            (df1 == df2).values.all()
            for df1, df2 in zip(samplesList[0], samplesList[1])
        )
        assert all(
            (df1 == df2).values.all()
            for df1, df2 in zip(samplesList[2], samplesList[3])
        )
