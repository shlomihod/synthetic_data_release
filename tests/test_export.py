"""A template file for writing a test for a new export generative model"""
from unittest import TestCase
from warnings import filterwarnings

filterwarnings("ignore")

from os import path

cwd = path.dirname(__file__)


from collections import Counter
from os import path
from warnings import simplefilter

import numpy as np

from export.bayesian_network import BayesianNetDS, PrivBayesDS
from export.transformer import MyTransformer
from generative_models.data_synthesiser import BayesianNet, PrivBayes
from predictive_models.predictive_model import RandForestClassTask
from utils.datagen import load_local_data_as_df
from utils.logging import LOGGER
from utils.utils import set_random_seed

simplefilter("ignore", category=FutureWarning)
simplefilter("ignore", category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42
DATASET_SIZE = 1000
NUM_SYNTHETIC_DATASETS = 10
DEGREE = 2
EPSILON = 1
HISTOGRM_BINS = 25
PREDICT_COLUMN = "RISK_MORTALITY"


class TestExport(TestCase):
    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, "../data/texas"))
        self.sizeS = len(self.raw)

        self.rawTrain = self.raw.query(
            "DISCHARGE in ['2013Q1', '2013Q2', '2013Q3', '2013Q4']"
        )

        self.rawTest = self.raw.query(
            "DISCHARGE in ['2014Q1', '2014Q2', '2014Q3', '2014Q4']"
        )

        rIdx = np.random.choice(
            list(self.rawTrain.index), size=DATASET_SIZE, replace=False
        ).tolist()
        self.rawTout = self.rawTrain.loc[rIdx]

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
            PrivBayesDS(degree=DEGREE, epsilon=EPSILON, secure=False),
            PrivBayesDS(degree=DEGREE, epsilon=EPSILON, secure=True),
        ]
        print(f"gmList = {gmList}")

        samplesList = []
        accuracyList = []

        transformer = MyTransformer(self.metadata, histogram_bins=25)
        encoded_df = transformer.encode(self.rawTout)

        for GenModel in gmList:
            print(f"{SEED=}")
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

        LOGGER.info(f"Start: Testing Utility...")
        for datasets in samplesList:
            accuracies = []
            for synth_df in datasets:
                set_random_seed(SEED)
                model = RandForestClassTask(self.metadata, PREDICT_COLUMN)
                model.train(synth_df)
                accuracies.append(np.mean(model.evaluate(self.rawTest)))

            accuracyList.append(np.mean(accuracies))

        model = RandForestClassTask(self.metadata, PREDICT_COLUMN)
        model.train(self.rawTout)
        print(
            "Accuracy of the raw dataset:",
            np.mean(model.evaluate(self.rawTest)),
        )

        print("Accuracy of the synthetic datasets:", accuracyList)
        LOGGER.info(f"Finish: Testing Utility...")

        assert gmList[0].bayesian_network == gmList[1].bayesian_network
        assert gmList[2].bayesian_network == gmList[3].bayesian_network

        assert (
            gmList[0].conditional_probabilities == gmList[1].conditional_probabilities
        )
        assert (
            gmList[2].conditional_probabilities == gmList[3].conditional_probabilities
        )

        assert all(
            (df1 == df2).values.all()
            for df1, df2 in zip(samplesList[0], samplesList[1])
        )
        assert all(
            (df1 == df2).values.all()
            for df1, df2 in zip(samplesList[2], samplesList[3])
        )

        assert accuracyList[0] == accuracyList[1]
        assert accuracyList[2] == accuracyList[3] 

        # PrivBayesDS(secure=False) and PrivBayesDS(secure=True) should be different
        assert gmList[3].bayesian_network != gmList[4].bayesian_network
        assert (
            gmList[3].conditional_probabilities != gmList[4].conditional_probabilities
        )
        assert all(
            (df1 != df2).values.any()
            for df1, df2 in zip(samplesList[3], samplesList[4])
        )

        assert accuracyList[3] != accuracyList[4]

        print("PrivBayesDS(secure=False)")
        gmList[3].show()
        # print(Counter(mech for mech, *_ in gmList[3].transcript))

        print("PrivBayesDS(secure=True)")
        gmList[4].show()
        # print(Counter(mech for mech, *_ in gmList[4].transcript))
