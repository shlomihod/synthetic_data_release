"""Generative models adapted from https://github.com/DataResponsibly/DataSynthesizer"""
# Copyright <2018> <dataresponsibly.com>

# Based on:
# http://github.com/shlomihod/synthetic_data_release


import itertools as it
import logging
import secrets
import warnings

import numpy as np
import pandas as pd
from opendp.measurements import make_base_laplace
from opendp.mod import enable_features
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

LOGGER = logging.getLogger(__name__)

CSPRNG = secrets.SystemRandom()


class GenerativeModel:
    """Parent class for all generative models"""

    def fit(self, data):
        """Fit a generative model to the input dataset"""
        return NotImplementedError("Method needs to be overwritten by a subclass.")

    def generate_samples(self, nsamples):
        """Generate a synthetic dataset of size nsamples"""
        return NotImplementedError("Method needs to be overwritten by a subclass.")


def mutual_information(labels_x: pd.Series, labels_y: pd.DataFrame):
    """Mutual information of distributions in format of pd.Series or pd.DataFrame.

    Parameters
    ----------
    labels_x : pd.Series
    labels_y : pd.DataFrame
    """
    if labels_y.shape[1] == 1:
        labels_y = labels_y.iloc[:, 0]
    else:
        labels_y = labels_y.apply(lambda x: " ".join(x.values), axis=1)

    return mutual_info_score(labels_x, labels_y)


def pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes."""
    sorted_columns = sorted(dataset.columns)
    mi_df = pd.DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(
                dataset[row].astype(str),
                dataset[col].astype(str),
                average_method="arithmetic",
            )
    return mi_df


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if np.isinf(summation):
            return normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation
    else:
        return np.full_like(distribution, 1 / distribution.size)


def display_bayesian_network(bn):
    length = 0
    for child, _ in bn:
        if len(child) > length:
            length = len(child)

    print("Constructed Bayesian network:")
    for child, parents in bn:
        print("    {0:{width}} has parents {1}.".format(child, parents, width=length))


def bayes_worker(paras):
    child, V, num_parents, split, dataset = paras
    parents_pair_list = []
    mutual_info_list = []

    if split + num_parents - 1 < len(V):
        for other_parents in it.combinations(
            V[split + 1 :], num_parents - 1  # noqa: E203
        ):
            parents = list(other_parents)
            parents.append(V[split])
            parents_pair_list.append((child, parents))
            mi = mutual_information(dataset[child], dataset[parents])
            mutual_info_list.append(mi)

    return parents_pair_list, mutual_info_list


def calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary):
    """Sensitivity function for Bayesian network construction. PrivBayes Lemma 1.
    Parameters
    ----------
    num_tuples : int
        Number of tuples in sensitive dataset.
    Return
    --------
    int
        Sensitivity value.
    """
    if attr_to_is_binary[child] or (
        len(parents) == 1 and attr_to_is_binary[parents[0]]
    ):
        a = np.log(num_tuples) / num_tuples
        b = (num_tuples - 1) / num_tuples
        b_inv = num_tuples / (num_tuples - 1)
        return a + b * np.log(b_inv)
    else:
        a = (2 / num_tuples) * np.log((num_tuples + 1) / 2)
        b = (1 - 1 / num_tuples) * np.log(1 + 2 / (num_tuples - 1))
        return a + b


def calculate_delta(num_attributes, sensitivity, epsilon):
    """Computing delta, which is a factor when applying differential privacy.
    More info is in PrivBayes Section 4.2 "A First-Cut Solution".
    Parameters
    ----------
    num_attributes : int
        Number of attributes in dataset.
    sensitivity : float
        Sensitivity of removing one tuple.
    epsilon : float
        Parameter of differential privacy.
    """
    return (num_attributes - 1) * sensitivity / epsilon


def exponential_mechanism(
    epsilon,
    mutual_info_list,
    parents_pair_list,
    attr_to_is_binary,
    num_tuples,
    num_attributes,
):
    """Applied in Exponential Mechanism to sample outcomes."""
    delta_array = []
    for child, parents in parents_pair_list:
        sensitivity = calculate_sensitivity(
            num_tuples, child, parents, attr_to_is_binary
        )
        delta = calculate_delta(num_attributes, sensitivity, epsilon)
        delta_array.append(delta)

    mi_array = np.array(mutual_info_list) / (2 * np.array(delta_array))
    mi_array = np.exp(mi_array)
    mi_array = normalize_given_distribution(mi_array)
    return mi_array


class BayesianNetDS(GenerativeModel):
    """
    A BayesianNet model using non-private GreedyBayes to learn conditional probabilities
    """

    def __init__(self, degree, with_root_hist_query=False, sampling_seed=None):
        self.degree = degree
        self.with_root_hist_query = with_root_hist_query
        self.sampling_seed = sampling_seed

        self.bayesian_network = None
        self.conditional_probabilities = None
        self.trained = False

        self.ranges = None
        self.num_attributes = None
        self.num_records = None

        self.sampling_prng = np.random.random.__self__

        self.__name__ = "BayesianNet"

    def fit(self, df, ranges):
        assert isinstance(df, pd.DataFrame), (
            f"{self.__class__.__name__} expects pd.DataFrame as input data "
            f" but got {type(df)}"
        )
        assert (
            len(list(df)) >= 2
        ), "BayesianNet requires at least 2 attributes(i.e., columns) in dataset."
        LOGGER.info(f"Start training BayesianNet on data of shape {df.shape}...")
        self.num_attributes = df.shape[1]
        self.ranges = ranges
        self.num_records = df.shape[0]

        if self.trained:
            self.trained = False
            self.bayesian_network = None
            self.conditional_probabilities = None

        self.bayesian_network = self._greedy_bayes_linear(df, self.degree)

        self.show()

        self.conditional_probabilities = self._construct_conditional_probabilities(
            self.bayesian_network, df
        )

        LOGGER.info("Finished training Bayesian net")
        self.trained = True

    def generate_samples(self, nsamples):
        LOGGER.info(f"Generate synthetic dataset of size {nsamples}")
        assert self.trained, "Model must be fitted to some real data first"

        # Get samples for attributes modelled in Bayesian net
        df = self._generate_encoded_dataset(nsamples)

        return df

    def _generate_encoded_dataset(self, nsamples):
        encoded_df = pd.DataFrame(
            columns=self._get_sampling_order(self.bayesian_network)
        )

        bn_root_attr = self.bayesian_network[0][1][0]
        root_attr_dist = self.conditional_probabilities[bn_root_attr]
        encoded_df[bn_root_attr] = self.sampling_prng.choice(
            len(root_attr_dist), size=nsamples, p=root_attr_dist
        )

        for child, parents in self.bayesian_network:
            child_conditional_distributions = self.conditional_probabilities[child]

            for parents_instance in child_conditional_distributions.keys():
                dist = child_conditional_distributions[parents_instance]
                parents_instance = list(eval(parents_instance))

                filter_condition = ""
                for parent, value in zip(parents, parents_instance):
                    filter_condition += f"(encoded_df['{parent}']=={value})&"

                filter_condition = eval(filter_condition[:-1])
                size = encoded_df[filter_condition].shape[0]
                if size:
                    encoded_df.loc[filter_condition, child] = self.sampling_prng.choice(
                        len(dist), size=size, p=dist
                    )

            # Fill any nan values by sampling from marginal child distribution
            # marginal_dist = (self.DataDescriber
            #                      .attr_dict[child]
            #                      .distribution_probabilities)
            # null_idx = encoded_df[child].isnull()
            # encoded_df.loc[null_idx, child] = np.random.choice(len(marginal_dist),
            #                                                    size=null_idx.sum(),
            #                                                    p=marginal_dist)
        # THE PRIVOUS COMMENT DOES NOT MAKE SENSE,
        # BECAUSE THE NULL CELLS WILL BE OVERWRITTEN BY THE LOOOP ABOVE
        assert not encoded_df.isnull().values.any()

        encoded_df[encoded_df.columns] = encoded_df[encoded_df.columns].astype(int)

        return encoded_df

    def _get_sampling_order(self, bayesian_net):
        order = [bayesian_net[0][1][0]]
        for child, _ in bayesian_net:
            order.append(child)
        return order

    def _greedy_bayes_linear(self, encoded_df, k):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)

        root_attribute = np.random.choice(dataset.columns)
        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            for child, split in it.product(
                rest_attributes, range(len(V) - num_parents + 1)
            ):
                task = (child, V, num_parents, split, dataset)
                res = bayes_worker(task)
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            idx = mutual_info_list.index(max(mutual_info_list))

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _construct_conditional_probabilities(self, bayesian_network, encoded_dataset):
        k = len(bayesian_network[-1][1])
        conditional_distributions = {}

        # first k+1 attributes
        root = bayesian_network[0][1][0]
        kplus1_attributes = [root]
        if not self.with_root_hist_query:
            for child, _ in bayesian_network[:k]:
                kplus1_attributes.append(child)

        freqs_of_kplus1_attributes = self._get_attribute_frequency_counts(
            kplus1_attributes, encoded_dataset
        )

        # get distribution of root attribute
        root_marginal_freqs = (
            freqs_of_kplus1_attributes.loc[:, [root, "count"]]
            .groupby(root)
            .sum()["count"]
        )
        conditional_distributions[root] = normalize_given_distribution(
            root_marginal_freqs
        ).tolist()

        for idx, (child, parents) in enumerate(bayesian_network):
            conditional_distributions[child] = {}

            if not self.with_root_hist_query and idx < k:
                stats = freqs_of_kplus1_attributes.copy().loc[
                    :, parents + [child, "count"]
                ]
            else:
                stats = self._get_attribute_frequency_counts(
                    parents + [child], encoded_dataset
                )

            stats = pd.DataFrame(
                stats.loc[:, parents + [child, "count"]]
                .groupby(parents + [child])
                .sum()
            )

            if len(parents) == 1:
                for parent_instance in stats.index.levels[0]:
                    dist = normalize_given_distribution(
                        stats.loc[parent_instance]["count"]
                    ).tolist()
                    conditional_distributions[child][str([parent_instance])] = dist
            else:
                for parents_instance in it.product(*stats.index.levels[:-1]):
                    dist = normalize_given_distribution(
                        stats.loc[parents_instance]["count"]
                    ).tolist()
                    conditional_distributions[child][str(list(parents_instance))] = dist

        return conditional_distributions

    def _get_attribute_frequency_counts(self, attributes, encoded_dataset):
        # Get attribute counts for category it.combinations present in data
        counts = encoded_dataset.groupby(attributes).size()
        counts.name = "count"
        counts = counts.reset_index()

        # Get all possible attribute it.combinations
        attr_combs = [
            range(self.ranges[attr][1] - self.ranges[attr][0] + 1)
            for attr in attributes
        ]
        full_space = pd.DataFrame(
            columns=attributes, data=list(it.product(*attr_combs))
        )

        # stats.reset_index(inplace=True)
        full_counts = pd.merge(full_space, counts, how="left")
        full_counts.fillna(0, inplace=True)

        return full_counts

    def show(self):
        display_bayesian_network(self.bayesian_network)


class PrivBayesDS(BayesianNetDS):
    """
    A differentially private BayesianNet model using GreedyBayes
    """

    def __init__(
        self,
        degree,
        epsilon,
        epsilon_split,
        is_theta=False,
        with_root_hist_query=False,
        secure=True,
    ):
        assert not is_theta or with_root_hist_query, "is_theta requires root_hist_query"

        super().__init__(degree=degree, with_root_hist_query=with_root_hist_query)

        assert 0 < epsilon_split < 1

        self.transcript = []

        self.epsilon = float(epsilon)
        self.epsilon_split = float(epsilon_split)

        self.epsilon_em = self.epsilon_split * self.epsilon
        self.epsilon_hist = (1 - self.epsilon_split) * self.epsilon

        LOGGER.info(
            f"{self.epsilon=} | {self.epsilon_split=}"
            f" | {self.epsilon_em=} | {self.epsilon_hist=}"
        )

        self.is_theta = is_theta

        LOGGER.info(f"{self.is_theta=} | {self.with_root_hist_query=}")

        if not self.is_theta:
            self._greedy_bayes_linear = self._greedy_bayes_linear_DEGREE
        else:
            self._greedy_bayes_linear = self._greedy_bayes_linear_THETA

        self.secure = secure

        if not self.secure:
            warnings.warn(
                "Secure is set to False. This is not recommended for real-world use."
            )

        if secure:
            # Replacing the default global Numpy random generator with a local one
            # NOT SECURE, but it is ok for sampling
            # We store the state as part of the transcript for reproducibility
            self.sampling_prng = np.random.default_rng()
            self.transcript.append(
                ("sampling_seed", self.sampling_prng.bit_generator.state)
            )

        self.__name__ = f"PrivBayesEps{self.epsilon}"

    @property
    def laplace_noise_scale(self):
        if self.with_root_hist_query:
            return 2 * self.num_attributes / self.epsilon_hist
        else:
            return 2 * (self.num_attributes - self.degree) / self.epsilon_hist

    def _greedy_bayes_linear_DEGREE(self, encoded_df, k):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)
        num_tuples, num_attributes = dataset.shape

        attr_to_is_binary = {attr: dataset[attr].unique().size <= 2 for attr in dataset}

        if self.secure:
            LOGGER.info("Secure random for root attribute")
            root_attribute = CSPRNG.choice(dataset.columns)
        else:
            root_attribute = np.random.choice(dataset.columns)

        self.transcript.append(("root_attribute", dataset.columns, root_attribute))

        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            for child, split in it.product(
                rest_attributes, range(len(V) - num_parents + 1)
            ):
                task = (child, V, num_parents, split, dataset)
                res = bayes_worker(task)
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            sampling_distribution = exponential_mechanism(
                self.epsilon_em,
                mutual_info_list,
                parents_pair_list,
                attr_to_is_binary,
                num_tuples,
                num_attributes,
            )

            seq = list(range(len(mutual_info_list)))
            if self.secure:
                LOGGER.info("Secure random for EM")
                idxs = CSPRNG.choices(seq, weights=sampling_distribution, k=1)
                assert len(idxs) == 1
                idx = idxs[0]
            else:
                idx = np.random.choice(seq, p=sampling_distribution)

            self.transcript.append(
                (
                    "exponential_mechanism",
                    self.secure,
                    self.epsilon_em,
                    mutual_info_list,
                    parents_pair_list,
                    attr_to_is_binary,
                    num_tuples,
                    num_attributes,
                    sampling_distribution,
                    idx,
                )
            )

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _greedy_bayes_linear_THETA(self, encoded_df, theta):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)
        num_tuples, num_attributes = dataset.shape

        attr_to_is_binary = {attr: dataset[attr].unique().size <= 2 for attr in dataset}

        if self.secure:
            LOGGER.info("Secure random for root attribute")
            root_attribute = CSPRNG.choice(dataset.columns)
        else:
            root_attribute = np.random.choice(dataset.columns)

        self.transcript.append(("root_attribute", dataset.columns, root_attribute))

        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []

        tau = self._calc_tau_bound()

        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            print("#" * 80)
            # num_parents = min(len(V), k)
            for num_parents in range(1, len(V) + 1):
                for child, split in it.product(
                    rest_attributes, range(len(V) - num_parents + 1)
                ):
                    print(f"    {len(V)=} | {num_parents=} | {child=} | {split=}")
                    task = (child, V, num_parents, split, dataset)
                    # print("*" * 80)
                    # print(f"    {task=}")
                    res = bayes_worker(task)
                    print(f"    {res=}")
                    # print(f"{num_parents=}")
                    # print(f"{split=}")
                    # print(f"{child=}")
                    # assert len(res[0]) == 1

                    # assert child_parents_pair[0] == child
                    # print(f"{child_parents_pair=}")

                    # print(f"    {attributes=}")

                    for parents_pair, mutal_info in zip(res[0], res[1]):
                        assert parents_pair[0] == child
                        attributes = [parents_pair[0]] + parents_pair[1]
                        num_cells = self._calc_number_of_cells(attributes)
                        print(f"    {tau=} | {num_cells=} {tau / num_cells=}")
                        print(f"    {parents_pair=}")
                        if tau / num_cells >= theta:
                            parents_pair_list += [parents_pair]
                            mutual_info_list += [mutal_info]
            print(f"XXX {parents_pair_list=}")
            sampling_distribution = exponential_mechanism(
                self.epsilon_em,
                mutual_info_list,
                parents_pair_list,
                attr_to_is_binary,
                num_tuples,
                num_attributes,
            )

            seq = list(range(len(mutual_info_list)))
            if self.secure:
                LOGGER.info("Secure random for EM")
                idxs = CSPRNG.choices(seq, weights=sampling_distribution, k=1)
                assert len(idxs) == 1
                idx = idxs[0]
            else:
                idx = np.random.choice(seq, p=sampling_distribution)

            self.transcript.append(
                (
                    "exponential_mechanism",
                    self.secure,
                    self.epsilon_em,
                    mutual_info_list,
                    parents_pair_list,
                    attr_to_is_binary,
                    num_tuples,
                    num_attributes,
                    sampling_distribution,
                    idx,
                )
            )

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _get_attribute_frequency_counts(self, attributes, encoded_dataset):
        """Differentially private mechanism to get attribute frequency counts"""
        # Get attribute counts for category it.combinations present in data
        counts = encoded_dataset.groupby(attributes).size()
        counts.name = "count"
        counts = counts.reset_index()

        # Get all possible attribute it.combinations
        attr_combs = [
            range(self.ranges[attr][1] - self.ranges[attr][0] + 1)
            for attr in attributes
        ]
        full_space = pd.DataFrame(
            columns=attributes, data=list(it.product(*attr_combs))
        )
        full_counts = pd.merge(full_space, counts, how="left")
        full_counts.fillna(0, inplace=True)

        # Get Laplace noise sample
        if self.secure:
            LOGGER.info("Secure random for histogram")
            enable_features("floating-point", "contrib", "use-openssl")
            meas = make_base_laplace(self.laplace_noise_scale)
            noise_sample = [meas(0.0) for _ in range(full_counts.index.size)]
        else:
            noise_sample = np.random.laplace(
                0, scale=self.laplace_noise_scale, size=full_counts.index.size
            )

        self.transcript.append(
            (
                "noise_sample",
                self.secure,
                self.laplace_noise_scale,
                attributes,
                noise_sample,
            )
        )

        full_counts["count"] += noise_sample
        full_counts.loc[full_counts["count"] < 0, "count"] = 0

        return full_counts

    def _calc_number_of_cells(self, attributes):
        attrs_ranges = [
            self.ranges[attr][1] - self.ranges[attr][0] + 1 for attr in attributes
        ]
        return np.prod(attrs_ranges)

    def _calc_tau_bound(self):
        # bound = ep * tbl.size() / (4.0 * dim * theta);		// bound to nr. of cells
        tau = self.epsilon_hist * self.num_records / (2 * self.num_attributes)
        return tau

    def fit(self, df, ranges):
        if self.trained:
            self.transcript = []

        super().fit(df, ranges)
