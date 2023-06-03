import numpy as np
import pandas as pd

from generative_models.data_synthesiser import DataDescriber
from utils.constants import *


class MyTransformer:
    def __init__(self, metadata, histogram_bins):
        self.metadata = self._read_meta(metadata)
        self.histogram_bins = histogram_bins
        self.DataDescriber = DataDescriber(
            self.metadata, histogram_bins, infer_ranges=False
        )
        self.ranges = None

    def encode(self, df):
        self.DataDescriber.describe(df)
        self.ranges = {
            attr: (0, attr_d.domain_size - 1)
            for attr, attr_d in self.DataDescriber.attr_dict.items()
        }

        encoded_df = pd.DataFrame(columns=self.DataDescriber.attr_names)

        for attr_name, column in self.DataDescriber.attr_dict.items():
            encoded_df[attr_name] = column.encode_values_into_bin_idx()

        return encoded_df

    def decode(self, df):
        decoded_df = pd.DataFrame(columns=self.DataDescriber.attr_names)

        # Get samples for attributes modelled in Bayesian net

        for attr in self.DataDescriber.attr_names:
            column = self.DataDescriber.attr_dict[attr]
            assert attr in df
            assert all(np.isfinite(df[attr]))
            decoded_df[attr] = column.sample_values_from_binning_indices(df[attr])

        return decoded_df

    def _read_meta(self, metadata):
        """Read metadata from metadata file."""
        metadict = {}

        for cdict in metadata["columns"]:
            col = cdict["name"]
            coltype = cdict["type"]

            if coltype == FLOAT or coltype == INTEGER:
                metadict[col] = {
                    "type": coltype,
                    "min": cdict["min"],
                    "max": cdict["max"],
                }

            elif coltype == CATEGORICAL or coltype == ORDINAL:
                metadict[col] = {
                    "type": coltype,
                    "categories": cdict["i2s"],
                    "size": len(cdict["i2s"]),
                }

            else:
                raise ValueError(f"Unknown data type {coltype} for attribute {col}")

        return metadict
