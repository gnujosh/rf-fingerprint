# -----------------------------------------------------------------------------------
# ---------------------------------- UNCLASSIFIED -----------------------------------
# This code was developed under the Defense Advanced Research Projects Agency (DARPA)
# Radio Frequency Machine Learning System (RFMLS) program: contract FA8750-18-C-0150.
# Government Unlimited Rights
# BAE Systems, Inc. (C) 2020.
# -----------------------------------------------------------------------------------

"""wifi_ofdm_generator.py

keras generators for the wifi ofdm data generated in matlab
"""

# standard lib imports
import numpy as np
import os
from copy import deepcopy

# third party imports
from tensorflow.keras.utils import Sequence


class WifiData(Sequence):
    """80,000 simulated 802.11a/g digital signals generated with Matlabs wlan toolbox.

    802.11a/g has 8 different modulations:

        1. BPSK 1/2 rate --> 10,000 signals
        2. BPSK 3/4 rate --> 10,000 signals
        3. QPSK 1/2 rate --> 10,000 signals
        4. QPSK 3/4 rate --> 10,000 signals
        5. 16QAM 1/2 rate -> 10,000 signals
        6. 16QAM 3/4 rate -> 10,000 signals
        7. 64QAM 2/3 rate -> 10,000 signals
        8. 64QAM 3/4 rate -> 10,000 signals
    """

    def __init__(self, dataroot):
        """Return a Sequence generator for the entire data set with batch size of one.

        Arguments

            dataroot: (string) path to data source file directory.
        """
        self.samps_per_signal = 1600
        self.dataroot = dataroot
        # data files
        sourcefiles = [
            "OFDM_1_CBW20_10000_sigs_by_1600_samples_bpsk_halfRate_modulated.bin",
            "OFDM_2_CBW20_10000_sigs_by_1600_samples_bpsk_threeQtrRate_modulated.bin",
            "OFDM_3_CBW20_10000_sigs_by_1600_samples_qpsk_halfRate_modulated.bin",
            "OFDM_4_CBW20_10000_sigs_by_1600_samples_qpsk_threeQtrRate_modulated.bin",
            "OFDM_5_CBW20_10000_sigs_by_1600_samples_16qam_halfRate_modulated.bin",
            "OFDM_6_CBW20_10000_sigs_by_1600_samples_16qam_threeQtrRate_modulated.bin",
            "OFDM_7_CBW20_10000_sigs_by_1600_samples_64qam_twoThirdRate_modulated.bin",
            "OFDM_8_CBW20_10000_sigs_by_1600_samples_64qam_threeQtrRate_modulated.bin",
        ]
        self.sourcefiles = [os.path.join(dataroot, sf) for sf in sourcefiles]
        # open files for reading now and close on __delete__
        self.openfids = [open(sf, "rb") for sf in self.sourcefiles]
        self.sig_index = np.arange(80000)
        # evenly distribute signal index wrt modulation type
        self.mod_index = np.array(10000 * [i for i in range(8)])
        self.get_mod_name = {
            0: "bpsk_halfRate",
            1: "bpsk_threeQtrRate",
            2: "qpsk_halfRate",
            3: "qpsk_threeQtrRate",
            4: "16qam_halfRate",
            5: "16qam_threeQtrRate",
            6: "64qam_twoThirdRate",
            7: "64qam_threeQtrRate",
        }

    def __len__(self):
        return len(self.sig_index)

    def __getitem__(self, idx):
        """Returns idx signal as complex numpy array.
        
        Arguments

            idx: (integer) signal index in range [0,79999]
        """
        index = self.sig_index[idx]
        mod = self.mod_index[index]
        skip = int(index / 8)
        fid = self.openfids[mod]
        fid.seek(skip * 4 * self.samps_per_signal, 0)
        sig = np.fromfile(fid, dtype="int16", count=2 * self.samps_per_signal)
        # return sig.astype("float").view("complex128")
        sig = sig.astype("float")
        sig = sig[: self.samps_per_signal] + 1j * sig[self.samps_per_signal :]
        return sig

    def get_modulation(self, idx):
        """Return the modulation for the signal corresponding to idx."""
        index = self.sig_index[idx]
        mod = self.mod_index[index]
        return self.get_mod_name[mod]

    def __delete__(self):
        for fid in self.openfids:
            close(fid)

    def __getstate__(self):
        """make class pickleable."""
        state = self.__dict__.copy()
        del state["openfids"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.openfids = [open(sf, "rb") for sf in self.sourcefiles]


class WifiSlice(Sequence):
    """Sequence generator that slices the WifiData generator's data."""

    def __init__(self, wifigen, slice_start=0, slice_end=1):
        """Returns a WifiSlice generator that slices the WifiData generator.

        Arguments
            
            wifigen: (WifiData object) the wifi dataset generator to slice up.
            slice_start: (float in [0,1)) start of slice range.
            slice_end: (float in [0,1)) end of slice range.
        """
        self.wifigen = deepcopy(wifigen)
        self.start = int(slice_start * len(self.wifigen))
        self.end = int(slice_end * len(self.wifigen))
        self.sig_index = self.wifigen.sig_index[self.start : self.end].copy()

    def __len__(self):
        raise NotImplementedError("Please implement in WifiSlice child classes")

    def __getitem__(self, index):
        raise NotImplementedError("Please implement in WifiSlice child classes")


def minusOneToOne(x):
    return np.real(x) / 32767.0 + 1j * np.imag(x) / 32767.0


def identity(x):
    return x


class WifiReconstruct(WifiSlice):
    """Sequence generator that returns batches of WifiData for training reconstruction models."""

    def __init__(
        self,
        wifigen,
        batch_size=30,
        slice_start=0,
        slice_end=1,
        transformation=identity,
    ):
        """Returns a WifiSlice generator that slices the WifiData generator.

        Arguments
            
            wifigen: (WifiData object) the wifi dataset generator to slice up.
            batch_size: (positive integer) batch size to generate.
            slice_start: (float in [0,1)) start of slice range.
            slice_end: (float in [0,1)) end of slice range.
            transformation: (callable) transformation to perform on signals
        """
        super().__init__(wifigen, slice_start=slice_start, slice_end=slice_end)
        self.batch_size = batch_size
        # calculate length in batch_sizes
        self.length = len(self.sig_index) // batch_size
        self.extra = len(self.sig_index) % batch_size
        if self.extra > 0:
            self.length += (
                1
            )  # I'll pad last batch with randomly sampled signals from training set
        self.transformation = transformation

    def __len__(self):
        return self.length

    def __getitem__(self, batch_index):
        """Return a batch of training signals and reconstruction targets.
        
        Arguments
        
            batch_index: (positive integer) the batch index.
        """
        if self.extra and batch_index == self.length - 1:
            # last batch that needs extra signals from training data to make full batch
            random_indices = np.random.choice(
                self.sig_index, self.extra, replace=False
            ).tolist()
            num_collect = self.batch_size - self.extra
        else:
            random_indices = []
            num_collect = self.batch_size
        sig_indices = [
            self.sig_index[idx + batch_index * self.batch_size]
            for idx in range(num_collect)
        ] + random_indices
        x = np.array(
            [
                self.transformation(minusOneToOne(self.wifigen[idx]))
                for idx in sig_indices
            ]
        )
        y = x
        return (x, y)

    def get_batch_modulations(self, batch_index):
        """retun list of modulations names for the batch corresponding to batch_index.
        
        Note: will not work for last batch when self.extra > 0
        """
        sig_indices = [
            self.sig_index[idx + batch_index * self.batch_size]
            for idx in range(self.batch_size)
        ]
        modulations = [self.wifigen.get_modulation(si) for si in sig_indices]
        return modulations


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataroot = "/local/wifigenLarge"
    wifidata = WifiData(dataroot)
    data = WifiReconstruct(wifidata, batch_size=8)
    x, y = data[0]
    print(x[0, 0:5])
    m = data.get_batch_modulations(0)
    print(m[0])
    plt.plot(np.abs(sig[:, 0]))
    plt.show()
