"""
Dataset classes.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#          Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd

import time, torch  # XXX: For benchmarking workers, to be removed
from torch.utils.data import Dataset, ConcatDataset, Subset


class BaseDataset(Dataset):
    """A base dataset holds a mne.Raw, and a pandas.DataFrame with additional
    description, such as subject_id, session_id, run_id, or age or gender of
    subjects.

    Parameters
    ----------
    raw: mne.io.Raw
    description: pandas.Series
        holds additional description about the continuous signal / subject
    target_name: str | None
        name of the index in `description` that should be use to provide the
        target (e.g., to be used in a prediction task later on).
    """
    def __init__(self, raw, description, target_name=None):
        self.raw = raw
        self.description = description

        if target_name is None:
            self.target = None
        elif target_name in self.description:
            self.target = self.description[target_name]
        else:
            raise ValueError(f"'{target_name}' not in description.")

    def __getitem__(self, index):
        return self.raw[:, index][0], self.target

    def __len__(self):
        return len(self.raw)


class WindowsDataset(BaseDataset):
    """Applies a windower to a base dataset.

    Parameters
    ----------
    windows: mne.Epochs
        windows/supercrops obtained through the application of a windower to a
        BaseDataset
    description: pandas.Series
        holds additional info about the windows
    """
    def __init__(self, windows, description, transform=None):
        self.windows = windows
        self.description = description
        self.y = np.array(self.windows.metadata.loc[:,'target'])

        # XXX: Temporary hack - find another way to avoid skorch's
        #      incompatibility with more than 2 outputs
        self.output_crop_inds = False
        if self.output_crop_inds:
            self.crop_inds = np.array(self.windows.metadata.loc[:,
                                ['i_supercrop_in_trial', 'i_start_in_trial',
                                'i_stop_in_trial']])

        self.transform = transform

    def __getitem__(self, index):
        # if self.windows.preload:  # For speed
        #   X = self.windows._data[index].astype('float32')
        # else:
        X = self.windows.get_data(item=index)[0]
        if self.transform is not None:
            X = self.transform(X)
        X = X.astype('float32')
        y = self.y[index]

        if self.output_crop_inds:
            # necessary to cast as list to get list of
            # three tensors from batch, otherwise get single 2d-tensor...
            crop_inds = list(self.crop_inds[index])
            return X, y, crop_inds
        else:
            return X, y

    def __len__(self):
        return len(self.windows.events)

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value


class BaseConcatDataset(ConcatDataset):
    """A base class for concatenated datasets. Holds either mne.Raw or
    mne.Epoch in self.datasets and has a pandas DataFrame with additional
    description.

    XXX: The getter system should instead be implemented in a child class.

    Parameters
    ----------
    list_of_ds: list
        list of BaseDataset of WindowsDataset to be concatenated.
    """
    def __init__(self, list_of_ds, sampling_kind='supervised'):
        super().__init__(list_of_ds)
        self.description = pd.DataFrame(
            [ds.description for ds in list_of_ds]).reset_index(drop=True)

        self.sampling_kind = sampling_kind

    @property
    def sampling_kind(self):
        return self._sampling_kind

    @sampling_kind.setter
    def sampling_kind(self, sampling_kind):
        """Set the getter based on the kind of sampling that will be done.
        """
        self._sampling_kind = sampling_kind
        if sampling_kind == 'supervised':
            self.getter = self._getitem_single
        elif sampling_kind == 'contrastive':
            self.getter = self._getitem_contrastive
        elif sampling_kind == 'triplet':
            self.getter = self._getitem_triplet
        elif sampling_kind == 'cpc':
            self.getter = self._getitem_cpc
        else:
            raise NotImplementedError

    def _getitem_single(self, index):
        return super().__getitem__(index)

    def _getitem_contrastive(self, index):
        """E.g., relative positioning.
        """
        idx1, idx2, y = index
        return (super().__getitem__(idx1)[0], super().__getitem__(idx2)[0]), y

    def _getitem_triplet(self, index):
        """E.g., temporal shuffling.
        """
        idx1, idx2, idx3, y = index
        return (
            super().__getitem__(idx1)[0],
            super().__getitem__(idx2)[0],
            super().__getitem__(idx2)[0]), y

    def _getitem_cpc(self, index):
        context_inds, predict_inds, y = index

        # List comprehensions sadly won't work because of scope of super()
        # XXX: Maybe this could changed to use map
        x1, x2 = list(), list()
        for i in context_inds:
            x1.append(super().__getitem__(i)[0])
        for i in predict_inds:
            x2.append(super().__getitem__(i)[0])

        x1 = np.stack(x1) if x1 else list()
        x2 = np.stack(x2) if x2 else list()

        return (x1, x2), y

    # def __getitem__(self, index):
    #     worker = torch.utils.data.get_worker_info()
    #     worker_id = worker.id if worker is not None else -1
    #     start = time.time()
    #     X, y = self.getter(index)
    #     end = time.time()
    #     return (X, worker_id, start, end), y

    def __getitem__(self, index):
        return self.getter(index)

    def split(self, some_property=None, split_ids=None):
        """Split the dataset based on some property listed in its description
        DataFrame or based on indices.

        Parameters
        ----------
        some_property: str
            some property which is listed in info DataFrame
        split_ids: list(int)
            list of indices to be combined in a subset

        Returns
        -------
        splits: dict{split_name: BaseConcatDataset}
            mapping of split name based on property or index based on split_ids
            to subset of the data
        """
        if split_ids is None and some_property is None:
            raise ValueError('Splitting requires defining ids or a property.')
        if isinstance(some_property, str) and isinstance(split_ids, list):
            split_ids = {
                split_i: np.where(self.description[some_property].isin(inds))[0]
                for split_i, inds in enumerate(split_ids)}
        elif split_ids is None:
            split_ids = {k: list(v) for k, v in self.description.groupby(
                some_property).groups.items()}
        else:
            split_ids = {split_i: split
                         for split_i, split in enumerate(split_ids)}

        return {split_name: BaseConcatDataset(
            [self.datasets[ds_ind] for ds_ind in ds_inds],
            sampling_kind=self.sampling_kind)
            for split_name, ds_inds in split_ids.items()}

    def subset(self, inds):
        """Create a subset with specific indices.

        This is a convenience method that also creates a metadata attribute.

        Parameters
        ----------
        inds : list
            List of indices to keep in the Subset.
        """
        raise NotImplementedError('Use a SubsetRandomSampler instead.')
        # ds = Subset(self, inds)
        # ds.metadata = self.metadata.iloc[inds, :]
        # return ds

    @property
    def metadata(self):
        return pd.concat([ds.windows.metadata for ds in self.datasets], axis=0)

    @property
    def metadata_desc(self):
        mds = list()
        for ds in self.datasets:
            md = ds.windows.metadata
            for k, v in ds.description.iteritems():
                md[k] = v
            mds.append(md)
        df = pd.concat(mds)
        return df
