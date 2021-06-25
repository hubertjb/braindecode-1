"""
Convenience functions for storing and loading of windows datasets.
"""

# Authors: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import os
import json
import warnings
from glob import glob

import mne
from mne.io.fiff import raw
import pandas as pd

from ..datasets.base import BaseDataset, BaseConcatDataset, WindowsDataset


def create_fname_id_from_description(description):
    # XXX: Enforce 'subject', 'recording' and/or 'run' in description?
    fname = list()
    if 'subject' in description:
        fname.append(f'subj{description["subject"]}')
    if 'recording' in description:
        fname.append(f'rec{description["recording"]}')
    if 'run' in description:
        fname.append(f'run{description["run"]}')

    if not fname:
        raise ValueError(
            'description must contain any of "subject", "recording" or "run".')

    return '_'.join(fname)


def save_base_dataset(dataset, path, overwrite=False):
    """Save a BaseDataset.

    Parameters
    ----------
    dataset : BaseDataset
        BaseDataset to serialize.
    path : str
        Directory in which the dataset should be saved.
    overwrite : bool
        If True, overwrite existing files with the same name. If False and
        files with the same name exist, an error will be raised.

    Returns
    -------
    str, str, str
        File names for the data, the description dictionary, and the target.
    """
    if not isinstance(dataset, BaseDataset):
        raise TypeError('dataset must inherit from BaseDataset.')

    if hasattr(dataset, 'raw'):
        dataset_type = 'raw'
    elif hasattr(dataset, 'windows'):
        dataset_type = 'windows'
    else:
        raise AttributeError(
            'dataset should have either a `raw` or `windows` attribute.')

    fnames_ = ['{}-raw.fif', '{}-epo.fif']

    fname_template = fnames_[0] if dataset_type == 'raw' else fnames_[1]
    basename = create_fname_id_from_description(dataset.description)
    fname = os.path.join(path, fname_template.format(basename))
    description_fname = os.path.join(path, f'{basename}-description.json')
    target_fname = os.path.join(path, f'{basename}-target_name.json')

    if not overwrite:
        if (os.path.exists(description_fname) or
                os.path.exists(target_fname)):
            raise FileExistsError(
                f'{description_fname} or {target_fname} already exist '
                f'under {path}.')
    else:
        for f in [fname, target_fname, description_fname]:
            if os.path.isfile(f):
                os.remove(f)

    getattr(dataset, dataset_type).save(fname, overwrite=overwrite)
    if dataset_type == 'raw':
        json.dump({'target_name': dataset.target_name},
                  open(target_fname, 'w'))
    dataset.description.to_json(description_fname)

    return fname, description_fname, target_fname


def load_base_dataset(fname, preload=False):
    """Load a BaseDataset.

    Parameters
    ----------
    path : str
        E.g. `my_data/subj0_rec1-raw.fif`
    preload : bool
        ...

    Returns
    -------
    ...
    """
    dirname, basename = os.path.split(fname)
    basename = basename[:-8]

    description_fname = os.path.join(dirname, basename + '-description.json')
    target_fname = os.path.join(dirname, basename + '-target_name.json')

    raw_or_epochs = _load_signals(fname, preload=preload)
    with open(description_fname, 'r') as f:
        description = json.load(f)

    if isinstance(raw_or_epochs, mne.io.Raw):
        with open(target_fname, 'r') as f:
            target_name = json.load(f)['target_name']
        ds = BaseDataset(raw_or_epochs, description=description,
                         target_name=target_name)
    elif isinstance(raw_or_epochs, mne.Epochs):
        ds = WindowsDataset(raw_or_epochs, description)

    # XXX What about transforms?

    return ds


def save_concat_dataset(path, concat_dataset, overwrite=False):
    warnings.warn('"save_concat_dataset()" is deprecated and will be removed in the future. '
                  'Use dataset.save() instead.')
    concat_dataset.save(path=path, overwrite=overwrite)


def load_concat_dataset(path, preload, ids_to_load=None, target_name=None):
    """Load a stored BaseConcatDataset of BaseDatasets or WindowsDatasets from
    files.

    Parameters
    ----------
    path: str
        Path to the directory of the .fif / -epo.fif and .json files.
    preload: bool
        Whether to preload the data.
    ids_to_load: None | list(int)
        Ids of specific files to load.
    target_name: None or str
        Load specific description column as target. If not given, take saved target name.

    Returns
    -------
    concat_dataset: BaseConcatDataset of BaseDatasets or WindowsDatasets
    """
    # assume we have a single concat dataset to load
    concat_of_raws = os.path.isfile(os.path.join(path, '0-raw.fif'))
    assert not (not concat_of_raws and target_name is not None), (
        'Setting a new target is only supported for raws.')
    concat_of_epochs = os.path.isfile(os.path.join(path, '0-epo.fif'))
    paths = [path]
    # assume we have multiple concat datasets to load
    if not (concat_of_raws or concat_of_epochs):
        concat_of_raws = os.path.isfile(os.path.join(path, '0', '0-raw.fif'))
        concat_of_epochs = os.path.isfile(os.path.join(path, '0', '0-epo.fif'))
        path = os.path.join(path, '*', '')
        paths = glob(path)
        paths = sorted(paths, key=lambda p: int(p.split(os.sep)[-2]))
        if ids_to_load is not None:
            paths = [paths[i] for i in ids_to_load]
        ids_to_load = None
    # if we have neither a single nor multiple datasets, something went wrong
    assert concat_of_raws or concat_of_epochs, (
        f'Expect either raw or epo to exist in {path} or in '
        f'{os.path.join(path, "0")}')

    datasets = []
    for path in paths:
        if concat_of_raws and target_name is None:
            target_file_name = os.path.join(path, 'target_name.json')
            target_name = json.load(open(target_file_name, "r"))['target_name']

        all_signals, description = _load_signals_and_description(
            path=path, preload=preload, raws=concat_of_raws,
            ids_to_load=ids_to_load
        )
        for i_signal, signal in enumerate(all_signals):
            if concat_of_raws:
                datasets.append(
                    BaseDataset(signal, description.iloc[i_signal],
                                target_name=target_name))
            else:
                datasets.append(
                    WindowsDataset(signal, description.iloc[i_signal])
                )
    return BaseConcatDataset(datasets)


def _load_signals_and_description(path, preload, raws, ids_to_load=None):
    all_signals = []
    file_name = "{}-raw.fif" if raws else "{}-epo.fif"
    description_df = pd.read_json(os.path.join(path, "description.json"))
    if ids_to_load is None:
        file_names = glob(os.path.join(path, f"*{file_name.lstrip('{}')}"))
        # Extract ids, e.g.,
        # '/home/schirrmr/data/preproced-tuh/all-sensors/11-raw.fif' ->
        # '11-raw.fif' -> 11
        ids_to_load = sorted(
            [int(os.path.split(f)[-1].split('-')[0]) for f in file_names])
    for i in ids_to_load:
        fif_file = os.path.join(path, file_name.format(i))
        all_signals.append(_load_signals(fif_file, preload))
    description_df = description_df.iloc[ids_to_load]
    return all_signals, description_df


def _load_signals(fif_file, preload):
    if fif_file.endswith('-raw.fif'):
        signals = mne.io.read_raw_fif(fif_file, preload=preload)
    elif fif_file.endswith('-epo.fif'):
        signals = mne.read_epochs(fif_file, preload=preload)
    else:
        raise ValueError('fif_file must end with raw.fif or epo.fif.')
    return signals
