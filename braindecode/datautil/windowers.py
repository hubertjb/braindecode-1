"""Get epochs from mne.Raw
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Lukas Gemein <l.gemein@gmail.com>
#          Simon Brandt <simonbrandt@protonmail.com>
#          David Sabbagh <dav.sabbagh@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
import pandas as pd
from joblib import Parallel, delayed

from ..datasets.base import WindowsDataset, BaseConcatDataset, BaseDataset


def create_windows_from_events(
        concat_ds, trial_start_offset_samples, trial_stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples, drop_samples,
        mapping=None, picks=None, preload=False, drop_bad_windows=True,
        transform=None, extract_annotations=False, n_jobs=1, max_nbytes='1M'):
    """Windower that creates supercrops/windows based on events in mne.Raw.

    The function fits supercrops of supercrop_size_samples in
    trial_start_offset_samples to trial_stop_offset_samples separated by
    supercrop_stride_samples. If the last supercrop does not end
    at trial_stop_offset_samples, it creates another overlapping supercrop that
    ends at trial_stop_offset_samples if drop_samples is set to False.

    in mne: tmin (s)                    trial onset        onset + duration (s)
    trial:  |--------------------------------|--------------------------------|
    here:   trial_start_offset_samples                trial_stop_offset_samples

    Parameters
    ----------
    concat_ds: BaseConcatDataset
        a concat of base datasets each holding raw and description
    trial_start_offset_samples: int
        start offset from original trial onsets in samples
    trial_stop_offset_samples: int
        stop offset from original trial stop in samples
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally divide the continuous signal
    mapping: dict(str: int)
        mapping from event description to target value
    picks: list | None
        channels to use.
    preload: bool
        if True, preload the data of the Epochs objects.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifiying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.
    transform: object | None
        Transform to be applied on-the-fly when calling __getitem__.
    extract_annotations: bool
        If True, extract overlapping ratios of the annotations in the Raw object
        and add them to the metadata DataFrame.
    n_jobs: int
        Number of jobs to parallelize the windowing.

    Returns
    -------
    windows_ds: WindowsDataset
        Dataset containing the extracted windows.
    """
    _check_windowing_arguments(
        trial_start_offset_samples, trial_stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples)

    def _apply_windowing(ds):
        mapping_ds = mapping  # To avoid hiding parent namespace's mapping
        if mapping_ds is None:
            # mapping event descriptions to integers from 0 on
            mapping_ds = {v: k for k, v in enumerate(
                np.unique(ds.raw.annotations.description))}

        events, _ = mne.events_from_annotations(ds.raw, mapping_ds)
        onsets = events[:, 0]
        # XXX: Find another way of taking duration into account?
        annots_inds = np.isin(
            ds.raw.annotations.description, list(mapping_ds.keys()))
        stops = onsets + (ds.raw.annotations.duration[annots_inds]
                          * ds.raw.info['sfreq']).astype(int)

        if stops[-1] + trial_stop_offset_samples > len(ds) + ds.raw.first_samp:
            raise ValueError('"trial_stop_offset_samples" too large. Stop of '
                             f'last trial ({stops[-1]}) + '
                             f'"trial_stop_offset_samples" '
                             f'({trial_stop_offset_samples}) must be smaller '
                             f'then length of recording {len(ds)}.')

        description = events[:, -1]
        i_trials, i_supercrop_in_trials, starts, stops = _compute_supercrop_inds(
            onsets, stops, trial_start_offset_samples,
            trial_stop_offset_samples, supercrop_size_samples,
            supercrop_stride_samples, drop_samples)

        events = [[start, supercrop_size_samples, description[i_trials[i_start]]]
                  for i_start, start in enumerate(starts)]
        events = np.array(events)

        if any(np.diff(events[:, 0]) <= 0):
            raise NotImplementedError('Trial overlap not implemented.')

        description = events[:, -1]

        metadata = pd.DataFrame({
            'i_supercrop_in_trial': i_supercrop_in_trials,
            'i_start_in_trial': starts,
            'i_stop_in_trial': stops,
            'target': description})
        if extract_annotations:
            annots_md = get_annotations_ratio(
                events, ds.raw, supercrop_size_samples)
            metadata = pd.concat([metadata, annots_md], axis=1)

        # supercrop size - 1, since tmax is inclusive
        mne_epochs = mne.Epochs(
            ds.raw, events, mapping_ds, baseline=None, tmin=0,
            tmax=(supercrop_size_samples - 1) / ds.raw.info['sfreq'],
            metadata=metadata, picks=picks, preload=preload,
            on_missing='warning')

        if drop_bad_windows:
            mne_epochs = mne_epochs.drop_bad(reject=None, flat={'eeg': 1e-6})

        return WindowsDataset(mne_epochs, ds.description, transform=transform)

    if n_jobs == 1:
        list_of_windows_ds = [
            _apply_windowing(ds) for ds in concat_ds.datasets]
    else:
        list_of_windows_ds = Parallel(n_jobs=n_jobs, max_nbytes=max_nbytes)(
            delayed(_apply_windowing)(ds) for ds in concat_ds.datasets)

    return BaseConcatDataset(list_of_windows_ds)


def get_fixed_length_window(ds, mapping, start_offset_samples,
                            stop_offset_samples, supercrop_size_samples,
                            supercrop_stride_samples, drop_samples,
                            drop_bad_windows, preload, picks, transform,
                            description=None, target=None, mne_out=False):
    if isinstance(ds, BaseDataset):
        raw = ds.raw
        description = ds.description
        target = ds.target
    elif isinstance(ds, mne.io.Raw):
        raw = ds
    else:
        raise ValueError('ds must be of type BaseDataset or mne.io.Raw')

    # already includes last incomplete supercrop start
    stop = (raw.n_times
            if stop_offset_samples is None
            else stop_offset_samples)
    last_allowed_ind = stop - supercrop_size_samples
    starts = np.arange(start_offset_samples, last_allowed_ind + 1,
                       supercrop_stride_samples) + raw.first_samp

    if not drop_samples and starts[-1] < last_allowed_ind + raw.first_samp:
        # if last supercrop does not end at trial stop, make it stop there
        starts = np.append(starts, last_allowed_ind + raw.first_samp)

    # TODO: handle multi-target case / non-integer target case
    target = -1 if target is None else target
    if mapping is not None:
        target = mapping[target]
    if type(target) in (float, np.float, np.float32):  # Most likely regression target
        target = int(target)
    if not isinstance(target, (np.integer, int)):
        raise ValueError(f"Mapping from '{target}' to int is required")

    fake_events = [[start, supercrop_size_samples, target]
                   for i_start, start in enumerate(starts)]
    metadata = pd.DataFrame({
        'i_supercrop_in_trial': np.arange(len(fake_events)),
        'i_start_in_trial': starts,
        'i_stop_in_trial': starts + supercrop_size_samples,
        'target': len(fake_events) * [target]
    })

    # supercrop size - 1, since tmax is inclusive
    mne_epochs = mne.Epochs(
        raw, fake_events, mapping, baseline=None, tmin=0,
        tmax=(supercrop_size_samples - 1) / raw.info['sfreq'],
        metadata=metadata, picks=picks, preload=preload,
        on_missing='warning')

    if drop_bad_windows:
        mne_epochs = mne_epochs.drop_bad(reject=None, flat={'eeg': 1e-6})

    if len(mne_epochs) == 0:  # Ignore empty sessions  XXX TEST MORE EXTENSIVELY!
        out = None
    elif mne_out:
        out = mne_epochs
    else:
        out = WindowsDataset(mne_epochs, description, transform=transform)

    return out


def create_fixed_length_windows(
        concat_ds, start_offset_samples, stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples, drop_samples,
        mapping=None, picks=None, preload=False, drop_bad_windows=True,
        transform=None, n_jobs=1, max_nbytes='1M'):
    """Windower that creates sliding supercrops/windows.

    Parameters
    ----------
    concat_ds: ConcatDataset
        a concat of base datasets each holding raw and descpription
    start_offset_samples: int
        start offset from beginning of recording in samples
    stop_offset_samples: int | None
        stop offset from beginning of recording in samples.
    supercrop_size_samples: int
        supercrop size
    supercrop_stride_samples: int
        stride between supercrops
    drop_samples: bool
        whether or not have a last overlapping supercrop/window, when
        supercrops/windows do not equally divide the continuous signal
    mapping: dict(str: int)
        mapping from event description to target value
    picks: list | None
        channels to use.
    preload: bool
        if True, preload the data of the Epochs objects.
    drop_bad_windows: bool
        If True, call `.drop_bad()` on the resulting mne.Epochs object. This
        step allows identifiying e.g., windows that fall outside of the
        continuous recording. It is suggested to run this step here as otherwise
        the BaseConcatDataset has to be updated as well.
    transform: object | None
        Transform to be applied on-the-fly when calling __getitem__.
    n_jobs: int
        Number of jobs to parallelize the windowing.

    Returns
    -------
    windows_ds: WindowsDataset
        Dataset containing the extracted windows.
    """
    _check_windowing_arguments(
        start_offset_samples, stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples)

    if n_jobs == 1:
        list_of_windows_ds = [
            get_fixed_length_window(
                ds, mapping, start_offset_samples, stop_offset_samples,
                supercrop_size_samples, supercrop_stride_samples, drop_samples,
                drop_bad_windows, preload, picks, transform)
            for ds in concat_ds.datasets]
    else:
        list_of_windows_ds = Parallel(n_jobs=n_jobs, max_nbytes=max_nbytes)(
            delayed(get_fixed_length_window)(
                ds, mapping, start_offset_samples, stop_offset_samples,
                supercrop_size_samples, supercrop_stride_samples, drop_samples,
                drop_bad_windows, preload, picks, transform)
            for ds in concat_ds.datasets)

    list_of_windows_ds = [ds for ds in list_of_windows_ds if ds is not None]

    return BaseConcatDataset(list_of_windows_ds)


def _compute_supercrop_inds(
        starts, stops, start_offset, stop_offset, size, stride, drop_samples):
    """Create supercrop starts from trial onsets (shifted by offset) to trial
    end separated by stride as long as supercrop size fits into trial

    Parameters
    ----------
    starts: array-like
        trial starts in samples
    stops: array-like
        trial stops in samples
    start_offset: int
        start offset from original trial onsets in samples
    stop_offset: int
        stop offset from original trial stop in samples
    size: int
        supercrop size
    stride: int
        stride between supercrops
    drop_samples: bool
        toggles of shifting last supercrop within range or dropping last samples

    Returns
    -------
    result_lists: (list, list, list, list)
        trial, i_supercrop_in_trial, start sample and stop sample of supercrops
    """

    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops

    starts += start_offset
    stops += stop_offset

    i_supercrop_in_trials, i_trials, supercrop_starts = [], [], []
    for start_i, (start, stop) in enumerate(zip(starts, stops)):
        # between original trial onsets (shifted by start_offset) and stops,
        # generate possible supercrop starts with given stride
        possible_starts = np.arange(
            start, stop, stride)

        # possible supercrop start is actually a start, if supercrop size fits
        # in trial start and stop
        for i_supercrop, s in enumerate(possible_starts):
            if (s + size) <= stop:
                supercrop_starts.append(s)
                i_supercrop_in_trials.append(i_supercrop)
                i_trials.append(start_i)

        # if the last supercrop start + supercrop size is not the same as
        # stop + stop_offset, create another supercrop that overlaps and stops
        # at onset + stop_offset
        if not drop_samples:
            if supercrop_starts[-1] + size != stop:
                supercrop_starts.append(stop - size)
                i_supercrop_in_trials.append(i_supercrop_in_trials[-1] + 1)
                i_trials.append(start_i)

    # update stops to now be event stops instead of trial stops
    supercrop_stops = np.array(supercrop_starts) + size
    if not (len(i_supercrop_in_trials) == len(supercrop_starts) ==
            len(supercrop_stops)):
        raise ValueError(f'{len(i_supercrop_in_trials)} == '
                         f'{len(supercrop_starts)} == {len(supercrop_stops)}')
    return i_trials, i_supercrop_in_trials, supercrop_starts, supercrop_stops


def _check_windowing_arguments(
        trial_start_offset_samples, trial_stop_offset_samples,
        supercrop_size_samples, supercrop_stride_samples):
    assert supercrop_size_samples > 0, (
        "supercrop size has to be larger than 0")
    assert supercrop_stride_samples > 0, (
        "supercrop stride has to be larger than 0")
    assert isinstance(trial_start_offset_samples, (int, np.integer))
    if trial_stop_offset_samples is not None:
        assert isinstance(trial_stop_offset_samples, (int, np.integer))
    assert isinstance(supercrop_size_samples, (int, np.integer))
    assert isinstance(supercrop_stride_samples, (int, np.integer))


def get_annotations_ratio(events, raw, win_len):
    """Get the ratio of each annotation inside events.

    Parameters
    ----------
    events : np.ndarray
        Events, shape (n_events, 3)
    annots : mne.Annotations
        Annotations.
    times : np.ndarray
        Sample times.
    win_len : int
        Number of samples in a window.

    Return
    ------
    pd.DataFrame
        DataFrame of shape (n_events, n_unique_annotations) containing the ratio
        of each annotation for each event.
    """
    unique_annots = {d: i for i, d in enumerate(np.unique(raw.annotations.description))}
    annots_events = np.zeros((events.shape[0], len(unique_annots)), dtype=np.float16)

    for a_type in unique_annots:
        grouped_a = raw.annotations[raw.annotations.description == a_type]
        trigger = np.zeros(len(raw.times), dtype=int)
        for a in grouped_a:
            onset = a['onset'] - raw.first_samp / raw.info['sfreq']
            mask = ((raw.times >= a['onset']) &
                    (raw.times < a['onset'] + a['duration']))  # This is probably a big bottleneck...
            trigger[mask] = 1
        for i, start in enumerate(events[:, 0]):
            annots_events[i, unique_annots[a_type]] = trigger[start:start + win_len].mean()

    return pd.DataFrame(annots_events, columns=unique_annots.keys())
