"""Benchmarking eager and lazy loading
======================================

In this example, we compare the execution time and memory requirements of 1)
eager loading, i.e., preloading the entire data into memory and 2) lazy loading,
i.e., only loading examples from disk when they are required. We also include
some other experiment parameters in the comparison for the sake of completeness
(e.g., `num_workers`, `cuda`, `batch_size`, etc.).

While eager loading might be required for some preprocessing steps that require
continuous data (e.g., temporal filtering, resampling), it also allows
fast access to the data during training. However, this might come at the expense
of large memory usage, and can ultimately become impossible if the dataset does
not fit into memory (e.g., the TUH EEG dataset's >1,5 TB of recordings will
not fit in the memory of most machines).

Lazy loading avoids this potential memory issue by loading examples from disk
when they are required. This means large datasets can be used for training,
however this introduces some file-reading overhead every time an example must
be extracted. Some preprocessing steps that require continuous data also have to
be implemented differently to accomodate the nature of windowed data. Overall
though, we can reduce the impact of lazy loading by using the `num_workers`
parameter of pytorch's `Dataloader` class, which dispatches the data loading to
multiple processes.

To enable lazy loading in braindecode, data files must be saved in an
MNE-compatible format (e.g., 'fif', 'edf', etc.), and the `Dataset` object must
have been instantiated with parameter `preload=False`.

NOTE:
This version was modified so that the different parts of the pipeline could be
timed.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import os
from itertools import product
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from skorch.callbacks import Callback, EpochScoring, Checkpoint, LoadInitState
from skorch.net import NeuralNet
from skorch.helper import predefined_split
from skorch.utils import to_numpy

from braindecode.datasets import TUHAbnormal
from braindecode.datautil.windowers import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet, Deep4Net, EEGResNet


mne.set_log_level('WARNING')  # avoid messages everytime a window is extracted

###############################################################################
# We start by setting two pytorch internal parameters that can affect the
# comparison::
N_JOBS = 8
torch.backends.cudnn.benchmark = True  # Enables automatic algorithm optimizations
torch.backends.cudnn.enabled = True
torch.set_num_threads(N_JOBS)  # Sets the available number of threads

###############################################################################
# Next, we define a few functions to automate the benchmarking.
# For the purpose of this example, we load some recordings from the TUH Abnormal
# corpus, extract sliding windows, and bundle them in a braindecode Dataset.
# We then train a neural network for a few epochs.
#
# Each one of these steps will be timed, so we can report the total time taken
# to prepare the data and train the model.

def load_example_data(preload, window_len_s, n_subjects=10, n_jobs=1):
    """Create windowed dataset from subjects of the TUH Abnormal dataset.

    Parameters
    ----------
    preload: bool
        If True, use eager loading, otherwise use lazy loading.
    n_subjects: int
        Number of subjects to load.

    Returns
    -------
    windows_ds: BaseConcatDataset
        Windowed data.

    .. warning::
        The recordings from the TUH Abnormal corpus do not all share the same
        sampling rate. The following assumes that the files have already been
        resampled to a common sampling rate.
    """
    subject_ids = list(range(n_subjects))
    ds = TUHAbnormal(
        TUH_PATH, subject_ids=subject_ids, target_name='pathological',
        preload=preload, n_jobs=n_jobs)

    fs = ds.datasets[0].raw.info['sfreq']
    window_len_samples = int(fs * window_len_s)
    window_stride_samples = int(fs * 4)
    # window_stride_samples = int(fs * window_len_s)
    windows_ds = create_fixed_length_windows(
        ds, start_offset_samples=0, stop_offset_samples=None,
        supercrop_size_samples=window_len_samples,
        supercrop_stride_samples=window_stride_samples, drop_samples=True,
        preload=preload, drop_bad_windows=True, n_jobs=n_jobs)

    return windows_ds


def create_example_model(n_channels, n_classes, window_len_samples,
                         kind='shallow', cuda=False):
    """Create model, loss and optimizer.

    Parameters
    ----------
    n_channels : int
        Number of channels in the input
    n_times : int
        Window length in the input
    n_classes : int
        Number of classes in the output
    kind : str
        'shallow' or 'deep'
    cuda : bool
        If True, move the model to a CUDA device.

    Returns
    -------
    model : torch.nn.Module
        Model to train.
    loss :
        Loss function
    optimizer :
        Optimizer
    """
    if kind == 'shallow':
        model = ShallowFBCSPNet(
            n_channels, n_classes, input_time_length=window_len_samples,
            n_filters_time=40, filter_time_length=25, n_filters_spat=40,
            pool_time_length=75, pool_time_stride=15, final_conv_length='auto',
            split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
            drop_prob=0.5)
    elif kind == 'deep':
        model = Deep4Net(
            n_channels, n_classes, input_time_length=window_len_samples,
            final_conv_length='auto', n_filters_time=25, n_filters_spat=25,
            filter_time_length=10, pool_time_length=3, pool_time_stride=3,
            n_filters_2=50, filter_length_2=10, n_filters_3=100,
            filter_length_3=10, n_filters_4=200, filter_length_4=10,
            first_pool_mode="max", later_pool_mode="max", drop_prob=0.5,
            double_time_convs=False, split_first_layer=True, batch_norm=True,
            batch_norm_alpha=0.1, stride_before_pool=False)
    elif kind == 'resnet':
        model = EEGResNet(
            n_channels, n_classes, input_time_length=window_len_samples,
            final_pool_length='auto', n_first_filters=24,
            n_layers_per_block=2, first_filter_length=3,
            split_first_layer=True, batch_norm_alpha=0.1,
            batch_norm_epsilon=1e-4)
    else:
        raise ValueError

    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters())
    loss = nn.NLLLoss()

    return model, loss, optimizer


def run_training(model, dataloader, loss, optimizer, n_epochs=1, cuda=False):
    """Run training loop.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    dataloader : torch.utils.data.Dataloader
        Data loader which will serve examples to the model during training.
    loss :
        Loss function.
    optimizer :
        Optimizer.
    n_epochs : int
        Number of epochs to train the model for.
    cuda : bool
        If True, move X and y to CUDA device.

    Returns
    -------
    model : torch.nn.Module
        Trained model.
    """
    for i in range(n_epochs):
        loss_vals = list()
        for X, y in dataloader:
            model.train()
            model.zero_grad()

            y = y.long()
            if cuda:
                X, y = X.cuda(), y.cuda()

            loss_val = loss(model(X), y)
            loss_vals.append(loss_val.item())

            loss_val.backward()
            optimizer.step()

        print(f'Epoch {i + 1} - mean training loss: {np.mean(loss_vals)}')

    return model


def run_training_time(model, dataloader, loss, optimizer, n_epochs=1, cuda=False):
    """Run training loop and time the loops.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    dataloader : torch.utils.data.Dataloader
        Data loader which will serve examples to the model during training. The
        underlying Dataset is expected to return (X, y, worker_id, start, end).
    loss :
        Loss function.
    optimizer :
        Optimizer.
    n_epochs : int
        Number of epochs to train the model for.
    cuda : bool
        If True, move X and y to CUDA device.

    Returns
    -------
    model : torch.nn.Module
        Trained model.
    times : list
        List of dictionaries containing the timing data of the workers and of
        the training itself.
    """
    times = list()
    for i in range(n_epochs):
        loss_vals = list()
        for X, y, worker_id, worker_start, worker_end in dataloader:
            start = time.time()

            model.train()
            model.zero_grad()

            y = y.long()
            if cuda:
                X, y = X.cuda(), y.cuda()

            loss_val = loss(model(X), y)
            loss_vals.append(loss_val.item())

            loss_val.backward()
            optimizer.step()

            end = time.time()

            assert len(set(worker_id.numpy())) == 1
            times.append({
                'worker_id': worker_id[0].item(),
                'start': worker_start[0].item(),
                'end': worker_end[-1].item()
            })
            times.append({
                'worker_id': -1,
                'start': start,
                'end': end
            })

        print(f'Epoch {i + 1} - mean training loss: {np.mean(loss_vals)}')

    return model, times


def run_training_skorch(model, dataset, batch_size, loss, optimizer,
                        num_workers=0, n_epochs=1, cuda=False):

    from sklearn.model_selection import train_test_split
    random_state = 42
    train_ind, test_ind = train_test_split(
        np.arange(dataset.description.shape[0]), test_size=0.2,
        random_state=random_state)
    train_ind, valid_ind = train_test_split(
        train_ind, test_size=0.3, random_state=random_state)
    split_ds = dataset.split(split_ids=[train_ind, valid_ind, test_ind])

    batch_timer = BatchTimer()
    cp = Checkpoint(dirname='exp1')

    # def target_extractor(y):
    #     return y.detach().cpu().numpy().argmax(axis=1)

    train_bal_acc = EpochScoring(
        scoring='balanced_accuracy', lower_is_better=False,
        name='train_bal_acc', on_train=True)
    valid_bal_acc = EpochScoring(
        scoring='balanced_accuracy', lower_is_better=False,
        name='valid_bal_acc', on_train=False)

    net = DebugNeuralNet(
        model,
        loss,
        # criterion__weight=train_weights,
        optimizer=optimizer,
        lr=1e-3,
        max_epochs=n_epochs,
        batch_size=batch_size,
        iterator_train__num_workers=num_workers,
        iterator_valid__num_workers=num_workers,
        train_split=predefined_split(split_ds[1]),
        device='cuda' if cuda else 'cpu',
        callbacks=[
            ('batch_timer', batch_timer),
            ('train_bal_acc', train_bal_acc),
            ('valid_bal_acc', valid_bal_acc),
            ('cp', cp)
        ]
        # callbacks=[('lr_scheduler', LRScheduler(
        #     policy=CyclicLR, base_lr=1e-4, max_lr=1e-2))]
    )
    net.fit(split_ds[0])

    # Load best model
    y_pred1 = net.predict(split_ds[2])
    net.initialize()
    net.load_params(checkpoint=cp)
    y_pred2 = net.predict(split_ds[2])

    # Evaluate test performance
    from sklearn.metrics import balanced_accuracy_score
    dataloader = DataLoader(
        split_ds[2], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_true, y_pred = net.get_ytrue_and_predict(
        dataloader, device='cuda' if cuda else 'cpu')
    test_bal_acc = balanced_accuracy_score(y_true, y_pred.argmax(axis=1))
    print(f'Test bal acc: {test_bal_acc:0.2f}')

    times = [item for sublist in net.history[:, 'batches'] for item in sublist]

    return model, times


import signal, subprocess
class GPUMonitor(object):
    """Launch a process to monitor GPU usage with nvidia-smi.

    Parameters
    ----------
    lms : int
        Interval between monitoring, in ms.
    dirname : str
        Directory where to save the output.
    fname : str
        File name.
    """
    def __init__(self, lms=500, dirname='.', fname=None):
        if fname is None:
            time_str = time.strftime('%Y-%m-%d-%H:%M:%S')
            fname = f'gpu_stats_{time_str}.csv'
        fname = os.path.join(dirname, fname)
        self.args = [
            'nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,'
            'utilization.memory,memory.total,memory.free,memory.used',
            '--format=csv,nounits', '-lms', str(lms), '-f', fname]

    def start(self):
        self.process = subprocess.Popen(self.args)

    def kill(self):
        self.process.send_signal(signal.SIGINT)


def time_workers(f):
    def get_and_time(index):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1

        start = time.time()
        X, y = f(index)
        end = time.time()

        return X, y, worker_id, start, end

    return get_and_time


class DebugNeuralNet(NeuralNet):
    """A NeuralNet with overriden methods for debugging and benchmarking.
    """
    # def get_loss(self, y_pred, y_true, *args, **kwargs):
    #     y_true, worker_id, worker_start, worker_end = y_true
    #     return super().get_loss(y_pred, y_true, *args, **kwargs)

    def predict(self, X):
        """Taken from NeuralNetClassifier class.
        """
        y_preds = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            y_preds.append(to_numpy(yp.max(-1)[-1]))
        y_pred = np.concatenate(y_preds, 0)
        return y_pred

    def get_ytrue_and_predict(self, dataloader, device='cpu'):
        self.module_.eval()
        y_preds, y_trues = list(), list()
        for X, y in dataloader:
            y_pred = self.module_(X[0].to(device))
            y_preds.append(y_pred.detach().cpu().numpy())
            y_trues.append(y[0].detach().cpu().numpy())  # Remove [0] if dataset returns X, y

        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)

        return y_trues, y_preds


class BatchTimer(Callback):
    """Callback to time batches.

    If the `y` returned by the dataloader is a list, this class expects it to
    contain (target, worker_id, worker_start_time, worker_end_time) and will
    log it as such.

    The following batch history parameters are created: batch_start_time,
    batch_end_time, worker_id, worker_start_time and worker_end_time.

    Parameters
    ----------
    Callback : [type]
        [description]
    """
    def initialize(self):
        pass

    def on_batch_begin(self, net, X=None, y=None, training=None, **kwargs):
        start = time.time()
        net.history.record_batch('batch_start_time', start)

        if isinstance(y, list):
            _, worker_id, worker_start, worker_end = y
            net.history.record_batch('worker_id', worker_id[0].item())
            net.history.record_batch('worker_start_time', worker_start[0].item())
            net.history.record_batch('worker_end_time', worker_end[-1].item())

    def on_batch_end(self, net, X=None, y=None, training=None, **kwargs):
        end = time.time()
        net.history.record_batch('batch_end_time', end)


if __name__ == '__main__':

    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    ############################################################################
    # Next, we define the different hyperparameters that we want to compare:

    PRELOAD = [False]  # True -> eager loading; False -> lazy loading
    N_SUBJECTS = [50]  # Number of recordings to load from the TUH Abnormal corpus
    WINDOW_LEN_S = [15]  # Window length, in seconds
    N_EPOCHS = [5]  # Number of epochs to train the model for
    BATCH_SIZE = [512]  # Training minibatch size
    MODEL = ['deep']

    NUM_WORKERS = [16]  # number of processes used by pytorch's Dataloader
    PIN_MEMORY = [False]  # whether to use pinned memory
    CUDA = [True]  # whether to use a CUDA device

    N_REPETITIONS = 1  #3 # Number of times to repeat the experiment (to get better time estimates)

    ############################################################################
    # The following path needs to be changed to your local folder containing the
    # TUH Abnormal corpus:
    TUH_PATH = '/storage/store/data/tuh_eeg/www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/'

    ############################################################################
    # We can finally cycle through all the different combinations of the parameters
    # we set above to evaluate their execution time:

    all_results = list()
    all_times = list()
    for (i, preload, n_subjects, win_len_s, n_epochs, batch_size, model_kind,
            num_workers, pin_memory, cuda) in product(
                range(N_REPETITIONS), PRELOAD, N_SUBJECTS, WINDOW_LEN_S, N_EPOCHS,
                BATCH_SIZE, MODEL, NUM_WORKERS, PIN_MEMORY, CUDA):

        results = {
            'repetition': i,
            'preload': preload,
            'n_subjects': n_subjects,
            'win_len_s': win_len_s,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'model_kind': model_kind,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'cuda': cuda
        }
        print(f'\nRepetition {i + 1}/{N_REPETITIONS}:\n{results}')

        # Load the dataset
        data_loading_start = time.time()
        dataset = load_example_data(
            preload, win_len_s, n_subjects=n_subjects, n_jobs=N_JOBS)
        data_loading_end = time.time()

        # Create the data loader
        training_setup_start = time.time()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
            num_workers=num_workers, worker_init_fn=None)

        # Instantiate model and optimizer
        n_channels = len(dataset.datasets[0].windows.ch_names)
        n_times = len(dataset.datasets[0].windows.times)
        n_classes = 2
        model, loss, optimizer = create_example_model(
            n_channels, n_classes, n_times, kind=model_kind, cuda=cuda)
        training_setup_end = time.time()

        # Start training loop
        model_training_start = time.time()
        trained_model, times = run_training_skorch(
            model, dataset, batch_size, nn.NLLLoss, optim.Adam,
            num_workers=num_workers, n_epochs=n_epochs, cuda=cuda)
        # trained_model, times = run_training_time(
        #     model, dataloader, loss, optimizer, n_epochs=n_epochs, cuda=cuda)
        all_times.extend(times)
        model_training_end = time.time()

        del dataset, model, loss, optimizer, trained_model

        # Record timing results
        results['data_preparation'] = data_loading_end - data_loading_start
        results['training_setup'] = training_setup_end - training_setup_start
        results ['model_training'] = model_training_end - model_training_start
        all_results.append(results)

    ###############################################################################
    # The results are formatted into a pandas DataFrame and saved locally as a CSV
    # file.

    results_df = pd.DataFrame(all_results)
    fname = 'lazy_vs_eager_loading_results_skorch.csv'
    results_df.to_csv(fname)
    print(f'Results saved under {fname}.')

    times_df = pd.DataFrame(all_times)
    fname = 'lazy_vs_eager_loading_worker_times_skorch.csv'
    times_df.to_csv(fname)
    print(f'Worker times saved under {fname}.')

    ###############################################################################
    # We can finally summarize this information into the following plot:

    print(results_df)

    sns.catplot(
        data=results_df, row='cuda', x='model_kind', y='model_training',
        hue='num_workers', col='preload', kind='strip')

    ###############################################################################
    # .. warning::
    #   The results of this comparison will change depending on the hyperparameters
    #   that were set above, and on the actual hardware that is being used.

    ###############################################################################
    # Generally speaking, we expect lazy loading to be slower than eager loading
    # during model training, but to potentially be pretty competitive if multiple
    # workers were enabled (i.e.., `num_workers > 0`). Training on a CUDA device
    # should also yield substantial speedups.


    gpu_monitor.kill()
