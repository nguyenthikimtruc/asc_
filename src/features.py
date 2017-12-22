#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import librosa
import scipy


def feature_extraction(y, fs=44100, statistics=True, mel_params=None):
    """Feature extraction, log mel spectrogram based features

    Outputs features in dict, format:

        {
            'feat': feature_matrix [shape=(frame count, feature vector size)],
            'stat': {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }
        }

    Parameters
    ----------
    y: numpy.array [shape=(signal_length, )]
        Audio

    fs: int > 0 [scalar]
        Sample rate
        (Default value=44100)

    statistics: bool
        Calculate feature statistics for extracted matrix
        (Default value=True)

    mel_params: dict or None
        Parameters for extraction of static  mel bands.


    Returns
    -------
    result: dict
        Feature dict

    """

    eps = numpy.spacing(1)

    # Windowing function
#    if mfcc_params['window'] == 'hamming_asymmetric':
#        window = scipy.signal.hamming(mfcc_params['n_fft'], sym=False)
#    elif mfcc_params['window'] == 'hamming_symmetric':
#        window = scipy.signal.hamming(mfcc_params['n_fft'], sym=True)
#    elif mfcc_params['window'] == 'hann_asymmetric':
#        window = scipy.signal.hann(mfcc_params['n_fft'], sym=False)
#    elif mfcc_params['window'] == 'hann_symmetric':
#        window = scipy.signal.hann(mfcc_params['n_fft'], sym=True)
#    else:
#        window = None

    # Calculate Static Coefficients
    power_spectrogram = numpy.abs(librosa.stft(y + eps,
                                               n_fft=mel_params['n_fft'],
                                               win_length=mel_params['win_length'],
                                               hop_length=mel_params['hop_length'],
                                               center=True))**2
    mel_basis = librosa.filters.mel(sr=fs,
                                    n_fft=mel_params['n_fft'],
                                    n_mels=mel_params['n_mels'],
                                    fmin=mel_params['fmin'],
                                    fmax=mel_params['fmax'])
                                    
    mel_spectrum = numpy.dot(mel_basis, power_spectrogram)
    log_mel_spectrogram = librosa.logamplitude(mel_spectrum)
                                
    # Collect the feature matrix
    feature_matrix = log_mel_spectrogram.T

    # Collect into data structure
    if statistics:
        return {
            'feat': feature_matrix,
            'stat': {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }
        }
    else:
        return {
            'feat': feature_matrix}


class FeatureNormalizer(object):
    """Feature normalizer class

    Accumulates feature statistics

    Examples
    --------

    >>> normalizer = FeatureNormalizer()
    >>> for feature_matrix in training_items:
    >>>     normalizer.accumulate(feature_matrix)
    >>>
    >>> normalizer.finalize()

    >>> for feature_matrix in test_items:
    >>>     feature_matrix_normalized = normalizer.normalize(feature_matrix)
    >>>     # used the features

    """
    def __init__(self, feature_matrix=None):
        """__init__ method.

        Parameters
        ----------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)] or None
            Feature matrix to be used in the initialization

        """
        if feature_matrix is None:
            self.N = 0
            self.mean = 0
            self.S1 = 0
            self.S2 = 0
            self.std = 0
        else:
            self.mean = numpy.mean(feature_matrix, axis=0)
            self.std = numpy.std(feature_matrix, axis=0)
            self.N = feature_matrix.shape[0]
            self.S1 = numpy.sum(feature_matrix, axis=0)
            self.S2 = numpy.sum(feature_matrix ** 2, axis=0)
            self.finalize()

    def __enter__(self):
        # Initialize Normalization class and return it
        self.N = 0
        self.mean = 0
        self.S1 = 0
        self.S2 = 0
        self.std = 0
        return self

    def __exit__(self, type, value, traceback):
        # Finalize accumulated calculation
        self.finalize()

    def accumulate(self, stat):
        """Accumalate statistics

        Input is statistics dict, format:

            {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }

        Parameters
        ----------
        stat : dict
            Statistics dict

        Returns
        -------
        nothing

        """
        self.N += stat['N']
        self.mean += stat['mean']
        self.S1 += stat['S1']
        self.S2 += stat['S2']

    def finalize(self):
        """Finalize statistics calculation

        Accumulated values are used to get mean and std for the seen feature data.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        # Finalize statistics
        self.mean = self.S1 / self.N
        self.std = numpy.sqrt((self.N * self.S2 - (self.S1 * self.S1)) / (self.N * (self.N - 1)))

        # In case we have very brain-death material we get std = Nan => 0.0
        self.std = numpy.nan_to_num(self.std)

        self.mean = numpy.reshape(self.mean, [1, -1])
        self.std = numpy.reshape(self.std, [1, -1])

    def normalize(self, feature_matrix):
        """Normalize feature matrix with internal statistics of the class

        Parameters
        ----------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Feature matrix to be normalized

        Returns
        -------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Normalized feature matrix

        """

        return (feature_matrix - self.mean) / self.std
