#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:04:25 2018

@author: truc
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2016::Acoustic Scene Classification / Baseline System
import pywt
from src.ui import *
from src.general import *
from src.files import *

from src.features import *
from src.dataset import *
from src.evaluation import *

import numpy
import csv
import argparse
import textwrap
import copy
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils
from keras.regularizers import l2

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

from sklearn.cluster import KMeans
#
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
bands =128
frames = 501
num_channels = 3
num_labels = 15

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


# activations = ['relu', 'tanh', 'linear']
# batch_sizes = [10, 20, 30, 40, 50, 60]
# epochs = [10, 20, 30]
# dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

optimisers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2016
            Task 1: Acoustic Scene Classification
            Baseline system
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description
                This is an baseline implementation for D-CASE 2016 challenge acoustic scene classification task.
                Features: MFCC (static+delta+acceleration)
                Classifier: GMM

        '''))

    # Setup argument handling
    parser.add_argument("-development", help="Use the system in the development mode", action='store_true',
                        default=True, dest='development')
    parser.add_argument("-challenge", help="Use the system in the challenge mode", action='store_true',
                        default=False, dest='challenge')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    args = parser.parse_args()

    # Load parameters from config file
    parameter_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  os.path.splitext(os.path.basename(__file__))[0]+'.yaml')
    params = load_parameters(parameter_file)
    params = process_parameters(params)
    make_folders(params)

    title("DCASE 2016::Acoustic Scene Classification / Baseline System")

    # Check if mode is defined
    if not (args.development or args.challenge):
        args.development = True
        args.challenge = False

    dataset_evaluation_mode = 'folds'
    if args.development and not args.challenge:
        print ("Running system in development mode")
        dataset_evaluation_mode = 'folds'
    elif not args.development and args.challenge:
        print ("Running system in challenge mode")
        dataset_evaluation_mode = 'full'

    # Get dataset container class
    dataset = eval(params['general']['development_dataset'])(data_path=params['path']['data'])

    # Fetch data over internet and setup the data
    # ==================================================
    if params['flow']['initialize']:
        dataset.fetch()

    # Extract features for all audio files in the dataset
    # ==================================================
            
    if params['flow']['extract_features']:
        section_header('Feature extraction')

        # Collect files in train sets and test sets
        files = []
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            for item_id, item in enumerate(dataset.train(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
            for item_id, item in enumerate(dataset.test(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
            for item_id, item in enumerate(dataset.crosseval(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
        files = sorted(files)

        # Go through files and make sure all features are extracted
        do_feature_extraction(files=files,
                              dataset=dataset,
                              feature_path=params['path']['features'],
                              params=params['features'],
                              overwrite=params['general']['overwrite'])

        foot()

    # Prepare feature normalizers
    # ==================================================
    if params['flow']['feature_normalizer']:
        section_header('Feature normalizer')

        do_feature_normalization(dataset=dataset,
                                 feature_normalizer_path=params['path']['feature_normalizers'],
                                 feature_path=params['path']['features'],
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 overwrite=params['general']['overwrite'])

        foot()

    # System training
    # ==================================================
    if params['flow']['train_system']:
        section_header('System training')

        do_system_training(dataset=dataset,                           
                           model_path=params['path']['models'],
                           feature_normalizer_path=params['path']['feature_normalizers'],
                           feature_path=params['path']['features'],
                           feature_params=params['features'],
                           classifier_params=params['classifier']['parameters'],
                           classifier_method=params['classifier']['method'],
                           dataset_evaluation_mode=dataset_evaluation_mode,
                           clean_audio_errors=params['classifier']['audio_error_handling']['clean_data'],
                           overwrite=params['general']['overwrite']
                           )

        foot()

    # System evaluation in development mode
    if args.development and not args.challenge:

        # System testing
        # ==================================================
        if params['flow']['test_system']:
            section_header('System testing')

            do_system_testing(dataset=dataset,                              
                              feature_path=params['path']['features'],
                              result_path=params['path']['results'],
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              classifier_method=params['classifier']['method'],
                              clean_audio_errors=params['recognizer']['audio_error_handling']['clean_data'],
                              overwrite=params['general']['overwrite']
                              )
            
            foot()

        # System evaluation
        # ==================================================
        if params['flow']['evaluate_system']:
            section_header('System evaluation')

            do_system_evaluation(dataset=dataset,
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 result_path=params['path']['results'])

            foot()

    # System evaluation with challenge data
    elif not args.development and args.challenge:
        # Fetch data over internet and setup the data
        challenge_dataset = eval(params['general']['challenge_dataset'])(data_path=params['path']['data'])
        if params['general']['challenge_submission_mode']:
            result_path = params['path']['challenge_results']
        else:
            result_path = params['path']['results']

        if params['flow']['initialize']:
            challenge_dataset.fetch()

        if not params['general']['challenge_submission_mode']:
            section_header('Feature extraction for challenge data')

            # Extract feature if not running in challenge submission mode.
            # Collect test files
            files = []
            for fold in challenge_dataset.folds(mode=dataset_evaluation_mode):
                for item_id, item in enumerate(dataset.test(fold)):
                    if item['file'] not in files:
                        files.append(item['file'])
            files = sorted(files)

            # Go through files and make sure all features are extracted
            do_feature_extraction(files=files,
                                  dataset=challenge_dataset,
                                  feature_path=params['path']['features'],
                                  params=params['features'],
                                  overwrite=params['general']['overwrite'])
            foot()

        # System testing
        if params['flow']['test_system']:
            section_header('System testing with challenge data')

            do_system_testing(dataset=challenge_dataset,
                              feature_path=params['path']['features'],
                              result_path=result_path,
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              classifier_method=params['classifier']['method'],
                              clean_audio_errors=params['recognizer']['audio_error_handling']['clean_data'],
                              overwrite=params['general']['overwrite'] or params['general']['challenge_submission_mode']
                              )
            foot()

            if params['general']['challenge_submission_mode']:
                print (" ")
                print ("Your results for the challenge data are stored at ["+params['path']['challenge_results']+"]")
                print (" ")

        # System evaluation if not in challenge submission mode
        if params['flow']['evaluate_system'] or params['general']['challenge_submission_mode']:
            section_header('System evaluation with challenge data')
            do_system_evaluation(dataset=challenge_dataset,
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 result_path=result_path)

            foot()

    return 0


def process_parameters(params):
    """Parameter post-processing.

    Parameters
    ----------
    params : dict
        parameters in dict

    Returns
    -------
    params : dict
        processed parameters

    """

    # Convert feature extraction window and hop sizes seconds to samples
    params['features']['mel']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mel']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])

    # Copy parameters for current classifier method
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]

    # Hash
    params['features']['hash'] = get_parameter_hash(params['features'])

    # Let's keep hashes backwards compatible after added parameters.
    # Only if error handling is used, they are included in the hash.
    classifier_params = copy.copy(params['classifier'])
    if not classifier_params['audio_error_handling']['clean_data']:
        del classifier_params['audio_error_handling']
    params['classifier']['hash'] = get_parameter_hash(classifier_params)

    params['recognizer']['hash'] = get_parameter_hash(params['recognizer'])

    # Paths
    params['path']['data'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['data'])
    params['path']['base'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['base'])

    # Features
    params['path']['features_'] = params['path']['features']
    params['path']['features'] = os.path.join(params['path']['base'],
                                              params['path']['features'],
                                              params['features']['hash'])

    # Feature normalizers
    params['path']['feature_normalizers_'] = params['path']['feature_normalizers']
    params['path']['feature_normalizers'] = os.path.join(params['path']['base'],
                                                         params['path']['feature_normalizers'],
                                                         params['features']['hash'])

    # Models
    params['path']['models_'] = params['path']['models']
    params['path']['models'] = os.path.join(params['path']['base'],
                                            params['path']['models'],
                                            params['features']['hash'],
                                            params['classifier']['hash'])
    # Results
    params['path']['results_'] = params['path']['results']
    params['path']['results'] = os.path.join(params['path']['base'],
                                             params['path']['results'],
                                             params['features']['hash'],
                                             params['classifier']['hash'],
                                             params['recognizer']['hash'])

    return params


def make_folders(params, parameter_filename='parameters.yaml'):
    """Create all needed folders, and saves parameters in yaml-file for easier manual browsing of data.

    Parameters
    ----------
    params : dict
        parameters in dict

    parameter_filename : str
        filename to save parameters used to generate the folder name

    Returns
    -------
    nothing

    """

    # Check that target path exists, create if not
    check_path(params['path']['features'])
    check_path(params['path']['feature_normalizers'])
    check_path(params['path']['models'])
    check_path(params['path']['results'])

    # Save parameters into folders to help manual browsing of files.

    # Features
    feature_parameter_filename = os.path.join(params['path']['features'], parameter_filename)
    if not os.path.isfile(feature_parameter_filename):
        save_parameters(feature_parameter_filename, params['features'])

    # Feature normalizers
    feature_normalizer_parameter_filename = os.path.join(params['path']['feature_normalizers'], parameter_filename)
    if not os.path.isfile(feature_normalizer_parameter_filename):
        save_parameters(feature_normalizer_parameter_filename, params['features'])

    # Models
    model_features_parameter_filename = os.path.join(params['path']['base'],
                                                     params['path']['models_'],
                                                     params['features']['hash'],
                                                     parameter_filename)
    if not os.path.isfile(model_features_parameter_filename):
        save_parameters(model_features_parameter_filename, params['features'])

    model_models_parameter_filename = os.path.join(params['path']['base'],
                                                   params['path']['models_'],
                                                   params['features']['hash'],
                                                   params['classifier']['hash'],
                                                   parameter_filename)
    if not os.path.isfile(model_models_parameter_filename):
        save_parameters(model_models_parameter_filename, params['classifier'])

    # Results
    # Save parameters into folders to help manual browsing of files.
    result_features_parameter_filename = os.path.join(params['path']['base'],
                                                      params['path']['results_'],
                                                      params['features']['hash'],
                                                      parameter_filename)
    if not os.path.isfile(result_features_parameter_filename):
        save_parameters(result_features_parameter_filename, params['features'])

    result_models_parameter_filename = os.path.join(params['path']['base'],
                                                    params['path']['results_'],
                                                    params['features']['hash'],
                                                    params['classifier']['hash'],
                                                    parameter_filename)
    if not os.path.isfile(result_models_parameter_filename):
        save_parameters(result_models_parameter_filename, params['classifier'])

    result_models_parameter_filename = os.path.join(params['path']['base'],
                                                    params['path']['results_'],
                                                    params['features']['hash'],
                                                    params['classifier']['hash'],
                                                    params['recognizer']['hash'],
                                                    parameter_filename)
    if not os.path.isfile(result_models_parameter_filename):
        save_parameters(result_models_parameter_filename, params['recognizer'])

def get_feature_filename(audio_file, path, extension='cpickle'):
    """Get feature filename

    Parameters
    ----------
    audio_file : str
        audio file name from which the features are extracted

    path :  str
        feature path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    feature_filename : str
        full feature filename

    """

    audio_filename = os.path.split(audio_file)[1]
    return os.path.join(path, os.path.splitext(audio_filename)[0] + '.' + extension)


def get_feature_normalizer_filename(fold, path, extension='cpickle'):
    """Get normalizer filename

    Parameters
    ----------
    fold : int >= 0
        evaluation fold number

    path :  str
        normalizer path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    normalizer_filename : str
        full normalizer filename

    """

    return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)


def get_model_container_filename(fold, path, extension='cpickle'):
    """Get model filename

    Parameters
    ----------
    fold : int >= 0
        evaluation fold number

    path :  str
        model path

    extension : str
        file extension
        (Default value='cpickle')get_keras_model_filename

    Returns
    -------
    model_filename : str
        full model filename

    """

    return os.path.join(path, 'model_fold' + str(fold) + '.' + extension)

def get_keras_model_filename(fold, path, extension='hdf5'):
    """Get model filename

    Parameters
    ----------
    fold : int >= 0
        evaluation fold number

    path :  str
        model path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    model_filename : str
        full model filename

    """

    return os.path.join(path, 'model_fold' + str(fold) + '.model.' + extension)


def get_result_filename(fold, path, extension='txt'):
    """Get result filename

    Parameters
    ----------
    fold : int >= 0
        evaluation fold number

    path :  str
        result path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    result_filename : str
        full result filename

    """

    if fold == 0:
        return os.path.join(path, 'results.' + extension)
    else:
        return os.path.join(path, 'results_fold' + str(fold) + '.' + extension)


def do_feature_extraction(files, dataset, feature_path, params, overwrite=False):
    """Feature extraction

    Parameters
    ----------
    files : list
        file list

    dataset : class
        dataset class

    feature_path : str
        path where the features are saved

    params : dict
        parameter dict

    overwrite : bool
        overwrite existing feature files
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    IOError
        Audio file not found.

    """

    # Check that target path exists, create if not
    check_path(feature_path)

    for file_id, audio_filename in enumerate(files):
        # Get feature filename
        current_feature_file = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)

        progress(title_text='Extracting',
                 percentage=(float(file_id) / len(files)),
                 note=os.path.split(audio_filename)[1])

        if not os.path.isfile(current_feature_file) or overwrite:
            # Load audio data
            if os.path.isfile(dataset.relative_to_absolute_path(audio_filename)):
                y, fs = load_audio(filename=dataset.relative_to_absolute_path(audio_filename), mono=True, fs=params['fs'])
            else:
                raise IOError("Audio file not found [%s]" % audio_filename)

            # Extract features
            feature_data = feature_extraction(y=y,
                                              fs=fs,
                                              mel_params=params['mel'])
            # Save
            save_data(current_feature_file, feature_data)


def do_feature_normalization(dataset, feature_normalizer_path, feature_path, dataset_evaluation_mode='folds', overwrite=False):
    """Feature normalization

    Calculated normalization factors for each evaluation fold based on the training material available.

    Parameters
    ----------
    dataset : class
        dataset class

    feature_normalizer_path : str
        path where the feature normalizers are saved.

    feature_path : str
        path where the features are saved.

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    overwrite : bool
        overwrite existing normalizers
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    IOErrorx
        Feature file not found.

    """

    # Check that target path exists, create if not
    check_path(feature_normalizer_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_normalizer_file = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)

        if not os.path.isfile(current_normalizer_file) or overwrite:
            # Initialize statistics
            file_count = len(dataset.train(fold))
            normalizer = FeatureNormalizer()

            for item_id, item in enumerate(dataset.train(fold)):
                progress(title_text='Collecting data',
                         fold=fold,
                         percentage=(float(item_id) / file_count),
                         note=os.path.split(item['file'])[1])
                # Load features
                if os.path.isfile(get_feature_filename(audio_file=item['file'], path=feature_path)):
                    feature_data = load_data(get_feature_filename(audio_file=item['file'], path=feature_path))['stat']
                else:
                    raise IOError("Feature file not found [%s]" % (item['file']))

                # Accumulate statistics
                normalizer.accumulate(feature_data)
            
            # Calculate normalization factors
            normalizer.finalize()

            # Save
            save_data(current_normalizer_file, normalizer)


def do_system_training(dataset, model_path, feature_normalizer_path, feature_path, feature_params, classifier_params,
                       dataset_evaluation_mode='folds', classifier_method='cnn', clean_audio_errors=False, overwrite=False):
    """System training

    model container format:

    {
        'normalizer': normalizer class
        'models' :
            {
                'office' : mixture.GMM class
                'home' : mixture.GMM class
                ...
            }
    }

    Parameters
    ----------
    dataset : class
        dataset class

    model_path : str
        path where the models are saved.

    feature_normalizer_path : str
        path where the feature normalizers are saved.

    feature_path : str
        path where the features are saved.

    feature_params : dict
        parameter dict

    classifier_params : dict
        parameter dict

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    classifier_method : str ['gmm']
        classifier method, currently only GMM supported
        (Default value='gmm')

    clean_audio_errors : bool
        Remove audio errors from the training data
        (Default value=False)

    overwrite : bool
        overwrite existing models
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    ValueError
        classifier_method is unknown.

    IOError
        Feature normalizer not found.
        Feature file not found.

    """

    if classifier_method != 'cnn':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(model_path)
    train_x = None
    train_y = None
    CrossEval_x = None
    CrossEval_y = None
    n_out = 15

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_model_file = get_keras_model_filename(fold=fold, path=model_path)
        current_model_container_file = get_model_container_filename(fold=fold, path=model_path)
        if not os.path.isfile(current_model_file) or overwrite:
            # Load normalizer
            feature_normalizer_filename = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)
            if os.path.isfile(feature_normalizer_filename):
                normalizer = load_data(feature_normalizer_filename)
            else:
                raise IOError("Feature normalizer not found [%s]" % feature_normalizer_filename)

            # Initialize model container
            model_container = {'normalizer': normalizer, 'models': {}, 'labels': None}

            # Collect training examples
            file_count = len(dataset.train(fold))
            data = {}

            for item_id, item in enumerate(dataset.train(fold)):
                progress(title_text='Collecting train data',
                         fold=fold,
                         percentage=(float(item_id) / file_count),
                         note=os.path.split(item['file'])[1])

                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                    raise IOError("Features not found [%s]" % (item['file']))

                # Scale features
                feature_data = model_container['normalizer'].normalize(feature_data)
                n_frames = feature_data.shape[0]
                n_feats = feature_data.shape[1]

                # Audio error removal
                if clean_audio_errors:
                    current_errors = dataset.file_error_meta(item['file'])
                    if current_errors:
                        removal_mask = numpy.ones((feature_data.shape[0]), dtype=bool)
                        for error_event in current_errors:
                            onset_frame = int(numpy.floor(error_event['event_onset'] / feature_params['hop_length_seconds']))
                            offset_frame = int(numpy.ceil(error_event['event_offset'] / feature_params['hop_length_seconds']))
                            if offset_frame > feature_data.shape[0]:
                                offset_frame = feature_data.shape[0]
                            removal_mask[onset_frame:offset_frame] = False
                        feature_data = feature_data[removal_mask, :]

                # Store features per class label
                feature_data = feature_data.T.flatten()[:, numpy.newaxis].T
                if item['scene_label'] not in data:
                    data[item['scene_label']] = feature_data
                else:
                    data[item['scene_label']] = numpy.vstack((data[item['scene_label']], feature_data))
                
#            data[item['scene_label']] = numpy.asarray(data[item['scene_label']]).reshape(len(data[item['scene_label']]), n_feats,n_frames)
         
            # Generate train_x and train_y with label for each frame
            for id_key, key in enumerate(data.keys()):
                train_x = data[key] if id_key==0 else numpy.vstack((train_x, data[key]))
                train_y = onehot(data[key],id_key,len(data.keys())) if id_key ==0 else numpy.vstack((train_y, onehot(data[key],id_key, len(data.keys()))))
            
            # save order of labels
            model_container['labels'] = data.keys()
            data = []
            
            # Collect cross evaluation examples
            crosseval_file_count = len(dataset.crosseval(fold))
            crosseval_data = {}
            for item_id, item in enumerate(dataset.crosseval(fold)):
                progress(title_text='Collecting evaluation data',
                         fold=fold,
                         percentage=(float(item_id) / crosseval_file_count),
                         note=os.path.split(item['file'])[1])

                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                    raise IOError("Features not found [%s]" % (item['file']))

                # Scale features
                feature_data = model_container['normalizer'].normalize(feature_data)
                n_frames = feature_data.shape[0]
                n_feats = feature_data.shape[1]

                # Audio error removal
                if clean_audio_errors:
                    current_errors = dataset.file_error_meta(item['file'])
                    if current_errors:
                        removal_mask = numpy.ones((feature_data.shape[0]), dtype=bool)
                        for error_event in current_errors:
                            onset_frame = int(numpy.floor(error_event['event_onset'] / feature_params['hop_length_seconds']))
                            offset_frame = int(numpy.ceil(error_event['event_offset'] / feature_params['hop_length_seconds']))
                            if offset_frame > feature_data.shape[0]:
                                offset_frame = feature_data.shape[0]
                            removal_mask[onset_frame:offset_frame] = False
                        feature_data = feature_data[removal_mask, :]

                # Store features per class label
                feature_data = feature_data.T.flatten()[:, numpy.newaxis].T
                if item['scene_label'] not in crosseval_data:
                    crosseval_data[item['scene_label']] = feature_data
                else:
                    crosseval_data[item['scene_label']] = numpy.vstack((crosseval_data[item['scene_label']], feature_data))
                        
            # Generate crosseval_x and crosseval_y with label for each frame
            for id_key, key in enumerate(crosseval_data.keys()):
                crosseval_x = crosseval_data[key] if id_key==0 else numpy.vstack((crosseval_x, crosseval_data[key]))
                crosseval_y = onehot(crosseval_data[key],id_key,len(crosseval_data.keys())) if id_key ==0 else numpy.vstack((crosseval_y, onehot(crosseval_data[key],id_key, len(crosseval_data.keys()))))
            
            train_x = train_x.reshape(len(train_x), n_feats, n_frames)
            train_x1 = []  
            train_x11 = []      
#            batch_size = 45
#            centers = [[1, 1], [-1, -1], [1, -1]]
#            n_clusters = len(centers)
#            X=train_x
#            len_train_x=len(train_x)
#            k_means = KMeans(init='k-means++', n_clusters=3, n_init=50)
#            k_means.fit(X)
#            k_means_labels = k_means.labels_
#            k_means_cluster_centers = k_means.cluster_centers_
#            k_means_labels_unique = np.unique(k_means_labels)
#            for k in range(n_clusters):
#                my_members = k_means_labels == k
##                cluster_center = k_means_cluster_centers[k]
#                X[k] = X[my_members]
#                         # Prepare training data and cross evaluation data shapes for classiffier
#                train_x1 =  train_x1.vstack(train_x,X[my_members])
#            crosseval_x = 
#            # Prepare training data and cross evaluation data shapes for classiffier
#            train_x = train_x1.reshape(len_train_x,1, n_frames,3)
            #%%
            for idx in range(len(train_x)):
                train_x1 = []
                X = train_x[idx]
                LL, (LH, HL, HH) = pywt.dwt2(X, 'bior1.3')
                for i, a in enumerate([LL, LH, HL, HH]):
                    if len(train_x1)==0:
                        train_x1 =  a
                    else: train_x1 = numpy.vstack((train_x1,a))
                    
                if len(train_x11)==0:
                        train_x11 =  train_x1
                else: train_x11 = numpy.vstack((train_x11,train_x1))
            # Prepare training data and cross evaluation data shapes for classiffier
            train_x = train_x11.reshape(len(train_x),4, 66, 253)

            # crosseval
            crosseval_x = crosseval_x.reshape(len(crosseval_x), n_feats, n_frames)
            crosseval_x1 = []  
            crosseval_x11 = []   
            for idx in range(len(crosseval_x)):
                crosseval_x1 = []
                X = crosseval_x[idx]
                LL, (LH, HL, HH) = pywt.dwt2(X, 'bior1.3')
                for i, a in enumerate([LL, LH, HL, HH]):
                    if len(crosseval_x1)==0:
                        crosseval_x1 =  a
                    else: crosseval_x1 = numpy.vstack((crosseval_x1,a))
                    
                if len(crosseval_x11)==0:
                        crosseval_x11 =  crosseval_x1
                else: crosseval_x11 = numpy.vstack((crosseval_x11,crosseval_x1))
            # Prepare training data and cross evaluation data shapes for classiffier
            crosseval_x = crosseval_x11.reshape(len(crosseval_x),4, 66, 253)
            
            
            
            
            
            
            
            
             #%%
            # Prepare training data and cross evaluation data shapes for classiffier
#            train_x = train_x.reshape(len(train_x),1, n_frames, n_feats)
#            crosseval_x = crosseval_x.reshape(len(crosseval_x),1, n_frames, n_feats)  
            

            # Train models for each class
            frames = train_x.shape[2]
            bands = train_x.shape[1]
            feature_size = bands * frames 
            num_labels = train_y.shape[-1]
            n_channels = 1
            f_size = 3
            
            model = build_model()
#            model = VGG_16()

            # earlystopping ends training when the validation loss stops improving
            earlystop = EarlyStopping(monitor='val_loss',min_delta=0, patience=20, verbose=0, mode='auto')
            
#            model.fit(train_x, train_y, validation_data=(crosseval_x, crosseval_y), callbacks=[earlystop], batch_size=128, epochs=150)
            model.fit(train_x, train_y, validation_data= None, callbacks=None, batch_size=64, epochs=30)

            # Save model and parameter
            save_data(current_model_container_file, model_container)
            save_model(current_model_file,model)

      

def VGG_16():
    # filters of size 1x1 
    f_size = 3
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(bands, frames, num_channels)))
    model.add(Convolution2D(64, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, f_size, strides=f_size, kernel_initializer='normal', padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, f_size, strides=f_size, kernel_initializer='normal', padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    
    # use the Adam optimiser
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    return model  

def build_model():
    
    model = Sequential()
    # input: 60x1501 data frames with 1 channels => (60,1501,1) tensors

    # filters of size 1x1 
    f_size = 3

    # first layer has 48 convolution filters 
#    model.add(Convolution2D(32, f_size, strides=f_size, kernel_initializer='normal', padding='same', input_shape=(1, frames, num_channels)))
    model.add(Convolution2D(32, f_size, strides=f_size, kernel_initializer='normal', padding='same', input_shape=(4, 66, 253),data_format='channels_first'))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))
    model.add(Dropout(0.5))

    # next layer has 96 convolution filters
    model.add(Convolution2D(64, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))
    model.add(Dropout(0.5))

    # next layer has 96 convolution filters
    model.add(Convolution2D(128, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))
    model.add(Dropout(0.5))
    
    # next layer has 96 convolution filters
    model.add(Convolution2D(256, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_last'))
    model.add(Dropout(0.5))
    
    # flatten output into a single dimension 
    # Keras will do shape inference automatically
    model.add(Flatten())

    # then a fully connected NN layer
    model.add(Dense(1024)) #1024
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # finally, an output layer with one node per class
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # use the Adam optimiser
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    return model     

def build_model_0():
    
    model = Sequential()
    # input: 60x1501 data frames with 1 channels => (60,1501,1) tensors

    # filters of size 1x1 
    f_size = 3

    # first layer has 48 convolution filters 
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same', input_shape=(bands, frames, num_channels)))
    model.add(Convolution2D(48, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
    model.add(Dropout(0.5))

    # next layer has 96 convolution filters
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Convolution2D(96, f_size, strides=f_size, kernel_initializer='normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))
    model.add(Dropout(0.5))

    # flatten output into a single dimension 
    # Keras will do shape inference automatically
    model.add(Flatten())

    # then a fully connected NN layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # finally, an output layer with one node per class
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    # use the Adam optimiser
#    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    return model    
  
            
def onehot(x, k, K):

    y = numpy.zeros((x.shape[0], K))    
    if k >= 0: y[:, k] = 1    
    return numpy.asarray(y,'float32')

def do_system_testing(dataset, result_path, feature_path, model_path, feature_params,
                      dataset_evaluation_mode='folds', classifier_method='gmm', clean_audio_errors=False, overwrite=False):
    """System testing.

    If extracted features are not found from disk, they are extracted but not saved.

    Parameters
    ----------
    dataset : class
        dataset class

    result_path : str
        path where the results are saved.

    feature_path : str
        path where the features are saved.

    model_path : str
        path where the models are saved.

    feature_params : dict
        parameter dict

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    classifier_method : str ['gmm']
        classifier method, currently only GMM supported
        (Default value='gmm')

    clean_audio_errors : bool
        Remove audio errors from the training data
        (Default value=False)

    overwrite : bool
        overwrite existing models
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    ValueError

        classifier_method is unknown.

    IOError
        Model file not found.
        Audio file not found.

    """

    if classifier_method != 'cnn':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(result_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_result_file = get_result_filename(fold=fold, path=result_path)
        if not os.path.isfile(current_result_file) or overwrite:
            results = []
            test_x = []
            # Load class model container
            model_container_filename = get_model_container_filename(fold=fold, path=model_path)           
            if os.path.isfile(model_container_filename):
                model_container = load_data(model_container_filename)
            else:
                raise IOError("Model container file not found [%s]" % model_container_filename)
                
            model_filename = get_keras_model_filename(fold=fold, path=model_path)    
            if os.path.isfile(model_filename):
                model = load_model(model_filename)
            else:
                raise IOError("Model file not found [%s]" % model_filename)
                
            normalizer = FeatureNormalizer()
            file_count = len(dataset.test(fold))
            for file_id, item in enumerate(dataset.test(fold)):
                progress(title_text='Testing',
                         fold=fold,
                         percentage=(float(file_id) / file_count),
                         note=os.path.split(item['file'])[1])
                
                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)

                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                    # Load audio
                    if os.path.isfile(dataset.relative_to_absolute_path(item['file'])):
                        y, fs = load_audio(filename=dataset.relative_to_absolute_path(item['file']), mono=True, fs=feature_params['fs'])
                    else:
                        raise IOError("Audio file not found [%s]" % (item['file']))
                    
                    feature_data = feature_extraction(y=y,
                                              fs=fs,mel_params=feature_params['mel'],
                                              statistics=False)
                                    
                # Scale features
                # if args.development:
                feature_data = model_container['normalizer'].normalize(feature_data)
                # args.challenge:
#                feature_data = model_container['normalizer'].normalize(feature_data['feat'])
                

                n_frames = feature_data.shape[0]
                n_feats = feature_data.shape[1]

                if clean_audio_errors:
                    current_errors = dataset.file_error_meta(item['file'])
                    if current_errors:
                        removal_mask = numpy.ones((feature_data.shape[0]), dtype=bool)
                        for error_event in current_errors:
                            onset_frame = int(numpy.floor(error_event['event_onset'] / feature_params['hop_length_seconds']))
                            offset_frame = int(numpy.ceil(error_event['event_offset'] / feature_params['hop_length_seconds']))
                            if offset_frame > feature_data.shape[0]:
                                offset_frame = feature_data.shape[0]
                            removal_mask[onset_frame:offset_frame] = False
                        feature_data = feature_data[removal_mask, :]
                
                # Store features per class label
                feature_data = feature_data.T.flatten()[:, numpy.newaxis].T
                test_x = feature_data if len(test_x)==0 else numpy.vstack((test_x, feature_data))
                
            test_x = test_x.reshape(len(test_x), n_feats, n_frames)
            test_x1 = []  
            test_x11 = []   
            for idx in range(len(test_x)):
                test_x1 = []
                X = test_x[idx]
                LL, (LH, HL, HH) = pywt.dwt2(X, 'bior1.3')
                for i, a in enumerate([LL, LH, HL, HH]):
                    if len(test_x1)==0:
                        test_x1 =  a
                    else: test_x1 = numpy.vstack((test_x1,a))
                    
                if len(test_x11)==0:
                        test_x11 =  test_x1
                else: test_x11 = numpy.vstack((test_x11,test_x1))
            # Prepare training data and cross evaluation data shapes for classiffier
            test_x = test_x11.reshape(len(test_x),4, 66, 253)
                
                    
#            test_x = test_x.reshape(len(test_x), 1, n_frames,n_feats)
            # Do classification for the block
            if classifier_method == 'cnn':
#                current_result = do_classification_cnn(test_x, model_container)
                y_probs = model.predict_proba(test_x,verbose=0)
                current_results = y_probs.argmax(axis=-1)
            else:
                raise ValueError("Unknown classifier method ["+classifier_method+"]")

                # Store the result
            for file_id, item in enumerate(dataset.test(fold)):
                str_current_result = model_container['labels'][current_results[file_id]]
                results.append((dataset.absolute_to_relative(item['file']), str_current_result))
                # Save testing results
                with open(current_result_file, 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for result_item in results:
                        writer.writerow(result_item)
            print('Finish saving results')

def do_classification_cnn(test_x, model_container):
    """CNN classification for give feature matrix

    model container format:

    {
        'normalizer': normalizer class
        'models' :
            {
                'office' : mixture.GMM class
                'home' : mixture.GMM class
                ...
            }
    }

    Parameters
    ----------
    feature_data : numpy.ndarray [shape=(t, feature vector length)]
        feature matrix

    model_container : dict
        model container

    Returns
    -------
    result : str
        classification result as scene label

    """

#    # Initialize log-likelihood matrix to -inf
#    logls = numpy.empty(len(model_container['models']))
#    logls.fill(-numpy.inf)
#
#    for label_id, label in enumerate(model_container['models']):
#        logls[label_id] = numpy.sum(model_container['models'][label].score(feature_data))
#
#    classification_result_id = numpy.argmax(logls)
#    return model_container['models'].keys()[classification_result_id]

    y_prob = model_container['model'].predict_proba(test_x,verbose=0)
    y_pred = y_prob.argmax(axis=-1)
#    y_true = np.argmax(test_y, 1)
    return y_pred


def do_system_evaluation(dataset, result_path, dataset_evaluation_mode='folds'):
    """System evaluation. Testing outputs are collected and evaluated. Evaluation results are printed.

    Parameters
    ----------
    dataset : class
        dataset class

    result_path : str
        path where the results are saved.

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    Returns
    -------
    nothing

    Raises
    -------
    IOError
        Result file not found

    """

    dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
    results_fold = []
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        dcase2016_scene_metric_fold = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
        results = []
        result_filename = get_result_filename(fold=fold, path=result_path)

        if os.path.isfile(result_filename):
            with open(result_filename, 'rt') as f:
                for row in csv.reader(f, delimiter='\t'):
                    results.append(row)
        else:
            raise IOError("Result file not found [%s]" % result_filename)

        y_true = []
        y_pred = []
        for result in results:
            y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
            y_pred.append(result[1])
        dcase2016_scene_metric.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        dcase2016_scene_metric_fold.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        results_fold.append(dcase2016_scene_metric_fold.results())
    results = dcase2016_scene_metric.results()

    print ("  File-wise evaluation, over %d folds" % dataset.fold_count)
    fold_labels = ''
    separator = '     =====================+======+======+==========+  +'
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_labels += " {:8s} |".format('Fold'+str(fold))
            separator += "==========+"
    print ("     {:20s} | {:4s} : {:4s} | {:8s} |  |".format('Scene label', 'Nref', 'Nsys', 'Accuracy')+fold_labels)
    print (separator)
    for label_id, label in enumerate(sorted(results['class_wise_accuracy'])):
        fold_values = ''
        if dataset.fold_count > 1:
            for fold in dataset.folds(mode=dataset_evaluation_mode):
                fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['class_wise_accuracy'][label] * 100)
        print ("     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format(label,
                                                                     results['class_wise_data'][label]['Nref'],
                                                                     results['class_wise_data'][label]['Nsys'],
                                                                     results['class_wise_accuracy'][label] * 100)+fold_values)
    print (separator)
    fold_values = ''
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['overall_accuracy'] * 100)

    print ("     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format('Overall accuracy',
                                                                 results['Nref'],
                                                                 results['Nsys'],
                                                                 results['overall_accuracy'] * 100)+fold_values)

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
