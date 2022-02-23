"""Module containing the TransformNode, which is used to perform various transforms on the result data.

These transform functions are relatively short and because of this are contained
within this single TransformNode. 
"""
import random
import sys
import logging
import pathlib
import pyvad
import transforms.perturbations as perturbations
import numpy as np
import librosa
import librosa.display
import pandas as pd
from multiprocessing import Lock
from nodes.loadsignalnode import load_audio_from_path
from .node import AQPNode
from pipeline import LOGGER_NAME
from qualitymetrics.visqol.nsim import nsim_map

LOGGER = logging.getLogger(LOGGER_NAME)

def df_columns_to_tuples(result: dict, target_key: str, output_key: str,
                         col_one: str, col_two: str, **kwargs):
    """Take two columns from a Pandas dataframe stored in the result dict and create a single list of tuples of the two columns.

    Parameters
    ----------
    result : dict
        The results dictionary.
    target_key : str
        Dict key containing the dataframe to operate on.
    output_key : str
        The key to assign the resulting list to.
    col_one : str
        The name of the first column to use.
    col_two : str
        The name of the second column to use.

    Returns
    -------
    None.

    """
    if not output_key:
        LOGGER.error("Function requires output_key to operate")
        sys.exit(-2)
    if not target_key:
        LOGGER.error("Function requires target_key to operate")
        sys.exit(-2)
    df = result[target_key]
    result[output_key] = list(zip(df[col_one], df[col_two]))
    
    
def column_to_iterable_entry(result, target_key, output_key, col_name, **kwargs):
    df = result[target_key]
    col = df[col_name]
    result[output_key] = col

def tuple_to_top_level(result: dict, target_key: str, 
                       reference_file_key: str='reference',
                       degraded_file_key: str='degraded', **kwargs):
    """
    Convert a two-item tuple to two top-level dict fields.

    Parameters
    ----------
    result : dict
        The results dictionary.
    target_key : str
        Dict key containing the tuple.
    reference_file_key : str, optional
        Key to assign the first tuple value to. The default is 'reference'.
    degraded_file_key : str, optional
        Key to assign the second tuple value to. The default is 'degraded'.

    Returns
    -------
    None.

    """
    file_names = result[target_key]
    result[reference_file_key] = file_names[0]
    result[degraded_file_key] = file_names[1]




def update_df(result: dict, target_key: str, 
                                 key: str, col_name: str='Ref_Wave', deg_col='Test_Wave',
                                 file_name_key: str='reference_file', test_file_name_key='degraded_file', **kwargs):
    """Update the dataframe being used based on the col_name and ref_file_name_key arguments, with the value stored at the key.

    Parameters
    ----------
    result : dict
        The results dictionary.
    target_key : str
        Key of the dataframe.
    key : str
        '.' separted template string used to represent the nested chain of keys
        required to retrieve the desired value. e.g. vnsims_mel.mos
        NOTE: This functionality will be changed soon, when the full port is done.
    col_name : str, optional
        The name of the column to search for the value at file_name_key.
        The default is 'ref_wav'.
    file_name_key : str, optional
        The key used to retrieve the file name that will be used to find the correct
        index in the dataframe. The default is 'reference_file'.

    Returns
    -------
    None.

    """
    df = result[target_key]
    index = df.index[((df[col_name] == result[file_name_key]) & (df[deg_col] == result[test_file_name_key]))]
    df.at[index, key] = result[key]


def extract_nsim_values(result, target_key, loop_data, value_key, output_col, **kwargs):
    df= result[target_key]
    loop_data = result[loop_data]
    # df[output_col] = ""
    values = []
    for key in loop_data:
        values.append(loop_data[key][value_key])
        # index = df.index[((df['Ref_Wave'] == key))]
    df[output_col] = values
        # df.iloc[index, output_col] = list(values)


def to_csv(result: dict, target_key: str, output_file_name: str, **kwargs):
    data = result.get(target_key, None)
    if data is None:
        LOGGER.error("Can't find data to write to csv.")
        return
    path_to_output_file = output_file_name[:output_file_name.rindex('/') + 1]
    pathlib.Path(path_to_output_file).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file_name)


def apply_function_to_col(result: dict, target_key: str, output_col: str = None, **kwargs):
    lock = Lock()
    df = result[target_key]
    modifier_function = SIGNAL_MODIFIERS[kwargs['function_name']]
    output_column = output_col if output_col else kwargs['signal_col']
    try:
        print(f'Adding column {output_column}')
        df[output_column] = df[kwargs['signal_col']].apply(modifier_function, **kwargs)
        lock.acquire()
        result['data'] = df
        lock.release()
    except (BaseException) as ex:
        print(ex)
    
    
def scale_by_constant(signal, constant, **kwargs): 
    return signal * constant

def apply_function_to_signal(result, target_key, **kwargs):
    signal = result[target_key]
    function = SIGNAL_MODIFIERS[kwargs['function_name']]
    signal = function(signal, **kwargs)
    result[target_key] = signal

def perform_vad(signal, sample_rate, **kwargs): 
    try:
        activity = pyvad.split(signal, sample_rate, fs_vad=sample_rate, hop_length=30, vad_mode=0)
        x=  np.concatenate([signal[edge[0]:edge[1]] for edge in activity], axis=0)
        return x
    except (BaseException) as ex:
        print(ex)
    
def create_spectrogram(signal, sample_rate, **kwargs): 
    spect = librosa.feature.melspectrogram(signal, sample_rate)
    # img = librosa.display.specshow(spect, x_axis='time', y_axis='mel', sr=sample_rate)
    return spect


def apply_func(result, target_key, function_name, **kwargs):
    func = SIGNAL_MODIFIERS[function_name]
    data = result[target_key]
    func(data, **kwargs)


def collect_columns_to_matrix(result, target_column_names, row_wise: bool = False, **kwargs):
    df = result['data']
    column_series = df[df.columns.intersection(target_column_names)]
    print(column_series)
    # print(f'Output matrix shape: {output_matrix.shape}')
    return None


SIGNAL_MODIFIERS = {
        'scale_by_constant': scale_by_constant,
        'add_constant': (lambda signal, constant, **kwargs: signal + constant),
        'load_signal': load_audio_from_path,
        'VAD': perform_vad,
        '_add_noise': perturbations._add_noise,
        'create_spectrogram': create_spectrogram
        }

    
# Used to assign the correct functions during deserialization from JSON.
FUNCTIONS = {
    'df_columns_to_tuples':  df_columns_to_tuples,
    'tuple_to_top_level': tuple_to_top_level,
    'update_df': update_df,
    'to_csv':  to_csv,
    "apply_function": apply_function_to_col,
    "apply_perturbation": perturbations.apply_perturbation,
    "apply_function_to_signal": apply_function_to_signal,
    "apply_func": apply_func,
    'collect_columns_to_matrix': collect_columns_to_matrix,
    'col_to_iterable_entry': column_to_iterable_entry,
    'extract_nsim_values': extract_nsim_values
}


class TransformNode(AQPNode):
    """Node which encapsulates some transform logic."""
    
    def __init__(self, id_: str, transform_name: str, function_args: dict,
                 target_key: str, output_key: str=None, 
                 draw_options: dict=None, **kwargs):
        super().__init__(id_, output_key=output_key, draw_options=draw_options)
        self.function = FUNCTIONS[transform_name]
        self.function_args = function_args
        self.target_key = target_key
        self.type_ = 'TransformNode'

    def execute(self, result: dict, **kwargs):
        """Execute the transform function stored by this node."""
        super().execute(result, **kwargs)
        args = {**self.__dict__(), **self.function_args,  **kwargs,}
        self.function(result, **args)
        return result

    def __dict__(self):
        """Create a dict representation of the node.

        Returns
        -------
        dict
            Dict representation of the node.

        """
        return {
                "output_key": self.output_key,
                "target_key": self.target_key
            }