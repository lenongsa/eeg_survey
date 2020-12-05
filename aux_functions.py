import pandas as pd
import numpy as np
from collections import Counter
import scipy.stats

def timeseries_to_pandas(dict_timeseries):
    """
    Read the UCR timeseries and convert them to pandas dataframes
    """

    case_id = []
    dim_id = []
    reading_id = []
    value = []
    class_id = []

    for dim in dict_timeseries.keys():
        for case in range(0, len(dict_timeseries[dim])):
            num_readings = len(dict_timeseries[dim][case]) - 1
            classification = dict_timeseries[dim][case][-1]

            case_id.extend([case + 1] * num_readings)
            dim_id.extend([dim] * num_readings)
            reading_id.extend(range(1, num_readings + 1))
            value.extend(list(dict_timeseries[dim][case])[0:-1])
            class_id.extend([classification] * num_readings)

    df = pd.DataFrame({'case_id': case_id,
                       'dim_id': dim_id,
                       'reading_id': reading_id,
                       'value': value,
                       'class_id': class_id})

    return df


def pandas_to_numpy(df):
    """
    Convert the pandas dataframes to numpy arrays of the form (n_samples, n_features, n_timestamps)
    """

    df_as_list = []

    for case in df.case_id.unique():
        df_case = df.loc[df.case_id == case]
        case_as_list = []

        for dim in df_case.dim_id.unique():
            values = list(df_case.loc[df_case.dim_id == dim, 'value'])
            case_as_list.append(values)

        df_as_list.append(case_as_list)

    response_as_list = list(df[['case_id', 'class_id']].drop_duplicates().class_id)

    return np.array(df_as_list), np.array(response_as_list)


def ResampleLinear1D(original, targetLen=40):
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original) - 1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int)  # Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor  # Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0 - index_rem) + val2 * index_rem
    assert (len(interp) == targetLen)
    return interp


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics
