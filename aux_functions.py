import pandas as pd
import numpy as np

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
