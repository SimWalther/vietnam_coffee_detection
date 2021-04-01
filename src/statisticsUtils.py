import pandas
import numpy as np
from sklearn import preprocessing
from seaborn import FacetGrid, boxplot
from rasterUtils import labels_values_from_raster_files


def draw_bands_boxplots(df, hue, title):
    g = FacetGrid(df, col="bands", col_wrap=4)
    g.fig.suptitle(title)
    g.map_dataframe(boxplot, x="bands", y="value", hue=hue)
    g.add_legend()
    g.set_titles(col_template="{col_name}")
    g.set(xticklabels=[])


def normalize_dataframe(dataframe, columns):
    scaler = preprocessing.MinMaxScaler()
    scaled_values = scaler.fit_transform(dataframe[columns])
    dataframe[columns] = scaled_values
    return dataframe


def dataframe_from_labels_values(labels_values, columns, group_names):
    # Create dataframes from those values
    df_list = [
        pandas.DataFrame(label_values, columns=columns).assign(group_name=group_names[i])
        for i, label_values
        in enumerate(labels_values)
    ]

    # Concatenate all those dataframe into one
    concat_df = pandas.concat(df_list)

    # Normalize data by columns
    return normalize_dataframe(concat_df, columns)


def statistics(raster_paths, labels, group_names, labels_coordinates_list, title='', describe_stats=False, asc_std_by_bands=False, draw_boxplot=True, nb_pixel_around=4):
    # Get values for each label we are interested in
    labels_values = labels_values_from_raster_files(labels, raster_paths, labels_coordinates_list, nb_pixel_around)

    columns = ['coastal aerosol', 'blue', 'green', 'red', 'nir', 'swir 1', 'swir 2', 'panchromatic']
    df = dataframe_from_labels_values(labels_values, columns, group_names)

    if draw_boxplot:
        melted_df = pandas.melt(df, id_vars=['group_name'], var_name=['bands'])
        draw_bands_boxplots(melted_df, 'group_name', title)

    if describe_stats:
        # Print statistical data pertaining to each time period
        for group_name in group_names:
            print('\n', group_name, '\n----------------\n', df.loc[df['group_name'] == group_name].describe())

    # TODO: extract this code in a function
    if asc_std_by_bands:
        # For each band, list standard deviations of each time period
        bands_std = {
            band: [
                df.loc[df['group_name'] == group_name][band].std() for group_name in group_names
            ] for band in columns
        }

        # For each band list time period by ascending standard deviations
        for band in bands_std:
            std_asc = list(map(lambda arg: (group_names[arg], bands_std[band][arg]), np.argsort(np.array(bands_std[band]))))

            print('\n', band, ':\n---------------')

            for std in std_asc:
                print(std[0], ": ", std[1])