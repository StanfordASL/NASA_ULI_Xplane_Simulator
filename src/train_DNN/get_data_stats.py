import pandas

DATA_DIR='/HD2Data/csandeep/NASA_ULI/updated/large_images/afternoon/afternoon_train/'

labels_file = DATA_DIR + '/labels.csv'

df = pandas.read_csv(labels_file)

unnorm_columns = ['distance_to_centerline_meters', 'heading_error_degrees', 'downtrack_position_meters']

normalized_columns = ['distance_to_centerline_NORMALIZED', 'heading_error_NORMALIZED', 'downtrack_position_NORMALIZED']

norm_df = df[normalized_columns]

print('norm_df: ', norm_df.describe())

unnorm_df = df[unnorm_columns]

print('unnorm_df: ', unnorm_df.describe())


