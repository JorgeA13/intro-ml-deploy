from dvc import api
import pandas as pd
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Fetching data...')

finantials_data_path = api.get_url('dataset/finantials.csv', remote='dataset-track')
finantials_data = pd.read_csv(finantials_data_path)

movie_data_path = api.get_url('dataset/movies.csv', remote='dataset-track')
movie_data = pd.read_csv(movie_data_path)

gross_data_path = api.get_url('dataset/opening_gross.csv', remote='dataset-track')
gross_data = pd.read_csv(gross_data_path)

numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
movie_data = movie_data[numeric_columns + ['movie_title']]

finantials_data = finantials_data[['movie_title', 'production_budget', 'worldwide_gross']]

fin_movie = pd.merge(finantials_data, movie_data, on='movie_title', how='left')
full_data = pd.merge(gross_data, fin_movie, on='movie_title', how='left')

full_data = full_data.drop(['movie_title', 'gross'], axis=1)
full_data_df = full_data.to_csv('dataset/full_data.csv', index=False)

logger.info('Data fetched and prepared')





