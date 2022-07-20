from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as np
import pandas as pd

from model_utils import update_model, save_metrics_report, get_model_performance

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Loading data for de model')
data = pd.read_csv('dataset/full_data.csv')


logger.info('Loading model')
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),
    ('core_model', GradientBoostingRegressor())
])


logger.info('Separating data in train and test df')
x = data.drop(['worldwide_gross'], axis=1)
y = data['worldwide_gross']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)


logger.info('Setting hyperparameters for regressor')
param_tuning = {
    'core_model__n_estimators': range(20, 301, 20)
}
grid_search = GridSearchCV(model, param_grid=param_tuning, scoring='r2', cv=5)


logger.info('Start the grid search for the best model')
grid_search.fit(x_train, y_train)


logger.info('Making cross validation for the result')
final_result = cross_validate(grid_search.best_estimator_, x_train, y_train, cv=5, return_train_score=True)
train_score = np.mean(final_result['train_score'])
validate_score = np.mean(final_result['test_score'])

assert train_score >= 0.80
assert validate_score >= 0.65

logger.info(f'Train Score: {train_score}')
logger.info(f'Validate Score: {validate_score}')


logger.info('Updating the model')
update_model(grid_search.best_estimator_)


logger.info('Generating model report')
test_score = grid_search.best_estimator_.score(x_test, y_test)
save_metrics_report(train_score=train_score.__float__(), test_score=test_score,
                    validate_score=validate_score.__float__(), model=grid_search.best_estimator_)

y_test_pred = grid_search.best_estimator_.predict(x_test)
get_model_performance(y_test=y_test, y_test_pred=y_test_pred)

logger.info('Training Finished')
