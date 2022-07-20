import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from joblib import dump
import seaborn as sns


def update_model(model: Pipeline) -> None:
    dump(model, 'model/model.pkl')


def save_metrics_report(train_score: float, validate_score: float,
                        test_score: float, model: Pipeline) -> None:
    with open('report.txt', 'w') as file:
        file.write(f'# Pipeline model description:\n')

        for key, value in model.named_steps.items():
            file.write(f'### {key} : {value.__repr__()}\n')

        file.write(f'### Train score: {train_score}\n')
        file.write(f'### Validation score: {validate_score}\n')
        file.write(f'### Test score: {test_score}\n')


def get_model_performance(y_test: pd.Series, y_test_pred: pd.Series) -> None:
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_test, y=y_test_pred, ax=ax)
    ax.set_xlabel('Real worldwide gross')
    ax.set_ylabel('Predicted worldwide gross')
    ax.set_title('Model performance')
    fig.savefig('model_behavior.png')