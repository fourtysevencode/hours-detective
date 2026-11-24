# hours-detective
Explored the dataset and engineered features to train a model via scikit-learn's HistGradientBoostingRegressor ensemble model to predict the hours spent on a hackclub project shipped.

## Try it out

Clone the repo and move to the project root by:

```bash
git clone https://github.com/fourtysevencode/hours-detective.git && cd hours-detective
```

Place a test.csv file containing features listed in [feature_list](feature_list.txt) and project instances in each row referred to by it's `project_id`

then in a terminal in the project root, run:

```bash
python predict.py
```

The results will be saved to `predictions.csv` in the project root.

## Credits

Shipped to [Haggle](https://haggle.hackclub.com) 2026

_with 💗 from fourtysevencode_
