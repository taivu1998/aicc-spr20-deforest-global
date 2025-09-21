import fire
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from util import *

REGIONS = ['na', 'la', 'eu', 'af', 'as', 'sea', 'oc']
LABELS = ['commodity driven deforestation', 'shifting agriculture', 'forestry', 'wildfire', 'urbanization']

def haversine(lat1, lon1, lat2, lon2):
    lat1 = lat1*np.pi/180.0
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    d = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))

def nearest_neighbor(lat1, lon1, lats, lons, preds=[]):
    dists = []
    for i, (lat2, lon2) in enumerate(zip(lats, lons)):
        if (lat1 == lat2 and lon1 == lon2) or (preds and preds[i] == -1):
            dists.append(np.nan)
        else:
            dists.append(haversine(lat1, lon1, lat2, lon2))
    return np.nanargmin(dists)

def get_predict_fn(train_data):
    def predict(row):
        probs = [row[f"ProbLabel{i + 1}"] for i in range(len(LABELS))]
        if np.max(probs) > 0.5:
            return np.argmax(probs) + 1
        else:
            nn_index = nearest_neighbor(row["lat"], row["lon"], train_data["lat"], train_data["lon"])
            return train_data["y"][nn_index]
    return predict

def predict_without_nn(row):
    probs = [row[f"ProbLabel{i + 1}"] for i in range(len(LABELS))]
    if np.max(probs) > 0.5:
        return np.argmax(probs) + 1
    else:
        return -1

def nn_on_test(preds, test_lat, test_lon):
    for i in range(len(preds)):
        if preds[i] == -1:
            nn_index = nearest_neighbor(test_lat[i], test_lon[i], test_lat, test_lon, preds.tolist())
            preds[i] = preds[nn_index]
    return preds

def add_logits(df, exp_name, split):
    logits = pd.read_csv(SANDBOX_DIR / exp_name / f"logits_{split}.csv")
    df = df.merge(logits, on="GoodeR_ID").reset_index(drop=True)
    return df

def add_pre_logits(df, exp_name, split):
    pre_logits = pd.read_csv(SANDBOX_DIR / exp_name / f"pre_logits_{split}.csv")
    df = df.merge(pre_logits, on="GoodeR_ID").reset_index(drop=True)
    return df

def load_data(split2path, region, logits, pre_logits, exp_name, linear, remove_aux, test_on_val, dl_model):
    ignore_cols = [
        LABEL_HEADER, "GoodeR_ID", LONGITUDE_HEADER,
        LATITUDE_HEADER, YEAR_HEADER, "loss_area",
        'Unnamed: 0', 'Unnamed: 0.1', 'closest_year',
        'composite', 'composite_is_landsat8', 'furthest_year',
        'image_paths', 'least_cloudy', 'loss_area_old',
        'num_imgs_downloaded', 'num_scenes', 'single_sc_annual_composites'
    ]
    keep_cols = ['label', 'GoodeR_ID', 'longitude', 'latitude', 'year', 'loss_area', 'Region']
    y_col = LABEL_HEADER
    area_col = "loss_area"

    region_df = pd.read_csv(
        DATA_BASE_DIR /
        "curtis" /
        "GoodeR_Boundaries_Region.csv"
    )
    region_df.columns = ['GoodeR_ID', 'Region']

    loss_df = pd.read_csv(
        DATA_BASE_DIR /
        "curtis" /
        "LossMaskFull.csv"
    )
    loss_df.columns = ['GoodeR_ID', 'loss_area']

    split2data = {}
    X_cols = []
    if dl_model:
        for split, path in split2path.items():
            logits = pd.read_csv(SANDBOX_DIR / exp_name / f"logits_{split}.csv")
            df = pd.read_csv(path)
            df = df.merge(logits, on="GoodeR_ID")
            for i, row in df.iterrows():
                row_logits = [row["0"], row["1"], row["2"], row["3"], row["4"]]
                probs = torch.nn.functional.softmax(torch.tensor(row_logits), dim=-1).tolist()
                df.loc[i, ["0", "1", "2", "3", "4"]] = [probs[0], probs[1], probs[2], probs[3], probs[4]]
            y = df[y_col].astype(int).values
            area = df[area_col].values
            lat = df[LATITUDE_HEADER].values
            lon = df[LONGITUDE_HEADER].values
            ids = df["GoodeR_ID"].values
            split2data[split] = {
                **{label: df[str(i)].tolist() for i, label in enumerate(LABELS)},
                **{"y": y, "area": area, "lat": lat, "lon": lon, "gooder_id": ids}
            }
    else:
        for split, path in split2path.items():
            # Read CSV from file
            df = pd.read_csv(path)

            # Drop the uncertain and unknown classes
            df = df[~df[LABEL_HEADER].isin([6, 7])].reset_index(drop=True)

            # Merge loss area from provided data
            df = df.drop(columns="loss_area").merge(loss_df, on="GoodeR_ID")

            # TODO: Drop examples with <0.5% tree cover loss

            if region != "global":
                df = df.merge(
                    region_df,
                    on="GoodeR_ID")
                df = df[df["Region"].astype(int) == REGIONS.index(
                    region) + 1].reset_index(drop=True)

            if remove_aux:
                df = df.drop(columns=[col for col in df.columns if col not in keep_cols])

            if logits:
                df = add_logits(df, exp_name, split)

            if pre_logits:
                df = add_pre_logits(df, exp_name, split)

            # Ignore columns that shouldn't be used when training
            X_cols = [col for col in df.columns
                      if col not in ignore_cols]

            # Keep X as dataframe in case we want to subset columns later
            X = df[X_cols]
            y = df[y_col].astype(int).values
            area = df[area_col].values
            lat = df[LATITUDE_HEADER].values
            lon = df[LONGITUDE_HEADER].values
            ids = df["GoodeR_ID"].values
            split2data[split] = {
                "X": X, "y": y, "area": area, "lat": lat, "lon": lon, "gooder_id": ids
            }

    # Combine train and valid splits
    if not test_on_val and not dl_model:
        split2data["train"]["X"] = np.concatenate(
            [split2data["train"]["X"], split2data["val"]["X"]
            ])
        split2data["train"]["y"] = np.concatenate(
            [split2data["train"]["y"], split2data["val"]["y"]
            ])
        split2data["train"]["area"] = np.concatenate(
            [split2data["train"]["area"], split2data["val"]["area"]
            ])
        split2data["train"]["lat"] = np.concatenate(
            [split2data["train"]["lat"], split2data["val"]["lat"]
            ])
        split2data["train"]["lon"] = np.concatenate(
            [split2data["train"]["lon"], split2data["val"]["lon"]
            ])
        del split2data["val"]

    if linear:
        scaler = StandardScaler()
        scaler.fit(split2data["train"]["X"])
        split2data["train"]["X"] = scaler.transform(split2data["train"]["X"])
        if test_on_val:
            split2data["val"]["X"] = scaler.transform(split2data["val"]["X"])
        else:
            split2data["test"]["X"] = scaler.transform(split2data["test"]["X"])

    return split2data, X_cols

def evaluate(predictions):
    y = predictions["TrueLabel"].values
    y_pred = predictions["FinalPred"].values
    area = predictions["area"].values
    for sample_weight, sample_weight_str in [(None, False), (area, True)]:
        report = classification_report(
            y, y_pred,
            sample_weight=sample_weight,
            digits=4
        )
        print(f"test scores weighted by loss area [{sample_weight_str}]")
        print(report)
        print(f"-" * 40)

def tune_models(split2data, nn_test, linear, cols, test_on_val, no_reg, dl_model):
    if linear:
        if no_reg:
            model2params = {
                LogisticRegression: {
                    'max_iter': [1000],
                    'class_weight': [None, "balanced"],
                    'penalty': ['none'],
                    'C': [0.1, 1.0, 100, 1000]
                }
            }
        else:
            model2params = {
                LogisticRegression: {
                    'max_iter': [1000],
                    'class_weight': [None, "balanced"],
                    'penalty': ['none', 'l2'],
                    'C': [0.1, 1.0, 100, 1000]
                }
            }
    else:
        model2params = {
            RandomForestClassifier: {
                'n_estimators': np.linspace(start=10, stop=100, num=4).astype(int),
                'max_depth': [None] + list(np.linspace(10, 110, num=6).astype(int)),
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, "balanced"],
                'random_state': [0]
            }
        }

    if test_on_val:
        test_set = "val"
    else:
        test_set = "test"

    predictions = pd.DataFrame()
    predictions['Index'] = np.arange(split2data[test_set]["y"].shape[0])
    predictions['TrueLabel'] = split2data[test_set]["y"]
    predictions['lat'] = split2data[test_set]["lat"]
    predictions['lon'] = split2data[test_set]["lon"]
    predictions['area'] = split2data[test_set]["area"]
    predictions['GoodeR_ID'] = split2data[test_set]["gooder_id"]

    if dl_model:
        for i, label in enumerate(LABELS):
            predictions[f"ProbLabel{LABELS.index(label) + 1}"] = split2data[test_set][label]
    else:
        for label in LABELS:
            train_X = split2data["train"]["X"]
            test_X = split2data[test_set]["X"]

            # Convert labels to a binary classification
            train_y = (split2data["train"]["y"] == (LABELS.index(label) + 1)).astype(int)
            test_y = (split2data[test_set]["y"] == (LABELS.index(label) + 1)).astype(int)

            for ModelClass, params in tqdm(model2params.items()):
                print(f"Training model {ModelClass.__name__}... for class {label}")
                model = ModelClass()

                # Perform 10-fold CV
                cv = GridSearchCV(
                    model,
                    param_grid=params,
                    cv=5,
                    scoring="f1_macro",
                    n_jobs=4,
                    verbose=1
                )
                if ModelClass == RandomForestClassifier:
                    cv.fit(train_X, train_y.ravel())
                else:
                    cv.fit(train_X, train_y)

                # Evaluate the best model on the train and test sets
                best_model = cv.best_estimator_
                predictions[f"ProbLabel{LABELS.index(label) + 1}"] = best_model.predict_proba(test_X)[:, 1]

    if linear:
        print("Feature coefficients")
        feature_importances = pd.Series(data=best_model.coef_[0], index=cols)
        feature_importances.sort_values(ascending=False,inplace=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(feature_importances)
    elif not dl_model:
        print("Feature importances")
        feature_importances = pd.Series(data=best_model.feature_importances_, index=cols)
        feature_importances.sort_values(ascending=False,inplace=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(feature_importances)
    
    # Set predictions to label of highest probability prediction if > 0.5
    # and label of nearest neighbor in predictions otherwise
    if nn_test:
        predictions["FinalPred"] = predictions.apply(predict_without_nn, axis=1)
        predictions["FinalPred"] = nn_on_test(predictions["FinalPred"], split2data[test_set]["lat"], split2data[test_set]["lon"])
    # Set predictions to label of highest probability prediction if > 0.5
    # and label of nearest neighbor in training set otherwise    
    else:
        predict = get_predict_fn(split2data["train"])
        predictions["FinalPred"] = predictions.apply(predict, axis=1)

    evaluate(predictions)

    return predictions

def train_baseline(global_model=True, nn_test=False, logits=False, pre_logits=False, exp_name=None, class_metrics=False, linear=False, remove_aux=False, test_on_val=False, output_pred=False, no_reg=False, dl_model=False):
    split2path = {
        "train": HANSEN_DIR / "curtis_v5" / "postdownload_meta_train_v5.csv",
        "val": HANSEN_DIR / "curtis_v5" / "postdownload_meta_val_v5.csv",
        "test": HANSEN_DIR / "curtis_v5" / "postdownload_meta_test_v5.csv"
    }

    if global_model:
        print("Training global model...")
        split2data, cols = load_data(
            split2path=split2path,
            region="global",
            logits=logits,
            pre_logits=pre_logits,
            exp_name=exp_name,
            linear=linear,
            remove_aux=remove_aux,
            test_on_val=test_on_val,
            dl_model=dl_model
        )
        predictions = tune_models(
            split2data=split2data,
            nn_test=nn_test,
            linear=linear,
            cols=cols,
            test_on_val=test_on_val,
            no_reg=no_reg,
            dl_model=dl_model
        )
    else:
        region_predictions = {}
        areas = {}
        for region in REGIONS:
            print(f"Training {region} model...")
            split2data, cols = load_data(
                split2path=split2path,
                region=region,
                logits=logits,
                pre_logits=pre_logits,
                exp_name=exp_name,
                linear=linear,
                remove_aux=remove_aux,
                test_on_val=test_on_val,
                dl_model=dl_model
            )
            predictions = tune_models(
                split2data=split2data,
                nn_test=nn_test,
                linear=linear,
                cols=cols,
                test_on_val=test_on_val,
                no_reg=no_reg,
                dl_model=dl_model
            )
            region_predictions[region] = predictions
            areas[region] = predictions['area'].sum()
        # Evaluate aggregated per-region models
        predictions = pd.concat(region_predictions)
        print("Aggregated metrics")
        evaluate(predictions)
        # Compute % area per region
        total = sum(areas.values())
        for region in areas:
            areas[region] /= total
        print("% area per region")
        print(areas)
    if output_pred:
        predictions.to_csv(DATA_BASE_DIR / f"curtis/predictions/{global_model}{logits}{pre_logits}{linear}{remove_aux}{test_on_val}{exp_name}_predictions.csv", index=False)
    if class_metrics:
        for label in range(5):
            class_predictions = predictions[(predictions['TrueLabel'] == (label + 1))]
            print("Metrics for class " + LABELS[label])
            evaluate(class_predictions)
