import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer

our_features = ['budget', 'vote_count', 'runtime', 'Year', 'Friday', 'Monday', 'Saturday',
                'Sunday', 'Thursday', 'Tuesday', 'Wednesday', '20th Century Fox',
                'Columbia Pictures', 'Metro', 'New Line Cinema', 'Paramount',
                'Universal Pictures', 'Warner Bros', 'Action', 'Adventure', 'Animation',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller',
                'War', 'en', 'holiday_month', 'based on novel or book', 'murder', 'woman director']


def load_data(_data):
    pd.options.mode.chained_assignment = None
    _data.drop(columns=["id", "homepage", "spoken_languages", "tagline"], inplace=True)
    _data['release_date'] = _data['release_date'].astype('datetime64[ns]')
    _data['Month'] = _data.release_date.dt.month
    _data['Year'] = _data.release_date.dt.year
    _data['Weekday'] = _data.release_date.dt.day_name()

    return _data


def filter_json(st):
    if len(str(st)) > 10:
        new_val = re.findall(r"(?<=name': ')[\w ]+", st)
        if new_val:
            return new_val
        else:
            return ""
    return ""


def filter_json_directors(st):
    if len(str(st)) > 10:
        new_val = re.findall(r"(?<=name': ')[\w ]+(?=', 'department': 'Directing', 'job': 'Director')", st)
        if new_val:
            return new_val
        else:
            return ""
    return ""


def filter_json_producer(st):
    if len(str(st)) > 10:
        new_val = re.findall(
            r"(?<=name': ')[\w ]+(?=', 'department': 'Production', 'job': 'Producer')", st)
        new_val2 = re.findall(
            r"(?<=name': ')[\w ]+(?=', 'department': 'Production', 'job': 'Executive)", st)
        lst = new_val + new_val2
        if new_val:
            return lst
        else:
            return ""
    return ""


def reg_fixing(_data):
    _data["genres"] = _data["genres"].apply(filter_json)
    _data["production_companies"] = _data["production_companies"].apply(filter_json)
    _data["keywords"] = _data["keywords"].apply(filter_json)
    _data["production_countries"] = _data["production_countries"].apply(filter_json)
    _data["cast"] = _data["cast"].apply(filter_json)
    _data["directors"] = _data["crew"].apply(filter_json_directors)
    _data["producers"] = _data["crew"].apply(filter_json_producer)
    return _data


def turn_to_dummies(_data):
    mlb = MultiLabelBinarizer()
    genres = pd.DataFrame(mlb.fit_transform(_data["genres"]), columns=mlb.classes_, index=_data.index)
    production_companies = pd.DataFrame(mlb.fit_transform(_data["production_companies"]), columns=mlb.classes_,
                                        index=_data.index)
    keywords = pd.DataFrame(mlb.fit_transform(_data["keywords"]), columns=mlb.classes_, index=_data.index)
    lang = pd.get_dummies(_data["original_language"])
    month = pd.get_dummies(_data["Month"])
    holiday_month = pd.DataFrame()
    holiday_month["holiday_month"] = np.zeros(len(_data))
    for i in range(1, 13):
        if i in (5, 6, 7, 11, 12) and str(i) in month.columns:
            holiday_month["holiday_month"] += month[i]
    Weekday = pd.get_dummies(_data["Weekday"])
    ready_data = pd.concat([Weekday, production_companies, genres, lang, holiday_month, keywords], axis=1)
    return ready_data


def run_process(_data):
    _data = load_data(_data)
    _data = reg_fixing(_data)
    dummy = turn_to_dummies(_data)
    clean_data = _data[["budget", "vote_count", "runtime", "Year"]]
    ready_data = pd.concat([clean_data, dummy], axis=1)
    ready_data = ready_data.dropna()
    missing_cols = sorted(set(our_features) - set(ready_data.columns))
    empty_df = pd.DataFrame(np.zeros((len(ready_data), len(missing_cols))), columns=missing_cols)
    ready_data = pd.concat([ready_data, empty_df], axis=1)
    ready_data = ready_data.reindex(sorted(ready_data.columns), axis=1)
    return ready_data
