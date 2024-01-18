import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def life_sq_predict(data, degree=4):
    df = data.copy()

    df_missing = df[df['life_sq'].isnull()]
    df_no_missing = df.dropna(subset=['life_sq'])

    X_train = df_no_missing[['full_sq']]
    y_train = df_no_missing['life_sq']

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    X_missing_poly = poly.transform(df_missing[['full_sq']])
    predicted_life_sq = model.predict(X_missing_poly)

    return predicted_life_sq

def build_year_predict(data, degree=3):
    df = data.copy()

    df_missing = df[df['build_year'].isnull()]
    df_no_missing = df.dropna(subset=['build_year'])

    X = df_no_missing[['price', 'year']]
    y = df_no_missing['build_year']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_missing = df_missing[['price', 'year']]
    X_missing_poly = poly.transform(X_missing)
    predicted_build_year = model.predict(X_missing_poly)

    return predicted_build_year

def num_room_predict(data, degree=2):
    df = data.copy()

    df_missing = df[df['num_room'].isnull()]
    df_no_missing = df.dropna(subset=['num_room'])

    X = df_no_missing[['price', 'full_sq', 'life_sq']]
    y = df_no_missing['num_room']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_missing = df_missing[['price', 'full_sq', 'life_sq']]
    X_missing_poly = poly.transform(X_missing)
    predicted_num_room = model.predict(X_missing_poly)

    return predicted_num_room

def full_sq_predict(data, degree=1):
    df = data.copy()

    df_missing = df[df['full_sq'].isnull()]
    df_no_missing = df.dropna(subset=['full_sq'])

    X = df_no_missing[['price', 'life_sq']]
    y = df_no_missing['full_sq']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_missing = df_missing[['price', 'life_sq']]
    X_missing_poly = poly.transform(X_missing)
    predicted_full_sq = model.predict(X_missing_poly)

    return predicted_full_sq

def apartment_condition_predict(data, degree=1):
    df = data.copy()

    df_missing = df[df['apartment_condition'].isnull()]
    df_no_missing = df.dropna(subset=['apartment_condition'])

    X = df_no_missing[['price']]
    y = df_no_missing['apartment_condition']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_missing = df_missing[['price']]
    X_missing_poly = poly.transform(X_missing)
    predicted_apartment_condition = model.predict(X_missing_poly)

    return predicted_apartment_condition

def preprocess(df,add_df):
    
    df.loc[:, 'year'] = pd.to_datetime(df['timestamp']).dt.year.astype(int)
    
    df.loc[df.apartment_condition == 33.0, 'apartment_condition'] = 3.0
    df.loc[df['build_year'] > 3000, 'build_year'] = 2007
    
    df = df[df['full_sq'] > 0]
    df = df.merge(add_df, how= 'left', on='id')
    
    df.loc[df.life_sq.isnull(),'life_sq'] = life_sq_predict(df)
    
    df.loc[df['full_sq'] < 10, 'full_sq'] = pd.NA
    df.loc[df.full_sq.isnull(),'full_sq'] = full_sq_predict(df)
    
    df.loc[df['build_year'] < 1800, 'build_year'] = pd.NA
    df.loc[df.build_year.isnull(),'build_year'] = build_year_predict(df)
    
    if df['num_room'].isna().any():
        df.loc[df.num_room.isnull(),'num_room'] = num_room_predict(df)
        
    df.loc[df.apartment_condition.isnull(),'apartment_condition'] = apartment_condition_predict(df)
    
    df.kitch_sq.fillna(df.kitch_sq.median(), inplace=True)
    df.floor.fillna(df.floor.median(), inplace=True)
    df.material.fillna(df.material.median(), inplace=True)
    
    df.num_room = df.num_room.astype(int)
    df.build_year = df.build_year.astype(int)
    df.apartment_condition = df.apartment_condition.astype(int)
    
    #df = df.fillna(0)
    
    return df


def feature_gen(df):
    df.timestamp = pd.to_datetime(df.timestamp)
    
    df['year'] = df.timestamp.dt.year.astype(int)
    df['month'] = df.timestamp.dt.month
    df['week_of_year'] = df.timestamp.dt.isocalendar().week.astype(int)
    df['day_of_week'] = df.timestamp.dt.weekday
    
    df["ratio_life_full_sq"] = df.life_sq / df.full_sq
    df["ratio_kitchen_full_sq"] = df.kitch_sq / df.full_sq
    df['diff_full_life_sq'] = df["full_sq"] - df["life_sq"]
    
    df['age'] = df.build_year - df.year
    
    df['is_top_floor'] = (df.floor == df.max_floor).astype(int)
    df['is_ground_floor'] = (df.floor == 1).astype(int)
    
    df['infrastructure'] = df.preschool_facilities + df.school_facilities + df.healthcare_facilities \
        + df.university_num + df.sport_objects_facilities + df.additional_education_facilities \
        + df.culture_objects_facilities + df.shopping_centers_facilities + df.office_num \
        + df.cafe_count + df.church_facilities + df.mosque + df.leisure_facilities
    
    df.drop(columns=['timestamp'],inplace=True)
    return df














def preprocess_test(df, add_df):
    df.loc[:, 'year'] = pd.to_datetime(df['timestamp']).dt.year.astype(int)
    
    df = df[df['full_sq'] > 0]
    df = df.merge(add_df, how= 'left', on='id')
    
    df.loc[df.life_sq.isnull(),'life_sq'] = life_sq_predict(df)
    
    df.material.fillna(df.material.median(), inplace=True)
    df.max_floor.fillna(df.max_floor.median(), inplace=True)
    
    df.loc[df['build_year'] < 1800, 'build_year'] = pd.NA
    df.loc[df.build_year.isnull(),'build_year'] = build_year_predict_test(df)
    
    if df['num_room'].isna().any():
        df.loc[df.num_room.isnull(),'num_room'] = num_room_predict_test(df)
        
    df.kitch_sq.fillna(df.kitch_sq.median(), inplace=True)
    df.floor.fillna(df.floor.median(), inplace=True)
    df.apartment_condition.fillna(df.apartment_condition.median(), inplace=True)
    
    df.num_room = df.num_room.astype(int)
    df.build_year = df.build_year.astype(int)
    df.apartment_condition = df.apartment_condition.astype(int)
    
    #df = df.fillna(0)
    
    return df


def build_year_predict_test(data, degree=3):
    df = data.copy()

    df_missing = df[df['build_year'].isnull()]
    df_no_missing = df.dropna(subset=['build_year'])

    X = df_no_missing[['year']]
    y = df_no_missing['build_year']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_missing = df_missing[['year']]
    X_missing_poly = poly.transform(X_missing)
    predicted_build_year = model.predict(X_missing_poly)

    return predicted_build_year


def num_room_predict_test(data, degree=2):
    df = data.copy()

    df_missing = df[df['num_room'].isnull()]
    df_no_missing = df.dropna(subset=['num_room'])

    X = df_no_missing[[ 'full_sq', 'life_sq']]
    y = df_no_missing['num_room']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_missing = df_missing[[ 'full_sq', 'life_sq']]
    X_missing_poly = poly.transform(X_missing)
    predicted_num_room = model.predict(X_missing_poly)

    return predicted_num_room
