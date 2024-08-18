import numpy as np
import re
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer  # Bu satır, IterativeImputer'ın kullanılabilmesi için gerekli
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

################################################
# 2. Data Preprocessing & Feature Engineering Functions
################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):

    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    low_limit = low_limit.astype(np.float32)
    up_limit = up_limit.astype(np.float32)

    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def optimize_dtypes(df):
    for column in df.columns:
        if df[column].dtype == 'int64':
            df[column] = df[column].astype('int32')
        elif df[column].dtype == 'float64':
            df[column] = df[column].astype('float32')
    return df

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # One-Hot Encoding işlemi
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

    # Yeni oluşturulan sütunları `int` veri tipine dönüştürme
    for col in dataframe.columns:
        if dataframe[col].dtype == 'bool':
            dataframe[col] = dataframe[col].astype(int)

    return dataframe
def extract_lab_result(lab_result, test_name):
    match = re.search(rf'{test_name}\s*([\d.]+)', lab_result)
    if match:
        value = match.group(1)
        try:
            return float(value)
        except ValueError:
            return None
    return None

test_names = ['NEU#', 'LYM#', 'MON#', 'PLT', 'CRP', 'HGB', 'HCT', 'BUN', 'KREATININ', 'KLOR', 'SODYUM', 'POTASYUM', 'KALSIYUM', 'EOS#', 'FHHB', 'SO2', 'WBC']

def data_preprocessing(df):
    df.columns = df.columns.str.strip()  # Başında ve sonunda olabilecek boşlukları kaldırır
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Geçersiz karakterleri kaldırır

    df['AGE'] = df['AGE'].astype(int)

    # Pnömoni sütunu olusturma
    df['DIAGNOSISCODE'] = df['DIAGNOSISCODE'].astype(str)
    df['PNOMONI'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if 'J18.9' in str(x) else 0)

    df['LABRESULTS'] = df['LABRESULTS'].astype(str)
    df['LABRESULTS'] = df['LABRESULTS'].str.upper()

    # LABSONUCLARI degiskeninden test sonuclarini al
    for test in test_names:
        col_name = re.sub(r'[\s\(\)#\\]', '', test)
        df[col_name] = df['LABRESULTS'].apply(lambda x: extract_lab_result(x, test))

    # TANILAR
    df['ASTIM'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if any(code in str(x)
                                                          for code in ['J45', 'J46']) else 0)
    df['KOAH'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if 'J44' in str(x) else 0)
    df['DM_2'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if 'E11' in str(x) else 0)
    df['KANSER'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if any(code in str(x)
                                                           for code in ['C', 'D49']) else 0)
    df['BOBREK_HASTASI'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if 'N18' in str(x) else 0)
    karaciger_kodlari = ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76', 'K77', 'C22', 'B15', 'B16', 'B17', 'B18',
                         'B19']
    df['KARACIGER'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if any(code in str(x) for code in karaciger_kodlari) else 0)
    df['ALZHEIMER'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if 'G30' in str(x) else 0)
    iskemik_kalp = ['I20', 'I21', 'I22', 'I23', 'I24', 'I25']
    df['ISKEMIK_KALP'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if any(code in str(x) for code in iskemik_kalp) else 0)
    df['KALP_YETMEZLIGI'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if 'I50' in str(x) else 0)
    df['AKUT_BRONSIT'] = df['DIAGNOSISCODE'].apply(lambda x: 1 if 'J20' in str(x) else 0)

    df.drop(['DIAGNOSIS', 'DIAGNOSISCODE', 'LABRESULTS'], axis=1, inplace=True)

    # PNOMONI degeri 0 olan ve ['NEU''LYM','MON','PLT','WBC'] degiskenlerinden hepsi ayni gozlemde Null olanlari sil

    filtered_df = df[(df['PNOMONI'] == 0) &
                     (df['NEU'].isnull()) &
                     (df['LYM'].isnull()) &
                     (df['MON'].isnull()) &
                     (df['PLT'].isnull()) &
                     (df['WBC'].isnull())]
    filtered_df.shape

    df = df.drop(filtered_df.index)

    # Değişken türlerinin ayrıştırılması
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if "PNOMONI" not in col]

    # Numerik degiskenlerde 0 degeri almamasi gereken degiskenlerin 0 olan degerlerini Nan'a cevirme.
    df[['EOS', 'CRP', 'MON']] = df[['EOS', 'CRP', 'MON']].replace(0, np.nan)

    #######################
    # Yüksek Korelasyonlu Değişkenlerin Silinmesi
    #######################
    # %90 uzerinde korrelasyon oldugu icin ['FHHB', 'SO2'] degiskenlerini siliyoruz. WBC degiskenini inflamasyon skoru olusturacagimiz icin daha sonra silecegiz.
    # HTC ve HGB yuksek korelasyonlu. Bunlardan bir degisken olusturduktan sonra 2'sinide silecegiz.
    drop_list = ['FHHB', 'SO2']
    df = df.drop(drop_list, axis=1)

    # Değişken türlerinin ayrıştırılması
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
    cat_cols = [col for col in cat_cols if "PNOMONI" not in col]

    # Outlier traşlama baskılama
    for col in num_cols:
        replace_with_thresholds(df, col)

    # Eksik Değer Oranı %50 ve Üzeri Feature'lari SIL
    # missing_ratio = df.isna().mean() * 100
    # columns_to_drop = missing_ratio[missing_ratio >= 50].index
    # df = df.drop(columns=columns_to_drop)
    # Modelleme surelerini dusurmek icin veri tiplerini degistir (int64 yerine int32 - float64 yerine float32)
    df = optimize_dtypes(df)
    # Missing Values Tahmine Dayalı Atama ile Doldurma
    # En iyi parametreler: {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
    # Sayısal olmayan sütunları sayısal değerlere dönüştürme
    # En iyi parametreler: {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
    # Sayısal olmayan sütunları sayısal değerlere dönüştürme
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    # Impute missing values using IterativeImputer

    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=0), max_iter=10, tol=1e-2, random_state=0)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Her sütunun min ve max değerlerini kontrol etme ve aralık dışındaki değerleri ortalama ile değiştirme
    for column in df_imputed.columns:
        min_value = df[column].min()
        max_value = df[column].max()
        mean_value = df[column].mean()
        df_imputed[column] = df_imputed[column].apply(lambda x: mean_value if x < min_value or x > max_value else x)
    # Sayısal olmayan sütunları orijinal veri çerçevesinden al
    for column in df.select_dtypes(include=['object']).columns:
        df_imputed[column] = df[column]
    # Sayısal olmayan sütunları orijinal değerlerine geri döndürme
    for column, le in label_encoders.items():
        df_imputed[column] = le.inverse_transform(df_imputed[column].astype(int))
    # Sütun isimlerini koruyarak veri çerçevesini birleştirme
    df_imputed.columns = df.columns
    df = df_imputed

    optimize_dtypes(df)

    # Sütun isimlerini koruyarak veri çerçevesini birleştirme
    # Modelleme surelerini dusurmek icin veri tiplerini degistir (int64 yerine int32 - float64 yerine float32)
    # Veri setini azaltarak dengeli hale getir
    # Hedef sayıları belirleme
    # Veri setini azaltarak dengeli hale getir
    # Hedef sayıları belirleme
    target_counts = {0: 1000, 1: 920}
    # RandomUnderSampler kullanarak veri setini dengeleme
    undersample = RandomUnderSampler(sampling_strategy=target_counts, random_state=10)
    X_under, y_under = undersample.fit_resample(df.drop('PNOMONI', axis=1), df['PNOMONI'])
    # Azaltılmış ve dengelenmiş veri setini birleştir
    df_1000_920 = pd.concat([X_under, y_under], axis=1)
    df = df_1000_920

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
    cat_cols = [col for col in cat_cols if "PNOMONI" not in col]

    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Yas kategorileri
    df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'CHILD_YOUNG'
    df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'MIDDLEAGE'
    df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'OLD'

    # Yeni değişkenlerin oluşturulması
    df['CRP_HGB'] = df['CRP'] * df['HGB']
    df['BUN_KREATININ'] = df['BUN'] / (df['KREATININ'] + 1e-6)
    df['ELEKTROLIT_DIFF'] = df['SODYUM'] - df['KALSIYUM']
    df['EOS_NLR_RATIO'] = df['EOS'] / (df['NEU'] + 1e-6)  # NLR yerine NEU kullanıldı
    df['POTASYUM_SODYUM_RATIO'] = df['POTASYUM'] / (df['SODYUM'] + 1e-6)
    df['LOG_CR'] = np.log(df['CRP'] + 1e-6)
    df['SQRT_KREATININ'] = np.sqrt(df['KREATININ'] + 1e-6)
    df['STD_HGB_KREATININ'] = df[['HGB', 'KREATININ']].std(axis=1)
    df['DISEASES'] = df[
        ['ASTIM_1.0', 'KOAH_1.0', 'DM_2_1.0', 'KANSER_1.0', 'BOBREK_HASTASI_1.0', 'KARACIGER_1.0', 'ALZHEIMER_1.0',
         'ISKEMIK_KALP_1.0', 'KALP_YETMEZLIGI_1.0', 'AKUT_BRONSIT_1.0']].sum(axis=1)
    # Ekstra yeni değişkenler
    df['WBC_PLT_RATIO'] = df['WBC'] / (df['PLT'] + 1e-6)  # Beyaz kan hücrelerinin trombositlere oranı
    df['LYM_NEU_RATIO'] = df['LYM'] / (df['NEU'] + 1e-6)  # Lenfositlerin nötrofillere oranı
    df['MON_EOS_RATIO'] = df['MON'] / (df['EOS'] + 1e-6)  # Monositlerin eozinofillere oranı
    df['PLT_HGB_PRODUCT'] = df['PLT'] * df['HGB']  # Trombositlerin hemoglobin ile çarpımı
    df['SODYUM_KLOR_DIFF'] = df['SODYUM'] - df['KLOR']  # Sodyum ve klor farkı
    df['WBC_LOG'] = np.log(df['WBC'] + 1e-6)  # WBC'nin logaritması
    df['HCT_HGB'] = (df['HCT'] + df['HGB']) / 2
    # Inflmasyon skorlari hesapla
    df['NLR'] = df['NEU'] / df['LYM']
    df['MLR'] = df['MON'] / df['LYM']
    df['PLR'] = df['PLT'] / df['LYM']
    df['d_NLR'] = df['NEU'] / (df['WBC'] - df['NEU'])


    # HTC_HGB olusturuldugu icin kaldirilabilir
    df.drop(['HCT', 'HGB'], axis=1, inplace=True)
    # NLR, MLR, PLR hesaplandığı için, NEU, LYM ve MON kaldırılabilir
    df.drop(columns=['NEU', 'LYM', 'MON'], inplace=True)
    # d_NLR hesaplandığı için WBC kaldırılabilir
    df.drop(columns=['WBC'], inplace=True)
    # CRP_HGB oluşturulduğu için CRP ve HGB kaldırılabilir
    df.drop(columns=['CRP'], inplace=True)
    # BUN_KREATININ oluşturulduğu için BUN ve KREATININ kaldırılabilir
    df.drop(columns=['BUN', 'KREATININ'], inplace=True)
    # ELEKTROLIT_DIFF ve POTASYUM_SODYUM_RATIO oluşturulduğu için SODYUM ve POTASYUM kaldırılabilir
    df.drop(columns=['SODYUM', 'POTASYUM'], inplace=True)
    # KALSIYUM'ın tek başına kullanımı ihtiyacını ortadan kaldırmak için kaldırılabilir
    df.drop(columns=['KALSIYUM'], inplace=True)
    # EOS_NLR_RATIO oluşturulduğu için EOS kaldırılabilir
    df.drop(columns=['EOS'], inplace=True)

    df.drop(columns=['ASTIM_1.0', 'KOAH_1.0', 'DM_2_1.0', 'KANSER_1.0',
                     'BOBREK_HASTASI_1.0', 'KARACIGER_1.0', 'ALZHEIMER_1.0',
                     'ISKEMIK_KALP_1.0', 'KALP_YETMEZLIGI_1.0', 'AKUT_BRONSIT_1.0'], inplace=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
    cat_cols = [col for col in cat_cols if "PNOMONI" not in col]

    for col in num_cols:
        replace_with_thresholds(df, col)

    drop_corr = ['HCT_HGB','d_NLR']
    df = df.drop(drop_corr, axis=1)

    optimize_dtypes(df)

    # One Hot Encoding
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    # Değişken türlerinin ayrıştırılması
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
    cat_cols = [col for col in cat_cols if "PNOMONI" not in col]

    df.to_csv('df_before_RobustScaler.csv', index=False)

    # Scaling Processing
    for col in num_cols:
        df[col] = RobustScaler().fit_transform(df[[col]])

    df.to_csv('df_after_RobustScaler.csv', index=False)

    y = df['PNOMONI']
    X = df.drop(['PNOMONI'], axis=1)
    return X, y

# def base_models(X, y):
#     print("Base Models....")
#     classifiers = [
#         ("RF", RandomForestClassifier(random_state=42, n_jobs=-1)),
#         ('XGBoost', XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)),
#         ('LightGBM', LGBMClassifier(random_state=42, n_jobs=-1)),
#         ('CatBoost', CatBoostClassifier(random_state=42, silent=True))
#     ]
#
#     cv_results = []
#     for name, classifier in classifiers:
#         cv_results.append((name, np.mean(cross_val_score(classifier, X, y, cv=5, scoring="roc_auc", n_jobs=1))))
#     for result in cv_results:
#         print(result[0], ":", result[1])
#         print("----------------------------------------")

# Hyperparameter Optimization
def hyperparameter_optimization(X, y):
    print("Hyperparameter Optimization....")
    classifiers = [
        ('RF', RandomForestClassifier(random_state=42, n_jobs=-1),
         {
             "max_depth": [None],
             "max_features": ['log2'],
             "min_samples_split": [2],
             "n_estimators": [500]
         }
         ),
        ('XGBoost', XGBClassifier(random_state=42, n_jobs=-1),
         {
             "learning_rate": [0.1],
             "max_depth": [8],
             "n_estimators": [100]
         }
         ),
        ('LightGBM', LGBMClassifier(random_state=42, n_jobs=-1, verbose =-1),
         {
             'learning_rate': [0.3],
             'n_estimators': [100],
             'force_col_wise': [True],
             'num_leaves': [31, 50, 70],
             'min_child_samples': [20, 30, 40],
             'min_child_weight': [0.001, 0.01, 0.1],
             'max_depth': [-1, 8, 10, 20]
         }
         ),
        ('CatBoost', CatBoostClassifier(random_state=42, silent=True),
         {
             'iterations': [1000],
             'learning_rate': [0.3],
             'depth': [6],
             'l2_leaf_reg': [3],
             'border_count': [64]
         }
         )
    ]

    cv = 5
    scoring = "roc_auc"
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1).fit(X, y)

        if hasattr(gs_best, 'best_params_'):
            final_model = classifier.set_params(**gs_best.best_params_)
            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
            print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
            best_models[name] = final_model
        else:
            print(f"GridSearchCV did not find best parameters for {name}.")

    return best_models

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[
        ('CatBoost', best_models["CatBoost"]),
        ('RF', best_models["RF"]),
        ('XGBoost', best_models["XGBoost"]),
        ('LightGBM', best_models["LightGBM"])],
        voting='soft', n_jobs=-1).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"], n_jobs=-1)
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

################################################
################################################
def main():
    df = pd.read_csv("ClinicalLabResults.csv")
    X,y = data_preprocessing(df)
    #base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "lungsureai.pkl")
    return voting_clf

if __name__ == "__main__":
    print("İşlem başladı")
    main()
