import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler


title_codes = {
    'Mr': 1,        # General adult male
    'Mrs': 2,       # General adult female
    'Miss': 3,      # General young female
    'Master': 4,    # General young male
    'Don': 5,       # Other male
    'Rev': 5,       # Other male
    'Dr': 5,        # Other male
    'Mme': 2,       # General adult female
    'Ms': 2,        # General adult female
    'Major': 5,     # Other male
    'Lady': 6,      # Other female
    'Sir': 5,       # Other male
    'Mlle': 3,      # General young female
    'Col': 5,       # Other male
    'Capt': 6,      # Other male
    'Countess': 6,  # Other female
    'Jonkheer': 5,  # Other male
    'Dona': 6       # Other Female
}

sex_codes = {
    'male': 0,
    'female': 1,
}

embarked_codes = {
    '': 0,   # Unknown
    'S': 1,  # Southampton
    'C': 2,  # Cherbourg
    'Q': 3,  # Queenstown
}


def import_data(train_data, test_data):
    '''
    Imports the train and test data and returns a dataframe
    '''

    df_train = pd.read_csv(train_data)
    df_test = pd.read_csv(test_data)
    df_train['Train'] = 1
    df_test['Test'] = 1
    df_test['Survived'] = 0
    df = pd.concat([df_train, df_test])
    df.columns = [x.lower() for x in df.columns]

    return df


def feature_engineering(df):

    def title(row):
        '''
        Uses a regular expression to search for a title in a string of text when
        applied to a pandas dataframe. Returns the title if successful and a
        blank string if not.
        '''

        title_search = re.search(' ([A-Za-z]+)\.', row['name'])
        if title_search:
            return title_search.group(1)
        else:
            return ""

    def infer_age(row):
        '''
        Uses the title of the passenger to infer an age value in the event of
        missing data.
        '''
        if(pd.isnull(row['age'])):

            if row['title'] == 1:    # Mr
                return 30
            elif row['title'] == 2:  # Mrs
                return 35
            elif row['title'] == 3:  # Miss
                return 21
            elif row['title'] == 4:  # Master
                return 4
            elif row['title'] == 5:  # Noble male
                return 40
            elif row['title'] == 6:  # Professional
                return 50
            elif row['title'] == 7:  # Noble female
                return 40

        else:
            return row['age']

    def family_sizer(row):
        '''Categorises the family size'''
        if row['family_size'] == 1:
            return 1
        elif row['family_size'] < 5:
            return 2
        else:
            return 3

    def group_sizer(row):
        '''Categorises the group size'''
        if row['group_size'] == 1:
            return 1
        elif row['group_size'] < 5:
            return 2
        else:
            return 3

    def is_alone(row):
        '''Determoines if someone is alone '''
        if row['family_size'] == 1:
            if row['group_size'] == 1:
                return 1
            else:
                return 0
        else:
            return 0

    def has_family(row):
        '''Determines if someone has a family'''
        if row['family_size'] >= 1:
            return 1
        else:
            return 0

    def has_group(row):
        '''Determines if someone has a group'''
        if row['group_size'] >= 1:
            return 1
        else:
            return 0

    # Basic feature engineering
    df['sex'] = df['sex'].replace(sex_codes)
    df['embarked'] = df['embarked'].replace(embarked_codes)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df = df.drop('cabin', axis=1)

    # Title
    df['title'] = df.apply(title, axis=1)
    df['title'] = df['title'].replace(title_codes)

    # Age
    df['age'] = df.apply(infer_age, axis=1)

    # Tickets
    ticket_split = df['ticket'].str.split().tolist()
    ticket_number = [x[-1] for x in ticket_split]
    df['ticket_number'] = ticket_number
    tb_ticket_counts = (
        df[['passengerid', 'ticket_number']]
        .groupby('ticket_number')
        .count()
        .reset_index()
        .rename(columns={'passengerid': 'group_size'})
    )
    df = pd.merge(
        left=df,
        right=tb_ticket_counts,
        how='left'
    )

    # Family Size
    df['family_size_cat'] = df.apply(family_sizer, axis=1)

    # Group Size
    df['group_size_cat'] = df.apply(group_sizer, axis=1)

    # Is alone?
    df['is_alone'] = df.apply(is_alone, axis=1)

    # Has family?
    df['has_family'] = df.apply(has_family, axis=1)

    # Has a group?
    df['has_group'] = df.apply(has_group, axis=1)

    # One-hot encoding
    df_sex = pd.get_dummies(df['sex'])
    df_sex.columns = [
        'sex_{}'.format(x) for x in df_sex.columns
    ]

    df_title = pd.get_dummies(df['title'])
    df_title.columns = [
        'title_{}'.format(x) for x in df_title.columns
    ]

    df_embarked = pd.get_dummies(df['embarked'])
    df_embarked.columns = [
        'embarked_{}'.format(int(x)) for x in df_embarked.columns
    ]

    df_family = pd.get_dummies(df['family_size_cat'])
    df_family.columns = [
        'family_size_{}'.format(x) for x in df_family.columns
    ]

    df_group = pd.get_dummies(df['group_size_cat'])
    df_group.columns = [
        'group_size_{}'.format(x) for x in df_group.columns
    ]

    # Merging the one-hot encoded variables
    df = pd.concat(
        [df, df_sex, df_title, df_embarked, df_family, df_group],
        axis=1
    )

    # Scaling

    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['true_fare'] = round(df['fare'] / df['group_size'], 2)
    scaler = MinMaxScaler()

    fare_array = df['true_fare'].as_matrix()
    age_array = df['age'].as_matrix()
    family_array = df['family_size'].as_matrix()
    group_array = df['group_size'].as_matrix()

    fare_scaled = scaler.fit_transform(fare_array.reshape(-1, 1))
    age_scaled = scaler.fit_transform(age_array.reshape(-1, 1))
    family_scaled = scaler.fit_transform(family_array.reshape(-1, 1))
    group_scaled = scaler.fit_transform(group_array.reshape(-1, 1))

    df['true_fare_scaled'] = fare_scaled
    df['age_scaled'] = age_scaled
    df['family_scaled'] = family_scaled
    df['group_scaled'] = group_scaled

    df.to_csv('./data/full_train_test.csv', index=False)

    return df
