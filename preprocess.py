"""
Mimic data version 1.4, Sept16 located at shala:/data2/aromanov/mimic_data_1_4_sept_2016/
Mimic data version 1.4, Sept16 located at ishkur:/mnt/data1/mimic_iii/mimic_data_1_4_sept_2016
"""
import logging
import re

import nltk
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_sentence(sentence):
    sentence = re.sub('\n ', '\n\n', sentence)
    fine_grained = sentence.split('\n\n')
    new_sentences = []
    for subsentence in fine_grained:
        new_sentence = re.sub('\n', ' ', subsentence)
        new_sentence = re.sub(' {2,}', ' ', new_sentence)
        if new_sentence:
            new_sentences.append(new_sentence)
    return new_sentences


def filter_text(text, **kwargs):
    """
    Lowercase and filter the given text using regular expressions.

    :param text: (str)
            Input text meant for filtering and processing
    :param kwargs:
            patterns: (list)
                List of string regex patterns corresponding to substrings that are to be removed
            num_token: (str)
                A string token substituting numbers in a text
    :return: Filtered text

    """

    text = text.lower()

    regexp_patterns = kwargs.get('patterns', ['\[\*\*.*\*\*\]',  # anonymization placeholder
                                              '(?<=\s)[0-9]+\)',
                                              '#',
                                              '\-\-\>',
                                              '\*',
                                              '\d{1,2}:\d{1,2} (am|pm)',  # time pattern
                                              '\_+',
                                              '\-+',
                                              '(?<=\n\s)[\w\s]+:',  # mini-titles with colons
                                              '[,@\(\)\%]',  # punctuation
                                              ])
    if isinstance(regexp_patterns, str):
        regexp_patterns = [regexp_patterns]

    # Remove anonymization placeholders
    for pattern in regexp_patterns:
        text = re.sub(pattern, ' ', text)

    text = re.sub('\n ', '\n', text)

    # Replace numbers with placeholders (numbers include floats, negatives and ranges)
    num_token = kwargs.get('num_token', ' <num> ')

    text = re.sub('^[0-9]+$', num_token, text)

    sentences = nltk.sent_tokenize(text)

    all_sentences = []
    for sentence in sentences:
        all_sentences.extend(check_sentence(sentence))
    return all_sentences


def tokenize_text(text, tokenizer, stop_words=True):
    """
    Tokenize the given text using a pre-specified tokenizer.
    :param text: (str)
            Analyzed text
    :param tokenizer: (object)
            Tokenizer
    :param stop_words: (bool)
            If True, stop-words are filtered
    :return: (list)
            List of tokens
    """
    tokens = tokenizer.tokenize(text)
    if stop_words:
        from nltk.corpus import stopwords
        words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in words]
    return tokens


def read_diagnoses(dir):
    """
    Read a table containing diagnoses for every admission id.
    :param dir: (str)
            Path to a folder containing source files
    :return: (pandas.DataFrame)
            Dataframe containing admission ids with primary ICD-9 codes.
    """
    df = pd.read_csv(dir + 'DIAGNOSES_ICD.csv')
    df['CATEGORY'] = df['ICD9_CODE'].apply(lambda x: str(x)[:3])

    # Extract only primary diagnoses (SEQ_NUM = 1)
    df = df[df['SEQ_NUM'] == 1]
    return df


def is_heart(row):
    """
    Check if diagnosis is related to heart
    :param row:
    :return:
    """
    if row[:2] == '42':
        return 0
    elif row[:3] == '410':
        return 1
    elif row[:3] == '414':
        return 2
    else:
        return 3


def read_and_pickle_notes(dir, target_loc):
    """
    Read NOTEEVENTS.csv table and convert it to a pandas dataframe.
    The raw text is filtered and tokenized; the unnecessary columns are dropped.
    The resulting dataframe is pickled to the specified location.

    :param dir: (str)
            Path to a folder containing source .csv files
    :param target_loc: (str)
            Path for pickling the dataframe
    :return: (pd.DataFrame)
            Dataframe describing patients
    """
    df = pd.read_csv(dir + 'NOTEEVENTS.csv')

    # Filter notes with NaN admissions
    df = df[~df['HADM_ID'].isnull()]
    df['HADM_ID'] = df['HADM_ID'].apply(int)

    # Filter notes marked as erroneous
    print('{} out of {} records are valid'.format(len(df[df['ISERROR'].isnull()]), len(df)))
    df = df[df['ISERROR'].isnull()]

    # Remove unnecessary columns
    df.drop(labels=['SUBJECT_ID', 'CHARTTIME', 'CHARTDATE', 'ISERROR',
                    'STORETIME', 'CATEGORY', 'DESCRIPTION', 'CGID', 'ROW_ID'],
            axis=1, inplace=True)

    # Map admission ids to diagnoses
    diagnoses = read_diagnoses(dir)
    diagnoses_dict = dict(zip(diagnoses['HADM_ID'], diagnoses['CATEGORY']))
    admissions_icd_known = diagnoses['HADM_ID'].values
    df = df[df['HADM_ID'].isin(admissions_icd_known)]
    df['ICD'] = df['HADM_ID'].apply(lambda x: diagnoses_dict[x])

    # Analyze the target variable and do the filtering
    print('The datatable contains {} unique codes'.format(len(df['ICD'].unique())))


    # Leave only heart diseases related codes (as the most frequent)
    df['HEART'] = df['ICD'].apply(is_heart)

    # Group texts by admission id
    grouped = pd.DataFrame(df.groupby(['HADM_ID', 'ICD'])['TEXT'].apply(lambda x: ' '.join(x)))
    grouped.reset_index(inplace=True)
    grouped['HEART'] = grouped['ICD'].apply(is_heart)

    grouped = grouped[grouped['HEART'] < 3]
    print('The target table contains {} patients'.format(len(grouped)))
    # Process the text
    grouped['SENTENCES'] = grouped['TEXT'].apply(filter_text)

    counts = grouped['ICD'].value_counts().to_dict()
    frequent_diagnoses = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print('10 most frequent codes are: ')
    for diagnosis in frequent_diagnoses:
        print('Code {} with count {}'.format(diagnosis[0], diagnosis[1]))

    # Pickle
    grouped.to_pickle(target_loc)
    return grouped


def read_pickled_notes(loc):
    """
    Load pickled dataframe
    :param loc: (str)
            Path to file
    :return: (pd.DataFrame) Deserialized object stored in file
    """
    df = pd.read_pickle(loc)
    print('{} entries in the dataset'.format(len(df)))
    return df


if __name__ == "__main__":
    data_dir = '/mnt/data1/mimic_iii/mimic_data_1_4_sept_2016/'
    target_loc = 'data/dataframe.pkl'
    patients = read_and_pickle_notes(data_dir, target_loc)
    print('Done')
