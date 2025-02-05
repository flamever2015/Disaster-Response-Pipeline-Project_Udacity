# import libraries
import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    输入：
         messages_filepath：消息数据的文件路径
         category_filepath：类别数据的文件路径
    输出：
         df：将消息和类别数据集合进行合并 并输出
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left',on = 'id')
    return df

def clean_data(df):

    '''
    输入：
         df：消息和类别的合并数据集
    输出：
         df：清洗后的合并数据集
    '''
    # categories_new = pd.concat([categories, categories['categories'].str.split(';', expand = True)], axis=1, names = row)
    # 分割 `categories`
    df_col_categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = np.array(df_col_categories.head(1)).tolist()
    row = row[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    def mapper(x):
      return str(x)[:-2]
    row_del2 = [mapper(x) for x in row]
    category_colnames = row_del2

    df_col_categories.columns = category_colnames
    # 转换类别值至数值 0 或 1
    for column in df_col_categories:
        # set each value to be the last character of the string
        df_col_categories[column] = df_col_categories[column].str[-1]
        # convert column from string to numeric
        df_col_categories[column] = df_col_categories[column].astype('int')

    # 替换 `df` `categories` 类别列
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df_all = pd.concat([df,df_col_categories],axis = 1)

    # 删除重复行
    # drop duplicates
    df_all_drop_ID_duplicates = df_all.drop_duplicates('id')
    df = df_all_drop_ID_duplicates
    return df

def save_data(df, database_filename):
    '''
     将df保存到sqlite db
     输入：
         df：清洗后的数据集
         database_filename：数据库名称，例如 DisasterMessages.db
     输出：
         SQLite数据库
     '''
    # table name
    table_name = 'DisasterResponse'
    # create engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save dataframe to database, relace if already exists
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
