
import time 
import numpy as np 
import pandas as pd 
import codecs


def divide2each_features(start_time):
    """
    transform the original data, which is collection of all price-volume features for the whole A stocks, 
    to divided csv file. Each file has the following form: columns is stock code, row is datetime
                000001 000002 0000003 ....
    2016-03-01   ***    ***
    2016-03-02   ***    ***
    .
    .
    """
    original_data0 = pd.read_csv(r"C:\Data\AStock\TRD_Dalyr_new.csv", index_col=1)
    original_data1 = pd.read_csv(r"C:\Data\AStock\TRD_Dalyr_new1.csv", index_col=1)
    original_data2 = pd.read_csv(r"C:\Data\AStock\TRD_Dalyr_new2.csv", index_col=1)
    original_data = pd.concat([original_data0, original_data1, original_data2], axis=0)
    original_data.columns = ['Stkcd', 'Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd', 'Dnvaltrd',
       'Dsmvosd', 'Dsmvtll', 'Dretwd', 'Dretnd', 'Adjprcwd', 'Adjprcnd',
       'Markettype', 'Capchgdt', 'Trdsta']
    # print(type(original_data.columns[0]))
    for col in ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd', 'Dnvaltrd',
       'Dsmvosd', 'Dsmvtll', 'Dretwd', 'Dretnd', 'Adjprcwd', 'Adjprcnd',
       'Markettype', 'Capchgdt', 'Trdsta']:
        catdf = original_data[['Stkcd', col]]
        catdf_group = catdf.groupby('Stkcd')
        df = pd.DataFrame()
        count = 0
        for i in catdf_group:
            count += 1
            if count % 500 == 0:
                print(count)
            x = i[1][col]
            x.name = str(i[0])
            df = pd.concat([df, x], axis=1, sort=False)
        print('\n', '*' * 100)
        print('Stock statistics:', count)
        df.to_csv('C:\\Data\\AStock\\'+col+'.csv')
        print('time costs:', time.time() - start_time)
        print('*' * 100, '\n')

    
start_time = time.time()
divide2each_features(start_time)