# -*- coding: utf-8 -*-
"""
Jaka Bizjak, Institut Jožef Stefan, Energy Efficiency Center

updated: 2024-04-24
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import holidays

def bloki_vrednosti(year, str_bloki, freq):
    '''
    Function calculates blok value for every timestamp in a given year and saves it in a csv file named 'year_bloki.txt'
    
    Function inputs:
    - year - year you want to get block values
    - str_bloki - string of csv file where definition of block values for different seasons and day types are
    - freq - frequency of measured electricity power in min

    '''
    loop_start = datetime.datetime.now() # stopwatch the time of loop calculation
    leto = np.arange(np.datetime64(str(year) + '-01-01'), np.datetime64(str(year+1) + '-01-01'), np.timedelta64(freq, "m"))
    
    bloki_vrednost = pd.read_csv(str_bloki) # razporeditev blokov glede na sezono, dan in uro - prebere cv file
    bloki = np.zeros(np.size(leto))
    
    df = pd.DataFrame(data=leto, columns=['datum']) # dataframe for given year with given frequency of timestamps
    df = df.assign(blok=bloki) # add column for bloki values
    
    # plot of daily blok values for different seasons and day types
    fig1, ax1 = plt.subplots()
    ax1.plot(bloki_vrednost.iloc[:,0], bloki_vrednost.iloc[:,1], label=' višja sezona - delovnik')
    ax1.plot(bloki_vrednost.iloc[:,0], bloki_vrednost.iloc[:,2], label=' nižja sezona - delovnik')
    ax1.plot(bloki_vrednost.iloc[:,0], bloki_vrednost.iloc[:,3], label=' višja sezona - vikend')
    ax1.plot(bloki_vrednost.iloc[:,0], bloki_vrednost.iloc[:,4], label=' nižja sezona - vikend')
    ax1.invert_yaxis()
    ax1.grid()
    ax1.legend()
    
    holidays_SI = holidays.Slovenia(year) # get list of public holidays in a given year
    
    # this loop assign correct blok value to a specific timestamp in a year according to daytype and season
    for i in range(0, np.size(leto)): 
        for j in range(0, bloki_vrednost.iloc[:,0].size):
            if df.iloc[i,0].month < 3 or df.iloc[i,0].month > 10: # visoka sezona
                if df.iloc[i,0].dayofweek < 5: # delovni dan   
                    if df.iloc[i,0].hour == bloki_vrednost.iloc[j,0]: 
                        df.iloc[i, 1] = bloki_vrednost.iloc[j,1]
                if df.iloc[i,0].dayofweek > 4 or df.iloc[i,0] in holidays_SI: # vikend in prazniki
                    if df.iloc[i,0].hour == bloki_vrednost.iloc[j,0]: 
                        df.iloc[i, 1] = bloki_vrednost.iloc[j,3]
            else: # nizka sezona
                if df.iloc[i,0].dayofweek < 5: # delovni dan   
                    if df.iloc[i,0].hour == bloki_vrednost.iloc[j,0]: 
                        df.iloc[i, 1] = bloki_vrednost.iloc[j,2]
                if df.iloc[i,0].dayofweek > 4 or df.iloc[i,0] in holidays_SI: # vikend in prazniki
                    if df.iloc[i,0].hour == bloki_vrednost.iloc[j,0]: 
                        df.iloc[i, 1] = bloki_vrednost.iloc[j,4]

    # save blok values for a given year in a separate csv file
    str_file = str(year) + '_bloki.txt'
    df.to_csv(str_file, index=False)
    loop_end = datetime.datetime.now()
    print('Time of function calculation', loop_end-loop_start)
