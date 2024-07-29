# -*- coding: utf-8 -*-
"""
Jaka Bizjak, Institut Jožef Stefan, Energy Efficiency Center
My python functions

updated: 2024-04-24

functions so far:
    - logp_h_diagram
    - bloki_omreznina
    - DegreeHours
    - ISO_52010
function from internet:
    - day_in_year
    - N_days
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import holidays
from CoolProp.CoolProp import PropsSI


def logp_h_diagram(refrigerant, p, h):
    '''
    Function plots logp-h diagram for given refrigerant and state points
    
    Function inputs:
    - refrigerant - name of refrigerant as string so CoolProp can recognise it
    - p - array of pressure for states 1-4 p[p_1, p_2, p_3, p_4]
    - h - array of enthalpy for states 1-4 h[h_1, h_2, h_3, h_4]

    '''
    loop_start = datetime.datetime.now() # stopwatch the time of loop calculation
    # get pressure and enthalpy of different states from arrays
    p_1 = p[0]
    p_2 = p[1]
    p_3 = p[2]
    p_4 = p[3]
    h_1 = h[0]
    h_2 = h[1]
    h_3 = h[2]
    h_4 = h[3]
    
    p_crit = PropsSI('pcrit', refrigerant) # get the critical pressure of refrigerant
    
    p_arr = np.linspace(0.5e5, p_crit, 50000) # array of pressure
    h_q0 = np.zeros(50000) # dew point curve
    h_q1 = np.zeros(50000) # boiling curve
    for i in range(0,49999):
        h_q0[i] = PropsSI('H','P',p_arr[i],'Q',0, refrigerant)
        h_q1[i] = PropsSI('H','P',p_arr[i],'Q',1, refrigerant)
        h_con = np.array([h_q0[-2], h_q1[-2]])
        p_con = np.array([p_arr[-2], p_arr[-2]])
    
    # array of state points
    ref = np.array([[h_1/1e3, p_1/1e5], # state 1
                    [h_2/1e3, p_2/1e5], # state 2
                    [h_3/1e3, p_3/1e5], # state 3
                    [h_4/1e3, p_4/1e5], # state 4
                    [h_1/1e3, p_1/1e5]]) # back to state 1
    
    # plot logp-h diagram with cooling cycle
    fig1, ax1 = plt.subplots()
    ax1.plot(h_q0[:-1]/1e3, p_arr[:-1]/1e5, 'k-', label = refrigerant) # dew ponit curve plotting
    ax1.plot(h_q1[:-1]/1e3, p_arr[:-1]/1e5, 'k-') # boiling curve plotting
    ax1.plot([h_q0[-2]/1e3, h_q1[-2]/1e3], [p_arr[-1]/1e5, p_arr[-1]/1e5], 'k-') # missing line between curves
    ax1.plot(ref[:,0], ref[:,1], 'r-o') # cooling cycle plotting
    ax1.set_yscale("log")
    ax1.grid()
    #ax1.legend()
    ax1.set_xlabel('$h$ [kJ/kg]')
    ax1.set_ylabel('$log p$ [bar]')
    ax1.set_ylim(bottom=1)
    loop_end = datetime.datetime.now()
    print('Time of function calculation', loop_end-loop_start)

def bloki_omreznina(year, str_bloki, freq):
    '''
    Function calculates block value for every timestamp in a given year and saves it in a csv file named 'year_bloki.txt'
    
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

    # save blok values fo a given year in a separate csv file
    str_file = str(year) + '_bloki.txt'
    df.to_csv(str_file, index=False)
    loop_end = datetime.datetime.now()
    print('Time of function calculation', loop_end-loop_start)


def DegreeHours(T_air, T_in, dT_eff, T_tr, N):
    ''' Function calculates the Degree Hour value based on yearly temperature of outside air, 
    desired indoor temperature and treshold temperature based on internal and external 
    heat gainings (from people, devices and sun irradiation)
    
    Function inputs:
    - T_air - np.array of yearly temperature (hourly values) in °C
    - T_in - desired indoor temperature in °C
    - dT_eff - temperature difference provided by internal gains in K
    - T_tr - treshold temperature in K (includes solar gains)
    - N - number of data per hour
    '''
    loop_start = datetime.datetime.now() # stopwatch the time of loop calculation
    # prepare arrays for desired values (temperature and hours in a year)
    T_air = np.round(T_air, 1)
    T_air_min = np.round(np.min(T_air), 1)
    T_air_max = np.round(np.max(T_air), 1)
    T_DH = np.round(np.arange(T_air_min, T_air_max+0.1, 0.1), 1)
    t_DH = np.zeros(np.size(T_DH))

    # calculate number of hours with given temperature in a year
    sum_t_DH = np.zeros(np.size(T_DH))
    for i in range(0, np.size(T_DH)):
        t_DH[i] = np.sum(T_air == T_DH[i])
       
    # summarize to get nice curve :)
    sum_t_DH[0] = t_DH[0]  
    for i in range(1, np.size(T_DH)):
        sum_t_DH[i] = sum_t_DH[i-1] + t_DH[i]
        
    # find position of treshold temperatrue
    try:
        ind = np.where(T_DH == T_tr)[0][0]
    except IndexError:
        ind = np.size(T_DH) - 1
    try:
        ind_2 = np.where(T_DH == (T_in-dT_eff))[0][0]
    except IndexError:
        ind_2 = np.size(T_DH) - 1
    t_tr = np.ones(np.size(T_DH))*sum_t_DH[ind]
    # plot yearly temperature profile
    fig1, ax1 = plt.subplots()
    ax1.plot(sum_t_DH, T_DH, label='yearly air temperature')
    ax1.plot(sum_t_DH, np.ones(np.size(T_DH))*T_in, label='indoor temperature', linestyle='--')
    ax1.plot(sum_t_DH, np.ones(np.size(T_DH))*T_in - dT_eff, label='effective indoor temperature', linestyle='--')
    ax1.plot(t_tr, T_DH, linestyle='--')
    ax1.plot(sum_t_DH, np.ones(np.size(T_DH))*T_tr, label='treshold temperature', linestyle='--', linewidth=1, color='k')
    ax1.fill_between(sum_t_DH[0:ind], np.ones(np.size(T_DH[0:ind]))*(T_in - dT_eff), T_DH[0:ind], label='degree hour value', color='grey')
    ax1.fill_between(sum_t_DH[ind:ind_2], np.ones(np.size(T_DH[ind:ind_2]))*(T_in - dT_eff), T_DH[ind:ind_2], label='solar gains', color='yellow')
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('$t$ [h]')
    ax1.set_ylabel('$T$ [°C]')
    ax1.set_title('Heating Degree Hours Value')

    # calculate degree hour value
    DH_i = np.zeros(np.size(sum_t_DH))
    for i in range(0, ind):
        DH_i[i] = ((T_in - dT_eff) - T_DH[i])*t_DH[i]
        if DH_i[i] < 0:
            DH_i[i] = 0

    DH_sol_gain_i = np.zeros(np.size(sum_t_DH))
    for i in range(ind, ind_2):
        DH_sol_gain_i[i] = ((T_in - dT_eff) - T_DH[i])*t_DH[i]
        if DH_sol_gain_i[i] < 0:
            DH_sol_gain_i[i] = 0

    DH = np.round(np.sum(DH_i)/N, 0)
    print('DegreeHours = ', DH, '°Ch')

    DH_sol_gain = np.sum(DH_sol_gain_i)/N
    print('solar gain = ', DH_sol_gain, '°Ch')
    loop_end = datetime.datetime.now()
    print('Time of function calculation', loop_end-loop_start)
    
    
def ISO_52010(location, slope, azimuth, G_sol_g, G_sol_b, G_sol_d, t_data, name):
    """ Function calculates solar irradiaton on given location, slope, azimuth and typical reference year 
    data for global solar irradiation by ISO 52010 standardand saves it as array in .txt file

    Function inputs:
    - location as np.array([latitude, longitude, time zone (+/-GMT)]),
    - slope in °, 
    - azimuth in ° (0° is south, + is east, - is west),
    - G_sol_g in W/m2 as np.array - typical reference year - global solar irradiance - if there is no data, value is -1,
    - G_sol_b in W/m2 as np.array - typical reference year - beam solar irradiance (normal incidence) - if there is no data, value is -1,
    - G_sol_d in W/m2 as np.array - typical reference year - difuse solar irradiance - if there is no data, value is -1,
    - t_data = time interval of G_sol_g data in min,
    - name = name of .txt file to save calculated solar irradiance)

    """
    loop_start = datetime.datetime.now() # stopwatch the time of loop calculation
    # time arrays
    day_len = int(60/t_data)*24
    N_data = 365 * day_len

    # some constants
    G_sol_c = 1370 # W/m2 - solar constant
    rho_g = 0.2 # / - ground reflectance
    K = 1.014 # rd^(-3) - constant
    lat = np.deg2rad(location[0]) # rad - latitude of location
    beta = np.deg2rad(slope) # rad - tilt angle of the inclined surface
    gamma = np.deg2rad(azimuth) # rad - orientation of inclined surface

    #defining array of days and hours
    days = np.linspace(1, 366, N_data+1) # array of days for a year
    hours = np.linspace(0, 8760, N_data+1) # array of hours for a year
    hours_day = np.tile(hours[0:day_len],365) # array of hours in a day for a year
    days = days[:-1]
    hours = hours[:-1]
    days = days.astype(int)

    # 6.4.1 calculations of the sun path

    # 6.4.1.1 solar declination
    R_de = np.deg2rad(360/365 * days) # rad - earth orbit deviation, eq. 2
    delta = np.deg2rad(0.33281 - 22.984*np.cos(R_de) - 0.3499*np.cos(2*R_de) - 0.1398*np.cos(3*R_de) + \
                3.7872*np.sin(R_de) + 0.03205*np.sin(2*R_de) + 0.07187*np.sin(3*R_de)) # rad - declination of earth, eq. 1

    # 6.4.1.2 equation of time
    t_eq = np.zeros(N_data)
    for i in range(0, N_data):
        if days[i] < 366:
            t_eq[i] = 0.45*(days[i]-359) # eq. 7
            if days[i] < 336:
                t_eq[i] = -6.3 -10*np.cos((days[i]-306)*0.036) # eq.6
                if days[i] < 241:
                    t_eq[i] = 1.4 - 5*np.cos((days[i]-135)*0.0449) # eq. 5
                    if days[i] < 136:
                        t_eq[i] = 5.2 + 9*np.cos((days[i]-43)*0.0357) # eq. 4
                        if days[i] < 21:
                            t_eq[i] = 2.6 +0.44* days[i] # eq 3

    # 6.4.1.3 time shift
    time_shift = location[-1] - location[1]/15 # h - time shift, eq. 8

    # 6.4.1.4 solar time
    t_sol = hours_day - t_eq/60 - time_shift # h - solar time, eq. 9

    # 6.4.1.5 solar hour angle
    omega = np.deg2rad(180/12 * (12.5 - t_sol)) # rad, eq. 10
    for i in range(0, N_data):
        if omega[i] > np.pi:
            omega[i] = np.deg2rad(180/12 * (12.5 - t_sol[i]) - 360) 
        if omega[i] < -np.pi:
            omega[i] = np.deg2rad(180/12 * (12.5 - t_sol[i]) + 360) 

    # 6.4.1.6 solar altitude and solar zenith angle
    alpha = np.arcsin(np.sin(delta)*np.sin(lat) + np.cos(delta)*np.cos(lat)*np.cos(omega)) # rad - solar altitude, eq. 11
    for i in range(0, N_data):
        if alpha[i] < np.deg2rad(3): # condition in standard is 0.0001 deg, but this is to strict - bad results when deviding by sin(alpha) or cos(theta)
            alpha[i] = 0
    theta = np.pi/2 - alpha # rad - solar zenith angle, eq. 12

    # 6.4.1.7 Solar azimuth angle
    sin_aux1 = (np.cos(delta)*np.sin(np.pi - omega))/(np.cos(np.arcsin(np.sin(alpha)))) # rad - first auxilairy angle, eq. 13
    cos_aux1 = (np.cos(lat)*np.sin(delta) + np.sin(lat)*np.cos(delta)*np.cos(np.pi - omega))/(np.cos(np.arcsin(np.sin(alpha))))
        # first auxiliary angle, eq. 14
    phi_aux2 = (np.arcsin(np.cos(delta)*np.sin(np.pi - omega)))/(np.cos(np.arcsin(np.sin(alpha)))) # rad - second auxilairy angle, eq. 15
    phi = - (np.pi + phi_aux2) # rad - solar azimuth angle, eq. 16
    for i in range(0, N_data):
        if sin_aux1[i] >= 0:
            if cos_aux1[i] > 0:
                phi[i] = np.pi - phi_aux2[i]
        if cos_aux1[i] < 0:
            phi[i] = phi_aux2[i]

    # 6.4.1.8 Solar angle of incidence on inclines surface
    theta_ic = np.arccos(np.sin(delta)*np.sin(lat)*np.cos(beta) \
                        - np.sin(delta)*np.cos(lat)*np.sin(beta)*np.cos(gamma) \
                        + np.cos(delta)*np.cos(lat)*np.cos(beta)*np.cos(omega) \
                        + np.cos(delta)*np.sin(lat)*np.sin(beta)*np.cos(gamma)*np.cos(omega) \
                        + np.cos(delta)*np.sin(beta)*np.sin(gamma)*np.sin(omega)) # rad - eq. 17

    # 6.4.1.9 Azimuth and tilt angle between sun and the inclines surface
    gamma_sol = omega - gamma # rad - azimuth between suna and surface, eq. 18
    for i in range(0, N_data):
        if gamma_sol[i] < -np.pi:
            gamma_sol[i] = 2*np.pi + omega[i] - gamma
        if gamma_sol[i] > np.pi:
            gamma_sol[i] = -2*np.pi + omega[i] - gamma

    beta_sol = beta - theta # rad tilt angle between sun and surface, eq. 19
    for i in range(0, N_data):
        if beta_sol[i] > np.pi:
            beta_sol[i] = -2*np.pi + beta - theta[i]
        if beta_sol[i] < -np.pi:
            beta_sol[i] = 2*np.pi + beta - theta[i]

    # 6.4.1.10 Air mass
    m = 1/(np.sin(alpha) + 0.15*(np.rad2deg(alpha) + 3.885)**(-1.253)) # / - air mass, eq. 21
    for i in range(0, N_data):
        if alpha[i] >= np.deg2rad(10):
            m[i] =  1/np.sin(alpha[i]) # eq.20

    # 6.4.2 Split between direct and diffusive irradiance

    # Method 1 - when in TRY data only G_sol_g is given - for example ARSO
    if np.size(G_sol_b) < 2:
        I_ex = G_sol_c*(1 + 0.033*np.cos(R_de)) # W/m2 - extra-terrestrial radiaton, eq. 27
        k_t = G_sol_g/(I_ex*np.cos(theta)) #  / - clearness index, eq. 24, checked with JRC Photovoltaic GIS
        for i in range(0, N_data):
            if theta[i] >= np.pi/2:
                k_t[i] = 0

        fr = np.ones(N_data)*0.165 # / - G_sol_d / G_sol_g - diffuse fraction, eq. 23
        for i in range(0, N_data):
            if k_t[i] <= 0.8:
                fr[i] = 0.9511 - 0.1604*k_t[i] + 4.388*k_t[i]**2 - 16.638*k_t[i]**3 + 12.336*k_t[i]**4
                if k_t[i] <= 0.22:
                    fr[i] = 1 - 0.09*k_t[i]

        G_sol_d = fr*G_sol_g # W/m2 - diffusive solar irradiance

        G_sol_b = np.zeros(N_data) # W/m2 - beam solar irradiance, eq. 25
        for i in range(0, N_data):
            if alpha[i] > 0:
                G_sol_b[i] = (G_sol_g[i] - G_sol_d[i])/np.sin(alpha[i])
    
    # Method 2 - when G_sol_b and G_sol_d in TRY data are given - fo example PVGIS          
    if np.size(G_sol_g) < 2:
        G_sol_g = G_sol_b*np.sin(alpha) + G_sol_d # W/m2 - eq. 22
        I_ex = G_sol_c*(1 + 0.033*np.cos(R_de)) # W/m2 - extra-terrestrial radiaton, eq. 27
        
        # save G_sol_g as txt file
        np.savetxt('G_sol_g.txt', G_sol_g)

    # 6.4.4 total solar irradiance at given orientation and solar angle

    # 6.4.4.1 Direct Irradiance
    I_dir = G_sol_b*np.cos(theta_ic) # W/m2 - direct irradiance, eq. 26
    for i in range(0, N_data):
        if I_dir[i] < 0:
            I_dir[i] = 0

    # 6.4.4.3 Diffuse irradiance
    a = np.cos(theta_ic) # parameter a, eq. 28
    for i in range(0, N_data):
        if a[i] < 0:
            a[i] = 0

    b = np.cos(theta) # parameter b, eq. 29
    for i in range(0, N_data):
        if b[i] < np.cos(np.deg2rad(85)):
            b[i] = np.cos(np.deg2rad(85))    

    epsilon = np.ones(N_data)*999 # / - clearness parameter, eq. 30
    for i in range(0, N_data):
        if G_sol_d[i] > 0:
            epsilon[i] = ((G_sol_d[i] + G_sol_b[i])/G_sol_d[i] + K*alpha[i]**3)/(1 + K*alpha[i]**3)

    # Table 8: Values for clearness index and brightness coefficients as function of clearness parameter
    f_11 = np.ones(N_data)*0.678
    f_12 = np.ones(N_data)*(-0.327)
    f_13 = np.ones(N_data)*(-0.25)
    f_21 = np.ones(N_data)*0.156
    f_22 = np.ones(N_data)*(-1.377)
    f_23 = np.ones(N_data)*0.251
    for i in range(0, N_data):
        if epsilon[i] < 6.2:
            f_11[i] = 1.06
            f_12[i] = -1.6
            f_13[i] = -0.359
            f_21[i] = 0.264
            f_22[i] = -1.127
            f_23[i] = 0.131
            if epsilon[i] < 4.5:
                f_11[i] = 1.132
                f_12[i] = -1.237
                f_13[i] = -0.412
                f_21[i] = 0.288
                f_22[i] = -0.823
                f_23[i] = 0.056
                if epsilon[i] < 2.8:
                    f_11[i] = 0.873
                    f_12[i] = -0.392
                    f_13[i] = -0.362
                    f_21[i] = 0.226
                    f_22[i] = -0.462
                    f_23[i] = 0.001
                    if epsilon[i] < 1.95:
                        f_11[i] = 0.568
                        f_12[i] = 0.187
                        f_13[i] = -0.295
                        f_21[i] = 0.109
                        f_22[i] = -0.152
                        f_23[i] = -0.014
                        if epsilon[i] < 1.5:
                            f_11[i] = 0.33
                            f_12[i] = 0.487
                            f_13[i] = -0.221
                            f_21[i] = 0.055
                            f_22[i] = -0.064
                            f_23[i] = -0.026
                            if epsilon[i] < 1.23:
                                f_11[i] = 0.13
                                f_12[i] = 0.683
                                f_13[i] = -0.151
                                f_21[i] = -0.019
                                f_22[i] = 0.066
                                f_23[i] = -0.029
                                if epsilon[i] < 1.065:
                                    f_11[i] = -0.008
                                    f_12[i] = 0.588
                                    f_13[i] = -0.062
                                    f_21[i] = -0.06
                                    f_22[i] = 0.072
                                    f_23[i] = -0.022

    DELTA = m*G_sol_d/I_ex # / - sky brightness parameter, eq. 31

    F_1 = f_11 + f_12*DELTA + f_13*theta # circumsolar brightness coefficient, eq. 32
    for i in range(0, N_data):
        if F_1[i] < 0:
            F_1[i] = 0

    F_2 = f_21 + f_22*DELTA + f_23*theta # horizontal brightness coefficient, eq. 33

    I_dif = G_sol_d * ((1 - F_1)*(1 + np.cos(beta))/2 + F_1*a/b + F_2*np.sin(beta)) # W/m2, eq. 34

    # 6.4.4.4 Diffusive solar irradiance due to ground reflection

    I_dif_grnd = (G_sol_d + G_sol_b*np.sin(alpha))*rho_g*(1 - np.cos(beta))/2 # W/m2, eq. 35

    # 6.4.4.5 Circumsolar irradiance
    I_circum = G_sol_d*F_1*a/b # W/m2 - circumsolar irradiance, eq. 36

    # 6.4.4.6 Calculates total direct solar irradiance
    I_dir_tot = I_dir + I_circum # W/m2, eq. 37

    # 6.4.4.7 Calculated total diffusive solar irradiance
    I_dif_tot = I_dif - I_circum + I_dif_grnd # W/m2, eq. 38

    # 6.4.4.7 Calculated total solar irradiance
    I_tot = I_dir_tot + I_dif_tot # W/m2, eq. 34

    # plot 1 - G_glob and I_tot yearly curves
    fig1, ax1 = plt.subplots()
    ax1.plot(np.linspace(0,N_data-1,N_data), I_tot, label='I_tot')
    ax1.plot(np.linspace(0,N_data-1,N_data), G_sol_g, label='G_glob')
    ax1.set_title(name)
    ax1.set(xlabel='$t$ [h]', ylabel='$I_{tot}$/$G_{glob}$ [W/m$^2$]')
    ax1.grid()
    ax1.legend(loc='upper left')
    
    # save I_tot as .txt file
    np.savetxt(name, I_tot)
    
    suma_I = np.sum(I_tot)
    suma_G = np.sum(G_sol_g)
    print('SUM I_tot =', np.round(suma_I/1000, 2), 'kWh/m2', ',', 'SUM G_glob  =', np.round(suma_G/1000, 2), 'kWh/m2')
    loop_end = datetime.datetime.now()
    print('Time of function calculation', loop_end-loop_start)

    
"""
Functions from internet

"""

def is_leap_year(year):
    """ if year is a leap year return True
        else return False """
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0

def day_in_year(Y,M,D):
    """ given year, month, day return day of year
        Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7 
        you have to be carefull, function also calculates day number of 35th of January for example"""
    if is_leap_year(Y):
        K = 1
    else:
        K = 2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N

def N_days(Y):
    if is_leap_year(Y):
        K = 366
    else:
        K = 365
    return K
