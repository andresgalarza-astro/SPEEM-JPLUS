''' Personal Package that contains the most useful routines defined as special
    functions during my PhD in Astronomy
    Created by: Andres Galarza
    Date: June 10th, 2019
    Version 0.0
'''

# Packages required to write function HourAngle_to_Deg
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
# Package required to plot data
import matplotlib.pyplot as plt
#
import os
# Package required to read Datasets
import pandas as pd
#
from  itertools import combinations


Reference = {'Full' : ['TILE_ID', 'NUMBER', 'RA', 'DEC','FWHM_WORLD'],
             'ID' : ['TILE_ID', 'NUMBER'],
             'Coords' : ['RA', 'DEC']}

Filters = {'JPLUS' : ['uJAVA', 'J0378', 'J0395', 'J0410', 'J0430', 'gSDSS', 'J0515',
                      'rSDSS','J0660','iSDSS','J0861', 'zSDSS'],
           'JPLUSnarrow' : ['uJAVA', 'J0378', 'J0395', 'J0410', 'J0430', 'J0515',
                            'J0660', 'J0861'],
           'JPLUS2' : ['uJAVA', 'F378', 'F395', 'F410', 'F430', 'gSDSS', 'F515',
                      'rSDSS','F660','iSDSS','F861', 'zSDSS'],
           'SPLUS' : ['uJAVA', 'F378', 'F395', 'F410', 'F430', 'G', 'F515',
                      'R','F660','I','F861', 'Z'],
           'SDSS' : ['gSDSS', 'rSDSS', 'iSDSS', 'zSDSS'],
           'WISE' : ['W1', 'W2', 'W3', 'W4', 'J', 'H', 'K'],
           '2MASS' : ['J', 'H', 'K'],
           'GAIA' : ['G', 'BP', 'RP'],
           'Ax' : ['Ax_uJAVA', 'Ax_J0378', 'Ax_J0395', 'Ax_J0410', 'Ax_J0430', 'Ax_gSDSS',
                   'Ax_J0515','Ax_rSDSS','Ax_J0660','Ax_iSDSS', 'Ax_J0861', 'Ax_zSDSS'],
           'zpt' : ['zpt_uJAVA', 'zpt_J0378', 'zpt_J0395', 'zpt_J0410', 'zpt_J0430', 'zpt_gSDSS',
                    'zpt_J0515','zpt_rSDSS','zpt_J0660','zpt_iSDSS', 'zpt_J0861', 'zpt_zSDSS']}

Stellar_Params = {'All' : ['Teff_GAIA', 'SPECTYPE_SUBCLASS',
                           'TEFF_ADOP', 'LOGG_ADOP', 'FEH_ADOP',
                           'TEFF_SPEC', 'LOGG_SPEC', 'FEH_SPEC'],
                  'stellar' : ['Teff_GAIA', 'subclass',
                               'teff', 'logg', 'feh'],
                  'lamost_dr6_mrs' : ['teff_lasp', 'teff_cnn', 'teff_lasp_err', 
                                      'logg_lasp', 'logg_cnn', 'logg_lasp_err',
                                      'feh_lasp', 'feh_cnn', 'feh_lasp_err',
                                      'c_fe', 'n_fe', 'o_fe', 'mg_fe', 'al_fe',
                                      'si_fe', 's_fe', 'ca_fe', 'ti_fe', 'cr_fe']}

'''
class JPLUS:

    usefulcolumns = ['TILE_ID', 'NUMBER', 'RA', 'DEC', 'uJAVA', 'J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861', 'gSDSS', 'rSDSS', 'iSDSS', 'zSDSS', 'CLASS_STAR', 'morph_prob_star']

    extracolumns = ['Teff_SPHINX', 'ERROR_Teff_SPHINX', 'parallax', 'parallax_error','BP_RP', 'EXT_BP_RP', 'Teff_GAIA']

    WISEcolumns = ['W1mag', 'W2mag', 'W3mag', 'W4mag', 'Jmag', 'Hmag', 'Kmag']

    MorphTypeColumns = ['CLASS', 'SUBCLASS']



    def __init__(self):
        self.data = usefulcolumns
        print("Modulo JPLUS cargado exitosamente")
    
    def add_other_columns(self):
        self.data = usefulcolumns + extracolumns
'''
def Path():
    return os.getcwd()

def ChangeFolder(CurrentPath,NewPath):
    
    Path = CurrentPath
    New_Path = NewPath
    os.chdir(New_Path)
    print('Last path: ', Path)
    #print('The root path is: ', directorio_raiz)
    print('New path is:', New_Path)
    '''directorio_raiz = '/home/usuario/Documents/'
    if (ActualPath != NewPath):
        directorio_nuevo = directorio_raiz + NewPath
        os.chdir(directorio_nuevo)
        print('The current path: ', directorio_actual)
        print('The root path is: ', directorio_raiz)
        print('The new path is:', directorio_nuevo)
    else:
        directorio_nuevo = directorio_raiz + NewPath
        os.chdir(directorio_nuevo)
        print('The current path: ', directorio_actual)
        print('The root path is: ', directorio_raiz)
        print('The new path is:', directorio_nuevo)'''
    
def Read_Dataset(filename,header='infer',column_names=None,columns=None,**kwargs):
    
    chunksize = 100000
    chunks = []
    for chunk in pd.read_csv(filename, chunksize = chunksize, header=header, 
                             names=column_names, usecols=columns, **kwargs):
        chunks.append(chunk)
        
    output = pd.concat(chunks, axis = 0) 

    return(output)
#----------------------------- Functions for Data Processing ----------------------------------------------------
def correctMags(df, filters=Filters['JPLUS'],Correction = Filters['Ax']):
    
    df_corrected = pd.concat([df[a].sub(df[b]) for a, b in 
                              zip(list(filters),list(Correction))],axis=1,keys=filters).round(3)
  
    #for a, b in zip(list(filters),list(Correction)):
    #    print(a + '-' + b)
        
    #return None
    return df_corrected
    
def add_id_columns(df1, df2, columns=Reference['Full']):
    
    df = df1.join(df2[columns])
    
    return df

def order_columns(df,reference=Reference['Full'],filters=Filters['JPLUS']):
    
    return df[reference + filters]

def add_wise_filters(df1, df2, columns=Filters['WISE']):
    
    df = df1.join(df2[columns])
    
    return df

def add_stellar_params(df1, df2, columns=Stellar_Params['stellar']):
    
    df = df1.join(df2[columns])
    
    return df

def remove_contaminants(df):
    
    df = df.replace(np.nan, 'OK', regex=True)
    dfcp = df[df.subclass.str.contains('R|S|W') == False].copy()
    #df_clean = df[df.SPECTYPE_SUBCLASS.str.contains('R|S|W') == False].copy()
    
    return dfcp

def remove_bad_values(df, Column = list(['feh','teff','logg'])):
    
    df_clean = df[(df[Column[0]].between(-5.0,1.0)) &
              (df[Column[1]].between(3500,10000)) &
              (df[Column[2]].between(0.0,5.0))].copy()
    
    return df_clean

def remove_spec_subclasses(df):
    
    #df['SPECTYPE_SUBCLASS'] = df.SPECTYPE_SUBCLASS.map(lambda x: x.rstrip(r'(\d)'))
    #df = df.assign(SPECTYPE_SUBCLASS = df.SPECTYPE_SUBCLASS.str.replace(r'(\d)', '')) # Replace Original Column
    df = df.assign(SPECTYPE_CLASS = df.subclass.str.replace(r'(\d)', '')) # Replace Original Column
    
    return df

def apply_pipeline(df):
    
    Working_df = (df.pipe(correctMags)            # MW Correction
                  .pipe(add_id_columns, df)       # Add TILE_ID, NUMBER, RA, DEC
                  .pipe(order_columns)            # Reorder columns
                  .pipe(add_wise_filters, df)     # Add WISE and 2MASS Filters
                  .pipe(add_stellar_params, df)   # Add TEFF_ADOP, FEH_ADOP, LOGG_ADOP to Dataframe
                  .pipe(remove_contaminants)      # Remove QSOs, BROADLINES, STARFORMING
                  .pipe(remove_bad_values)        # Remove -9999.0 values
                  .pipe(remove_spec_subclasses)   # Remove numerical reference i.e A0 -> A, F5 -> F, ...
                 )
    
    return Working_df
#----------------------------------------------------------------------
# Transform a list of coordinates from Hour-Angle to Deg and write a new file with the conversion
# http://docs.astropy.org/en/stable/coordinates/?fbclid=IwAR0dV17eI2HhYFS3afknsc1r0wWFnrW4M7e_2k0_5Mkp2GeyrXwTmK9kvbQ
def HourAngle_to_Deg(Filename1, Filename2):
    
    File1 = open(Filename1, "r")    # Name of the file to be readed
    File2 = open(Filename2, "w+")   # Name of the file to be written
    
    for Line in File1:
        c = SkyCoord(Line, unit=(u.hourangle, u.deg))           # Coordinates conversion from Hour-Angle to Dec
        line = "%8.5f\t"% (c.ra.deg) + "%8.5f\n"% (c.dec.deg)   # Setting the format type to be written on File2
        File2.write(line)

    File1.close()
    File2.close()

# Create JPLUS Colors Dataset
def createColors(Dataset, filters = Filters['JPLUS']):
    # Creating all the possible color combinations based on a set of Photometric Filters

    df = Dataset[filters].copy()

    cc = list(combinations(df.columns,2))

    df = pd.concat([df[c[0]].sub(df[c[1]]) for c in cc], axis=1, keys=cc)
    df.columns = df.columns.map(''.join)
    
    return df

def createColors2(Dataset):
    # Creating all the possible color combinations based on a set of Photometric Filters

    cc = list(combinations(Dataset.columns,2))

    df = pd.concat([Dataset[c[0]].sub(Dataset[c[1]]) for c in cc], axis=1, keys=cc)
    df.columns = df.columns.map(''.join)
    
    return df

def colorsComb(Dataset, filters = Filters['JPLUS']):
    # Creating all the possible color combinations based on a set of Photometric Filters

    df = Dataset[filters].copy()

    cc = list(combinations(df.columns,4))

    df = pd.concat([(df[c[0]].sub(df[c[1]])).sub(df[c[2]].sub(df[c[3]])) for c in cc], axis=1, keys=cc)
    df.columns = df.columns.map(''.join)
    
    return df

def colorsComb2(Dataset):
    # Creating all the possible color combinations based on a set of Photometric Filters

    cc = list(combinations(Dataset.columns,4))

    df = pd.concat([(Dataset[c[0]].sub(Dataset[c[1]])).sub(Dataset[c[2]].sub(Dataset[c[3]])) for c in cc], axis=1, keys=cc)
    df.columns = df.columns.map(''.join)
    
    return df

# Adjacent Subplots Function.
def AdjacentSubplots(X,Y,minX,maxX,label='label',title='title',xlabel='X_Label',ylabel='Y_Label',loc=4, frameon=True, handlelength=2, markerscale=1):
    
    x = np.linspace(minX,maxX,50)
    x1 = np.linspace(minX,maxX,50)
    y1 = np.zeros(len(x))
    
    fig, axs = plt.subplots(2, 1, sharex=True,gridspec_kw={'height_ratios': [3, 1]})
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.set_figwidth(15)
    fig.set_figheight(15)

    # Plot each graph, and manually set the y tick values
    axs[0].plot(X,Y,'ro',label=label)
    axs[0].plot(x,x,'b--')
    #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[0].grid(True)
    axs[0].set_xlim(minX,maxX)
    axs[0].set_ylim(minX,maxX)
    axs[0].set_ylabel(ylabel,fontsize=28)
    axs[0].set_title(title,fontsize=32)
    axs[0].legend(fontsize=30,loc=loc, frameon=frameon, handlelength=handlelength, markerscale=markerscale)
    axs[0].tick_params('y',labelsize=30)
    
    axs[1].plot(X,Y-X,'ro')
    axs[1].plot(x1,y1,'b--')
    #axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    axs[1].grid(True)
    axs[1].set_ylim(-np.max(np.abs(Y-X)), np.max(np.abs(Y-X)))
    axs[1].set_xlabel(xlabel,fontsize=28)
    axs[1].set_ylabel('DIFFERENCE',fontsize=28)
    axs[1].tick_params('both',labelsize=30)

    plt.tight_layout()
    plt.show()
'''
# Plot Adjacent Subplots. The dataset must be readed and it only works for metallicity
def AdjacentSubplots_FeH(X,Y):
    
    x = np.arange(-3.5,-1.7,0.01)
    x1 = np.arange(-3.5,-1.7,0.01)
    y1 = np.zeros(len(x))
    
    fig, axs = plt.subplots(2, 1, sharex=True,gridspec_kw={'height_ratios': [3, 1]})
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.set_figwidth(15)
    fig.set_figheight(15)

    # X = Dataset.FeH
    # Y = Dataset.FEH_SPEC

    # Plot each graph, and manually set the y tick values
    axs[0].plot(X,Y,'ro')
    axs[0].plot(x,x,'b--')
    #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[0].grid(True)
    axs[0].set_xlim(-3.5,-1.7)
    axs[0].set_ylim(-3.5,-1.7)
    axs[0].set_ylabel('[Fe/H]_SPEC',fontsize=18)
    axs[0].set_title('Random Forest Estimations VS [Fe/H] Spectroscopic for SDSS',fontsize=22)

    axs[1].plot(X,Y-X,'ro')
    axs[1].plot(x1,y1,'b--')
    #axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    axs[1].grid(True)
    axs[1].set_ylim(-0.5, 0.5)
    axs[1].set_xlabel('[Fe/H]_ADOP',fontsize=18)
    axs[1].set_ylabel('DIFFERENCE',fontsize=18)

    plt.show()

# Adjacent Subplots Function. The dataset must be readed and it only works for Teff
def AdjacentSubplots_Teff(X,Y,minX,maxX,label='label',title='title',xlabel='X_Label',ylabel='Y_Label'):
    
    x = np.linspace(minX,maxX,50)
    x1 = np.linspace(minX,maxX,50)
    y1 = np.zeros(len(x))
    
    fig, axs = plt.subplots(2, 1, sharex=True,gridspec_kw={'height_ratios': [3, 1]})
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)
    fig.set_figwidth(15)
    fig.set_figheight(15)

    # Plot each graph, and manually set the y tick values
    axs[0].plot(X,Y,'ro',label=label)
    axs[0].plot(x,x,'b--')
    #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[0].grid(True)
    axs[0].set_xlim(minX,maxX)
    axs[0].set_ylim(minX,maxX)
    axs[0].set_ylabel(ylabel,fontsize=28)
    axs[0].set_title(title,fontsize=32)
    axs[0].legend(fontsize=30)
    axs[0].tick_params('y',labelsize=18)
    
    axs[1].plot(X,Y-X,'ro')
    axs[1].plot(x1,y1,'b--')
    #axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    axs[1].grid(True)
    axs[1].set_ylim(-np.max(np.abs(Y-X)), np.max(np.abs(Y-X)))
    axs[1].set_xlabel(xlabel,fontsize=28)
    axs[1].set_ylabel('DIFFERENCE',fontsize=28)
    axs[1].tick_params('both',labelsize=18)
    
    plt.tight_layout()
    plt.show()
'''
# The next step is to create a better function tha combines the two AdjacentSubplots definitions
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/ganged_plots.html#sphx-glr-gallery-subplots-axes-and-figures-ganged-plots-py