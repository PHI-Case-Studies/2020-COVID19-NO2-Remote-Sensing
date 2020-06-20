
import pandas as pd
import geopandas as gpd
from os import listdir, rename, path, remove, mkdir
from os.path import isfile, join, getsize, exists
from netCDF4 import Dataset
import time
import numpy as np
import sys
import calendar
import datetime as dt
import re
from socket import timeout
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import urllib
import requests, json
from requests.auth import HTTPBasicAuth
import xmltodict
import pandas as pd
import geopandas as gpd
from shapely import wkt

'''
Module: s5p_no2_tools.py
============================================================================================
Disclaimer: The code is for demonstration purposes only. Users are responsible to check for 
    accuracy and revise to fit their objective.

nc_to_df adapted from read_tropomi_no2_and_dump_ascii.py by Justin Roberts-Pierel, 2015 
    of NASA ARSET
    Purpose of original code: To print all SDS from an TROPOMI file
    Modified by Vikalp Mishra & Pawan Gupta, May 10 2019 to read TROPOMI data
    Modified by Herman Tolentino, May 8, 2020 to as steps 1 and 2 of pipeline to process 
        TROPOMI NO2 data
============================================================================================
'''
def create_project(project_name='default'):
    """
    Purpose: Create project subfolder
    Parameters:
        project_name (string): subfolder to create
    Returns:
        project_name (string)
    """
    if not exists(project_name):
        try:
            mkdir(project_name)
        except OSError:
            print (f"Creation of the directory {project_name} failed.")
            return ''
        else:
            print (f"Successfully created the directory {project_name}.")
            return project_name
    else:
        print(f"Directory {project_name} exists.")
        return project_name
    
def get_place_boundingbox(place_gdf, buffer):
    """
    Purpose: Determine the bounding box of a place GeoDataFrame (polygon)
    Parameters:
        place_gdf (GeoDataFrame): place GeoDataFrame to get bounding box for. This should be
            a Level 0 polygon from GADM.
        buffer (int): buffer in miles to add to place geometry
    Returns:
        bbox (GeoDataFrame): GeoDataFrame containing the bounding box as geometry
    """
    bbox = gpd.GeoDataFrame(geometry=place_gdf['geometry'].buffer(buffer).envelope, crs=place_gdf.crs)\
    .reset_index()

    return bbox
    
def filter_swath_set(swath_set_gdf, place_gdf):
    """
    Purpose: Reduce the number of swaths based on place constraint.
    Parameters:
        swath_set_gdf (GeoDataFrame): A GeoDataFrame output from sending a query to the 
                                      sentinel_api_query() function. This GeoDataFrame contains
                                      a geometry for the swath. This geometry should contain
                                      the place_gdf geometry below.
        place_gdf (GeoDataFrame):     A GeoDataFrame that should be contained within the
                                      swath_set_gdf geometry.
    Returns:
        filtered_gdf (GeoDataFrame): A subset of swath_set_gdf representing geometries that
                                     contain the place_gdf geometry.
    """
    filtered_gdf = gpd.sjoin(swath_set_gdf, place_gdf, how='right', op='contains').reset_index()
    filtered_gdf = filtered_gdf.drop(columns=['level_0','index_left','index'])
    return filtered_gdf

def geometry_to_wkt(place_gdf):
    """
    Purpose: Sentinel 5P Data Access hub requires a constraining polygon filter to 
             retrieve a smaller number of satellite image swaths.
    
    Parameters:
        place_gdf (GeoDataFrame): Target place for obtaining NO2 levels. Place GDF should be
                                  a simple, GADM level 0 polygon.

    Returns:
        wkt_string (string): string containing polygon vertices in WKT format
    """
    # get geometry convex hull and simplify
    geometry = place_gdf.reset_index()['geometry'].convex_hull.simplify(tolerance=0.05)
    
    # convert to WKT
    wkt_string = wkt.dumps(geometry[0])
    
    return wkt_string

def date_from_week(weekstring='2019-W01'):
    d = weekstring
    r = dt.datetime.strptime(d + '-1', "%Y-W%W-%w")
    return r

def add_days(start, numDays=1):
    end = start + dt.timedelta(days=numDays)
    startDate = start.strftime("%Y-%m-%d")
    endDate = end.strftime("%Y-%m-%d")
    return [startDate, endDate]

def nc_to_df(ncfile):
    """
    Purpose: This converts a TROPOMI NO2 file to a Pandas DataFrame.

    Notes:
    This was adapted from read_tropomi_no2_and_dump_ascii.py by Justin Roberts-Pierel, 2015 
    of NASA ARSET.
    
    Parameters:
    ncfile: NetCD4 file from Copernicus S5P Open Data Hub

    Returns:
    dataframe: data from NetCD4 file
    """
    
    try:
        f = open(ncfile, 'r')
    except OSError:
        print('cannot open', ncfile)
    
    
    df = pd.DataFrame()
        
    # read the data
    if 'NO2___' in ncfile and 'S5P' in ncfile:
        tic = time.perf_counter()
        FILENAME = ncfile
        print(ncfile+' is a TROPOMI NO2 file.')

        #this is how you access the data tree in an NetCD4 file
        SDS_NAME='nitrogendioxide_tropospheric_column'
        file = Dataset(ncfile,'r')
        grp='PRODUCT' 
        ds=file
        grp='PRODUCT'
        lat= ds.groups[grp].variables['latitude'][0][:][:]
        lon= ds.groups[grp].variables['longitude'][0][:][:]
        data= ds.groups[grp].variables[SDS_NAME]      
        #get necessary attributes 
        fv=data._FillValue

        #get scan time and turn it into a vector
        scan_time= ds.groups[grp].variables['time_utc']
        # scan_time=geolocation['Time'][:].ravel()

        year = np.zeros(lat.shape)
        mth = np.zeros(lat.shape)
        doy = np.zeros(lat.shape)
        hr = np.zeros(lat.shape)
        mn = np.zeros(lat.shape)
        sec = np.zeros(lat.shape)
        strdatetime = np.zeros(lat.shape)

        for i in range(0,lat.shape[0]):
            t = scan_time[0][i].split('.')[0]
            t1 = t.replace('T',' ')
            t2 = dt.datetime.strptime(t,'%Y-%m-%dT%H:%M:%S')
            t3 = t2.strftime("%s")
            #y = t2.year
            #m = t2.month
            #d = t2.day
            #h = t2.hour
            #m = t2.minute
            #s = t2.second

            #year[i][:] = y
            #mth[i][:] = m
            #doy[i][:] = d
            #hr[i][:] = h
            #mn[i][:] = m
            #sec[i][:] = s
            strdatetime[i][:] = t3
        vlist = list(file[grp].variables.keys())
        #df['Year'] = year.ravel()
        #df['Month'] = mth.ravel()
        #df['Day'] = doy.ravel()
        #df['Hour'] = hr.ravel()
        #df['Minute'] = mn.ravel()
        #df['Second'] = sec.ravel()
        df['UnixTimestamp'] = strdatetime.ravel()
        df['DateTime'] = pd.to_datetime(df['UnixTimestamp'], unit='s')
        df[['Date','Time']] = df['DateTime'].astype(str).str.split(' ',expand=True)
        # This for loop saves all of the SDS in the dictionary at the top 
        #    (dependent on file type) to the array (with titles)
        for i in range(0,len(vlist)):
            SDS_NAME=vlist[(i)] # The name of the sds to read
            #get current SDS data, or exit program if the SDS is not found in the file
            #try:
            sds=ds.groups[grp].variables[SDS_NAME]
            if len(sds.shape) == 3:
                print(SDS_NAME,sds.shape)
                # get attributes for current SDS
                if 'qa' in SDS_NAME:
                    scale=sds.scale_factor
                else: scale = 1.0
                fv=sds._FillValue

                # get SDS data as a vector
                data=sds[:].ravel()
                # The next few lines change fill value/missing value to NaN so 
                #     that we can multiply valid values by the scale factor, 
                #     then back to fill values for saving
                data=data.astype(float)
                data=(data)*scale
                data[np.isnan(data)]=fv
                data[data==float(fv)]=np.nan

                df[SDS_NAME] = data
        toc = time.perf_counter()
        elapsed_time = toc-tic
        print("Processed "+ncfile+" in "+str(elapsed_time/60)+" minutes")
    else:
        raise NameError('Not a TROPOMI NO2 file name.')

    return df

def polygon_filter(input_df, filter_gdf):
    """
    Purpose: This removes records from the TROPOMI NO2 Pandas DataFrame that
        is not found within the filter polygons

    Parameters:
    input_df: Pandas DataFrame containing NO2 data coming from nc_to_df() 
    filter_gdf: GeoPandas GeoDataFrame containing geometries to constrain
        NO2 records. Be sure to create the spatial index for filter_gdf to
        speed up sjoin operation. You can do this by calling
        filter_gdf.sindex before feeding filter_gdf into:
        polygon_filter(input_df=input_df, filter_gdf=filter_gdf)

    Returns:
    geodataframe: Filtered GeoPandas GeoDataFrame
    """
    print('To speed up the polygon_filter() operation, did you create the spatial index for filter_gdf?')
    tic = time.perf_counter()
    output_gdf = pd.DataFrame()
    print('Processing input dataframe...')
    crs = filter_gdf.crs
    # 1. Convert input_df to gdf
    gdf1 = gpd.GeoDataFrame(input_df, geometry=gpd.points_from_xy(input_df.longitude, input_df.latitude),crs=crs)
    print('Original NO2 DataFrame length:', len(gdf1))
    # 2. Find out intersection between African Countries GeoDataFrames (geometry) and
    #       NO2 GeoDataFrames using Geopandas sjoin (as GeoDataFrame, gdf2)
    sjoin_gdf = gpd.sjoin(gdf1, filter_gdf, how='inner', op='intersects')
    print('Filtered NO2 GeoDataFrame length:', len(sjoin_gdf))
    toc = time.perf_counter()
    elapsed_time = toc-tic
    print("Processed NO2 DataFrame sjoin in "+str(elapsed_time/60)+" minutes")

    return sjoin_gdf

def get_filename_from_cd(cd):
    """
    Purpose: Get filename from content-disposition (cd)
    
    Parameters:
        cd (string): content-disposition

    Returns:
        fname[0] (string): filename
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]

def download_nc_file(url, auth, savedir, logging, refresh):
    """
    Purpose: Download NetCD4 files from URL
    
    Parameters:
    url: string, download url obtained from Sentinel 5P Open Data Hub search results
    auth: dictionary of 'user' and 'password'
    savedir: string, path to save NetCD4 files
    logging: boolean, turn logging on or off
    refresh: boolean, overwrite previously downloaded files (helps save time if False)

    Returns:
    filename: string filename of NetCD4 file
    """
    user = auth['user']
    password = auth['password']
    filename = 'temp.nc'
    logfile = 'nc.log'
    try:
        refresh=refresh
        headers = {
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
        }
        tic = time.perf_counter()
        with open(savedir+'/'+filename, 'wb') as f:
            response = requests.get(url, auth=(user, password), stream=True, headers=headers)
            filename0 = get_filename_from_cd(response.headers.get('content-disposition')).replace('"','')
            if path.exists(savedir+'/'+filename0):
                print('File '+filename0+' exists.')
                if refresh==False:
                    filename0_size = getsize(savedir+'/'+filename0)
                    print('Filename size:', filename0_size,' bytes')
                    if filename0_size > 0:
                        remove(savedir+'/'+filename)
                        return filename0
            print('Downloading '+filename0+'...')
            total = response.headers.get('content-length')

            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)
                for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50*downloaded/total)
                    sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                    sys.stdout.flush()
        if logging==True:
            with open(logfile, 'a+') as l:
                # Move read cursor to the start of file.
                l.seek(0)
                # If file is not empty then append '\n'
                data = l.read(100)
                if len(data) > 0 :
                    l.write("\n")
                # Append text at the end of file
                l.write(filename0)
        sys.stdout.write('\n')
        rename(savedir+'/'+filename, savedir+'/'+filename0)
        toc = time.perf_counter()
        elapsed_time = toc-tic
        print('Success: Saved '+filename0+' to '+savedir+'.')
        print('Download time, seconds: '+str(elapsed_time))
        delays = [7, 4, 6, 2, 10, 15, 19, 23]
        delay = np.random.choice(delays)
        print('Delaying for '+str(delay)+' seconds...')
        time.sleep(delay)
        return filename0
    except:
        print('Something went wrong.')
        
def batch_download_nc_files(auth, savedir, url_file, numfiles, logging, refresh):
    """
    Purpose: For batch downloading nc files from the Copernicus S5P Data Access Hub
    
    Parameters:
        auth (dict): authentication dictionary, {'user':'myusername', 'password':'mypassword'}
        savedir (string): directory used to save NetCD4  files
        url_file (string): file containing NetCD4 download URLS
        numfiles (int)
        logging (bool)
        refresh (bool)
    Returns:
        df (DataFrame)
    """
    savedir=savedir
    url_file = url_file
    df = pd.read_csv(url_file)
    df['ncfile'] = ''
    counter=0
    numfiles = numfiles
    if numfiles > 0:
        print('Processing '+str(numfiles)+((' files...') if numfiles > 1 else ' file...'))
    else:
        print('Processing '+str(len(df))+((' files...') if len(df) > 1 else ' file...'))
    for index, row in df.iterrows():
        url = row['URL']
        filename = download_nc_file(url=url, 
                                    auth=auth, 
                                    savedir='NO2-NetCD4',
                                    logging=logging,
                                    refresh=refresh)
        if filename:
            print('Downloaded file no:',counter+1)
            print(row['URL'], filename)
            df.loc[index,'ncfile'] = filename
        counter += 1
        if numfiles > 0:
            if counter >= numfiles:
                return df
        delays = [7, 4, 6, 2, 10, 15, 19, 23]
        delay = np.random.choice(delays)
        print('Delaying for '+str(delay)+' seconds...')
        time.sleep(delay)
    return df
        
def harpconvert(input_filename, input_dir, output_dir):
    """
    Purpose: This converts a TROPOMI NO2 NetCD4 file to a HDF5 (Level 3 Analysis).
    
    Notes: harp convert command adapted from Google Earth Engine site:
            https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2
           This assumes harp has been installed using: conda install -c stcorp 'harp'.
    
    Parameters:
    input_filename: NetCD4 file from Copernicus S5P Open Data Hub
    input_dir: Directory where input_filename is located
    output_dir: Directory where hdf5 file output is saved

    Returns:
    dictionary: NetCD4 filename, processing time in seconds, stdout, stderr

    """
    tic = time.perf_counter()
    input_filename = input_filename
    output_filename = input_filename[:-3]+'.h5'
    input_path = input_dir+'/'+input_filename
    output_path = output_dir+'/'+output_filename
    cmd = "harpconvert --format hdf5 --hdf5-compression 9 \
        -a 'tropospheric_NO2_column_number_density_validity>50;derive(datetime_stop {time})' \
        %s %s" % (input_path, output_path)
    process = subprocess.Popen(['bash','-c', cmd],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    filesize = getsize(output_path)
    fs = f'{filesize:,}'
    stdout, stderr = process.communicate()
    toc = time.perf_counter()
    elapsed_time = toc-tic
    status_dict = {}
    status_dict['input_filename'] = input_filename
    status_dict['output_filesize'] = fs
    status_dict['elapsed_time'] = elapsed_time
    status_dict['stdout'] = stdout
    status_dict['stderr'] = stderr
    return status_dict

def batch_assemble_filtered_pickles(filtered_dir):
    tic = time.perf_counter()
    filtered_dir = filtered_dir
    pickle_files = [f for f in listdir(filtered_dir) if isfile(join(filtered_dir, f))]

    full_df = pd.DataFrame()
    df_len = []
    for i in range(0, len(pickle_files)):
        print(pickle_files[i])
        df = pd.read_pickle('NO2-Filtered/'+pickle_files[i])
        df_len.append(len(df))
        full_df = pd.concat([df,full_df],axis=0)
    toc = time.perf_counter()
    elapsed_time = toc-tic
    print('Assembly time, minutes: '+str(elapsed_time/60))
 
    return full_df

def plot_maps(iso3, filter_gdf, filelist, colormap, sensing_date):
    """
    Parameters:
        iso3 (string): 3-letter ISO code for country
        filter_gdf (GeoDataFrame): 
        filelist (list): 
        colormap (string): colormap for colorbar
        sensing_date (string):
    Returns:
        Matplotlib Plot
    """
    crs = filter_gdf.crs
    gdf_sjoin_list = []
    country_gdf = filter_gdf[filter_gdf['iso3']==iso3]
    country_name = list(country_gdf.loc[country_gdf['iso3']==iso3,'name'].unique())[0]
    for file in filelist:
        gdf_sjoin = pd.read_pickle(file).set_geometry('geometry')
        gdf_sjoin.crs = crs
        gdf_sjoin.drop(columns=['index_right'], inplace=True)
        gdf_countries_sjoin = gpd.sjoin(gdf_sjoin, country_gdf, how='inner', op='intersects')
        if len(gdf_countries_sjoin) > 0:
            gdf_sjoin_list.append(gdf_countries_sjoin)
    swaths = len(gdf_sjoin_list)
    print('Using '+str(swaths)+' swaths.')
    
    qa_vmax = []
    qa_vmin = []
    column='qa_value'
    for gdf in gdf_sjoin_list:
        qa_vmax.append(gdf[column].max())
        qa_vmin.append(gdf[column].min())
    vmax_qa = max(qa_vmax)
    vmin_qa = min(qa_vmin)
    
    no2_vmax = []
    no2_vmin = []
    column='nitrogendioxide_tropospheric_column'
    for gdf in gdf_sjoin_list:
        no2_vmax.append(gdf[column].max())
        no2_vmin.append(gdf[column].min())
    vmax_no2 = max(no2_vmax)
    vmin_no2 = min(no2_vmin)

    colormap=colormap
    fig, ax= plt.subplots(sharex=True, sharey=True, figsize=(8,6), constrained_layout=True)
    for gdf in gdf_sjoin_list:
        gdf.plot(cmap=plt.get_cmap(colormap), ax=ax,
                 column='qa_value', vmin=vmin_qa, vmax=vmax_qa, alpha=0.9)
    country_gdf.plot(ax=ax, alpha=0.1, color='None')

    # add colorbar
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin_qa, vmax=vmax_qa))
    sm._A = []
    fig.colorbar(sm, cax=cax)
    plt.suptitle('Tropospheric NO2, QA Value ('+country_name+', '+sensing_date+')')

    fig, ax= plt.subplots(sharex=True, sharey=True, figsize=(8,6), constrained_layout=True)
    for gdf in gdf_sjoin_list:
        gdf.plot(cmap=plt.get_cmap(colormap), ax=ax,
                 column='nitrogendioxide_tropospheric_column_precision_kernel', 
                 vmin=vmin_no2, vmax=vmax_no2, alpha=0.9)
    country_gdf.plot(ax=ax, alpha=0.1, color='None',legend=False)

    # add colorbar
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin_no2, vmax=vmax_no2))
    sm._A = []
    fig.colorbar(sm, cax=cax)
    plt.suptitle('Tropospheric NO2, Tropospheric Column, moles/m2 ('+country_name+', '+sensing_date+')')
    

def sentinel_api_query(query_dict,silentmode=False):
    """
    Author: Herman Tolentino MD
    Parameters:
        query_dict: dictionary of API query variables (see below)
    query_dict = {
        'startDate': startDate,             # YYYY-MM-DD
        'endDate': endDate,                 # YYYY-MM-DD
        'productType': productType,         # L2__NO2___
        'processingLevel': processingLevel, # L2
        'platformName': platformName,       # Sentinel-5
        'processingMode': processingMode,   # Offline
        'polygon': polygon,                 # Shapefile in WKT format
        'startPage': startPage,             # 0
        'numRows': numRows,                 # 50-100 (usual number for continent level)
        'dhus_url': dhus_url,               # Data Access Hub URL for S5P
        'username': username,               # s5pguest
        'password': password                # s5pguest
    }
    """
    delays = [7, 4, 6, 2, 10, 15, 19, 23]
    delay = np.random.choice(delays)
    print('Delaying for '+str(delay)+' seconds...') if silentmode==False else None
    time.sleep(delay)
    # STEP 1: Unpack query_dict and check data (TODO)
    polygon = query_dict['polygon']
    startDate = query_dict['startDate']
    endDate = query_dict['endDate']
    platformName = query_dict['platformName']
    productType = query_dict['productType']
    processingLevel = query_dict['processingLevel']
    processingMode = query_dict['processingMode']
    dhus_url = query_dict['dhus_url']
    startPage = query_dict['startPage']
    numRows = query_dict['numRows']
    username = query_dict['username']
    password = query_dict['password']
    # check data here before proceeding (TODO)
    
    # STEP 2: Construct query string for API
    query=f'( footprint:"Intersects({polygon})") AND \
        ( beginPosition:[{startDate}T00:00:00.000Z TO {endDate}T23:59:59.999Z] AND \
        endPosition:[{startDate}T00:00:00.000Z TO {endDate}T23:59:59.999Z] ) \
        AND ( (platformname:{platformName} AND producttype:{productType} \
        AND processinglevel:{processingLevel} AND processingmode:{processingMode}))'.replace('  ',' ')
    print('query:', query) if silentmode==False else None
    
    quoted = urllib.parse.quote_plus(query)
    print('quoted:', quoted) if silentmode==False else None
    
    # STEP 3: Send query to API and get convert XML response to dictionary
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    response = requests.get(f'{dhus_url}dhus/search?q={quoted}&start={startPage}&rows={numRows}', \
                     auth=HTTPBasicAuth(username, password),\
                    headers=headers)
    print('headers:', response.headers) if silentmode==False else None
    print('cookie:', response.headers['Set-Cookie']) if silentmode==False else None
    # Convert XML response to dictionary
    my_dict = xmltodict.parse(response.text)
    results = int(my_dict['feed']['opensearch:totalResults'])
    if results > numRows:
        print('==========================================================================================')
        print('WARNING: Returned results '+str(results)+' exceeds requested number of rows ('+str(numRows)+'). ')
        print('==========================================================================================')
    itemsperpage = int(my_dict['feed']['opensearch:itemsPerPage'])
    print('search term:', my_dict['feed']['opensearch:Query']['@searchTerms']) if silentmode==False else None

    # STEP 4: Store dictionary items in Pandas DataFrame
    record_dict = {}
    study_df = pd.DataFrame()
    for item in my_dict['feed']['entry']:
        print(item)
        #print(item['title'])
        #print('summary:', item['summary'])
        #print('date:', item['date'][0])
        print('=========') if silentmode==False else None
        gmldict = xmltodict.parse(item['str'][1]['#text'])
        crs = gmldict['gml:Polygon']['@srsName'].split('#')
        record_dict['ingestiondate'] = item['date'][0]['#text']
        record_dict['beginposition'] = item['date'][1]['#text']
        record_dict['endposition'] = item['date'][2]['#text']
        record_dict['orbitnumber'] = item['int']['#text']
        record_dict['filename'] = item['str'][0]['#text']
        record_dict['crs'] = 'epsg:'+crs[1]
        record_dict['format'] = item['str'][2]['#text']
        record_dict['identifier'] = item['str'][3]['#text']
        record_dict['instrumentname'] = item['str'][4]['#text']
        record_dict['instrumentshortname'] = item['str'][5]['#text']
        record_dict['footprint'] = item['str'][6]['#text']
        record_dict['mission'] = item['str'][7]['#text']
        record_dict['platformname'] = item['str'][8]['#text']
        record_dict['platformserialidentifier'] = item['str'][9]['#text']
        record_dict['platformshortname'] = item['str'][10]['#text']
        record_dict['processinglevel'] = item['str'][11]['#text']
        record_dict['processingmode'] = item['str'][12]['#text']
        record_dict['processingmodeabbreviation'] = item['str'][13]['#text']
        record_dict['processorversion'] = item['str'][14]['#text']
        record_dict['producttype'] = item['str'][15]['#text']
        record_dict['producttypedescription'] = item['str'][16]['#text']
        record_dict['revisionnumber'] = item['str'][17]['#text']
        record_dict['size'] = item['str'][18]['#text']
        record_dict['uuid'] = item['str'][19]['#text']
        record_dict['downloadurl'] = item['link'][0]['@href']
        print(record_dict) if silentmode==False else None
        study_df = study_df.append(record_dict, ignore_index=True)
        print('\n\n') if silentmode==False else None
        
    # STEP 5: Convert DataFrame to GeoDataFrame and return GeoDataFrame
    study_df['geometry'] = study_df['footprint']
    study_df['geometry'] = study_df['geometry'].apply(wkt.loads)
    study_df['beginposition'] = study_df['beginposition'].replace('T',' ').astype('datetime64[ns]')
    study_df['endposition'] = study_df['endposition'].replace('T',' ').astype('datetime64[ns]')
    study_df['startdate'] = study_df['beginposition'].dt.strftime('%Y-%m-%d')
    study_df['enddate'] = study_df['endposition'].dt.strftime('%Y-%m-%d')
    crs = {'init': 'epsg:4326'}
    study_gdf = gpd.GeoDataFrame(study_df, crs=crs, geometry=study_df.geometry)
    
    return study_gdf

def show_colormap():

    cmaps = [('Perceptually Uniform Sequential', [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
             ('Sequential', [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('Sequential (2)', [
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                'hot', 'afmhot', 'gist_heat', 'copper']),
             ('Diverging', [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
             ('Qualitative', [
                'Pastel1', 'Pastel2', 'Paired', 'Accent',
                'Dark2', 'Set1', 'Set2', 'Set3',
                'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('Miscellaneous', [
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(gradient, cmap_category, cmap_list)

    plt.show()
    return True

def plot_color_gradients(gradient, cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-.01, .5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()
