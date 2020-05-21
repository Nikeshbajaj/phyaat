'''API to download dataset.

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, six, time, collections
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlopen, urlretrieve
import tarfile

from pathlib import Path

try:
    import queue
except ImportError:
    import Queue as queue
    

def progressBar(i,L,title=''):
    rol =['|','\\','-','/']
    pf = 100*i/float(L)
    bar = ']['+'#'*int(pf/2)+''+' '*(50-int(pf/2))+'] '+title
    pp = str(int(pf))
    pp = ' '*(3-len(pp))+pp+'%['+rol[int(pf)%len(rol)]
    print(pp+bar,end='\r', flush=True)
    if pf==100: print('')
        
        
def download_data(baseDir='../Phyaat', subject=1,verbose=1,overwrite=False):
    """Download Phyaat dataset.

    # Arguments
        path: loacal path where you want to store the data
        relative to `../phyaat/dataset`).
        subject: int, Dataset of of subject will be downloaded (default=1)
               : -1 for downloading dataset of all the subjects (1-25)
        
    # Path
        Path of the dataset.

    # Raises
        ValueError: in case `subject` is not int or -1<subject>25
    """
    DataPath = 'https://github.com/Nikeshbajaj/PhyaatDataset/raw/master/Signals/'
    
    try:
        datadir1 = os.path.join(baseDir, 'phyaat_dataset')
        datadir = os.path.join(datadir1, 'Signals')
        path    = Path(datadir)
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print('NOTE:: Path :  \"'+ baseDir +'\" is not accessible. Creating  \"phyaat\" in  "/tmp\" directory for dataset' )
        baseDir = os.path.join('/tmp','phyaat')
        datadir1 = os.path.join(baseDir, 'phyaat_dataset')
        datadir = os.path.join(datadir1, 'Signals')
        path    = Path(datadir)
        path.mkdir(parents=True, exist_ok=True)

    #print(datadir)
    
    assert isinstance(subject, int)
    assert (subject>=-1 and subject<=25)
    assert subject!=0
    
    if subject==-1:
        if verbose: print('Downloading data from', DataPath)
        for i in range(1,26):
            ifpath = _download_1S(subject=i,datadir=datadir,DataPath=DataPath,
                     ExtractAndRemove=True,verbose=verbose,overwrite=overwrite)
    else:
        ifpath = _download_1S(subject=subject,datadir=datadir,DataPath=DataPath,
                  ExtractAndRemove=True,verbose=verbose,overwrite=overwrite)
    
    # Download additional files
    origin1  = DataPath.replace('Signals/','README.md')
    fpath1   = datadir1 + '/README.md'
    
    if not os.path.exists(fpath1):
        fpath1  = _download_sFile(origin1,fpath1,bar=False)
    #origin2  = DataPath + 'Demographics.csv'
    return datadir
    
def _download_1S(subject,datadir,DataPath,ExtractAndRemove=True,verbose=1,overwrite=False):
    
    fname = 'S'+str(subject) +'.tar.gz'
    
    fpath = os.path.join(datadir, fname)
    fpathD = os.path.join(datadir, 'S'+str(subject))
    fpathS = os.path.join(fpathD, 'S'+str(subject)+'_Signals.csv')
    fpathT = os.path.join(fpathD, 'S'+str(subject)+'_Textscore.csv')
    if os.path.exists(fpath) and not(overwrite):
        if verbose: 
            print('File already exist in ...')
            print('  - ',fpathD)
            print('To overwrite the download.. set "overwrite=True"')
        
    elif os.path.exists(fpathS) and os.path.exists(fpathT) and not(overwrite):
        if verbose:
            print('Signal file and Score file already exist in directory...')
            print('  - ',datadir)
            print('To overwrite the download.. set "overwrite=True"')
    else:
        origin  = DataPath + fname
        if verbose: print('Downloading data from', origin)
        ifpath  = _download_sFile(origin,fpath)
    
        if ExtractAndRemove:
            if verbose: print('\n Extracting .tar.gz...')
            tar = tarfile.open(ifpath)
            tar.extractall(datadir)
            tar.close()
            os.remove(ifpath)
            if verbose: print(".tar.gz File Removed!")
        return ifpath

def _download_sFile(origin,fpath,bar=True):
    sub = origin.split('/')[-1].split('.')[0]
    class ProgressTracker(object):
        progbar = None

    def dl_progress(count, block_size, total_size):
        #print(count, block_size, count * block_size, total_size)
        if bar:
            progressBar(count * block_size,total_size,title=sub)
        else:
            pass

    error_msg = 'URL fetch failure on {} : {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath, dl_progress)
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if os.path.exists(fpath):
            os.remove(fpath)
        raise
    ProgressTracker.progbar = None
    
    return fpath