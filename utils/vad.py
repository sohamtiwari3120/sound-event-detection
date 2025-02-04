"""
SUMMARY:  SUMMARY:  Acoustic event detection with voice activity detection (VAD)
AUTHOR:   Qiuqiang Kong
Created:  2016.06.15
Modified: -
--------------------------------------
"""
import numpy as np


def activity_detection(x, thres, low_thres=None, n_smooth=1, n_salt=0):
    """Activity detection. 
    
    Args:
      x: array
      thres:    float, threshold
      low_thres:float, second lower threshold
      n_smooth: integar, number of frames to smooth. 
      n_salt:   integar, number of frames equal or shorter this value will be 
                removed. Set this value to 0 means do not use delete_salt_noise. 
    
    Return: list of [bgn, fin]
    """
    
    #print(x, '\n')
    locts = np.where(x > thres)[0]
    #print(locts, '\n')
    
    # Find pairs of [bgn, fin]
    bgn_fin_pairs = find_bgn_fin_pairs(locts)
    #print(bgn_fin_pairs, '\n')
    
    # Second threshold
    if low_thres is not None:
        bgn_fin_pairs = activity_detection_with_second_thres(
            x, bgn_fin_pairs, low_thres)
    
    # Smooth
    bgn_fin_pairs = smooth(bgn_fin_pairs, n_smooth)
    
    # Remove salt noise
    bgn_fin_pairs = remove_salt_noise(bgn_fin_pairs, n_salt)
    
    #print(np.array(bgn_fin_pairs).shape)
    return bgn_fin_pairs

def activity_detection_binary(x, overlap_value, sample_duration, thres, low_thres=None, n_smooth=1, n_salt=0):
    """Activity detection.

    Args:
      x: array
      thres:    float, threshold
      low_thres:float, second lower threshold
      n_smooth: integar, number of frames to smooth.
      n_salt:   integar, number of frames equal or shorter this value will be
                removed. Set this value to 0 means do not use delete_salt_noise.

    Return: list of [bgn, fin]
    """
    
    all_pairs = []
    overlap_interval = int(100 * overlap_value)
    interval = (sample_duration * 100) - overlap_interval
    
    #print(x, '\n')
    #print(x.shape, '\n')
    all_locts = []
    for i in range(0, x.shape[0]-overlap_interval, overlap_interval):
        if i < interval:
            num_overlaps = i//overlap_interval + 1
        elif i >= x.shape[0] - interval:
            num_overlaps = ((x.shape[0] - i) // overlap_interval) + 1
        else:
            num_overlaps = sample_duration
        
        #print(num_overlaps, '\n')
        #print(x[i:i+overlap_interval], '\n')
        locts = np.where(x[i:i+overlap_interval] >= (num_overlaps))[0]
        if len(locts) != 0:
            #print('BEFORE', locts, '\n')
            locts = [x+i for x in locts]
            #print('AFTER', locts, '\n')
            all_locts.extend(locts)
        else:
            continue

    # Find pairs of [bgn, fin]
    bgn_fin_pairs = find_bgn_fin_pairs(all_locts)
    #print(bgn_fin_pairs, '\n')

    # Second threshold
    if low_thres is not None:
        bgn_fin_pairs = activity_detection_with_second_thres(
            x, bgn_fin_pairs, low_thres)

    # Smooth
    bgn_fin_pairs = smooth(bgn_fin_pairs, n_smooth)

    # Remove salt noise
    bgn_fin_pairs = remove_salt_noise(bgn_fin_pairs, n_salt)
    
    #all_pairs.extend(bgn_fin_pairs)
    
    #print(np.array(all_pairs).shape)

    return bgn_fin_pairs
        
def find_bgn_fin_pairs(locts):
    """Find pairs of [bgn, fin] from loctation array
    """
    if len(locts)==0:
        return []
        
    else:
        bgns = [locts[0]]
        fins = []
        for i1 in range(1, len(locts)):
            if locts[i1] - locts[i1 - 1] > 1:
                fins.append(locts[i1 - 1] + 1)
                bgns.append(locts[i1] + 1)
        fins.append(locts[-1])
            
    assert len(bgns)==len(fins)
    
    lists = []
    
    for i1 in range(len(bgns)):
        lists.append([bgns[i1], fins[i1]])
        
    return lists


def activity_detection_with_second_thres(x, bgn_fin_pairs, thres):
    """Double threshold method. 
    """

    new_bgn_fin_pairs = []
    
    for [bgn, fin] in bgn_fin_pairs:
        
        while(bgn != -1):
            if x[bgn] < thres:
                break                
            bgn -= 1
        
        while(fin != len(x)):
            if x[fin] < thres:
                break
            fin += 1
                
        new_bgn_fin_pairs.append([bgn + 1, fin])
        
    new_bgn_fin_pairs = smooth(new_bgn_fin_pairs, n_smooth=1)
        
    return new_bgn_fin_pairs


def smooth(bgn_fin_pairs, n_smooth):
    """Smooth the [bgn, fin] pairs. 
    """

    new_bgn_fin_pairs = []
    
    if len(bgn_fin_pairs) == 0:
        return []
    
    [mem_bgn, fin] = bgn_fin_pairs[0]
    
    for n in range(1, len(bgn_fin_pairs)):
        
        [pre_bgn, pre_fin] = bgn_fin_pairs[n - 1]
        [bgn, fin] = bgn_fin_pairs[n]
        
        if bgn - pre_fin <= n_smooth:
            pass
            
        else:
            new_bgn_fin_pairs.append([mem_bgn, pre_fin])
            mem_bgn = bgn
            
    new_bgn_fin_pairs.append([mem_bgn, fin])
    
    return new_bgn_fin_pairs
    
    
def remove_salt_noise(bgn_fin_pairs, n_salt):
    """Remove salt noise
    """

    new_bgn_fin_pairs = []
    
    for [bgn, fin] in bgn_fin_pairs:
        if fin - bgn <= n_salt:
            pass
            
        else:
            new_bgn_fin_pairs.append([bgn, fin])
            
    return new_bgn_fin_pairs
