import os
import json
import shutil
import argparse
import subprocess
import pandas as pd
from glob import glob
from tqdm import tqdm
from youtube_dl import YoutubeDL

def main(args):
    
    data_type = args.data_type
    workspace = args.workspace
    start_index = args.start_index
    stop_index = args.stop_index
    
    # Create directories
    data_path = os.path.join(workspace, 'dataset', data_type)
    os.makedirs(data_path, exist_ok=True)
    
    csv_path = os.path.join(workspace, 'metadata', 'groundtruth_strong_label_{data_type}_set.csv'.format(data_type))
    df = pd.read_csv(csv_path, header=None)
    distinct_files = df['segment_id'].unique()
    distinct_set = [(s[:s.rindex('_')], int(s[s.rindex('_')+1:])/1000) for s in distinct_files]
    print(len(distinct_set))
    total_num = len(distinct_set)
    root = os.getcwd()
    print('CWD:', root)
    
    # Extract videos from YouTube
    audio_downloader = YoutubeDL({'format':'bestaudio'})
    error_count = 0
    error_files = []
    print(f'start_index = {start_index}')
    print(f'stop_index = {stop_index}')
    if stop_index is not None:
        subset = distinct_set[start_index:stop_index]
    else:
        subset = distinct_set[start_index:]
        
    ctr = 0
    mode = 'w'
    if start_index > 0:
        mode = 'a'
    with open(f"{data_type}_download_{start_index}-{stop_index}.txt", mode) as f:
        for file in tqdm(subset):
            try:
                f.flush()
                ctr+=1
                f.write(f'Starting file: {start_index+ctr}\n')
                URL = 'https://www.youtube.com/watch?v={}'.format(file[0])
                command = "ffmpeg -ss " + str(int(file[-1])) + " -t 10 -i $(youtube-dl -f 'bestaudio' -g " + URL + ") -ar " + str(16000) + " -- \"" + '{}/{}_{}'.format(data_path, file[0], int(file[-1])) + ".wav\""
                print('COMMAND:', command)
                os.system((command))
            except Exception:
                error_count += 1
                error_files.append(file[0])
                f.write("Couldn\'t download the audio\n")
        start_index+=ctr
        f.write(f'Number of files that could not be downloaded: {error_count}\n')
        f.write(f"Could not download the following files:\n{error_files}\n")
        f.write(f'Next start_index = {start_index}\n')
        if(start_index == total_num):
            f.write(f'All files have been downloaded for {data_type} set.\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract AudioSet')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--data_type', type=str, required=True, choices=['training', 'testing'])
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--stop_index', type=int)
    args = parser.parse_args()

    main(args)
