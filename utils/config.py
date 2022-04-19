sample_rate = 16000
audio_duration = 10     # Audio clips have durations of 10 seconds
audio_samples = sample_rate * audio_duration

# Hyper-parameters follow [1] Kong, Q., Cao, Y., Iqbal, T., Wang,
# Y., Wang, W. and Plumbley, M. D., 2019. PANNs: Large-Scale Pretrained Audio
# Neural Networks for Audio Pattern Recognition. arXiv preprint arXiv:1912.10211.
mel_bins = 64
fmin = 50
fmax = sample_rate/2
window_size = 1024 * sample_rate//32000
hop_size = 320 * sample_rate//32000
frames_per_second = sample_rate // hop_size
time_steps = frames_per_second * audio_duration
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

# ID of classes
# ids = ['/m/028ght', '/m/0lyf6', '/m/07rkbfh', '/m/053hz1', '/m/0ytgt', '/m/0l15bq', '/m/01h8n0', '/m/01b_21', '/m/03qtwd', '/m/0463cq4', '/m/02zsn', '/m/01j3sz', '/m/05zppz', '/m/06h7j', '/m/03qc9zr', '/m/07p6fty', '/m/01hsr_', '/m/07pbtc8', '/m/02rtxlg', '/m/05x_td', '/m/02mfyn', '/m/03j1ly', '/m/014zdl', '/m/032s66','/m/03kmc9']

# Name of classes
labels = ['Motor vehicle (road)', 'Explosion', 'Gunshot, gunfire', 'Screaming', 'Siren', 'Breaking', 'Crowd', 'Crying, sobbing'] 
siren_fine_labels = ["Emergency vehicle", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Police car (siren)", "Civil defense siren"]
broad_lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}

classes_num = len(labels)
def lb_to_idx(label):
    if label in labels:
        return broad_lb_to_idx[label]
    elif label in siren_fine_labels:
        return broad_lb_to_idx['Siren']
    else:
        raise Exception(f'Invalid label -{label} provided. Should be one of {labels} or {siren_fine_labels}')
        
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
