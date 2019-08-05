#!/usr/bin/python3

# Automatic sound file alignment tool
# Author: Jakub Paliesek
# License: MIT

import numpy as np
import sys, getopt
import scipy.io.wavfile as wf

def usage():
    print('Usage:')
    print('-s [number]  manually set first sample of cross correlation (default is 48000)')
    print('-l [number]  manually set length of window for cross correlation samples (default is 10000)')
    print('-c [number]  manually set channel number for cross correlation samples (default is 0)')
    print('-f           full cross correlation of entire tracks (takes very long time)')

def errquit(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print('use --help for more info', file=sys.stderr)
    sys.exit(1)

start_sample = 48000
nsamples = 10000
channel = 0
full = False

try:
    opts, args = getopt.getopt(sys.argv[1:], 's:l:c:fh', ['help'])
except getopt.GetoptError as err:
    # print help information and exit:
    errquit(err)  # will print something like "option -a not recognized"
for o, a in opts:
    if o == "-s":
        start_sample = int(a)
    elif o == '-l':
        nsamples = int(a)
    elif o == '-c':
        channel = int(a)
    elif o == '-f':
        full = True
    elif o in ('-h', '--help'):
        usage()
        sys.exit()
    else:
        errquit('Unknown option.')


base_fs, base_wav_pcm = wf.read(args[0])
base_wav = base_wav_pcm / np.max(np.abs(base_wav_pcm))  # normalized samples of first wav
if len(base_wav.shape) == 1:  # make mono track to have 2 dimensions as well
    base_wav = np.expand_dims(base_wav, 1)
base_xcorr_samples = np.transpose(base_wav)[channel]
if full:
    base_xcorr_samples = base_xcorr_samples[start_sample:start_sample+nsamples]

shifts = [(0, base_wav, args[0][:-4])]

for f in args[1:]:
    fs, wav_pcm = wf.read(f)
    if (fs != base_fs):
        errquit('Sample rates do not match.')
    wav = wav_pcm / np.max(np.abs(wav_pcm))
    if len(wav.shape) == 1:  # make mono track to have 2 dimensions as well
        wav = np.expand_dims(wav, 1)
    xcorr_samples = np.transpose(wav)[channel]
    if full:
        xcorr_samples = xcorr_samples[start_sample:start_sample+nsamples]
    xcorr = np.correlate(base_xcorr_samples, xcorr_samples, 'full')
    shift = np.argmax(xcorr) - nsamples + 1  # lag of f relative to base wav
    shifts.append((shift, wav, f[:-4]))
    
min_shift = min(shifts, key=lambda x: x[0])[0]

for w in shifts:
    sound = np.pad(w[1], ((w[0]+np.abs(min_shift), 0), (0, 0)))
    print('Shifting', w[2], 'by', w[0], 'samples')
    wf.write(w[2]+'.align.wav', base_fs, sound)

