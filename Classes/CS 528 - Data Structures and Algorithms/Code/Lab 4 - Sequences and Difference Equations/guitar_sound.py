# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:07:02 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""
from scitools.sound import max_amplitude
import numpy as np
import wave, sys, os, subprocess


def write(data, filename, channels=1, sample_width=2, sample_rate=44100):
    """
    Taken from scitools.sound module
    Writes the array data to the specified file.
    The array data type can be arbitrary as it will be
    converted to numpy.int16 in this function.
    """
    
    ofile = wave.open(filename, 'w')
    ofile.setnchannels(channels)
    ofile.setsampwidth(sample_width)
    ofile.setframerate(sample_rate)
    ofile.writeframesraw(data.astype(np.int16).tostring())
    ofile.close()
    
    
#had to copy (and slightly modify) scitools.sound.play
def play(file, sample_rate=44100, player=None):
    """
    Play a file with array data.  (The array is first written to file
    using the write function so the data type can be arbitrary.)  The
    player is chosen by the programs 'open' on Mac and 'start' on
    Windows. On Linux, try different open programs for various
    distributions. If keyword argument `player` is given, only this
    spesific command is run.
    """
    #write(data, tmpfile, sample_rate = sample_rate)

    if player:
        msg = 'Unable to open sound file with %s' %player
        if sys.platform[:3] == 'win':
            status = os.system('%s %s' %(player, file))
        else:
            status, output = subprocess.getstatusoutput('%s %s' %(player, file))
            msg += '\nError message:\n%s' %output
        if status != 0:
            raise OSError(msg)
        return

    if sys.platform[:5] == 'linux':
        open_commands = ['gnome-open', 'kmfclient exec', 'exo-open', 'xdg-open', 'open']
        for cmd in open_commands:
            status, output = subprocess.getstatusoutput('%s %s' %(cmd, file))
            if status == 0:
                break
        if status != 0:
            raise OSError('Unable to open sound file, try to set player'\
                              ' keyword argument.')

    elif sys.platform == 'darwin':
        try:
            subprocess.getstatusoutput('afplay %s' %file)
        except:
            subprocess.getstatusoutput('open %s' % file)
    else:
        # assume windows
        os.system('start %s' %file)


#given start values of x_0, x_1, ..., x_p:
#x_n=1/2(x_{n-p}+x_{n-p-1}), n=p+1, ..., N
#makes a guitar sound (apparently)
def solve(x, p):
    N = len(x)
    x_n=x
    for i in range(p+1, N, 1):
        #print("i={0}, p={1}, i-p={2} x[i-p]={3}".format(i, p, i-p, x[i-p]))
        x_n[i] = (0.5*(x[i-p]+x[i-p-1]))    
    return x_n


#sample rate
r = 44100

#440Hz tone
p = int(r/440)

#x_0 = 1, x_1 = x_2 = x_3 = ... = x_N, N = 3*r
x1=[1] + [0 for _ in range(3*r-1)]
x1 = solve(x1, p)
x1 = [x*max_amplitude for x in x1]

write(np.array(x1), 'Atone.wav', sample_rate=r)

#392Hz tone
p=int(r/392)

x2 = np.random.uniform(-1, 1, 2*r)
x2 = solve(x2, p)
x2 = [x*max_amplitude for x in x2]

write(np.array(x2), 'Gtone.wav', sample_rate=r)

data = x1 + x2

write(np.array(data), 'Chord.wav', sample_rate=r)
play('Chord.wav')
