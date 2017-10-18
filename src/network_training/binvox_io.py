'''
binvox.py
Updated: 8/28/17

README:

Script defines methods used to read and write numpy arrays from binvox files.

'''
import numpy as np

def read_binvox(file_path):
    '''
    Method load binvox file as numpy array.

    '''
    with open(file_path, 'rb') as fp:
        line = fp.readline().strip()
        if not line.startswith(b'#binvox'): raise IOError('Not a binvox file')
        dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
        translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
        scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
        fp.readline()
        raw_data = np.frombuffer(fp.read(), dtype=np.uint8)

    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)

    return data

def write_binvox(file_path, np_array):
    '''
    Method write numpy array as binvox file.

    '''
    with open(file_path, 'wb') as fp:
        dims = np_array.shape
        scale = 1.0
        translate = 0
        voxels_flat = np_array.flatten()
        fp.write('#binvox 1\n'.encode('ascii'))
        fp.write(('dim '+' '.join(map(str, dims))+'\n').encode('ascii'))
        fp.write(('translate 0 0 0\n').encode('ascii'))
        fp.write(('scale 1.0\n').encode('ascii'))
        fp.write('data\n'.encode('ascii'))

        # keep a sort of state machine for writing run length encoding
        state = voxels_flat[0]
        ctr = 0
        for c in voxels_flat:
            if c==state:
                ctr += 1
                # if ctr hits max, dump
                if ctr==255:
                    fp.write(int(state).to_bytes(1,'big'))
                    fp.write(int(ctr).to_bytes(1,'big'))
                    ctr = 0
            else:
                # if switch state, dump
                fp.write(int(state).to_bytes(1,'big'))
                fp.write(int(ctr).to_bytes(1,'big'))
                state = c
                ctr = 1

        # flush out remainders
        if ctr > 0:
            fp.write(int(state).to_bytes(1,'big'))
            fp.write(int(ctr).to_bytes(1,'big'))
