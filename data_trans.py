import numpy as np
import pandas as pd

filename = {"data1": "GSE71858.npy",  #
            "data2": "GSE60361.npy",  #
            "data3": "GSE71585.npy",
            "data4": "GSE62270.npy",  #
            "data5": "GSE48968.npy",  #
            "data6": "GSE52529.npy",  #
            "data7": "GSE77564.npy",
            "data8": "GSE78779.npy",  #
            "data9": "GSE10247.npy",  #
            "data10": "GSE69405.npy"}

#for _, f_name in filename.items():
#    if f_name[-3:] == "txt":
#        _file = pd.read_table(f_name, sep='\s+', index_col=False)
#    else:
#        _file = pd.read_csv(f_name, index_col=False)
#    _file_t = np.array(_file)
#    _file_t = np.delete(_file_t, 0, 1)
#    file_t = np.around(_file_t.astype(np.float), decimals=7)
#    np.savetxt(f_name[0:8] + '.txt', _file_t)
    #np.save(f_name[0:8] + ".npy", _file_t)
    #print np.loadtxt(f_name[0:7] + '.txt').shape

for f_key, f_name in filename.items():
    data = np.load("./Data/"+f_name)
    print f_key, data.shape
