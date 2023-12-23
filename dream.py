from utils.io import load_pickle, write_pickle
import numpy as np

def main():
    output = load_pickle('output/dream1.pickle')
    base = {}
    for key, value in output.items(): 
        _, code = key.split('-')
        code = 'DREAM-' + code
        base[code] = value 

    for i in range(2, 6):
        output = load_pickle(f'output/dream{i}.pickle')
        for key, value in output.items():
            _, code = key.split('-')
            code = 'DREAM-' + code
            for method, result in value.items():
                for metric, vlist in result.items():
                    if np.isnan(vlist[0]):
                        vlist[0] = 0
                    base[code][method][metric] += vlist

    write_pickle(base, 'output/dream.pickle')

main()