import json
import os

from utils.sensor import Sensor, run_feature_extraction

BLENDER_DATA_ROOT = r'.\eyelink_data_converter\blender_data'
DESIGN = 'grid15'

if __name__ == '__main__':
    #%%load test data
    design_path = 'designs\{}.json'.format(DESIGN)
    print(design_path)
    with open(design_path, 'r') as f:
        sensor = Sensor(json.load(f))

    data_exp = {}
    for exp in os.listdir(BLENDER_DATA_ROOT):
        exp = 3
        exp_path = '{root}\{exp}\{exp}.csv'.format(root=BLENDER_DATA_ROOT, exp=exp)
        fpath_sr = exp_path.replace('.csv', '_%s.csv'%DESIGN)
        print(fpath_sr)
        print(exp_path)
        data_exp[exp] = run_feature_extraction(exp_path, sensor, fpath_sr=fpath_sr)
        break