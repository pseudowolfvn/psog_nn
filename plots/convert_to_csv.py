import pandas as pd
from sklearn.externals import joblib

model_name = 'finetune_{}_({}, {}, {}, {})'
subjs_4_suffix = '_[\'{}\', \'{}\', \'{}\', \'{}\'].pkl'
subjs_3_suffix = '_[\'{}\', \'{}\', \'{}\'].pkl'

data = pd.DataFrame(columns=['subj', 'arch', 'appr', 'rep', 'acc'])

def subj_to_suffix(subj):
    div4 = ((subj - 1) // 4) * 4
    div3 = (subj // 3) * 3
    if subj < 5:
        return subjs_4_suffix.format(1, 2, 3, 4)
    elif subj < 10:
        return subjs_4_suffix.format(5, 6, 7, 8)
    elif subj < 14:
        return subjs_4_suffix.format(10, 11, 12, 13)
    elif subj < 18:
        return subjs_4_suffix.format(14, 15, 16, 17)
    elif subj < 21:
        return subjs_3_suffix.format(18, 19, 20)
    else:
        return subjs_3_suffix.format(21, 22, 23)

for subj in range(1, 23 + 1):
    if subj == 9:
        continue
    mlp_data = joblib.load(
        model_name.format('mlp', 0, 0, 6, 20) +
        subj_to_suffix(subj)
    ) 
    
    cnn_data = joblib.load(
        model_name.format('cnn', 4, 4, 4, 20) +
        subj_to_suffix(subj)
    )

    for ind,acc in enumerate(mlp_data[str(subj)]['ft']['data']):
        data = data.append({
            'subj': subj,
            'arch': 1,
            'appr': 1,
            'rep': ind + 1,
            'acc': acc
        }, ignore_index=True)

    for ind,acc in enumerate(mlp_data[str(subj)]['scr']['data']):
        data = data.append({
            'subj': subj,
            'arch': 1,
            'appr': 2,
            'rep': ind + 1,
            'acc': acc
        }, ignore_index=True)    

    for ind,acc in enumerate(cnn_data[str(subj)]['ft_fc']['data']):
        data = data.append({
            'subj': subj,
            'arch': 2,
            'appr': 1,
            'rep': ind + 1,
            'acc': acc
        }, ignore_index=True)    

    for ind,acc in enumerate(cnn_data[str(subj)]['scr']['data']):
        data = data.append({
            'subj': subj,
            'arch': 2,
            'appr': 2,
            'rep': ind + 1,
            'acc': acc
        }, ignore_index=True)

data['subj'] = data['subj'].astype(int)
data['arch'] = data['arch'].astype(int)
data['appr'] = data['appr'].astype(int)
data['rep'] = data['rep'].astype(int)

data.to_csv('for_analysis_lp.csv', sep=',', index=False)