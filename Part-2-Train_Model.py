import pandas as pd
import numpy as np
import pickle
import os
import multiprocessing as mp
import statsmodels.api as sm
from argparse import ArgumentParser, RawTextHelpFormatter
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import utils
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


# 参数
def __init__():
    parser = ArgumentParser(
        formatter_class = RawTextHelpFormatter,
        description = 'Build and train models.'
    )
    parser.add_argument(
        '-v', '--version', action = 'version', version = '%(prog)s 1.0.0'
    )
    parser.add_argument(
        '-pro', '--proteins-matrix', type = str, required = True, metavar = '<str>',
        help = 'Path to the proteins matrix file.'
    )
    parser.add_argument(
        '-marker', '--marker-list', type = str, required = True, metavar = '<str>',
        help = 'Path to the brain region marker proteins list file.'
    )
    parser.add_argument(
        '-r', '--pearson-r', type = float, default = 0.05, required = False, metavar = '<float>',
        help = 'Filter models with pearsonr value.\nDefault: 0.05'
    )
    parser.add_argument(
        '-nbootstrap', '--bootstrap-times', type = int, default = 100, required = False, metavar = '<int>',
        help = 'Bootstrap training times.\nDefault: 100'
    )
    parser.add_argument(
        '-k', '--k-fold', default = 5, type = int, required = False, metavar = '<int>',
        help = 'K-fold training.\nDefault: 5'
    )
    parser.add_argument(
        '-t', '--threads-use', default = 2, type = int, required = False, metavar = '<int>',
        help = 'Threads numbers to use.\nDefault: 2'
    )
    parser.add_argument(
        '-o', '--output', type = str, required = True, metavar = '<str>',
        help = 'Path to the output file.'
    )
    return parser.parse_args()

def LassoTrain(i, npx, label, region, k):
    reg = Lasso(max_iter=50000)
    param_grid = {'alpha': np.array(range(1, 1000, 1)) / 1000}
    grid_search = GridSearchCV(reg, param_grid, cv=k, scoring='r2', n_jobs=1)
    gs = grid_search.fit(npx, label)
    params = gs.param_grid['alpha']
    scores = gs.cv_results_['mean_test_score']
    # select alpha with performance of 0.95
    scale_score = scores / np.max(scores)
    selected_idx = np.where(scale_score >= 0.95)[0][-1]
    alpha = params[selected_idx]
    lasso_model = Lasso(alpha=alpha, max_iter=50000)
    lasso_model.fit(npx, label)
    save_dir = out_dir + f"GTEx_Age_models/{region}/"
    save_fname = f"lasso_bs{i}_aging_model.pkl"
    savefp = os.path.join(save_dir, save_fname)
    with open(savefp, 'wb') as f:
        pickle.dump(lasso_model, f)
    return lasso_model

def parallel_LassoTrain(input_list):
    i, re_npx, re_label, region, k = input_list
    print(f"第{i}次Bootstrap采样\n")
    return LassoTrain(i, re_npx, re_label, region, k)

def BootStrap(samples, npx, label, region, n_bootstrap, k, t):
    # 创建进程池
    pool = mp.Pool(t)
    input_list = []
    for i in range(n_bootstrap):
        resamples = utils.resample(samples)
        re_npx = npx.loc[resamples]
        re_label = label.loc[resamples]
        input_list.append((i, re_npx, re_label, region, k))
    results = [pool.apply_async(parallel_LassoTrain, args=(input_arg,)) for input_arg in input_list]
    models = [res.get() for res in results]
    pool.close()
    pool.join()
    return models

# 训练模型
def train(df_pro, df_markers, r, n, k, t, output):
    selected_regions = []
    all_regions = df_markers['Group'].unique()
    for region in all_regions:  
        try:
            save_dir = os.path.join(output, f"GTEx_Age_models/{region}/")
            os.makedirs(save_dir, exist_ok=True)
            region_markers = df_markers[df_markers['Group'] == region]
            specific_proteins = region_markers['Gene'].tolist()
            filtered_columns = df_pro.filter(items=specific_proteins)
            data = pd.concat([df_pro.iloc[:, :3], filtered_columns], axis=1)
            X = data.drop(['Age', 'ID'], axis=1)
            y = data['Age']
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
            train_indices = X_train.index
            data['Flag'] = 'Test' 
            data.loc[train_indices, 'Flag'] = 'Train'
            data.to_csv(f'{save_dir}/train_data.tsv', sep='\t', index=False)
            # 模型训练
            idx = X_train.index
            t_models = BootStrap(idx, X_train, y_train, region, n, k, t)
            # 计算pearsonr
            pre_ages = np.mean(np.array([m.predict(X_test) for m in t_models]), axis=0)
            pre_data = pd.DataFrame({
                'ID': data.loc[X_test.index, 'ID'].values,
                'Pre_age': pre_ages,
                'Real_age': data.loc[X_test.index, 'Age'].values
            })
            pre_data.to_csv(save_dir + "pre_ages.tsv", sep='\t', index=False)
            r_value, p_value = pearsonr(pre_data['Real_age'], pre_data['Pre_age'])
            mse = mean_squared_error(pre_data['Real_age'], pre_data['Pre_age'])
            rmse = np.sqrt(mse)
            if r_value > 0.05:
                selected_regions.append(region)
                print(f"Region {region} selected with r={r_value:.2f}, p={p_value:.1e}, rmse={rmse:.2f}")
            else:
                print(f"Region {region} rejected (r={r_value:.2f}), p={p_value:.1e}, rmse={rmse:.2f}")
                
        except Exception as e:
            print(f"Error processing {region}: {str(e)}")
            continue
        
    with open(os.path.join(output, 'selected_regions.txt'), 'w') as f:
        f.write('\n'.join(selected_regions))
        

if __name__ == "__main__":
    parameters = __init__()
    train(parameters.proteins_matrix, parameters.marker_list, parameters.pearson_r, parameters.bootstrap_times,
          parameters.k_fold, parameters.threads_use, parameters.output)
    