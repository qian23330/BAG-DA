import pandas as pd
import pickle
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats


# 参数
def __init__():
    parser = ArgumentParser(
        formatter_class = RawTextHelpFormatter,
        description = 'Predict Brain age gap.'
    )
    parser.add_argument(
        '-v', '--version', action = 'version', version = '%(prog)s 1.0.0'
    )
    parser.add_argument(
        '-regions', '--models-name', type = str, required = True, metavar = '<str>',
        help = 'Path to the models filtered name file.'
    )
    parser.add_argument(
        '-path2', '--part2-path', type = str, required = True, metavar = '<str>',
        help = 'Path to the part2 result directory.'
    )
    parser.add_argument(
        '-o', '--output', type = str, required = True, metavar = '<str>',
        help = 'Path to the output file.'
    )
    return parser.parse_args()

# 计算BAG
def calculate(regions, path, output):
    with open(regions) as f:
        model_names = [line.strip() for line in f]
    for region in model_names:
        save_dir = os.path.join(path, f"GTEx_Age_models/{region}/")
        pre_data = pd.read_csv(save_dir + "pre_ages.tsv", sep='\t')
        # 使用 LOWESS 回归拟合预测年龄和真实年龄的关系
        lowess_result = lowess(
            pre_data['Pre_age'],  
            pre_data['Real_age'],  
            frac=2/3,             
            it=3,                
            return_sorted=False
        )
        pre_data['Lowess_age'] = lowess_result
        # 计算年龄差（BAG）：预测年龄 - LOWESS 回归估计的年龄
        pre_data['BAG'] = pre_data['Pre_age'] - pre_data['Lowess_age']
        pre_data['Zscored(BAG)'] = stats.zscore(pre_data['BAG'])
        output_dir = f"{output}/{region}"
        os.makedirs(output_dir, exist_ok=True)
        pre_data.to_csv(output_dir + 'test.healthy_BAG.tsv', sep='\t', index=False)
        

if __name__ == "__main__":
    parameters = __init__()
    calculate(parameters.models_name, parameters.part2_path, parameters.output)
    
