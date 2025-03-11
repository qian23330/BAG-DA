import pandas as pd
import os
import pickle
import numpy as np
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from datetime import datetime
from lifelines import CoxPHFitter


# 参数
def __init__():
    parser = ArgumentParser(
        formatter_class = RawTextHelpFormatter,
        description = 'Calculate BAG-Disease risk associations.'
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
        '-path3', '--part3-path', type = str, required = True, metavar = '<str>',
        help = 'Path to the part3 result directory.'
    )
    parser.add_argument(
        '-disease', '--disease-pro', type = str, required = True, metavar = '<str>',
        help = 'Path to the disease proteins matrix directory.'
    )
    parser.add_argument(
        '-p', '--p-value', default = 0.05, type = float, required = False, metavar = '<float>',
        help = 'P-value filtered for associations.\nDefault: 0.05'
    )
    parser.add_argument(
        '-o', '--output', type = str, required = True, metavar = '<str>',
        help = 'Path to the output file.'
    )
    return parser.parse_args()

# 既往疾病BAG计算
def prevalent_dis_BAG(regions, path2, disease_pro):
    with open(regions) as f:
        model_names = [line.strip() for line in f]
    disease = 'prevalent'
    base_dir = path2
    disease_dir = f'{disease_pro}/{disease}_protein_datasets/'
    disease_annotation_path = f"{disease_pro}/UKB-disease_matched_filtered_{disease}.tsv"
    disease_annotation = pd.read_csv(disease_annotation_path, sep='\t')
    disease_annotation = disease_annotation[['FID', 'Disease']]
    disease_annotation.rename(columns={'FID': 'Disease_ID'}, inplace=True)
    disease_name_map = dict(zip(disease_annotation['Disease_ID'], disease_annotation['Disease']))
    all_pre_data = []
    for region in model_names:
        # 计算疾病人群BAG
        model_dir = os.path.join(base_dir, f"GTEx_Age_models/{region}/")
        if not os.path.exists(model_dir):
            print(f"Warning: {model_dir} not found, skipping")
            continue
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not model_files:
            print(f"Warning: No models found in {model_dir}, skipping")
            continue
        t_models = []
        for model_file in model_files:
            with open(os.path.join(model_dir, model_file), 'rb') as file:
                t_models.append(pickle.load(file))
        pro_dir = os.path.join(base_dir, f"GTEx_Age_models/{region}/")
        pro_data = pd.read_csv(pro_dir + "train_data.tsv", sep='\t')
        proteins = pro_data.drop(columns=['ID', 'Age', 'Flag']).columns.to_list()
        for disease_file in os.listdir(disease_dir):
            if not disease_file.endswith('.tsv'):
                continue
            disease_id = disease_file.split('_')[0]
            dis = pd.read_csv(os.path.join(disease_dir, disease_file), sep='\t')
            data = dis.filter(items=proteins)
            if data.empty:
                print(f"Warning: No features left in {disease_file}, skipping")
                continue
            pre_ages = np.mean([model.predict(data) for model in t_models], axis=0)
            pre_data = pd.DataFrame({
                'eid': dis['eid'].values,
                'Sex': dis['Sex'].values,
                'Real_age': dis['Age'].values,
                'Pre_age': pre_ages,
                'Disease_ID': disease_id
            })
            lowess_result = lowess(
                pre_data['Pre_age'],
                pre_data['Real_age'],
                frac=2/3,
                it=3,
                return_sorted=False
            )
            pre_data['Lowess_age'] = lowess_result
            pre_data['BAG'] = pre_data['Pre_age'] - pre_data['Lowess_age']
            pre_data['Zscored(BAG)'] = stats.zscore(pre_data['BAG'])
            pre_data['Model'] = region
            all_pre_data.append(pre_data)
            
        final_df = pd.concat(all_pre_data, ignore_index=True)
        final_df['Group'] = "Prevalent Disease"
        final_df['Disease_Name'] = final_df['Disease_ID'].map(disease_name_map)
        return final_df
    
# 既往疾病Associations计算
def prevalent_dis_association(BAG_file, regions, path2, path3, disease_pro, p, output):
    with open(regions) as f:
        model_names = [line.strip() for line in f]
    disease = 'prevalent'
    out_dir = output
    base_dir = path2
    disease_dir = f'{disease_pro}/{disease}_protein_datasets/'
    train_data_path = os.path.join(base_dir, f"GTEx_Age_models/{model_names[0]}/")
    train_data = pd.read_csv(train_data_path, sep='\t')
    train_data = train_data[['ID', 'Sex']]
    disease_data = BAG_file
    results = []
    for region in model_names:
        control_data_path = os.path.join(path3, f"{region}/test.healthy_BAG.tsv")
        if not os.path.exists(control_data_path):
            print(f"Warning: {control_data_path} not found, skipping")
            continue
        control_data = pd.read_csv(control_data_path, sep='\t')
        control_data = control_data.merge(train_data, on='ID', how='left')
        control_data['Trait'] = 0  # 对照组
        disease_data['Trait'] = 1  # 疾病组

        for disease_id, disease_name in disease_data[['Disease_ID', 'Disease_Name']].drop_duplicates().values:
            current_disease_data = disease_data[
                (disease_data['Disease_ID'] == disease_id) & 
                (disease_data['Model'] == region)
            ]
            combined_data = pd.concat([current_disease_data, control_data], ignore_index=True)
            if combined_data.empty:
                print(f"Warning: No data for Disease_ID={disease_id}, Model={region}, skipping")
                continue
            X = combined_data[['Trait', 'Sex', 'Real_age']]  # 自变量：Trait (患病状态), Sex, Real_age
            X = sm.add_constant(X)  # 添加截距项
            y = combined_data['BAG']  # 因变量：dAge
            # 拟合线性回归模型
            model = sm.OLS(y, X).fit()
            beta = model.params['Trait']  # Trait 的回归系数
            se = model.bse['Trait']       # Trait 的标准误
            p = model.pvalues['Trait']    # Trait 的 p 值
            # 计算置信区间 (95%)
            ci_lower = beta - 1.96 * se
            ci_upper = beta + 1.96 * se
            control_n = len(control_data)
            case_n = len(current_disease_data)
            results.append({
                'Disease_ID': disease_id,
                'Disease_Name': disease_name,
                'Group': "Prevalent Disease",
                'Model': region,
                'Control_N': control_n,
                'Case_N': case_n,
                'beta': beta,
                'se': se,
                'p': p,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

    results_df = pd.DataFrame(results)
    # 筛选显著结果
    significant_results = results_df[results_df['p'] < p]
    output_path = os.path.join(out_dir, "test.prevalent_disease.associations.tsv")
    significant_results.to_csv(output_path, sep='\t', index=False)
    print(f"既往疾病分析完成，结果已保存至: {out_dir}")
 
# 新发疾病BAG计算   
def incident_dis_BAG(regions, path2, disease_pro):
    with open(regions) as f:
        model_names = [line.strip() for line in f]
    disease = 'incident'
    base_dir = path2
    disease_dir = f'{disease_pro}/{disease}_protein_datasets/'
    disease_annotation_path = f"{disease_pro}/UKB-disease_matched_filtered_{disease}.tsv"
    disease_annotation = pd.read_csv(disease_annotation_path, sep='\t')
    disease_annotation = disease_annotation[['FID', 'Disease']]
    disease_annotation.rename(columns={'FID': 'Disease_ID'}, inplace=True)
    disease_name_map = dict(zip(disease_annotation['Disease_ID'], disease_annotation['Disease']))
    all_pre_data = []
    for region in model_names:
        # 计算疾病人群BAG
        model_dir = os.path.join(base_dir, f"GTEx_Age_models/{region}/")
        if not os.path.exists(model_dir):
            print(f"Warning: {model_dir} not found, skipping")
            continue
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not model_files:
            print(f"Warning: No models found in {model_dir}, skipping")
            continue
        t_models = []
        for model_file in model_files:
            with open(os.path.join(model_dir, model_file), 'rb') as file:
                t_models.append(pickle.load(file))
        pro_dir = os.path.join(base_dir, f"GTEx_Age_models/{region}/")
        pro_data = pd.read_csv(pro_dir + "train_data.tsv", sep='\t')
        proteins = pro_data.drop(columns=['ID', 'Age', 'Flag']).columns.to_list()
        for disease_file in os.listdir(disease_dir):
            if not disease_file.endswith('.tsv'):
                continue
            disease_id = disease_file.split('_')[0]
            dis = pd.read_csv(os.path.join(disease_dir, disease_file), sep='\t')
            data = dis.filter(items=proteins)
            if data.empty:
                print(f"Warning: No features left in {disease_file}, skipping")
                continue
            pre_ages = np.mean([model.predict(data) for model in t_models], axis=0)
            pre_data = pd.DataFrame({
            'eid': dis['eid'].values,        
            'Sex': dis['Sex'].values,
            'Event': dis['Event'].values,
            'Time': dis['Time'].values,
            'Real_age': dis['Age'].values,
            'Pre_age': pre_ages,
            'Disease_ID': disease_id        
            })
            lowess_result = lowess(
                pre_data['Pre_age'],
                pre_data['Real_age'],
                frac=2/3,
                it=3,
                return_sorted=False
            )
            pre_data['Lowess_age'] = lowess_result
            pre_data['BAG'] = pre_data['Pre_age'] - pre_data['Lowess_age']
            pre_data['Zscored(BAG)'] = stats.zscore(pre_data['BAG'])
            pre_data['Model'] = region
            all_pre_data.append(pre_data)
            
        final_df = pd.concat(all_pre_data, ignore_index=True)
        final_df['Group'] = "Incident Disease"
        final_df['Disease_Name'] = final_df['Disease_ID'].map(disease_name_map)
        return final_df
    
# 新发疾病Associations计算
def incident_dis_association(BAG_file, regions, path2, path3, disease_pro, p, output):
    with open(regions) as f:
        model_names = [line.strip() for line in f]
    disease = 'prevalent'
    out_dir = output
    base_dir = path2
    disease_dir = f'{disease_pro}/{disease}_protein_datasets/'
    disease_data = BAG_file
    control_blood_date_path = "./test_data/disease_data/53.csv"  # 对照组采血日期文件
    mortality_path = "./test_data/disease_data/Mortality_pheno_death.csv"  # 死亡数据文件
    mortality_data = pd.read_csv(mortality_path, usecols=['eid', 'date_of_death'])
    mortality_data['date_of_death'] = pd.to_datetime(mortality_data['date_of_death'])
    control_blood_date = pd.read_csv(control_blood_date_path, usecols=['eid', '53-0.0'])
    control_blood_date.rename(columns={'53-0.0': 'Time'}, inplace=True)
    control_blood_date['Time'] = pd.to_datetime(control_blood_date['Time'])
    control_data = control_blood_date.merge(mortality_data, on='eid', how='left')
    # 处理对照组数据：计算生存时间并排除提前死亡的个体
    control_data['event'] = 0  # 默认未患病
    control_data['duration'] = (study_end_date - control_data['Time']).dt.days / 365.25
    # 检查是否存在提前死亡的个体
    mask_death = control_data['date_of_death'] < study_end_date
    control_data.loc[mask_death, 'duration'] = (control_data.loc[mask_death, 'date_of_death'] - control_data.loc[mask_death, 'Time']).dt.days / 365.25
    # 排除在研究结束前死亡的个体（因为无法观察到他们是否患病）
    control_data = control_data[~mask_death]

    results = []
    for region in model_names:
        control_data_path = os.path.join(path3, f"{region}/test.healthy_BAG.tsv")
        if not os.path.exists(control_data_path):
            print(f"Warning: {control_data_path} not found, skipping")
            continue
        control_dage = pd.read_csv(control_data_path, sep='\t')
        control_combined = control_data.merge(control_dage, left_on='eid', right_on='ID')
        for disease_id, disease_name in disease_data[['Disease_ID', 'Disease_Name']].drop_duplicates().values:
            current_disease = disease_data[
                (disease_data['Disease_ID'] == disease_id) & 
                (disease_data['Model'] == region)
            ]
            current_disease['Event'] = pd.to_datetime(current_disease['Event'])
            current_disease['Time'] = pd.to_datetime(current_disease['Time'])
            current_disease['duration'] = (current_disease['Event'] - current_disease['Time']).dt.days / 365.25
            current_disease = current_disease[current_disease['duration'] > 0]
            current_disease['event'] = 1
            combined = pd.concat([current_disease, control_combined], ignore_index=True)
            combined = combined[['duration', 'event', 'BAG', 'Sex', 'Real_age']].dropna()
            # Cox 回归
            cph = CoxPHFitter()
            cph.fit(combined, duration_col='duration', event_col='event', formula='BAG + Sex + Real_age')
            coef = cph.summary.loc['BAG', 'coef']
            hr = np.exp(coef)
            se = cph.summary.loc['BAG', 'se(coef)']
            p = cph.summary.loc['BAG', 'p']
            ci_lower = cph.summary.loc['BAG', 'coef lower 95%']
            ci_upper = cph.summary.loc['BAG', 'coef upper 95%']
            results.append({
                'Disease_ID': disease_id,
                'Disease_Name': disease_name,
                'Model': region,
                'Control_N': len(control_combined),
                'Case_N': len(current_disease),
                'hr': hr,
                'se': se,
                'hr_lower': np.exp(ci_lower),
                'hr_upper': np.exp(ci_upper),
                'p': p
            })

    results_df = pd.DataFrame(results)
    # 筛选显著结果 (p < 0.05)
    significant_results = results_df[results_df['p'] < p]
    significant_results.to_csv(os.path.join(out_dir, "test.incident_disease.associations.tsv"), sep='\t', index=False)
    print(f"新发疾病分析完成，结果已保存至: {out_dir}")
    
    
if __name__ == "__main__":
    parameters = __init__()
    df_prevalent = prevalent_dis_BAG(parameters.models_name, parameters.part2_path, parameters.disease_pro)
    prevalent_dis_association(df_prevalent, parameters.models_name, parameters.part2_path, parameters.part3_path,
                              parameters.disease_pro, parameters.p_value, parameters.output)
    df_incident = incident_dis_BAG(parameters.models_name, parameters.part2_path, parameters.disease_pro)
    incident_dis_association(df_incident, parameters.models_name, parameters.part2_path, parameters.part3_path,
                              parameters.disease_pro, parameters.p_value, parameters.output)
    
    