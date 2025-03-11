import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from argparse import ArgumentParser, RawTextHelpFormatter


# 参数
def __init__():
    parser = ArgumentParser(
        formatter_class = RawTextHelpFormatter,
        description = 'Identify marker proteins.'
    )
    parser.add_argument(
        '-v', '--version', action = 'version', version = '%(prog)s 1.0.0'
    )
    parser.add_argument(
        '-gtex', '--GTEx-data', type = str, required = True, metavar = '<str>',
        help = 'Path to the GTEx matrix file.'
    )
    parser.add_argument(
        '-pro', '--proteins-matrix', type = str, required = True, metavar = '<str>',
        help = 'Path to the proteins matrix file.'
    )
    parser.add_argument(
        '-organ', '--organ-fc', default = 2.0, type = float, required = False, metavar = '<float>',
        help = 'Organ marker identify foldchange.\nDefault: 2.0'
    )
    parser.add_argument(
        '-brain', '--brain-fc', default = 1.5, type = float, required = False, metavar = '<float>',
        help = 'Brain region marker identify foldchange.\nDefault: 1.5'
    )
    parser.add_argument(
        '-o', '--output', type = str, required = True, metavar = '<str>',
        help = 'Path to the output file.'
    )
    return parser.parse_args()

# 归一化处理
def process_tissue(tissue, GTEx_path):
    file_path = f'{GTEx_path}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.{tissue}'
    data = pd.read_csv(file_path, sep='\t', index_col='Description').drop(columns=['Name'])
    clinical_df = pd.DataFrame({'condition': ['dummy']*data.shape[1]}, index=data.columns)
    dds = DeseqDataSet(
        counts=data.T,  
        clinical=clinical_df,
        design_factors="condition", 
        refit_cooks=False,           
        n_cpus=2                   
    )
    dds.deseq2()
    normalized_counts = dds.normalized_count_df.T 
    output_path = f'./GTEx_normalized/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.normalized.{tissue}'
    normalized_counts.to_csv(output_path, sep='\t', index=True, header=True)

# 生成表达矩阵
def process_mean(GTEx_path):
    data1 = pd.DataFrame()
    for t in ts:
        data = pd.read_csv('./GTEx_normalized/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.normalized.' + t, header=0, sep='\t')
        data1[t] = data.mean(axis=1)
    data = data = pd.read_csv(f'{GTEx_path}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.Liver', header=0, sep='\t')
    data1.index = data['Description']
    return data1

# 鉴定器官marker基因
def organ_marker(df, fc):
    columns_to_drop = ['Cells_Cultured_fibroblasts', 'Cells_EBV-transformed_lymphocytes', 'Cervix_Ectocervix',
                   'Cervix_Endocervix', 'Ovary', 'Uterus', 'Vagina', 'Fallopian_Tube', 'Testis', 'Prostate',
                   'Breast_Mammary_Tissue', 'Nerve_Tibial']
    df = df.drop(columns=columns_to_drop)
    # 合并来源于同一个器官的组织, 定义组织到器官的映射关系
    tissue_to_organ_map = {
        'Adipose': ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum'],
        'Adrenal': ['Adrenal_Gland'],
        'Artery': ['Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial'],
        'Bladder': ['Bladder'],
        'Brain': ['Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
                'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
                'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia',
                'Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra'],
        'Intestine': ['Colon_Sigmoid', 'Colon_Transverse', 'Small_Intestine_Terminal_Ileum'],
        'Esophagus': ['Esophagus_Gastroesophageal_Junction', 'Esophagus_Mucosa', 'Esophagus_Muscularis'],
        'Heart': ['Heart_Atrial_Appendage', 'Heart_Left_Ventricle'],
        'Kidney': ['Kidney_Cortex', 'Kidney_Medulla'],
        'Liver': ['Liver'],
        'Lung': ['Lung'],
        'Salivary': ['Minor_Salivary_Gland'],
        'Muscle': ['Muscle_Skeletal'],
        'Pancreas': ['Pancreas'],
        'Pituitary': ['Pituitary'],
        'Skin': ['Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg'],
        'Immune': ['Spleen', 'Whole_Blood'],
        'Stomach': ['Stomach'],
        'Thyroid': ['Thyroid']
    }
    merged_data = {'Description': df['Description']}
    for organ, tissues in tissue_to_organ_map.items():
        merged_data[organ] = df[tissues].max(axis=1)
    merged_df = pd.DataFrame(merged_data)
    merged_df.set_index("Description", inplace=True)
    result = []
    for organ in merged_df.columns:
        organ_expression = merged_df[organ]
        other_organs_max_expression = merged_df.drop(columns=[organ]).max(axis=1)
        markers = merged_df.index[organ_expression >= fc * other_organs_max_expression]
        for gene in markers:
            result.append({"Group": organ, "Gene": gene})
    result_df = pd.DataFrame(result)
    print("Organ Marker Selection Done.")
    return result_df

# 脑区marker蛋白鉴定
def brain_marker_pro(df_marker, df_matrix, df_pro, fc, output):
    marker_genes = df_marker
    brain_markers = marker_genes[marker_genes['Group'] == 'Brain']
    expression_data = df_matrix
    brain_genes = brain_markers['Gene'].tolist()
    filtered_expression = expression_data[expression_data['Description'].isin(brain_genes)]
    columns_to_keep = ['Description', 'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia',
                'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9',
                'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia',
                'Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra']
    filtered_matrix = filtered_expression[columns_to_keep]
    filtered_data = filtered_matrix[(filtered_matrix.iloc[:, 1:] != 0).any(axis=1)]
    df = filtered_data
    # 合并来源于同一个脑区的组织
    tissue_to_region_map = {
        'Cerebral_cortex': ['Brain_Anterior_cingulate_cortex_BA24', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9'],
        'Cerebral_nuclei': ['Brain_Caudate_basal_ganglia', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia', 'Brain_Amygdala'],
        'Cerebellum': ['Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum'],
        'Hippocampus': ['Brain_Hippocampus'],
        'Hypothalamus': ['Brain_Hypothalamus'],
        'Substantia_nigra': ['Brain_Substantia_nigra'],
        'Spinal_cord': ['Brain_Spinal_cord_cervical_c-1']
    }
    merged_data = {'Description': df['Description']}
    for region, tissues in tissue_to_region_map.items():
        merged_data[region] = df[tissues].max(axis=1)
    merged_df = pd.DataFrame(merged_data)
    merged_df.set_index("Description", inplace=True)
    result = []
    for region in merged_df.columns:
        region_expression = merged_df[region]
        other_regions_max_expression = merged_df.drop(columns=[region]).max(axis=1)
        markers = merged_df.index[region_expression >= fc * other_regions_max_expression]
        for gene in markers:
            result.append({"Group": region, "Gene": gene})
    result_df = pd.DataFrame(result)
    print("Brain Region Marker Selection Done.")
    # 映射marker蛋白
    genelist = df_pro.iloc[:, 3:].columns.tolist()
    df_markers = df_markers[df_markers['Gene'].isin(genelist)]
    df_markers.to_csv(f'{output}/test.marker_proteins.tsv', sep='\t', index=False)
    

if __name__ == "__main__":
    parameters = __init__()
    with open('tissues') as f:
        tissues = [line.strip() for line in f]
    os.makedirs('./GTEx_normalized', exist_ok=True)
    for tissue in tissues:
        print(f"Processing {tissue}...")
        process_tissue(tissue, parameters.GTEx_data)
    df_mean = process_mean(parameters.GTEx_data)
    organ_markers = organ_marker(df_mean, parameters.organ_fc)
    brain_marker_pro(organ_markers, df_mean, parameters.proteins_matrix, parameters.brain_fc, parameters.output)
    
