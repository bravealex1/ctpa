# import pandas as pd 
# import os
# answer_path = '/media/brownradx/ssd_2t/Zhusi_projects/CTPALM_eval/eval_output_v2_1600_nogt'
# accession_result_df1 = pd.read_json(os.path.join(answer_path, 'accession_findingsimpression.jsonl'), lines=True)
# accession_result_df2 = pd.read_json(os.path.join(answer_path, 'accession_findings_to_impression.jsonl'), lines=True)

# findings_df = accession_result_df1[['AccessionNumber_md5', 'findings_gt', 'findings_accession_pred']]
# impression_df = pd.merge(accession_result_df1[['AccessionNumber_md5', 'impression_gt',]], accession_result_df2[['AccessionNumber_md5', 'impression_accession_pred']])

# # findings_df.to_excel('brown_test_findings.xlsx')
# # impression_df.to_excel('brown_test_impression.xlsx')

# base_path = '/media/brownradx/ssd_2t/Zhusi_projects/M3D_data'
# all_data_df = pd.read_excel(base_path + '/brown_CTPA_PE_all.xlsx')


# test_data = all_data_df[all_data_df['split']=='test']
# pe_test_data = test_data[~test_data['PESI'].isna()]

# pe_test_finding_df = findings_df[findings_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]
# pe_test_impression_df = impression_df[impression_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]

# nope_test_finding_df = findings_df[~findings_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]
# nope_test_impression_df = impression_df[~impression_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]

# score_path = '/media/brownradx/ssd_2t/Zhusi_projects/CTPALM_eval/eval_output_v2_1600_scores/'
# findings_socre_df = pd.read_csv(score_path + 'accession_findings_scores.csv')
# impression_socre_df = pd.read_csv(score_path + 'accession_impression_scores.csv')

# no_pe_test_findings_score_df = findings_socre_df[findings_socre_df['AccessionNumber_md5'].isin(nope_test_finding_df['AccessionNumber_md5'])]
# no_pe_test_impression_socre_df = impression_socre_df[impression_socre_df['AccessionNumber_md5'].isin(nope_test_impression_df['AccessionNumber_md5'])]

# nope_test_finding_df_s = pd.merge(nope_test_finding_df, no_pe_test_findings_score_df, how='outer', on='AccessionNumber_md5')
# nope_test_impression_df_s = pd.merge(nope_test_impression_df, no_pe_test_impression_socre_df, how='outer', on='AccessionNumber_md5')

# sort_by = 'bert_f1'
# sort_by = 'bleu4'
# nope_test_finding_df_s = nope_test_finding_df_s.sort_values(sort_by, ascending=True)
# # nope_test_impression_df_s = nope_test_impression_df_s.sort_values(sort_by, ascending=True)
# # nope_test_impression_df_choice = nope_test_impression_df_s[:len(pe_test_impression_df)]

# nope_test_impression_df_s['AccessionNumber_md5'] = pd.Categorical(nope_test_impression_df_s['AccessionNumber_md5'], categories=nope_test_finding_df_s['AccessionNumber_md5'].values.tolist(), ordered=True)
# sorted_nope_test_impression_df_s = nope_test_impression_df_s.sort_values(by='AccessionNumber_md5')

# eval_number = len(pe_test_finding_df)
# eval_number = 50

# pe_test_finding_df_choice = pe_test_finding_df[:eval_number]
# pe_test_impression_df_choice = pe_test_impression_df[:eval_number]
# nope_test_finding_df_choice = nope_test_finding_df_s[:eval_number]
# nope_test_impression_df_choice = sorted_nope_test_impression_df_s[:eval_number]


# print(nope_test_finding_df_choice.iloc[0]['findings_gt'],'\n\n', nope_test_finding_df_choice.iloc[0]['findings_accession_pred'])

# eval_findings_df = pd.concat([pe_test_finding_df_choice, nope_test_finding_df_choice[['AccessionNumber_md5', 'findings_gt', 'findings_accession_pred']]] )
# eval_findings_df = eval_findings_df.sample(frac=1, replace=False)

# eval_impression_df = pd.concat([pe_test_impression_df_choice, nope_test_impression_df_choice[['AccessionNumber_md5', 'impression_gt', 'impression_accession_pred']]] )
# # eval_impression_df = eval_impression_df.sample(frac=1, replace=False)

# eval_impression_df['AccessionNumber_md5'] = pd.Categorical(eval_impression_df['AccessionNumber_md5'], categories=eval_findings_df['AccessionNumber_md5'].values.tolist(), ordered=True)
# eval_impression_df = eval_impression_df.sort_values(by='AccessionNumber_md5')


# eval_findings_df = eval_findings_df.rename(columns={'findings_accession_pred':'gen_report','findings_gt':'gt_report'})
# eval_impression_df = eval_impression_df.rename(columns={'impression_accession_pred':'gen_report','impression_gt':'gt_report'})

# eval_findings_df = eval_findings_df.reset_index(drop=True)
# eval_impression_df = eval_impression_df.reset_index(drop=True)

# eval_findings_df.to_excel('brown_test_findings_eval_100.xlsx')
# eval_impression_df.to_excel('brown_test_impression_eval_100.xlsx')


import pandas as pd
import os
from io import StringIO

# Define paths
answer_path = 'C:/Users/alexvanhalen/OneDrive/Desktop/PE_CTPA_survey/PE_CTPA_survey/survey_app/eval_output_v2_1600_nogt'
base_path = 'C:/Users/alexvanhalen/OneDrive/Desktop/PE_CTPA_survey/PE_CTPA_survey/survey_app/M3D_data'
score_path = 'C:/Users/alexvanhalen/OneDrive/Desktop/PE_CTPA_survey/PE_CTPA_survey/survey_app/eval_output_v2_1600_scores'

# Function to safely read JSONL files
def safe_read_jsonl(file_path):
    try:
        if os.path.exists(file_path):
            return pd.read_json(file_path, lines=True)
        else:
            print(f"File not found: {file_path}")
            return pd.DataFrame()  # Return an empty DataFrame if the file is not found
    except ValueError as e:
        print(f"Error reading JSONL file {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Read JSONL files using the updated function
accession_result_df1 = safe_read_jsonl(os.path.join(answer_path, 'accession_findingsimpression.jsonl'))
accession_result_df2 = safe_read_jsonl(os.path.join(answer_path, 'accession_findings_to_impression.jsonl'))

# Proceed if data was successfully read
if not accession_result_df1.empty and not accession_result_df2.empty:
    findings_df = accession_result_df1[['AccessionNumber_md5', 'findings_gt', 'findings_accession_pred']]
    impression_df = pd.merge(
        accession_result_df1[['AccessionNumber_md5', 'impression_gt']],
        accession_result_df2[['AccessionNumber_md5', 'impression_accession_pred']],
        on='AccessionNumber_md5',
        how='inner'
    )

    # Read Excel and CSV files
    all_data_df = pd.read_excel(os.path.join(base_path, 'brown_CTPA_PE_all.xlsx'))
    findings_socre_df = pd.read_csv(os.path.join(score_path, 'accession_findings_scores.csv'))
    impression_socre_df = pd.read_csv(os.path.join(score_path, 'accession_impression_scores.csv'))

    # Filter the data based on test split and PESI values
    test_data = all_data_df[all_data_df['split'] == 'test']
    pe_test_data = test_data[~test_data['PESI'].isna()]

    # Filter findings and impressions based on test data
    pe_test_finding_df = findings_df[findings_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]
    pe_test_impression_df = impression_df[impression_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]

    nope_test_finding_df = findings_df[~findings_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]
    nope_test_impression_df = impression_df[~impression_df['AccessionNumber_md5'].isin(pe_test_data['AccessionNumber_md5'])]

    # Merge scores with the findings and impressions
    no_pe_test_findings_score_df = findings_socre_df[findings_socre_df['AccessionNumber_md5'].isin(nope_test_finding_df['AccessionNumber_md5'])]
    no_pe_test_impression_socre_df = impression_socre_df[impression_socre_df['AccessionNumber_md5'].isin(nope_test_impression_df['AccessionNumber_md5'])]

    nope_test_finding_df_s = pd.merge(nope_test_finding_df, no_pe_test_findings_score_df, how='outer', on='AccessionNumber_md5')
    nope_test_impression_df_s = pd.merge(nope_test_impression_df, no_pe_test_impression_socre_df, how='outer', on='AccessionNumber_md5')

    # Sort and prepare data
    sort_by = 'bleu4'
    nope_test_finding_df_s = nope_test_finding_df_s.sort_values(sort_by, ascending=True)
    nope_test_impression_df_s['AccessionNumber_md5'] = pd.Categorical(
        nope_test_impression_df_s['AccessionNumber_md5'],
        categories=nope_test_finding_df_s['AccessionNumber_md5'].values.tolist(),
        ordered=True
    )
    sorted_nope_test_impression_df_s = nope_test_impression_df_s.sort_values(by='AccessionNumber_md5')

    # Define evaluation numbers
    eval_number = 50
    pe_test_finding_df_choice = pe_test_finding_df[:eval_number]
    pe_test_impression_df_choice = pe_test_impression_df[:eval_number]
    nope_test_finding_df_choice = nope_test_finding_df_s[:eval_number]
    nope_test_impression_df_choice = sorted_nope_test_impression_df_s[:eval_number]

    # Print sample output
    print(nope_test_finding_df_choice.iloc[0]['findings_gt'], '\n\n', nope_test_finding_df_choice.iloc[0]['findings_accession_pred'])

    # Create final evaluation dataframes
    eval_findings_df = pd.concat([pe_test_finding_df_choice, nope_test_finding_df_choice[['AccessionNumber_md5', 'findings_gt', 'findings_accession_pred']]]).sample(frac=1, replace=False).reset_index(drop=True)
    eval_impression_df = pd.concat([pe_test_impression_df_choice, nope_test_impression_df_choice[['AccessionNumber_md5', 'impression_gt', 'impression_accession_pred']]])

    eval_impression_df['AccessionNumber_md5'] = pd.Categorical(eval_impression_df['AccessionNumber_md5'], categories=eval_findings_df['AccessionNumber_md5'].values.tolist(), ordered=True)
    eval_impression_df = eval_impression_df.sort_values(by='AccessionNumber_md5')

    # Rename columns
    eval_findings_df = eval_findings_df.rename(columns={'findings_accession_pred': 'gen_report', 'findings_gt': 'gt_report'})
    eval_impression_df = eval_impression_df.rename(columns={'impression_accession_pred': 'gen_report', 'impression_gt': 'gt_report'})

    # Save to Excel
    eval_findings_df.to_excel('brown_test_findings_eval_100.xlsx', index=False)
    eval_impression_df.to_excel('brown_test_impression_eval_100.xlsx', index=False)

else:
    print("One or more JSONL files could not be read correctly. Please check the file paths and formats.")