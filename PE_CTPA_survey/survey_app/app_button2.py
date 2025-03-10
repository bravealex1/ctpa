# from flask import Flask, render_template, session, request, redirect, url_for
# import pandas as pd
# import time
# import openai
# import json
# import re
# import nltk
# from sklearn.metrics.pairwise import cosine_similarity
# from difflib import SequenceMatcher
# from transformers import AutoTokenizer, AutoModelForTokenClassification

# # DeepSeek API: sk-3beb69261fac4be09ac453135b7b12b2

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # Load Medical-NER model
# try:
#     ner_tokenizer = AutoTokenizer.from_pretrained("blaze999/Medical-NER")
#     ner_model = AutoModelForTokenClassification.from_pretrained("blaze999/Medical-NER")
# except Exception as e:
#     print(f"Error loading Medical-NER model: {str(e)}")
#     ner_model = None

# # Set OpenAI API key
# openai.api_key = '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'

# # Load dataset
# label_path = 'C:\\Users\\alexvanhalen\\OneDrive\\Desktop\\PE_CTPA_survey\\PE_CTPA_survey\\brown_test_RG_eval_100.xlsx'
# labels_df = pd.read_excel(label_path)

# # Prepare survey data
# image_reports = []
# for i in range(len(labels_df)):
#     d = labels_df.iloc[i]
#     image_reports.append({
#         'name': d['AccessionNumber_md5'],
#         'findings': d['findings_accession_pred'],
#         'impression': d['impression_accession_pred']
#     })

# # Medical-NER analysis function
# def analyze_with_medical_ner(text):
#     if not ner_model or not text:
#         return {"error": "Model not available"}
    
#     inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     outputs = ner_model(**inputs)
#     predictions = outputs.logits.argmax(dim=-1)[0].tolist()
#     tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
#     label_map = {0: 'O', 1: 'B-MEDICAL', 2: 'I-MEDICAL',
#                  3: 'B-DISEASE', 4: 'I-DISEASE',
#                  5: 'B-TREATMENT', 6: 'I-TREATMENT'}
    
#     entities = []
#     current_entity = None
#     for token, prediction in zip(tokens, predictions):
#         label = label_map.get(prediction, 'O')
#         if label.startswith('B-'):
#             if current_entity:
#                 entities.append(current_entity)
#             current_entity = {
#                 'text': token.replace('##', ''),
#                 'type': label[2:],
#                 'start': len(entities),
#                 'end': len(entities) + 1
#             }
#         elif label.startswith('I-') and current_entity:
#             current_entity['text'] += ' ' + token.replace('##', '')
#             current_entity['end'] += 1
#         else:
#             if current_entity:
#                 entities.append(current_entity)
#                 current_entity = None

#     target_conditions = {
#         'PE': ['pulmonary embolism', 'pe'],
#         'Pneumonia': ['pneumonia', 'pneumonitis'],
#         'Cancer': ['cancer', 'malignancy', 'neoplasm']
#     }
    
#     analysis = {condition: False for condition in target_conditions}
#     for entity in entities:
#         entity_text = entity['text'].lower()
#         for condition, keywords in target_conditions.items():
#             if any(kw in entity_text for kw in keywords):
#                 analysis[condition] = True
#                 break
    
#     return {
#         'entities': entities,
#         'condition_analysis': analysis,
#         'PE_detected': analysis['PE'],
#         'other_conditions': [k for k,v in analysis.items() if v and k != 'PE']
#     }

# # OpenAI integration functions
# def evaluate_with_openai(gt, pred):
#     prompt = f"""
# Compare these radiology reports:
# Ground Truth: {gt}
# Prediction: {pred}

# Identify errors in these categories:
# 1) False positive findings
# 2) False negative omissions
# 3) Incorrect location
# 4) Incorrect severity

# Rate severity (1-4) and provide corrections. Format as JSON with:
# - error_type
# - severity
# - correction
# """
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             temperature=0
#         )
#         return json.loads(response.choices[0].message['content'])
#     except Exception as e:
#         return {"error": str(e)}

# def combined_analysis(text, ground_truth):
#     try:
#         ner_results = analyze_with_medical_ner(text)
#         # Add fallback for missing keys
#         pe_detected = ner_results.get('PE_detected', False)
#         gpt_results = evaluate_with_openai(ground_truth, text)
        
#         # Safe check for GPT corrections
#         gpt_correction = gpt_results.get('correction', '').lower()
#         pe_in_correction = 'pulmonary embolism' in gpt_correction or 'pe' in gpt_correction
        
#         return {
#             'ner': ner_results,
#             'gpt': gpt_results,
#             'consensus_pe': pe_detected and pe_in_correction
#         }
#     except Exception as e:
#         print(f"Analysis error: {str(e)}")
#         return {
#             'ner': {'error': str(e)},
#             'gpt': {'error': 'Analysis failed'},
#             'consensus_pe': False
#         }

# # Flask routes
# @app.route('/')
# def index():
#     session['start_time'] = time.time()
#     return redirect(url_for('survey', index=0))

# @app.route('/survey/<int:index>')
# def survey(index):
#     if index < len(image_reports):
#         item = image_reports[index]
#         return render_template(
#             'survey_button2.html',
#             item=item,
#             index=index,
#             length=len(image_reports),
#             analysis_summary={
#                 'total_reports': len(image_reports),
#                 'findings_similarity': 'N/A',
#                 'impression_similarity': 'N/A',
#                 'most_frequent_error': 'Pending'
#             }
#         )
#     return "Survey complete!"

# @app.route('/submit', methods=['POST'])
# def submit():
#     index = int(request.form['index'])
#     image_name = request.form['image_name']
#     modified_findings = request.form.get('modified_findings', '')
#     modified_impressions = request.form.get('modified_impressions', '')
    
#     if not modified_findings or not modified_impressions:
#         return "Error: Empty findings/impressions", 400
    
#     # Get ground truth data
#     row = labels_df[labels_df['AccessionNumber_md5'] == image_name].iloc[0]
#     findings_gt = row['findings_gt']
#     impressions_gt = row['impression_gt']
    
#     # Perform combined analysis
#     findings_analysis = combined_analysis(modified_findings, findings_gt)
#     impressions_analysis = combined_analysis(modified_impressions, impressions_gt)
    
#     # Save results
#     labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'modified_findings'] = modified_findings
#     labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'modified_impressions'] = modified_impressions
#     labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'full_analysis'] = str({
#         'findings': findings_analysis,
#         'impressions': impressions_analysis
#     })
    
#     # Save to Excel
#     labels_df.to_excel('modified_reports.xlsx', index=False)
    
#     next_index = index + 1
#     if next_index < len(image_reports):
#         return redirect(url_for('survey', index=next_index))
    
#     total_time = time.time() - session['start_time']
#     return f"Survey completed in {int(total_time//60)}m {int(total_time%60)}s"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)






# from flask import Flask, render_template, session, request, redirect, url_for
# import pandas as pd
# import time
# import openai
# import json
# import requests  # Added for DeepSeek API
# # Add this with other imports at the top
# import textwrap

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # API Keys
# DEEPSEEK_API_KEY = 'sk-3beb69261fac4be09ac453135b7b12b2'  # Replace with your DeepSeek API key
# openai.api_key = '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'

# # Load dataset
# label_path = 'C:\\Users\\alexvanhalen\\OneDrive\\Desktop\\PE_CTPA_survey\\PE_CTPA_survey\\brown_test_RG_eval_100.xlsx'
# labels_df = pd.read_excel(label_path)

# # Prepare survey data
# image_reports = []
# for i in range(len(labels_df)):
#     d = labels_df.iloc[i]
#     image_reports.append({
#         'name': d['AccessionNumber_md5'],
#         'findings': d['findings_accession_pred'],
#         'impression': d['impression_accession_pred']
#     })

# # DeepSeek analysis function
# def analyze_with_deepseek(text):
#     if not text:
#         return {"error": "No text provided"}
    
#     headers = {
#         "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
#         "Content-Type": "application/json"
#     }
    
#     prompt = f"""Analyze this radiology report and identify if any of these conditions are present:
#     1. Pulmonary Embolism (PE)
#     2. Pneumonia
#     3. Cancer
    
#     Respond with JSON format containing:
#     - "PE_detected": boolean
#     - "Pneumonia_detected": boolean
#     - "Cancer_detected": boolean
#     - "confidence_level": percentage (0-100)
    
#     Report text: {text}"""
    
#     payload = {
#         "model": "deepseek-chat",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.1,
#         "max_tokens": 200,
#         "response_format": {"type": "json_object"}
#     }
    
#     try:
#         response = requests.post(
#             "https://api.deepseek.com/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )
#         response.raise_for_status()
#         result = response.json()
#         content = json.loads(result['choices'][0]['message']['content'])
        
#         return {
#             'PE_detected': content.get('PE_detected', False),
#             'other_conditions': [
#                 cond for cond in ['Pneumonia', 'Cancer'] 
#                 if content.get(f'{cond}_detected', False)
#             ],
#             'confidence': content.get('confidence_level', 0),
#             'full_analysis': content
#         }
#     except Exception as e:
#         return {"error": str(e)}

# # OpenAI integration (kept for comparison)
# def evaluate_with_openai(gt, pred):
#     prompt = f"""Compare these radiology reports:
#     Ground Truth: {gt}
#     Prediction: {pred}

#     Identify errors in these categories:
#     1) False positive findings
#     2) False negative omissions
#     3) Incorrect location
#     4) Incorrect severity

#     Rate severity (1-4) and provide corrections. Format as JSON with:
#     - error_type
#     - severity
#     - correction
#     """
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             temperature=0
#         )
#         return json.loads(response.choices[0].message['content'])
#     except Exception as e:
#         return {"error": str(e)}

# def combined_analysis(text, ground_truth):
#     try:
#         deepseek_results = analyze_with_deepseek(text)
#         gpt_results = evaluate_with_openai(ground_truth, text)
        
#         # Check for PE mentions in GPT corrections
#         gpt_correction = gpt_results.get('correction', '').lower()
#         pe_in_correction = any(term in gpt_correction for term in ['pulmonary embolism', 'pe'])
        
#         return {
#             'deepseek': deepseek_results,
#             'gpt': gpt_results,
#             'consensus_pe': deepseek_results.get('PE_detected', False) and pe_in_correction,
#             'confidence': deepseek_results.get('confidence', 0)
#         }
#     except Exception as e:
#         print(f"Analysis error: {str(e)}")
#         return {
#             'deepseek': {'error': str(e)},
#             'gpt': {'error': 'Analysis failed'},
#             'consensus_pe': False,
#             'confidence': 0
#         }



# # Flask routes remain the same
# @app.route('/')
# def index():
#     session['start_time'] = time.time()
#     return redirect(url_for('survey', index=0))

# @app.route('/survey/<int:index>')
# def survey(index):
#     if index < len(image_reports):
#         item = image_reports[index]
#         return render_template(
#             'survey_button2.html',
#             item=item,
#             index=index,
#             length=len(image_reports),
#             analysis_summary={
#                 'total_reports': len(image_reports),
#                 'findings_similarity': 'N/A',
#                 'impression_similarity': 'N/A',
#                 'most_frequent_error': 'Pending'
#             }
#         )
#     return "Survey complete!"

# @app.route('/submit', methods=['POST'])
# def submit():
#     index = int(request.form['index'])
#     image_name = request.form['image_name']
#     modified_findings = request.form.get('modified_findings', '')
#     modified_impressions = request.form.get('modified_impressions', '')
    
#     if not modified_findings or not modified_impressions:
#         return "Error: Empty findings/impressions", 400
    
#     # Get ground truth data
#     row = labels_df[labels_df['AccessionNumber_md5'] == image_name].iloc[0]
#     findings_gt = row['findings_gt']
#     impressions_gt = row['impression_gt']
    
#     # Perform combined analysis
#     findings_analysis = combined_analysis(modified_findings, findings_gt)
#     impressions_analysis = combined_analysis(modified_impressions, impressions_gt)
    
#     # Save results
#     labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'modified_findings'] = modified_findings
#     labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'modified_impressions'] = modified_impressions
#     labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'full_analysis'] = str({
#         'findings': findings_analysis,
#         'impressions': impressions_analysis
#     })
    
#     # Save to Excel
#     labels_df.to_excel('modified_reports.xlsx', index=False)
    
#     next_index = index + 1
#     if next_index < len(image_reports):
#         return redirect(url_for('survey', index=next_index))
    
#     total_time = time.time() - session['start_time']
#     return f"Survey completed in {int(total_time//60)}m {int(total_time%60)}s"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




# from flask import Flask, render_template, session, request, redirect, url_for
# import pandas as pd
# import time
# import openai
# import json
# import requests
# import os

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # # API Keys
# DEEPSEEK_API_KEY = 'sk-3beb69261fac4be09ac453135b7b12b2'  # Replace with your DeepSeek API key
# openai.api_key = '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'

# # Load dataset
# label_path = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\brown_test_RG_eval_100.xlsx'
# labels_df = pd.read_excel(label_path)

# # Prepare survey data using AI-generated reports
# image_reports = []
# for i in range(len(labels_df)):
#     d = labels_df.iloc[i]
#     image_reports.append({
#         'name': d['AccessionNumber_md5'],
#         'findings': d['findings_accession_pred'],
#         'impression': d['impression_accession_pred']
#     })

# def analyze_with_deepseek(text):
#     if not text:
#         return {"error": "No text provided"}
    
#     headers = {
#         "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
#         "Content-Type": "application/json"
#     }
    
#     prompt = f"""Analyze this radiology report and identify if any of these conditions are present:
# 1. Pulmonary Embolism (PE)
# 2. Pneumonia
# 3. Cancer

# Respond with JSON format containing:
# - "PE_detected": boolean
# - "Pneumonia_detected": boolean
# - "Cancer_detected": boolean
# - "confidence_level": percentage (0-100)

# Report text: {text}"""
    
#     payload = {
#         "model": "deepseek-chat",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.1,
#         "max_tokens": 200,
#         "response_format": {"type": "json_object"}
#     }
    
#     try:
#         response = requests.post(
#             "https://api.deepseek.com/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )
#         response.raise_for_status()
#         result = response.json()
#         content = json.loads(result['choices'][0]['message']['content'])
#         return {
#             'PE_detected': content.get('PE_detected', False),
#             'other_conditions': [
#                 cond for cond in ['Pneumonia', 'Cancer'] 
#                 if content.get(f'{cond}_detected', False)
#             ],
#             'confidence': content.get('confidence_level', 0),
#             'full_analysis': content
#         }
#     except Exception as e:
#         return {"error": str(e)}

# def evaluate_with_openai(gt, pred):
#     prompt = f"""Compare these radiology reports:
# Ground Truth: {gt}
# Prediction: {pred}

# Identify errors in these categories:
# 1) False positive findings
# 2) False negative omissions
# 3) Incorrect location
# 4) Incorrect severity

# Rate severity (1-4) and provide corrections. Format as JSON with:
# - error_type
# - severity
# - correction
# """
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             response_format={"type": "json_object"},
#             temperature=0
#         )
#         return json.loads(response.choices[0].message['content'])
#     except Exception as e:
#         return {"error": str(e)}

# def combined_analysis(text, ground_truth):
#     try:
#         deepseek_results = analyze_with_deepseek(text)
#         gpt_results = evaluate_with_openai(ground_truth, text)
#         gpt_correction = gpt_results.get('correction', '').lower()
#         pe_in_correction = any(term in gpt_correction for term in ['pulmonary embolism', 'pe'])
#         return {
#             'deepseek': deepseek_results,
#             'gpt': gpt_results,
#             'consensus_pe': deepseek_results.get('PE_detected', False) and pe_in_correction,
#             'confidence': deepseek_results.get('confidence', 0)
#         }
#     except Exception as e:
#         print(f"Analysis error: {str(e)}")
#         return {
#             'deepseek': {'error': str(e)},
#             'gpt': {'error': 'Analysis failed'},
#             'consensus_pe': False,
#             'confidence': 0
#         }

# @app.route('/')
# def index():
#     session['start_time'] = time.time()
#     return redirect(url_for('survey', index=0))

# @app.route('/survey/<int:index>')
# def survey(index):
#     if index < len(image_reports):
#         item = image_reports[index]
#         # Retrieve ground truth for this image from the dataset
#         row = labels_df[labels_df['AccessionNumber_md5'] == item['name']].iloc[0]
#         gt_findings = row['findings_gt']
#         gt_impression = row['impression_gt']
#         analysis_summary = {
#             'total_reports': len(image_reports),
#             'findings_similarity': 'N/A',
#             'impression_similarity': 'N/A',
#             'most_frequent_error': 'Pending'
#         }
#         return render_template(
#             'survey_button3.html',
#             item=item,
#             index=index,
#             length=len(image_reports),
#             gt_findings=gt_findings,
#             gt_impression=gt_impression,
#             analysis_summary=analysis_summary
#         )
#     return "Survey complete!"

# @app.route('/submit', methods=['POST'])
# def submit():
#     index = int(request.form['index'])
#     image_name = request.form['image_name']
    
#     # Retrieve evaluation choices from the radiologist
#     evaluation_choice_findings = request.form.get('evaluation_choice_findings', '')
#     evaluation_choice_impressions = request.form.get('evaluation_choice_impressions', '')
#     error_category_findings = request.form.get('error_category_findings', '')
#     error_reasoning_findings = request.form.get('error_reasoning_findings', '')
#     error_category_impressions = request.form.get('error_category_impressions', '')
#     error_reasoning_impressions = request.form.get('error_reasoning_impressions', '')
    
#     # Handle file upload (if any)
#     uploaded_image = request.files.get('uploaded_image')
#     image_path = ""
#     if uploaded_image and uploaded_image.filename != "":
#         upload_folder = 'uploads'
#         if not os.path.exists(upload_folder):
#             os.makedirs(upload_folder)
#         image_path = os.path.join(upload_folder, uploaded_image.filename)
#         uploaded_image.save(image_path)
    
#     # Get ground truth from the dataset
#     row = labels_df[labels_df['AccessionNumber_md5'] == image_name].iloc[0]
#     findings_gt = row['findings_gt']
#     impressions_gt = row['impression_gt']
    
#     # Retrieve AI-generated report from image_reports based on the image name
#     ai_item = next((report for report in image_reports if report['name'] == image_name), None)
#     if ai_item is None:
#         return "Error: Image report not found", 404
    
#     # Perform combined analysis on the AI-generated report compared to the ground truth
#     findings_analysis = combined_analysis(ai_item['findings'], findings_gt)
#     impressions_analysis = combined_analysis(ai_item['impression'], impressions_gt)
    
#     # Compile the evaluation report to save
#     evaluation_report = {
#         'image_name': image_name,
#         'evaluation_choice_findings': evaluation_choice_findings,
#         'evaluation_choice_impressions': evaluation_choice_impressions,
#         'error_category_findings': error_category_findings,
#         'error_reasoning_findings': error_reasoning_findings,
#         'error_category_impressions': error_category_impressions,
#         'error_reasoning_impressions': error_reasoning_impressions,
#         'findings_analysis': findings_analysis,
#         'impressions_analysis': impressions_analysis,
#         'uploaded_image_path': image_path
#     }
    
#     # Save the evaluation report in the dataframe and then to Excel
#     labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'evaluation_report'] = json.dumps(evaluation_report)
#     labels_df.to_excel('modified_reports.xlsx', index=False)
    
#     next_index = index + 1
#     if next_index < len(image_reports):
#         return redirect(url_for('survey', index=next_index))
    
#     total_time = time.time() - session['start_time']
#     return f"Survey completed in {int(total_time//60)}m {int(total_time%60)}s"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)



from flask import Flask, render_template, session, request, redirect, url_for
import pandas as pd
import time
import json
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load dataset
label_path = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\brown_test_RG_eval_100.xlsx'
labels_df = pd.read_excel(label_path)

# Prepare survey data (using AI-generated reports)
image_reports = []
for i in range(len(labels_df)):
    d = labels_df.iloc[i]
    image_reports.append({
        'name': d['AccessionNumber_md5'],
        'findings': d['findings_accession_pred'],
        'impression': d['impression_accession_pred']
    })

@app.route('/')
def index():
    session['start_time'] = time.time()
    return redirect(url_for('survey', index=0))

@app.route('/survey/<int:index>')
def survey(index):
    if index < len(image_reports):
        item = image_reports[index]
        # Retrieve ground truth for internal evaluation (not editable by the radiologist)
        row = labels_df[labels_df['AccessionNumber_md5'] == item['name']].iloc[0]
        gt_findings = row['findings_gt']
        gt_impression = row['impression_gt']
        analysis_summary = {
            'total_reports': len(image_reports),
            'findings_similarity': 'N/A',
            'impression_similarity': 'N/A',
            'most_frequent_error': 'Pending'
        }
        return render_template(
            'survey_button4.html',
            item=item,
            index=index,
            length=len(image_reports),
            gt_findings=gt_findings,
            gt_impression=gt_impression,
            analysis_summary=analysis_summary
        )
    return "Survey complete!"

@app.route('/submit', methods=['POST'])
def submit():
    index = int(request.form['index'])
    image_name = request.form['image_name']
    
    # Get radiologist's evaluation inputs
    eval_choice_findings = request.form.get('evaluation_choice_findings', '')
    error_category_findings = request.form.get('error_category_findings', '')
    error_severity_findings = request.form.get('error_severity_findings', '')
    error_reasoning_findings = request.form.get('error_reasoning_findings', '')
    
    eval_choice_impressions = request.form.get('evaluation_choice_impressions', '')
    error_category_impressions = request.form.get('error_category_impressions', '')
    error_severity_impressions = request.form.get('error_severity_impressions', '')
    error_reasoning_impressions = request.form.get('error_reasoning_impressions', '')
    
    # Handle file upload (if provided)
    uploaded_image = request.files.get('uploaded_image')
    image_path = ""
    if uploaded_image and uploaded_image.filename != "":
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        image_path = os.path.join(upload_folder, uploaded_image.filename)
        uploaded_image.save(image_path)
    
    # Compile evaluation report (tracking only radiologist inputs)
    evaluation_report = {
        'image_name': image_name,
        'evaluation_choice_findings': eval_choice_findings,
        'error_category_findings': error_category_findings,
        'error_severity_findings': error_severity_findings,
        'error_reasoning_findings': error_reasoning_findings,
        'evaluation_choice_impressions': eval_choice_impressions,
        'error_category_impressions': error_category_impressions,
        'error_severity_impressions': error_severity_impressions,
        'error_reasoning_impressions': error_reasoning_impressions,
        'uploaded_image_path': image_path
    }
    
    # Save the evaluation report into the DataFrame and export to Excel
    labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'evaluation_report'] = json.dumps(evaluation_report)
    labels_df.to_excel('modified_reports.xlsx', index=False)
    
    next_index = index + 1
    if next_index < len(image_reports):
        return redirect(url_for('survey', index=next_index))
    
    total_time = time.time() - session['start_time']
    return f"Survey completed in {int(total_time//60)}m {int(total_time%60)}s"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

