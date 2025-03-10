# # app.py – Flask backend for radiologist evaluation
# from flask import Flask, session, render_template, request, redirect, url_for
# import os, json, random, time
# from datetime import datetime
# import pandas as pd

# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

# # Set up provided paths
# label_path = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\Ext_Val_XNAT.xlsx'
# BASE_IMAGE_DIR = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\2D_picture'

# # Load the evaluation labels from the Excel file
# try:
#     labels_df = pd.read_excel(label_path)
# except Exception as e:
#     raise RuntimeError(f"Failed to load dataset: {str(e)}")

# # For this example, we assume that the Excel file (Ext_Val_XNAT.xlsx) contains a column named 'XNATSessionID'
# # and that for each XNATSessionID there are 32 slices stored in a subfolder under BASE_IMAGE_DIR.
# # We build a simple list of cases from the Excel file.
# cases = labels_df['XNATSessionID'].unique().tolist()
# total_cases = len(cases)

# # Helper function: Save annotations for a given case as a JSON file.
# def save_annotations(case_id, annotations):
#     os.makedirs('evaluations', exist_ok=True)
#     save_path = os.path.join('evaluations', f"{case_id}_annotations.json")
#     with open(save_path, 'w') as f:
#         json.dump(annotations, f, indent=2)

# # Landing page – resume if possible
# @app.route('/')
# def index():
#     last_case = session.get('last_case', 0)
#     if last_case >= total_cases:
#         return "<h3>All cases have been evaluated. Thank you!</h3>"
#     return render_template('index.html', resume_case=last_case, total_cases=total_cases)

# # Evaluation route for each case
# @app.route('/case/<int:case_index>', methods=['GET', 'POST'])
# def evaluate_case(case_index):
#     if case_index >= total_cases:
#         return "<h3>Evaluation complete. Thank you!</h3>"
    
#     case_id = cases[case_index]
#     session['last_case'] = case_index  # track progress

#     if request.method == 'GET':
#         # Load report texts for this case.
#         # Assume each case folder is named after the XNATSessionID (e.g. "C-78")
#         case_folder = os.path.join("static", "cases", case_id)
#         ai_report_path = os.path.join(case_folder, "ai_report.txt")
#         human_report_path = os.path.join(case_folder, "human_report.txt")
#         if not (os.path.exists(ai_report_path) and os.path.exists(human_report_path)):
#             return f"<h3>Case {case_id} data not found.</h3>"
#         with open(ai_report_path, 'r') as f:
#             ai_text = f.read()
#         with open(human_report_path, 'r') as f:
#             human_text = f.read()
        
#         # Randomize report assignment for blinded evaluation: assign Report A vs B randomly.
#         assignments = session.get('assignments', {})
#         if str(case_index) in assignments:
#             assign_A = assignments[str(case_index)]
#         else:
#             assign_A = random.choice([True, False])
#             assignments[str(case_index)] = assign_A
#             session['assignments'] = assignments
#         if assign_A:
#             reportA_text = ai_text
#             reportB_text = human_text
#         else:
#             reportA_text = human_text
#             reportB_text = ai_text

#         # Render the evaluation template.
#         return render_template('evaluate.html',
#                                case_index=case_index,
#                                total_cases=total_cases,
#                                case_id=case_id,
#                                reportA=reportA_text,
#                                reportB=reportB_text,
#                                slice_count=32)  # each case has 32 slices

#     # POST: Save corrections/annotations and then move to next case.
#     corrections_json = request.form.get('corrections', '[]')
#     try:
#         corrections = json.loads(corrections_json)
#     except:
#         corrections = []
    
#     # For simplicity, we simply save the corrections to a JSON file.
#     save_annotations(case_id, corrections)
    
#     # (Optionally, you can also update a "reviewed" report file here if needed.)
    
#     # Move to next case.
#     next_case = case_index + 1
#     session['last_case'] = next_case
#     session.modified = True
#     return redirect(url_for('evaluate_case', case_index=next_case))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)





# from flask import Flask, session, render_template, request, redirect, url_for, send_from_directory
# import os, json, random
# import pandas as pd

# app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

# # ----------------------------------------------------------------------
# # 1) Set up your base directory and CSV paths
# # ----------------------------------------------------------------------
# label_path = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\Ext_Val_XNAT.csv'
# BASE_IMAGE_DIR = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\2D_picture'

# # ----------------------------------------------------------------------
# # 2) Load CSV and filter the desired cases (only if XNATSessionID is in CSV)
# # ----------------------------------------------------------------------
# try:
#     labels_df = pd.read_csv(label_path)
# except Exception as e:
#     raise RuntimeError(f"Failed to load dataset: {str(e)}")

# labels_df['XNATSessionID'] = labels_df['XNATSessionID'].astype(str)
# desired_cases = ['C-78', 'C-134', 'C-154']
# cases = [case for case in desired_cases if case in labels_df['XNATSessionID'].values]
# total_cases = len(cases)
# print("Evaluating cases:", cases)

# def extract_findings(report_text):
#     """
#     Extracts and returns the findings section from a report.
#     (If the text contains "findings:" (case-insensitive), only that section is returned.)
#     """
#     lower_text = report_text.lower()
#     idx = lower_text.find("findings:")
#     if idx != -1:
#         return report_text[idx:]
#     return report_text

# # ----------------------------------------------------------------------
# # 3) Serve images from your local folder.
# #    This route allows <img src="/slice/<filename>"> to fetch images
# #    from BASE_IMAGE_DIR.
# # ----------------------------------------------------------------------
# @app.route('/slice/<path:filename>')
# def slice_image(filename):
#     return send_from_directory(BASE_IMAGE_DIR, filename)

# # ----------------------------------------------------------------------
# # Helper function to save data (annotations or edited reports) for a given case
# # ----------------------------------------------------------------------
# def save_data(case_id, data, prefix="annotations"):
#     os.makedirs('evaluations', exist_ok=True)
#     save_path = os.path.join('evaluations', f"{case_id}_{prefix}.json")
#     with open(save_path, 'w') as f:
#         json.dump(data, f, indent=2)

# # ----------------------------------------------------------------------
# # Landing page – resume if possible
# # ----------------------------------------------------------------------
# @app.route('/')
# def index():
#     last_case = session.get('last_case', 0)
#     if last_case >= total_cases:
#         return "<h3>All cases have been evaluated. Thank you!</h3>"
#     return f'''
#         <h3>Welcome to the PE CTPA Survey</h3>
#         <p>There are {total_cases} case(s) to evaluate.</p>
#         <a href="{url_for('evaluate_case', case_index=last_case)}">Start Evaluation Session</a>
#         <br><br>
#         <a href="{url_for('error_correction', case_index=last_case)}">Start Error Correction Session</a>
#         <br><br>
#         <a href="{url_for('turing_test', case_index=last_case)}">Start Turing Test Session</a>
#     '''

# # ----------------------------------------------------------------------
# # 4a) Evaluation Session (Blind Turing-Test Style)
# # ----------------------------------------------------------------------
# @app.route('/case/<int:case_index>', methods=['GET', 'POST'])
# def evaluate_case(case_index):
#     if case_index >= total_cases:
#         return "<h3>Evaluation complete. Thank you!</h3>"
#     case_id = cases[case_index]
#     session['last_case'] = case_index  # track progress

#     # Retrieve the row for this case from CSV
#     row = labels_df[labels_df['XNATSessionID'] == case_id]
#     if row.empty:
#         return f"<h3>Case {case_id} not found in CSV.</h3>"
#     # Get the two reports from the CSV columns
#     ground_truth_report = row['Ground Truth'].values[0]
#     ai_report = row['pred'].values[0]
#     # Extract only the findings section
#     gt_finding = extract_findings(ground_truth_report)
#     ai_finding = extract_findings(ai_report)
#     # Randomize assignment (blind evaluation)
#     assignments = session.get('assignments', {})
#     if str(case_index) in assignments:
#         assign_A = assignments[str(case_index)]
#     else:
#         assign_A = random.choice([True, False])
#         assignments[str(case_index)] = assign_A
#         session['assignments'] = assignments
#     if assign_A:
#         reportA_text = ai_finding
#         reportB_text = gt_finding
#     else:
#         reportA_text = gt_finding
#         reportB_text = ai_finding

#     # Gather the 32 slice images for this case (e.g., C-78_img_slice_1.png ... C-78_img_slice_32.png)
#     slice_images = []
#     for i in range(1, 33):
#         filename = f"{case_id}_img_slice_{i}.png"
#         if os.path.exists(os.path.join(BASE_IMAGE_DIR, filename)):
#             slice_images.append(filename)
#     slice_images.sort()

#     return render_template('evaluate.html',
#                            case_index=case_index,
#                            total_cases=total_cases,
#                            case_id=case_id,
#                            reportA=reportA_text,
#                            reportB=reportB_text,
#                            slice_images=slice_images)

# # ----------------------------------------------------------------------
# # 4b) Error Correction Session (for editing the reports)
# # ----------------------------------------------------------------------
# @app.route('/error_correction/<int:case_index>', methods=['GET', 'POST'])
# def error_correction(case_index):
#     if case_index >= total_cases:
#         return "<h3>Error Correction: All cases processed. Thank you!</h3>"
#     case_id = cases[case_index]
#     # Retrieve the reports from CSV
#     row = labels_df[labels_df['XNATSessionID'] == case_id]
#     if row.empty:
#         return f"<h3>Case {case_id} not found in CSV.</h3>"
#     gt_report = row['Ground Truth'].values[0]
#     ai_report = row['pred'].values[0]
#     # Extract findings only
#     gt_finding = extract_findings(gt_report)
#     ai_finding = extract_findings(ai_report)
#     # Randomize assignment (separate from evaluation)
#     assignments = session.get('error_assignments', {})
#     if str(case_index) in assignments:
#         assign_A = assignments[str(case_index)]
#     else:
#         assign_A = random.choice([True, False])
#         assignments[str(case_index)] = assign_A
#         session['error_assignments'] = assignments
#     if assign_A:
#         reportA_text = ai_finding
#         reportB_text = gt_finding
#     else:
#         reportA_text = gt_finding
#         reportB_text = ai_finding

#     if request.method == 'POST':
#         corrections_json = request.form.get('corrections', '[]')
#         try:
#             corrections = json.loads(corrections_json)
#         except:
#             corrections = []
#         # Save the corrections (edited reports) separately
#         save_data(case_id, corrections, prefix="edited")
#         next_case = case_index + 1
#         session['last_case'] = next_case
#         session.modified = True
#         return redirect(url_for('error_correction', case_index=next_case))

#     return render_template('error_correction.html',
#                            case_index=case_index,
#                            total_cases=total_cases,
#                            case_id=case_id,
#                            reportA=reportA_text,
#                            reportB=reportB_text)

# # ----------------------------------------------------------------------
# # 4c) Turing Test Session (for independent evaluation of the edited reports)
# # ----------------------------------------------------------------------
# @app.route('/turing_test/<int:case_index>', methods=['GET', 'POST'])
# def turing_test(case_index):
#     if case_index >= total_cases:
#         return "<h3>Turing Test complete. Thank you!</h3>"
#     case_id = cases[case_index]
#     # Attempt to load the edited reports if available; otherwise fallback to CSV
#     edited_path = os.path.join('evaluations', f"{case_id}_edited.json")
#     if os.path.exists(edited_path):
#         with open(edited_path, 'r') as f:
#             edited_data = json.load(f)
#         reportA_text = edited_data.get('reportA', '')
#         reportB_text = edited_data.get('reportB', '')
#     else:
#         row = labels_df[labels_df['XNATSessionID'] == case_id]
#         if row.empty:
#             return f"<h3>Case {case_id} not found in CSV.</h3>"
#         gt_report = row['Ground Truth'].values[0]
#         ai_report = row['pred'].values[0]
#         gt_finding = extract_findings(gt_report)
#         ai_finding = extract_findings(ai_report)
#         assignments = session.get('turing_assignments', {})
#         if str(case_index) in assignments:
#             assign_A = assignments[str(case_index)]
#         else:
#             assign_A = random.choice([True, False])
#             assignments[str(case_index)] = assign_A
#             session['turing_assignments'] = assignments
#         if assign_A:
#             reportA_text = ai_finding
#             reportB_text = gt_finding
#         else:
#             reportA_text = gt_finding
#             reportB_text = ai_finding

#     if request.method == 'POST':
#         ai_choice = request.form.get('ai_selection')
#         # Save the Turing test decision (e.g., in a file or database)
#         save_data(case_id, {"ai_choice": ai_choice}, prefix="turing")
#         next_case = case_index + 1
#         session['last_case'] = next_case
#         session.modified = True  # Ensure session changes are saved
#         return redirect(url_for('turing_test', case_index=next_case))

#     return render_template('turing_test.html',
#                            case_index=case_index,
#                            total_cases=total_cases,
#                            case_id=case_id,
#                            reportA=reportA_text,
#                            reportB=reportB_text)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, session, render_template, request, redirect, url_for, send_from_directory
import os, json, random
import pandas as pd

app = Flask(__name__)

# ----------------------------------------------------------------------
# 1) Set up your base directory and CSV paths
# ----------------------------------------------------------------------
label_path = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\Ext_Val_XNAT.csv'
BASE_IMAGE_DIR = r'C:\Users\alexvanhalen\OneDrive\Desktop\PE_CTPA_survey\PE_CTPA_survey\2D_picture'

# ----------------------------------------------------------------------
# 2) Load CSV and filter the desired cases (only if XNATSessionID is in CSV)
# ----------------------------------------------------------------------
try:
    labels_df = pd.read_csv(label_path)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {str(e)}")

labels_df['XNATSessionID'] = labels_df['XNATSessionID'].astype(str)
desired_cases = ['C-78', 'C-134', 'C-154']
cases = [case for case in desired_cases if case in labels_df['XNATSessionID'].values]
total_cases = len(cases)
print("Evaluating cases:", cases)

def extract_findings(report_text):
    """
    Extracts and returns the findings section from a report.
    If the text contains "findings:" (case-insensitive), only that section is returned.
    """
    lower_text = report_text.lower()
    idx = lower_text.find("findings:")
    if idx != -1:
        return report_text[idx:]
    return report_text

# ----------------------------------------------------------------------
# 3) Serve images from your local folder.
#    This route allows <img src="/slice/<filename>"> to fetch images
#    from BASE_IMAGE_DIR.
# ----------------------------------------------------------------------
@app.route('/slice/<path:filename>')
def slice_image(filename):
    return send_from_directory(BASE_IMAGE_DIR, filename)

# ----------------------------------------------------------------------
# Helper function to save annotations for a given case
# ----------------------------------------------------------------------
def save_annotations(case_id, annotations):
    os.makedirs('evaluations', exist_ok=True)
    save_path = os.path.join('evaluations', f"{case_id}_annotations.json")
    with open(save_path, 'w') as f:
        json.dump(annotations, f, indent=2)

# ----------------------------------------------------------------------
# Reset route to clear session and restart evaluation
# ----------------------------------------------------------------------
@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

# ----------------------------------------------------------------------
# Landing page – resume if possible, with a reset option
# ----------------------------------------------------------------------
@app.route('/')
def index():
    last_case = session.get('last_case', 0)
    if last_case >= total_cases:
        return f'''
            <h3>All selected cases have been evaluated. Thank you!</h3>
            <a href="{url_for('reset')}">Reset Evaluation</a>
        '''
    return f'''
        <h3>Welcome to the PE CTPA Survey</h3>
        <p>We have {total_cases} case(s) to evaluate.</p>
        <a href="{url_for('evaluate_case', case_index=last_case)}">Start the evaluation</a>
        <br><br>
        <a href="{url_for('reset')}">Reset Evaluation</a>
    '''

# ----------------------------------------------------------------------
# Evaluation route for each case
# ----------------------------------------------------------------------
@app.route('/case/<int:case_index>', methods=['GET', 'POST'])
def evaluate_case(case_index):
    if case_index >= total_cases:
        return f'''
            <h3>Evaluation complete. Thank you!</h3>
            <a href="{url_for('reset')}">Reset Evaluation</a>
        '''
    case_id = cases[case_index]
    session['last_case'] = case_index  # track progress

    if request.method == 'GET':
        # Load text reports (from CSV in this example)
        row = labels_df[labels_df['XNATSessionID'] == case_id]
        if row.empty:
            return f"<h3>Case {case_id} not found in CSV.</h3>"
        # Get the two reports from the CSV columns
        ground_truth_report = row['Ground Truth'].values[0]
        ai_report = row['pred'].values[0]
        # Extract only the findings section
        gt_finding = extract_findings(ground_truth_report)
        ai_finding = extract_findings(ai_report)
        # Randomize assignment (blind evaluation)
        assignments = session.get('assignments', {})
        if str(case_index) in assignments:
            assign_A = assignments[str(case_index)]
        else:
            assign_A = random.choice([True, False])
            assignments[str(case_index)] = assign_A
            session['assignments'] = assignments
        if assign_A:
            reportA_text = ai_finding
            reportB_text = gt_finding
        else:
            reportA_text = gt_finding
            reportB_text = ai_finding

        # Gather the 32 slice images for this case (e.g. C-78_img_slice_1.png ... C-78_img_slice_32.png)
        slice_images = []
        for i in range(1, 33):
            filename = f"{case_id}_img_slice_{i}.png"
            if os.path.exists(os.path.join(BASE_IMAGE_DIR, filename)):
                slice_images.append(filename)
        slice_images.sort()

        return render_template('index.html',
                               case_index=case_index,
                               total_cases=total_cases,
                               case_id=case_id,
                               reportA=reportA_text,
                               reportB=reportB_text,
                               slice_images=slice_images)

    # POST: Save corrections/annotations, move to next case
    corrections_json = request.form.get('corrections', '[]')
    try:
        corrections = json.loads(corrections_json)
    except:
        corrections = []
    
    save_annotations(case_id, corrections)
    
    next_case = case_index + 1
    session['last_case'] = next_case
    session.modified = True  # Ensure session changes are saved
    return redirect(url_for('evaluate_case', case_index=next_case))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

