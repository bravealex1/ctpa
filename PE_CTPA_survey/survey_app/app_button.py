# from flask import Flask, render_template, request
# import pandas as pd
# import time
# import csv

# app = Flask(__name__)

# # Load the labels data from CSV
# label_path = 'C:\\Users\\alexvanhalen\\OneDrive\\Desktop\\PE_CTPA_survey\\PE_CTPA_survey\\brown_test_RG_eval_100.xlsx'
# labels_df = pd.read_excel(label_path)

# # Preparing data for the survey
# image_reports = []
# for i_re in range(len(labels_df)):
#     d = labels_df.iloc[i_re, :]
#     image_path = d['AccessionNumber_md5']
#     reports_gt_f = d['findings_gt']
#     reports_AI_f = d['findings_accession_pred']
#     reports_gt_i = d['impression_gt']
#     reports_AI_i = d['impression_accession_pred']
#     AA = {'name': image_path,
#         'reports': [reports_gt_f, reports_AI_f, reports_gt_i, reports_AI_i]}
#     image_reports.append(AA)

# with open('responses.csv', 'a', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["AccessionNumber_md5", "report_quality_findings", "report_quality_impression", "confidence", "response_time"])

# @app.route('/')
# def index():
#     # Display the first image and report set
#     return render_template('survey_button.html', item=image_reports[0], index=0, length=len(image_reports), start_time=time.time())

# @app.route('/submit', methods=['POST'])
# def submit():
#     index = int(request.form['index'])
#     # print(index)
#     response_time = time.time() - float(request.form['start_time'])
#     image_name = request.form['image_name']

#     # Access form data safely using get method
#     report_quality_f = request.form.get('finding_quality')
#     report_quality_i = request.form.get('impression_quality')
#     confidence = request.form.get('confidence')
    
#     # Record the response in a CSV file
#     with open('responses.csv', 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             image_name,
#             report_quality_f, 
#             report_quality_i, 
#             confidence, 
#             response_time])

#     # Move to the next image/report set or finish
#     next_index = index + 1
#     if next_index < len(image_reports):
#         return render_template('survey_button.html', item=image_reports[next_index], index=next_index, length=len(image_reports), start_time=time.time())
#     else:
#         return "Thank you for participating!"

# if __name__ == '__main__':
 
#     # app.run(debug=True, host='0.0.0.0', port=5000)
#     app.run(host='0.0.0.0', port=5000)



# # TODO
# # keep one report. For now, keep Generated FINDINGS Report
# # Modify and save the report in the same floder
# # Remove the score and submit button from UI
# # Time needs to be recorded (change the excel sheet name to time elapse from users fist click the UI to them finishing the survey)




# from flask import Flask, render_template, request, session
# import pandas as pd
# import time

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# # Load the labels data from Excel
# label_path = 'C:\\Users\\alexvanhalen\\OneDrive\\Desktop\\PE_CTPA_survey\\PE_CTPA_survey\\brown_test_RG_eval_100.xlsx'
# labels_df = pd.read_excel(label_path)

# # Prepare data for the survey by keeping only the Generated FINDINGS Report
# image_reports = []
# for i in range(len(labels_df)):
#     # Extract each row as a Series
#     d = labels_df.iloc[i]
#     # Get the AccessionNumber (unique identifier)
#     image_name = d['AccessionNumber_md5']
#     # Keep only the Generated FINDINGS Report
#     generated_report = d['findings_accession_pred']
#     # Append the report to the list
#     image_reports.append({
#         'name': image_name,
#         'report': generated_report
#     })

# # Create a copy of the DataFrame to store modified reports
# modified_labels_df = labels_df.copy()

# @app.route('/')
# def index():
#     # Record the start time when the user first accesses the UI
#     session['start_time'] = time.time()
#     # Display the first report to the user
#     return render_template('survey_button.html', item=image_reports[0], index=0, length=len(image_reports))

# @app.route('/submit', methods=['POST'])
# def submit():
#     # Retrieve the index of the current report
#     index = int(request.form['index'])
#     # Get the unique identifier of the report
#     image_name = request.form['image_name']
#     # Get the modified report from the form data
#     modified_report = request.form.get('modified_report')

#     # Update the modified_labels_df with the modified report
#     modified_labels_df.loc[modified_labels_df['AccessionNumber_md5'] == image_name, 'findings_accession_pred'] = modified_report

#     # Determine if there are more reports to display
#     next_index = index + 1
#     if next_index < len(image_reports):
#         # If yes, render the next report
#         return render_template('survey_button.html', item=image_reports[next_index], index=next_index, length=len(image_reports))
#     else:
#         # If no, calculate the total time elapsed
#         total_time = time.time() - session['start_time']
#         # Format the total time as minutes and seconds
#         minutes, seconds = divmod(int(total_time), 60)
#         # Save the modified reports back to an Excel file in the same folder
#         output_filename = f'modified_reports_{minutes}m{seconds}s.xlsx'
#         modified_labels_df.to_excel(output_filename, index=False)
#         # Inform the user that the survey is complete and display the total time taken
#         return f"Thank you for participating! Total time taken: {minutes} minutes and {seconds} seconds."

# if __name__ == '__main__':
#     # Run the Flask app on all available IP addresses on port 5000
#     app.run(host='0.0.0.0', port=5000)





# from flask import Flask, render_template, request, session
# import pandas as pd
# import time
# import openai
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# # Set your OpenAI API key
# openai.api_key = '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'  # Replace with your actual OpenAI API key

# # Load the dataset
# label_path = 'C:\\Users\\alexvanhalen\\OneDrive\\Desktop\\PE_CTPA_survey\\PE_CTPA_survey\\brown_test_RG_eval_100.xlsx'
# labels_df = pd.read_excel(label_path)

# # Prepare data for the survey by keeping only the Generated FINDINGS Report
# image_reports = []
# for i in range(len(labels_df)):
#     d = labels_df.iloc[i]
#     image_name = d['AccessionNumber_md5']
#     generated_report = d['findings_accession_pred']
#     image_reports.append({
#         'name': image_name,
#         'report': generated_report
#     })

# # Create a copy of the DataFrame to store modified reports
# modified_labels_df = labels_df.copy()

# # Function to get embeddings from OpenAI
# def get_embedding(text, model="text-embedding-ada-002"):
#     response = openai.Embedding.create(input=[text], model=model)
#     return response['data'][0]['embedding']

# # Function to calculate semantic similarity
# def calculate_semantic_similarity(gt_text, pred_text):
#     gt_embedding = get_embedding(gt_text)
#     pred_embedding = get_embedding(pred_text)
#     similarity = cosine_similarity([gt_embedding], [pred_embedding])[0][0]
#     return similarity

# # Function to generate analysis summary
# def generate_analysis_summary():
#     total_reports = len(labels_df)
#     findings_similarities = []
#     impression_similarities = []
#     for _, row in labels_df.iterrows():
#         findings_similarity = calculate_semantic_similarity(row['findings_gt'], row['findings_accession_pred'])
#         impression_similarity = calculate_semantic_similarity(row['impression_gt'], row['impression_accession_pred'])
#         findings_similarities.append(findings_similarity)
#         impression_similarities.append(impression_similarity)
#     average_findings_similarity = np.mean(findings_similarities)
#     average_impression_similarity = np.mean(impression_similarities)
#     most_frequent_error = "Impression" if np.mean(impression_similarities) < np.mean(findings_similarities) else "Findings"
#     return {
#         "total_reports": total_reports,
#         "average_findings_similarity": round(average_findings_similarity, 3),
#         "average_impression_similarity": round(average_impression_similarity, 3),
#         "most_frequent_error": most_frequent_error,
#     }

# @app.route('/')
# def index():
#     # Generate analysis summary
#     analysis_summary = generate_analysis_summary()
#     # Record the start time when the user first accesses the UI
#     session['start_time'] = time.time()
    
#     analysis_summary = {
#         'total_reports': len(image_reports),
#         'average_findings_similarity': 0.0,
#         'average_impression_similarity': 0.0,
#         'most_frequent_significant_error': ''
#     }
    
#     # Display the first report to the user
#     return render_template(
#         'survey_button.html',
#         item=image_reports[0],
#         index=0,
#         length=len(image_reports),
#         analysis_summary=analysis_summary
#     )

# @app.route('/submit', methods=['POST'])
# def submit():
#     # Retrieve the index of the current report
#     index = int(request.form['index'])
#     image_name = request.form['image_name']
#     modified_report = request.form.get('modified_report')

#     # Update the modified_labels_df with the modified report
#     modified_labels_df.loc[modified_labels_df['AccessionNumber_md5'] == image_name, 'findings_accession_pred'] = modified_report

#     # Perform semantic similarity analysis
#     gt_text = labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_gt'].values[0]
#     similarity = calculate_semantic_similarity(gt_text, modified_report)
    
#     # Update or initialize the analysis summary stored in the session
#     analysis_summary = session.get('analysis_summary', {
#         'total_reports': 0,
#         'total_similarity': 0.0,
#         'average_similarity': 0.0
#     })
#     analysis_summary['total_reports'] += 1
#     analysis_summary['total_similarity'] += similarity
#     analysis_summary['average_similarity'] = analysis_summary['total_similarity'] / analysis_summary['total_reports']
#     session['analysis_summary'] = analysis_summary

#     # Determine if there are more reports to display
#     next_index = index + 1
#     if next_index < len(image_reports):
#         return render_template(
#             'survey_button.html',
#             item=image_reports[next_index],
#             index=next_index,
#             length=len(image_reports),
#             similarity=similarity,
#             analysis_summary=analysis_summary  # Pass the updated analysis summary
#         )
#     else:
#         # Save the modified reports back to an Excel file
#         total_time = time.time() - session['start_time']
#         minutes, seconds = divmod(int(total_time), 60)
#         output_filename = f'modified_reports_{minutes}m{seconds}s.xlsx'
#         modified_labels_df.to_excel(output_filename, index=False)

#         # Inform the user that the survey is complete
#         return f"Survey complete! Total time: {minutes} minutes and {seconds} seconds. Results saved to {output_filename}."


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)




from flask import Flask, render_template, session, request, redirect, url_for
import pandas as pd
import time
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Set your OpenAI API key
openai.api_key = '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'  # Replace with your actual OpenAI API key

# Load the dataset
label_path = 'C:\\Users\\alexvanhalen\\OneDrive\\Desktop\\PE_CTPA_survey\\PE_CTPA_survey\\brown_test_RG_eval_100.xlsx'
labels_df = pd.read_excel(label_path)

# Prepare data for the survey by keeping only the Generated FINDINGS Report
image_reports = []
for i in range(len(labels_df)):
    d = labels_df.iloc[i]
    image_name = d['AccessionNumber_md5']
    generated_findings_report = d['findings_accession_pred']
    generated_impressions_report = d['impression_accession_pred']
    image_reports.append({
        'name': image_name,
        'findings': generated_findings_report,
        'impression': generated_impressions_report
    })

# Create a copy of the DataFrame to store modified reports
modified_labels_df = labels_df.copy()
modified_labels_df['findings_similarity'] = 0.0
modified_labels_df['impression_similarity'] = 0.0
modified_labels_df['most_frequent_error'] = ''

# Function to get embeddings from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error obtaining embedding: {e}")
        return None

# Function to calculate semantic similarity
def calculate_semantic_similarity(gt_text, pred_text):
    gt_embedding = get_embedding(gt_text)
    pred_embedding = get_embedding(pred_text)
    similarity = cosine_similarity([gt_embedding], [pred_embedding])[0][0]
    return similarity

findings_similarities = []
impression_similarities = []
total_reports = len(labels_df)
# Function to generate analysis summary
def generate_analysis_summary():
    findings_similarities.clear()
    impression_similarities.clear()
    for _, row in labels_df.iterrows():
        findings_similarity = calculate_semantic_similarity(
            row['findings_gt'], row['findings_accession_pred']
        )
        impression_similarity = calculate_semantic_similarity(
            row['impression_gt'], row['impression_accession_pred']
        )
        findings_similarities.append(findings_similarity)
        impression_similarities.append(impression_similarity)
    
    most_frequent_error = (
        "Impression" if np.mean(impression_similarities) < np.mean(findings_similarities)
        else "Findings"
    )
    return {
        "total_reports": len(labels_df),  # Ensure total_reports is included
        "findings_similarity": round(np.mean(findings_similarities), 3),
        "impression_similarity": round(np.mean(impression_similarities), 3),
        "most_frequent_error": most_frequent_error,
    }

# Function to generate a concise summary using OpenAI's API
def generate_summary(analysis_summary):
    prompt = (
        "Analyze the ground truth and predicted reports from the provided dataset. "
        "Generate a concise summary (50 words) that highlights key insights, similarities."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def index():
    # Initialize the analysis summary in the session
    session['analysis_summary'] = {
        'total_reports': 0,
        'findings_similarity': 0.0,
        'impression_similarity': 0.0,
        'significant_errors': {},
        'most_frequent_error': 'xxx'
    }
    session['start_time'] = time.time()
    return redirect(url_for('survey', index=0))

@app.route('/survey/<int:index>')
def survey(index):
    if index < len(image_reports):
        item = image_reports[index]
        analysis_summary = session.get('analysis_summary', {})
        return render_template(
            'survey_button.html',
            item=item,
            index=index,
            length=len(image_reports),
            analysis_summary=analysis_summary
        )
    else:
        return "No more reports to display."


@app.route('/submit', methods=['POST'])
def submit():
    index = int(request.form['index'])
    image_name = request.form['image_name']
    modified_findings = request.form.get('modified_findings')  # Separate user input for findings
    modified_impressions = request.form.get('modified_impressions')  # Separate user input for impressions

    # Update modified reports in the DataFrame
    modified_labels_df.loc[
        modified_labels_df['AccessionNumber_md5'] == image_name, 'findings_accession_pred'
    ] = modified_findings
    modified_labels_df.loc[
        modified_labels_df['AccessionNumber_md5'] == image_name, 'impression_accession_pred'
    ] = modified_impressions

    # Calculate findings similarity
    findings_gt = labels_df.loc[
        labels_df['AccessionNumber_md5'] == image_name, 'findings_gt'
    ].values[0]
    findings_similarity = calculate_semantic_similarity(findings_gt, modified_findings)
    modified_labels_df.loc[
        modified_labels_df['AccessionNumber_md5'] == image_name, 'findings_similarity'
    ] = findings_similarity

    # Calculate impression similarity
    impression_gt = labels_df.loc[
        labels_df['AccessionNumber_md5'] == image_name, 'impression_gt'
    ].values[0]
    impression_similarity = calculate_semantic_similarity(impression_gt, modified_impressions)
    modified_labels_df.loc[
        modified_labels_df['AccessionNumber_md5'] == image_name, 'impression_similarity'
    ] = impression_similarity

    # Update session analysis summary dynamically
    findings_similarities.append(findings_similarity)
    impression_similarities.append(impression_similarity)
    analysis_summary = generate_analysis_summary()

    analysis_summary = {
        "total_reports": len(findings_similarities),
        "findings_similarity": round(np.mean(findings_similarities), 3),
        "impression_similarity": round(np.mean(impression_similarities), 3),
        "most_frequent_error": (
            "Impression" if np.mean(impression_similarities) < np.mean(findings_similarities)
            else "Findings"
        ),
    }
    session['analysis_summary'] = analysis_summary
    
    # Generate summary text
    summary_text = generate_summary(analysis_summary)

    # Proceed to the next report
    next_index = index + 1
    if next_index < len(image_reports):
        return render_template(
            'survey_button.html',
            item=image_reports[next_index],
            index=next_index,
            length=len(image_reports),
            findings_similarity=round(findings_similarity, 3),
            impression_similarity=round(impression_similarity, 3),
            analysis_summary=analysis_summary
        )
    else:
        # Save and finish
        total_time = time.time() - session['start_time']
        minutes, seconds = divmod(int(total_time), 60)
        output_filename = f'modified_reports_{minutes}m{seconds}s.xlsx'

        # Save only relevant columns to Excel
        relevant_columns = [
            'AccessionNumber_md5', 'findings_similarity', 'impression_similarity'
        ]
        output_df = modified_labels_df[relevant_columns]
        output_df.to_excel(output_filename, index=False)
        
        # Save the results with the summary
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            modified_labels_df.to_excel(writer, sheet_name='Reports', index=False)
            worksheet = writer.sheets['Reports']

            # Add a blank line and summary
            worksheet.append([])
            worksheet.append(['Generated Summary'])
            worksheet.append([summary_text])


        return f"Survey complete! Total time: {minutes} minutes and {seconds} seconds. Results saved to {output_filename}."



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
