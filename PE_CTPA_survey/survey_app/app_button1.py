from flask import Flask, render_template, session, request, redirect, url_for
import pandas as pd
import time
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher
import nltk
nltk.data.path.append('C:/Users/alexvanhalen/nltk_data')  # Replace with your path
nltk.download('punkt_tab', download_dir='C:/Users/alexvanhalen/nltk_data')  # Optional download_dir
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from difflib import SequenceMatcher

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management

# Set your OpenAI API key
openai.api_key = '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'  # Replace with your actual OpenAI API key

# Load the dataset
label_path = 'C:\\Users\\alexvanhalen\\OneDrive\\Desktop\\PE_CTPA_survey\\PE_CTPA_survey\\brown_test_RG_eval_100.xlsx'  # Path to the input dataset
labels_df = pd.read_excel(label_path)  # Read the dataset into a DataFrame

# Prepare data for the survey
image_reports = []
for i in range(len(labels_df)):
    # Extract relevant data from each row
    d = labels_df.iloc[i]
    image_name = d['AccessionNumber_md5']  # Unique identifier for the report
    generated_findings_report = d['findings_accession_pred']  # Predicted findings
    generated_impressions_report = d['impression_accession_pred']  # Predicted impressions
    image_reports.append({
        'name': image_name,
        'findings': generated_findings_report,
        'impression': generated_impressions_report
    })

# Function to calculate semantic similarity using embeddings from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def calculate_semantic_similarity(text1, text2):
    # Compute cosine similarity between two text embeddings
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Function to calculate edit distance (degree of textual changes)
def calculate_edit_distance(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

# # Function to track changes made by users
# def track_changes(original, modified):
#     changes = []
#     original_lines = original.splitlines()
#     modified_lines = modified.splitlines()
    
#     for i, (o_line, m_line) in enumerate(zip(original_lines, modified_lines), start=1):
#         if o_line != m_line:
#             changes.append({
#                 'line_number': i,
#                 'original': o_line,
#                 'modified': m_line
#             })
#     return changes

# Function to normalize text for word comparison
def normalize_text(text):
    # Remove punctuation and convert to lowercase
    return re.findall(r'\b\w+\b', text.lower())

# Function to calculate detailed word changes
def calculate_word_changes(original, modified):
    original_words = normalize_text(original)
    modified_words = normalize_text(modified)

    matcher = SequenceMatcher(None, original_words, modified_words)
    changes = {
        "insertions": 0,
        "deletions": 0,
        "replacements": 0
    }

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            changes["replacements"] += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            changes["deletions"] += i2 - i1
        elif tag == "insert":
            changes["insertions"] += j2 - j1

    # Total word changes
    total_changes = changes["insertions"] + changes["deletions"] + changes["replacements"]
    return changes, total_changes


# Calculate changes in words and sentences
def calculate_changes(original, modified):
    original_words = set(word_tokenize(original))
    modified_words = set(word_tokenize(modified))
    word_changes = len(original_words.symmetric_difference(modified_words))

    original_sentences = set(sent_tokenize(original))
    modified_sentences = set(sent_tokenize(modified))
    sentence_changes = len(original_sentences.symmetric_difference(modified_sentences))

    return word_changes, sentence_changes

def evaluate_with_openai(gt, pred):
    # We'll only list categories 1–4 since we are not using subtypes 5 and 6:
    #   1) False prediction of finding (false positive)
    #   2) Omission of finding (false negative)
    #   3) Incorrect location/position of finding
    #   4) Incorrect severity of finding
    #
    # Additionally, ask for a severity rating:
    #   1) Not actionable
    #   2) Actionable nonurgent error
    #   3) Urgent error
    #   4) Emergent error
    #
    # For each error, we also request text corrections.

    prompt = f"""
You are reviewing two radiology reports:
- A ground truth report (what the finding/impression should be)
- A predicted report (what was actually stated)

We have four main error categories (if an error exists):
  1) False prediction of finding (false positive)
  2) Omission of finding (false negative)
  3) Incorrect location/position of finding
  4) Incorrect severity of finding

We are not including categories about prior comparisons (5 and 6) because this task does not use information from prior reports.

Next, rate the severity of each error on a scale:
  1) Not actionable
  2) Actionable nonurgent error
  3) Urgent error
  4) Emergent error

Finally, if an error is found, provide a short text correction that fixes it.

Please:
1. Compare sentence by sentence.
2. If no error, say "No error".
3. If there is an error, specify which category (1–4) and severity (1–4).
4. Provide a short text correction.

Ground Truth:
{gt}

Prediction:
{pred}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        return response.choices[0].message['content'].strip()

    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
        return "Invalid request: Check input format."
    except openai.error.RateLimitError:
        print("Rate limit exceeded. Please try again later.")
        return "Rate limit exceeded."
    except openai.error.AuthenticationError:
        print("Authentication error: Check your API key.")
        return "Authentication error."
    except Exception as e:
        print(f"Unexpected error: {e.__class__.__name__} - {str(e)}")
        return "Error evaluating changes."



@app.route('/')
def index():
    # Start the session timer when the survey begins
    session['start_time'] = time.time()
    return redirect(url_for('survey', index=0))  # Redirect to the first report

@app.route('/survey/<int:index>')
def survey(index):
    if index < len(image_reports):
        # Display the current report for editing
        item = image_reports[index]
        analysis_summary = {
            'total_reports': len(image_reports),
            'findings_similarity': 0.0,  # Placeholder for now
            'impression_similarity': 0.0,  # Placeholder for now
            'most_frequent_error': "None"  # Placeholder for now
        }
        return render_template(
            'survey_button1.html',
            item=item,
            index=index,
            length=len(image_reports),
            analysis_summary=analysis_summary
        )
    else:
        return "No more reports to display."  # End of survey

@app.route('/submit', methods=['POST'])
def submit():
    # Get current report index and user modifications
    index = int(request.form['index'])
    image_name = request.form['image_name']
    modified_findings = request.form.get('modified_findings')  # Edited findings
    modified_impressions = request.form.get('modified_impressions')  # Edited impressions

    # Retrieve original data for comparison
    original_findings = labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_accession_pred'].values[0]
    original_impressions = labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impression_accession_pred'].values[0]
    findings_gt = labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_gt'].values[0]
    findings_pred = modified_findings
    impressions_gt = labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impression_gt'].values[0]
    impressions_pred = modified_impressions

    # Calculate detailed word changes
    findings_word_changes, findings_total_changes = calculate_word_changes(original_findings, modified_findings)
    impressions_word_changes, impressions_total_changes = calculate_word_changes(original_impressions, modified_impressions)
    
    # Calculate changes
    findings_word_changes, findings_sentence_changes = calculate_changes(original_findings, modified_findings)
    impressions_word_changes, impressions_sentence_changes = calculate_changes(original_impressions, modified_impressions)
    
    # Use OpenAI to evaluate changes
    findings_evaluation = evaluate_with_openai(findings_gt, findings_pred)
    impressions_evaluation = evaluate_with_openai(impressions_gt, impressions_pred)

    # Save changes back to the DataFrame
    labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'modified_findings'] = modified_findings
    labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'modified_impressions'] = modified_impressions
    
    labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_word_changes'] = findings_word_changes
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_total_changes'] = findings_total_changes
    labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_word_changes'] = impressions_word_changes
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_total_changes'] = impressions_total_changes
    
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_changes'] = str(findings_word_changes)
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_changes'] = findings_sentence_changes
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_changes'] = str(impressions_word_changes)
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_changes'] = impressions_sentence_changes
    
    labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_evaluations'] = findings_evaluation
    labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_evaluations'] = impressions_evaluation

    # # Calculate semantic similarity and edit distance for user modifications
    # findings_similarity = calculate_semantic_similarity(original_findings, modified_findings)
    # impressions_similarity = calculate_semantic_similarity(original_impressions, modified_impressions)

    # findings_edit_distance = calculate_edit_distance(original_findings, modified_findings)
    # impressions_edit_distance = calculate_edit_distance(original_impressions, modified_impressions)

    # # Store similarity and edit distance back in the DataFrame
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_similarity'] = findings_similarity
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_similarity'] = impressions_similarity
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'findings_edit_distance'] = findings_edit_distance
    # labels_df.loc[labels_df['AccessionNumber_md5'] == image_name, 'impressions_edit_distance'] = impressions_edit_distance

    # Save updated DataFrame to Excel after each submission
    labels_df.to_excel('modified_reports.xlsx', index=False)

    # Move to the next report or finish the survey
    next_index = index + 1
    if next_index < len(image_reports):
        return redirect(url_for('survey', index=next_index))
    else:
        total_time = time.time() - session['start_time']
        minutes, seconds = divmod(int(total_time), 60)
        return f"Survey complete! Total time: {minutes} minutes and {seconds} seconds."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
