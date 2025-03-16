import streamlit as st
import os
import json, random
import pandas as pd

# --- Initialize Session State ---
if "last_case" not in st.session_state:
    st.session_state.last_case = 0
if "assignments" not in st.session_state:
    st.session_state.assignments = {}
if "page" not in st.session_state:
    st.session_state.page = "index"
if "current_slice" not in st.session_state:
    st.session_state.current_slice = 0
if "corrections" not in st.session_state:
    st.session_state["corrections"] = []  # list of correction entries

# --- Set up base directory and CSV paths ---
label_path = 'PE_CTPA_survey/Ext_Val_XNAT.csv'
BASE_IMAGE_DIR = 'PE_CTPA_survey/2D_picture'

# --- Load CSV and filter desired cases ---
try:
    labels_df = pd.read_csv(label_path)
except Exception as e:
    st.error(f"Failed to load dataset: {str(e)}")
    st.stop()

labels_df['XNATSessionID'] = labels_df['XNATSessionID'].astype(str)
desired_cases = ['C-78', 'C-134', 'C-154']
cases = [case for case in desired_cases if case in labels_df['XNATSessionID'].values]
total_cases = len(cases)
st.write("Evaluating cases:", cases)

def extract_findings(report_text):
    """
    Extracts and returns the findings section from a report.
    If "findings:" is present (case-insensitive), returns that section.
    """
    lower_text = report_text.lower()
    idx = lower_text.find("findings:")
    if idx != -1:
        return report_text[idx:]
    return report_text

def save_annotations(case_id, annotations):
    os.makedirs('evaluations', exist_ok=True)
    save_path = os.path.join('evaluations', f"{case_id}_annotations.json")
    with open(save_path, 'w') as f:
        json.dump(annotations, f, indent=2)

# --- Helper function for the image carousel ---
def display_slice_carousel(case_id):
    # Build list of slice image paths (one per slice)
    slice_images = []
    for i in range(1, 33):
        filename = f"{case_id}_img_slice_{i}.png"
        full_path = os.path.join(BASE_IMAGE_DIR, filename)
        if os.path.exists(full_path):
            slice_images.append(full_path)
    slice_images.sort()
    
    total_slices = len(slice_images)
    if total_slices == 0:
        st.info("No slices found")
        return

    # Create three columns: Prev button, image, Next button
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("⟨ Prev"):
            if st.session_state.current_slice > 0:
                st.session_state.current_slice -= 1
                st.rerun()
    with col2:
        st.image(slice_images[st.session_state.current_slice], use_column_width=True)
        st.caption(f"Slice {st.session_state.current_slice + 1} of {total_slices}")
    with col3:
        if st.button("Next ⟩"):
            if st.session_state.current_slice < total_slices - 1:
                st.session_state.current_slice += 1
                st.rerun()

# --- Revised Evaluation Page ---
def evaluate_case():
    case_index = st.session_state.last_case
    if case_index >= total_cases:
        st.markdown("### Evaluation complete. Thank you!")
        if st.button("Reset Evaluation"):
            reset_evaluation()
        return

    case_id = cases[case_index]
    st.session_state.last_case = case_index  # track progress

    # Load CSV row for the current case
    row = labels_df[labels_df['XNATSessionID'] == case_id]
    if row.empty:
        st.error(f"Case {case_id} not found in CSV.")
        return

    ground_truth_report = row['Ground Truth'].values[0]
    ai_report = row['pred'].values[0]
    gt_finding = extract_findings(ground_truth_report)
    ai_finding = extract_findings(ai_report)

    assignments = st.session_state.get('assignments', {})
    if str(case_index) in assignments:
        assign_A = assignments[str(case_index)]
    else:
        assign_A = random.choice([True, False])
        assignments[str(case_index)] = assign_A
        st.session_state.assignments = assignments

    if assign_A:
        reportA_text = ai_report
        reportB_text = ground_truth_report
    else:
        reportA_text = ground_truth_report
        reportB_text = ai_report

    st.markdown(f"### Evaluating Case: **{case_id}** (Case {case_index+1} of {total_cases})")
    
    st.markdown("#### Report A")
    st.text_area("Report A", reportA_text, height=300, key=f"reportA_{case_id}")
    st.markdown("#### Report B")
    st.text_area("Report B", reportB_text, height=300, key=f"reportB_{case_id}")
    
    st.markdown("#### Slice Images")
    display_slice_carousel(case_id)
    
    # --- Correction / Annotations Input ---
    st.markdown("#### Corrections / Annotations")
    # For demonstration, we use a hardcoded list of organs.
    # In practice, you could parse the report to detect these automatically.
    detected_organs = [
        "LIVER", "PORTAL VEIN", "INTRAHEPATIC IVC", "INTRAHEPATIC BILE DUCTS", 
        "COMMON BILE DUCT", "GALLBLADDER", "PANCREAS", "RIGHT KIDNEY", "OTHER FINDINGS"
    ]
    st.write("Select the organs you want to correct:")
    selected_organs = []
    for organ in detected_organs:
        if st.checkbox(organ, key=f"organ_{organ}"):
            selected_organs.append(organ)
    
    if selected_organs:
        st.write("Enter correction details:")
        # Let the user choose one organ from the selected ones for this correction entry.
        organ_to_correct = st.selectbox("Organ to correct", options=selected_organs, key="organ_select")
        predefined_reasons = ["Measurement error", "Misinterpretation", "Missing finding", "Other"]
        reason_choice = st.selectbox("Reason for disagreement", options=predefined_reasons, key="reason_choice")
        if reason_choice == "Other":
            reason_detail = st.text_input("Specify other reason", key="reason_text")
            final_reason = reason_detail if reason_detail else "Other"
        else:
            final_reason = reason_choice
        correction_text = st.text_area("Correction details", key="correction_text")
        
        if st.button("Add Correction"):
            new_correction = {
                "organ": organ_to_correct,
                "reason": final_reason,
                "details": correction_text
            }
            st.session_state["corrections"].append(new_correction)
            st.success("Correction added!")
            st.rerun()
    
    # Display the corrections in a table
    if st.session_state["corrections"]:
        st.markdown("#### Current Corrections")
        corrections_df = pd.DataFrame(st.session_state["corrections"])
        st.table(corrections_df)
    
    # Button to submit all corrections
    if st.button("Submit All Corrections"):
        try:
            corrections = st.session_state["corrections"]
            save_annotations(case_id, corrections)
            st.success("Annotations saved.")
            # Reset corrections list for next case
            st.session_state["corrections"] = []
            st.session_state.last_case = case_index + 1
            st.session_state.current_slice = 0
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save annotations: {e}")

# --- Reset function ---
def reset_evaluation():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- Landing Page ---
def index():
    last_case = st.session_state.last_case
    if last_case >= total_cases:
        st.markdown("### All selected cases have been evaluated. Thank you!")
        if st.button("Reset Evaluation"):
            reset_evaluation()
    else:
        st.markdown("### Welcome to the PE CTPA Survey")
        st.markdown(f"We have **{total_cases}** case(s) to evaluate.")
        if st.button("Start Evaluation"):
            st.session_state.page = "evaluate_case"
            st.rerun()

# --- Main Navigation ---
if st.session_state.page == "index":
    index()
elif st.session_state.page == "evaluate_case":
    evaluate_case()
else:
    index()
