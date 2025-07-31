Of course\! I've substantially enhanced the Streamlit application based on your requirements.

This advanced version includes:

  * **A professional UI** with a blue color scheme matching the Brafe logo.
  * **The official Brafe logo** integrated into the UI and the final PDF reports.
  * **Multiple Root Cause Analysis (RCA) techniques** (5 Whys, Fishbone Diagram, Pareto Analysis) with descriptions.
  * **Digital signature pads** for capturing approvals on reports.
  * **Enhanced PDF reports** that dynamically include the chosen RCA technique details and signatures.
  * **More robust code structure**, including data loading from GitHub for persistence.

Here is the complete, advanced code for your application.

-----

### **Instructions for Setup**

1.  **Save the Logo**: Save the logo you provided as `brafe-logo.png` in the same directory as your Python script.

2.  **Create a Configuration File**: In your project directory, create a new folder named `.streamlit`. Inside that folder, create a file named `config.toml` and add the following content to set the blue color theme.

    ```toml
    # .streamlit/config.toml
    [theme]
    primaryColor="#002855"
    backgroundColor="#F0F2F6"
    secondaryBackgroundColor="#FFFFFF"
    textColor="#1E1E1E"
    font="sans serif"
    ```

3.  **Create a Secrets File**: For deploying on Streamlit Cloud, create a file named `.streamlit/secrets.toml` to store your GitHub credentials securely.

    ```toml
    # .streamlit/secrets.toml
    GITHUB_USER = "YourGitHubUsername"
    GITHUB_REPO = "YourGitHubRepoName"
    GITHUB_TOKEN = "YourGitHubPersonalAccessToken"
    ```

4.  **Install Required Libraries**: You'll need to install all the necessary Python libraries.

    ```bash
    pip install streamlit streamlit-option-menu pandas fpdf2 Pillow PyGithub streamlit-drawable-canvas matplotlib plotly
    ```

-----

### **Advanced Application Code (`app.py`)**

```python
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from fpdf import FPDF
from PIL import Image
import io
import base64
import os
from datetime import datetime
from github import Github, InputGitTreeElement, UnknownObjectException
import plotly.express as px
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Brafe Engineering QMS",
    page_icon="brafe-logo.png",
    layout="wide"
)

# --- INITIALIZE SESSION STATE ---
def init_session_state():
    defaults = {
        'rca_records': [],
        'capa_records': [],
        'car_counter': 1,
        'user': 'Default User',
        'data_loaded': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- GITHUB INTEGRATION ---
import os # Make sure this is at the top of your file

@st.cache_resource
def get_github_repo():
    """Returns a connection to the GitHub repository."""
    try:
        # Read credentials from environment variables on Render
        token = os.environ.get("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")
        user = os.environ.get("maliktabishooo")
        repo_name = os.environ.get("brafe-rca-capa")

        if not all([token, user, repo_name]):
            st.sidebar.error("GitHub credentials not found in environment variables.")
            return None
            
        g = Github(token)
        repo = g.get_repo(f"{user}/{repo_name}")
        return repo
    except Exception as e:
        st.sidebar.error(f"GitHub connection failed: {e}")
        return None

def load_data_from_github():
    """Loads RCA and CAPA data from CSV files in the GitHub repo."""
    if st.session_state.data_loaded:
        return
    repo = get_github_repo()
    if not repo:
        return

    for file_name, state_key in [("rca_data.csv", "rca_records"), ("capa_data.csv", "capa_records")]:
        try:
            content = repo.get_contents(file_name).decoded_content.decode()
            df = pd.read_csv(io.StringIO(content))
            # Convert specific columns from string representations of lists/dicts back to objects
            if state_key == 'rca_records':
                df['five_whys'] = df['five_whys'].apply(eval)
                df['images'] = df['images'].apply(eval)
                df['technique_details'] = df['technique_details'].apply(eval)
            st.session_state[state_key] = df.to_dict('records')
        except UnknownObjectException:
            st.sidebar.warning(f"'{file_name}' not found. Starting fresh.")
        except Exception as e:
            st.sidebar.error(f"Error loading {file_name}: {e}")

    # Set initial CAR counter based on loaded data
    if st.session_state.capa_records:
        last_car = max([rec.get('car_number', 'CAR-2025-000') for rec in st.session_state.capa_records])
        try:
            st.session_state.car_counter = int(last_car.split('-')[-1]) + 1
        except (ValueError, IndexError):
            st.session_state.car_counter = len(st.session_state.capa_records) + 1

    st.session_state.data_loaded = True

def save_data_to_github():
    """Saves the current RCA and CAPA data to CSV files on GitHub."""
    repo = get_github_repo()
    if not repo:
        st.error("Cannot save: GitHub repository not connected.")
        return False

    try:
        # Convert session state records to DataFrames
        rca_df = pd.DataFrame(st.session_state.rca_records)
        capa_df = pd.DataFrame(st.session_state.capa_records)

        # Create file elements for commit
        files_to_commit = [
            InputGitTreeElement("rca_data.csv", "100644", "blob", rca_df.to_csv(index=False)),
            InputGitTreeElement("capa_data.csv", "100644", "blob", capa_df.to_csv(index=False))
        ]

        # Commit to GitHub
        commit_message = f"QMS data update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        master_ref = repo.get_git_ref("heads/main")
        base_tree = repo.get_git_tree(master_ref.object.sha)
        tree = repo.create_git_tree(files_to_commit, base_tree)
        parent = repo.get_git_commit(master_ref.object.sha)
        commit = repo.create_git_commit(commit_message, tree, [parent])
        master_ref.edit(commit.sha)

        st.success("Data successfully saved to GitHub!")
        return True
    except Exception as e:
        st.error(f"Error saving to GitHub: {e}")
        return False

# --- PDF REPORT GENERATOR ---
class PDF(FPDF):
    def header(self):
        self.image('brafe-logo.png', 10, 8, 40)
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Corrective Action Report (CAR)', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(0, 40, 85) # Brafe Blue
        self.set_text_color(255, 255, 255)
        self.cell(0, 7, title, 0, 1, 'L', True)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def chapter_body(self, data):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, data)
        self.ln()

    def key_value_pair(self, key, value):
        self.set_font('Arial', 'B', 11)
        self.cell(50, 6, key)
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, str(value))
        self.ln(2)

def generate_pdf(rca_data, capa_data):
    pdf = PDF()
    pdf.add_page()

    # --- Document Header ---
    pdf.chapter_title('1. Report Details')
    pdf.key_value_pair('CAR Number:', capa_data.get('car_number', 'N/A'))
    pdf.key_value_pair('Generated By:', rca_data.get('generated_by', 'N/A'))
    pdf.key_value_pair('Generated Date:', datetime.now().strftime("%Y-%m-%d"))
    pdf.key_value_pair('Record Type:', rca_data.get('record_type', 'N/A'))
    if rca_data.get('record_type') == 'Customer':
        pdf.key_value_pair('Customer Name:', rca_data.get('customer_name', 'N/A'))
    pdf.key_value_pair('PO Number:', rca_data.get('po_number', 'N/A'))
    pdf.key_value_pair('Work Order:', rca_data.get('work_order', 'N/A'))
    pdf.ln(5)

    # --- Problem Description ---
    pdf.chapter_title('2. Problem Description')
    pdf.chapter_body(rca_data.get('problem_description', ''))
    pdf.ln(5)

    # --- Root Cause Analysis ---
    pdf.chapter_title(f"3. Root Cause Analysis ({rca_data.get('rca_technique', 'N/A')})")
    technique = rca_data.get('rca_technique')
    details = rca_data.get('technique_details', {})

    if technique == '5 Whys':
        for i, why in enumerate(details.get('whys', []), 1):
            pdf.key_value_pair(f'Why {i}:', why)
    elif technique == 'Fishbone Diagram':
        for category, causes in details.get('fishbone', {}).items():
            if causes:
                pdf.key_value_pair(f'{category}:', ', '.join(causes))
    elif technique == 'Pareto Analysis':
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 5, "Pareto analysis identifies the most significant factors in a set of data.")
        pdf.ln(2)
        if 'pareto_chart_path' in details and os.path.exists(details['pareto_chart_path']):
            pdf.image(details['pareto_chart_path'], w=180)
            os.remove(details['pareto_chart_path']) # Clean up temp file
    pdf.ln(5)

    # --- CAPA Details ---
    pdf.chapter_title('4. Corrective & Preventive Actions')
    pdf.key_value_pair('Corrective Action:', capa_data.get('corrective_action', ''))
    pdf.key_value_pair('Preventive Action:', capa_data.get('preventive_action', ''))
    pdf.key_value_pair('Responsible Person:', capa_data.get('responsible', ''))
    pdf.key_value_pair('Due Date:', capa_data.get('due_date', ''))
    pdf.key_value_pair('Status:', capa_data.get('status', ''))
    pdf.ln(5)

    # --- Signatures ---
    pdf.chapter_title('5. Approvals')
    sig_y_pos = pdf.get_y()
    
    def add_signature(label, sig_data, x_pos):
        pdf.set_font('Arial', 'B', 10)
        pdf.set_y(sig_y_pos)
        pdf.set_x(x_pos)
        pdf.cell(80, 5, label, 0, 2, 'C')
        if sig_data:
            try:
                img_bytes = base64.b64decode(sig_data.split(",")[1])
                img = Image.open(io.BytesIO(img_bytes))
                temp_sig_path = f"temp_sig_{label}.png"
                img.save(temp_sig_path)
                pdf.image(temp_sig_path, x=x_pos + 15, y=sig_y_pos + 8, w=50)
                os.remove(temp_sig_path)
            except Exception:
                pass
        pdf.set_font('Arial', 'I', 9)
        pdf.set_y(sig_y_pos + 35)
        pdf.set_x(x_pos)
        pdf.cell(80, 5, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
        pdf.line(x_pos + 10, sig_y_pos + 35, x_pos + 70, sig_y_pos + 35)

    add_signature("Responsible Person", capa_data.get('responsible_sig'), 20)
    add_signature("QA Approval", capa_data.get('approver_sig'), 110)
    pdf.ln(40)


    # --- Evidence Images ---
    if rca_data.get('images'):
        pdf.add_page()
        pdf.chapter_title('6. Evidence Photos')
        for img_data in rca_data['images']:
            try:
                img_bytes = base64.b64decode(img_data.split(",")[1])
                img = Image.open(io.BytesIO(img_bytes))
                temp_file = "temp_evidence.png"
                img.save(temp_file)
                pdf.image(temp_file, x=10, w=180)
                pdf.ln(5)
                os.remove(temp_file)
            except Exception as e:
                pdf.cell(0, 10, f"Error processing image: {e}", 0, 1)

    return pdf.output(dest='S').encode('latin1')

# --- UI RENDERING FUNCTIONS ---
def render_dashboard():
    st.title("📊 QMS Dashboard")
    st.markdown("Overview of the Quality Management System status at Brafe Engineering.")

    rca_df = pd.DataFrame(st.session_state.rca_records)
    capa_df = pd.DataFrame(st.session_state.capa_records)

    if rca_df.empty:
        st.info("No data available. Create your first RCA record to get started.")
        return

    # KPI Metrics
    c1, c2, c3 = st.columns(3)
    open_capas = capa_df[capa_df['status'] != 'Closed'] if not capa_df.empty else pd.DataFrame()
    c1.metric("Total RCA Records", len(rca_df))
    c2.metric("Total CAPA Records", len(capa_df))
    c3.metric("Open CAPAs", len(open_capas))

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CAPA Status Distribution")
        if not capa_df.empty:
            status_counts = capa_df['status'].value_counts()
            fig = px.pie(status_counts, values=status_counts.values, names=status_counts.index, hole=.3)
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.text("No CAPA data.")

    with c2:
        st.subheader("RCA Record Types")
        type_counts = rca_df['record_type'].value_counts()
        fig = px.bar(type_counts, x=type_counts.index, y=type_counts.values, labels={'x': 'Record Type', 'y': 'Count'})
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Recent Activity Log")
    if not capa_df.empty and not rca_df.empty:
        merged_df = pd.merge(rca_df, capa_df, left_on='id', right_on='rca_id', how='left', suffixes=('_rca', '_capa'))
        st.dataframe(merged_df[['id_rca', 'problem_description', 'status', 'due_date']].tail(), use_container_width=True)
    else:
        st.dataframe(rca_df[['id', 'problem_description', 'record_type']].tail(), use_container_width=True)

def render_create_rca():
    st.title("📝 Create Root Cause Analysis")
    
    rca_techniques = {
        "5 Whys": "A simple, iterative technique to explore the cause-and-effect relationships underlying a problem. Ask 'Why?' repeatedly until the root cause is identified.",
        "Fishbone Diagram": "Also known as an Ishikawa diagram, it helps visualize potential causes of a problem by grouping them into major categories (e.g., Manpower, Method, Machine, Material, Measurement, Environment).",
        "Pareto Analysis": "A statistical technique that uses the 80/20 rule to identify the most significant factors from a list of many. It helps prioritize efforts on the 'vital few' causes that have the largest impact."
    }

    with st.form("rca_form"):
        st.subheader("1. General Information")
        c1, c2 = st.columns(2)
        with c1:
            record_type = st.radio("Record Type", ["Internal", "Customer"])
            customer_name = st.text_input("Customer Name", disabled=(record_type == "Internal"))
        with c2:
            po_number = st.text_input("Purchase Order (PO)")
            work_order = st.text_input("Work Order")

        st.subheader("2. Problem Details")
        problem_description = st.text_area("Problem Description", height=100, placeholder="Clearly describe the issue, what happened, and where it was observed.")
        
        st.subheader("3. Root Cause Analysis Technique")
        technique = st.selectbox("Select RCA Technique", options=list(rca_techniques.keys()))
        with st.expander("What is this technique?"):
            st.info(rca_techniques[technique])

        # Dynamic form for selected RCA technique
        technique_details = {}
        if technique == '5 Whys':
            whys = [st.text_input(f"Why {i+1}?", key=f"why{i}") for i in range(5)]
            technique_details['whys'] = [w for w in whys if w]
        
        elif technique == 'Fishbone Diagram':
            fishbone_data = {}
            categories = ['Manpower', 'Method', 'Machine', 'Material', 'Measurement', 'Environment']
            cols = st.columns(3)
            for i, cat in enumerate(categories):
                with cols[i % 3]:
                    fishbone_data[cat] = st.text_area(f"Causes for {cat}", height=100, key=f"fishbone_{cat}").split('\n')
                    fishbone_data[cat] = [c.strip() for c in fishbone_data[cat] if c.strip()]
            technique_details['fishbone'] = fishbone_data

        elif technique == 'Pareto Analysis':
            st.markdown("Enter defect types and their frequency. The system will generate a Pareto chart.")
            if 'pareto_items' not in st.session_state:
                st.session_state.pareto_items = [{"cause": "", "frequency": 1}]
            
            pareto_data = []
            for i, item in enumerate(st.session_state.pareto_items):
                c1, c2, c3 = st.columns([4, 2, 1])
                cause = c1.text_input("Cause", value=item["cause"], key=f"cause_{i}")
                freq = c2.number_input("Frequency", value=item["frequency"], min_value=1, key=f"freq_{i}")
                if c3.button("🗑️", key=f"del_{i}"):
                    st.session_state.pareto_items.pop(i)
                    st.rerun()
                pareto_data.append({"cause": cause, "frequency": freq})
            
            if st.button("Add Cause", use_container_width=True):
                st.session_state.pareto_items.append({"cause": "", "frequency": 1})
                st.rerun()
            technique_details['pareto'] = pareto_data

        st.subheader("4. Evidence")
        images = st.file_uploader("Upload Evidence Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        st.divider()
        submitted = st.form_submit_button("✅ Save RCA Record", use_container_width=True)

        if submitted:
            if not problem_description or not technique:
                st.error("Problem Description and an RCA technique are required!")
            else:
                image_data = []
                for img in images:
                    img_bytes = img.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode()
                    image_data.append(f"data:{img.type};base64,{img_base64}")
                
                rca_record = {
                    "id": f"RCA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    "record_type": record_type,
                    "customer_name": customer_name if record_type == "Customer" else "",
                    "po_number": po_number,
                    "work_order": work_order,
                    "problem_description": problem_description,
                    "rca_technique": technique,
                    "technique_details": technique_details,
                    "five_whys": technique_details.get('whys', []), # Legacy compatibility
                    "generated_by": st.session_state.user,
                    "created_at": datetime.now().isoformat(),
                    "images": image_data
                }
                st.session_state.rca_records.append(rca_record)
                st.success("RCA record created successfully! You can now create a CAPA for it.")
                if 'pareto_items' in st.session_state:
                    del st.session_state.pareto_items # Clean up

def render_create_capa():
    st.title("🛡️ Create Corrective/Preventive Action (CAPA)")

    if not st.session_state.rca_records:
        st.warning("No RCA records found. Please create an RCA first.")
        return

    rca_options = {rca['id']: f"{rca['id']} - {rca.get('problem_description', 'N/A')[:50]}..." for rca in st.session_state.rca_records}
    selected_rca_id = st.selectbox("Select the RCA record to address:", options=list(rca_options.keys()), format_func=lambda x: rca_options[x])
    
    rca_data = next((r for r in st.session_state.rca_records if r['id'] == selected_rca_id), None)

    if rca_data:
        with st.expander("Selected RCA Details", expanded=True):
            st.write(f"**Problem:** {rca_data['problem_description']}")
            st.write(f"**RCA Technique:** {rca_data.get('rca_technique', 'N/A')}")

        with st.form("capa_form"):
            st.subheader("1. Action Plan")
            car_number = st.text_input("CAR Number", value=f"CAR-{datetime.now().year}-{st.session_state.car_counter:03d}", disabled=True)
            
            c1, c2 = st.columns(2)
            with c1:
                corrective_action = st.text_area("Corrective Action", height=150, help="Actions to eliminate the cause of the detected non-conformity.")
            with c2:
                preventive_action = st.text_area("Preventive Action", height=150, help="Actions to prevent the recurrence of the non-conformity.")

            st.subheader("2. Assignment and Status")
            c1, c2, c3 = st.columns(3)
            with c1:
                responsible = st.text_input("Responsible Person/Team")
            with c2:
                due_date = st.date_input("Due Date")
            with c3:
                status = st.selectbox("Status", ["Open", "In Progress", "Completed", "Closed"])

            st.subheader("3. Signatures")
            c1, c2 = st.columns(2)
            with c1:
                st.write("Responsible Person's Signature")
                responsible_sig = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2, stroke_color="#000000",
                    background_color="#FFFFFF", height=150, key="responsible_sig_canvas"
                )
            with c2:
                st.write("QA Approver's Signature")
                approver_sig = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)", stroke_width=2, stroke_color="#000000",
                    background_color="#FFFFFF", height=150, key="approver_sig_canvas"
                )

            submitted = st.form_submit_button("✅ Save CAPA Record", use_container_width=True)

            if submitted:
                if not corrective_action or not preventive_action or not responsible:
                    st.error("Corrective/Preventive Actions and Responsible Person are required!")
                else:
                    capa_record = {
                        "id": f"CAPA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        "rca_id": selected_rca_id,
                        "car_number": car_number,
                        "corrective_action": corrective_action,
                        "preventive_action": preventive_action,
                        "responsible": responsible,
                        "due_date": due_date.isoformat(),
                        "status": status,
                        "responsible_sig": responsible_sig.image_data if responsible_sig else None,
                        "approver_sig": approver_sig.image_data if approver_sig else None,
                        "created_at": datetime.now().isoformat()
                    }
                    st.session_state.capa_records.append(capa_record)
                    st.session_state.car_counter += 1
                    st.success("CAPA record created successfully!")

def render_generate_report():
    st.title("📄 Generate CAR Report")

    if not st.session_state.capa_records:
        st.warning("No CAPA records found. Create a CAPA first.")
        return

    capa_options = {capa['car_number']: f"{capa['car_number']} - RCA: {capa['rca_id']}" for capa in st.session_state.capa_records}
    selected_car_number = st.selectbox("Select a CAR to generate a PDF report:", options=list(capa_options.keys()), format_func=lambda x: capa_options[x])

    capa_data = next((c for c in st.session_state.capa_records if c.get('car_number') == selected_car_number), None)
    
    if capa_data:
        rca_data = next((r for r in st.session_state.rca_records if r['id'] == capa_data['rca_id']), None)
        if not rca_data:
            st.error("Linked RCA record not found! Cannot generate report.")
            return

        st.subheader(f"Preview for {selected_car_number}")
        # Special handling for Pareto chart generation before PDF
        if rca_data.get('rca_technique') == 'Pareto Analysis':
            pareto_details = rca_data.get('technique_details', {}).get('pareto', [])
            if pareto_details:
                df = pd.DataFrame(pareto_details).sort_values('frequency', ascending=False)
                df['cumulative_percentage'] = (df['frequency'].cumsum() / df['frequency'].sum()) * 100
                
                fig, ax1 = plt.subplots()
                ax1.bar(df['cause'], df['frequency'], color='C0')
                ax1.set_ylabel('Frequency', color='C0')
                ax1.tick_params(axis='y', labelcolor='C0')
                ax1.tick_params(axis='x', rotation=45)
                
                ax2 = ax1.twinx()
                ax2.plot(df['cause'], df['cumulative_percentage'], color='C1', marker='o', ms=5)
                ax2.set_ylabel('Cumulative Percentage', color='C1')
                ax2.tick_params(axis='y', labelcolor='C1')
                ax2.set_ylim([0, 110])
                
                plt.title('Pareto Analysis')
                fig.tight_layout()
                
                # Save to a temporary file for PDF embedding
                chart_path = f"temp_pareto_{rca_data['id']}.png"
                fig.savefig(chart_path)
                rca_data['technique_details']['pareto_chart_path'] = chart_path


        st.info("Click the button below to generate and download the formal PDF document.")
        if st.button("🚀 Generate PDF Report", use_container_width=True):
            with st.spinner("Creating your report..."):
                pdf_bytes = generate_pdf(rca_data, capa_data)
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=f"{capa_data['car_number']}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    else:
        st.warning("Please select a valid CAR.")

def render_settings():
    st.title("⚙️ System Settings")

    st.subheader("User Preferences")
    username = st.text_input("Your Name (for 'Generated By' field)", value=st.session_state.get('user', ''))
    if st.button("Save User Preference"):
        st.session_state.user = username
        st.success("Username updated!")

    st.divider()

    st.subheader("System Data Management")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save All Data to GitHub", use_container_width=True):
            save_data_to_github()

    with c2:
        if st.button("⚠️ Reset All Local Data", type="primary", use_container_width=True):
            if st.checkbox("Confirm data reset"):
                for key in ['rca_records', 'capa_records', 'car_counter', 'data_loaded']:
                    if key in st.session_state:
                        del st.session_state[key]
                init_session_state()
                st.success("All local session data has been reset!")
                st.rerun()

    st.info(f"""
    **Current Session Info:**
    - **RCA Records:** {len(st.session_state.rca_records)}
    - **CAPA Records:** {len(st.session_state.capa_records)}
    - **Next CAR Number:** CAR-{datetime.now().year}-{st.session_state.car_counter:03d}
    - **GitHub Connection:** {'Active' if get_github_repo() else 'Inactive'}
    """)

# --- MAIN APP LOGIC ---
def main():
    init_session_state()

    # Load data from GitHub at the start of the session
    if 'GITHUB_TOKEN' in st.secrets and not st.session_state.data_loaded:
        with st.spinner("Loading data from repository..."):
            load_data_from_github()

    with st.sidebar:
        st.image("brafe-logo.png")
        st.markdown(f"Welcome, **{st.session_state.user}**")
        selected = option_menu(
            menu_title="Main Menu",
            options=["Dashboard", "Create RCA", "Create CAPA", "Generate Report", "Settings"],
            icons=["speedometer2", "journal-plus", "shield-check", "file-earmark-pdf", "gear-wide-connected"],
            menu_icon="cast",
            default_index=0
        )
        st.sidebar.divider()
        st.sidebar.info("This QMS application is for internal use at Brafe Engineering.")

    page_map = {
        "Dashboard": render_dashboard,
        "Create RCA": render_create_rca,
        "Create CAPA": render_create_capa,
        "Generate Report": render_generate_report,
        "Settings": render_settings
    }
    
    page_map[selected]()

if __name__ == "__main__":
    main()
```
