import itertools
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import sys
import re

# Define the parameter values in a dictionary
parameters = {
    "base_death_rate_untreated_sam": [0.01, 0.03, 0.05, 0.08, 0.1],
    "mod_wast_incidence__coef": [0.1, 0.3, 0.5, 0.7, 0.9],
    "progression_to_sev_wast__coef": [1, 5, 10, 15, 20],
    "prob_death_after_SAMcare__as_prop_of_death_rate_untreated_sam": [0.85, 0.7, 0.55, 0.4]
}

# Create the parameter combinations
param_names = list(parameters.keys())
param_values = list(parameters.values())
pars_combinations = list(itertools.product(*param_values))

# Function to create a PDF with parameter values
def create_parameter_page(params):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=(200, 45))  # Smaller page size
    c.setFont("Helvetica", 5)  # Set font and size
    y_position = 35
    for name, value in zip(param_names, params):
        c.drawString(10, y_position, f"{name} = {value}")
        y_position -= 10  # Move down for the next parameter
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Base directory path
BASE_PATH = Path("/home/eva/PycharmProjects/TLOmodel/outputs/sejjej5@ucl.ac.uk/wasting/")

# Function to extract the indices from the file name
def extract_indices(file_name):
    match = re.search(r'_(\d+)_(\d+)\.pdf$', file_name)
    return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)

# Process each PDF
def process_pdfs(in_folder_name):
    pdf_dir = BASE_PATH / in_folder_name / "_outcome_figures"
    output_dir = BASE_PATH / in_folder_name / "_outputs_with_pars"
    output_dir.mkdir(exist_ok=True)
    print(f"Processing PDFs in directory: {pdf_dir}")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    # Sort the PDF files by the extracted indices
    pdf_files.sort(key=lambda x: extract_indices(x.name))

    # Extract the draw indices from the file names
    existing_draws = {extract_indices(pdf_file.name)[0] for pdf_file in pdf_files}

    # Filter out the parameter combinations for the missing draws
    filtered_pars_combinations = [params for i, params in enumerate(pars_combinations) if i in existing_draws]

    for pdf_file, params in zip(pdf_files, filtered_pars_combinations):
        print(f"Processing file: {pdf_file}")
        reader = PdfReader(str(pdf_file))
        writer = PdfWriter()

        # Create the parameter page
        parameter_page = create_parameter_page(params)
        parameter_reader = PdfReader(parameter_page)
        writer.add_page(parameter_reader.pages[0])

        # Add the original pages
        for page in reader.pages:
            writer.add_page(page)

        # Save the new PDF with _pars added to the original name
        output_pdf = output_dir / f"{pdf_file.stem}_pars.pdf"
        with open(output_pdf, "wb") as f:
            writer.write(f)
        print(f"Saved updated PDF: {output_pdf}")

    print("PDFs have been updated with parameter pages.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_pars_page_ToAllPDFs.py <folder_name>")
    else:
        folder_name = sys.argv[1]
        process_pdfs(folder_name)

