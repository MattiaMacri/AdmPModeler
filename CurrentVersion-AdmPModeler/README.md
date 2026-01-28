# AdmPModeler (Current Version)

⚠️ This folder contains the current version of AdmPModeler.  
It represents an improved and extended implementation of the project, designed to be more controllable, human-readable, and performance-efficient compared to the initial version.

Here, the current version of AdmPModeler will be referred simply as "AdmPModeler".

---

AdmPModeler is a Python-based tool designed to model administrative procedures from their textual descriptions.

In its current version, AdmPModeler extracts structured process models and provides their representation in three complementary formats:

- **BPMN** – ensures a clear and human-readable understanding of the administrative process  
- **CSV** – contains complete structured information, including process descriptions and activity-level metadata  
- **HTML** – integrates the readability of BPMN with the informational completeness of CSV  

---

## Workflow Overview

The current version takes as input an administrative procedure described in natural language and automatically creates a dedicated folder for that procedure containing the three output files:

- `.bpmn`
- `.csv`
- `.html`

To achieve this, the system relies on three prompt-based extraction steps, two of which use **structured outputs** (JSON-based rather than free-text responses) to improve reliability and controllability.

---

## Repository Structure

├── AdmPModeler.py # Main script to extract process models and output them in BPMN, CSV and HTML formats
├── Requirements.txt # Dependencies for running the tool
├── InputOutputExamples/ # Example inputs and outputs
│ └── Each subfolder contains:
│ ├── input_procedure.docx
│ ├── output_model.csv
│ ├── output_model.bpmn
│ └── output_model.html
---

