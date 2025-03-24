# AdmPModeler

AdmPModeler is a Python-based tool designed to model administrative procedures from their textual descriptions. The goal is to extract structured process models and provide both CSV outputs and visual representations of the derived workflows.

## Features

- **Text-to-Model Conversion**: Parses textual descriptions of administrative procedures and outputs structured process models in CSV format.
- **Flowchart Visualization**: Generates graphical flowcharts of the extracted processes for better understanding and analysis.
- **Case Study Support**: Includes example results from a case study for reference and validation.

## Repository Structure

```
.
├── AdmPModeler.py              # Main script to extract process models and output them in CSV format
├── FlowchartVisualization.py   # Script to visualize process models as flowcharts
├── Requirements.txt            # Dependencies for running both scripts
├── ProcedureTable/             # Folder containing results from the case study
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/.../AdmPModeler.git
cd AdmPModeler
```

2. Install the required packages:

```bash
pip install -r Requirements.txt
```

## Usage

1. **Model the Procedure**

Run the `AdmPModeler.py` script to generate a CSV model from a textual description:

```bash
python AdmPModeler.py
```

2. **Visualize the Process**

Use the `FlowchartVisualization.py` script to generate a visual flowchart of the model:

```bash
python FlowchartVisualization.py
```

## Example Results

Check the `ProcedureTable/` directory for example outputs from a real-world case study.
