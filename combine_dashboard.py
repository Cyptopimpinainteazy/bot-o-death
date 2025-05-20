#!/usr/bin/env python3
"""
Utility script to combine the dashboard parts into a single file
"""
import os

# Define the output file
output_file = "quantum_dashboard.py"

# Define the input files
input_files = [
    "enhanced_dashboard/data_manager.py",
    "enhanced_dashboard/dashboard_ui.py",
    "enhanced_dashboard/dashboard_ui_part2.py",
    "enhanced_dashboard/dashboard_ui_part3.py"
]

# Check if all files exist
missing_files = [f for f in input_files if not os.path.exists(f)]
if missing_files:
    print(f"Error: The following files are missing: {missing_files}")
    exit(1)

# Read the content of all files
contents = []

# First file (data_manager.py) - keep everything
with open(input_files[0], 'r') as f:
    contents.append(f.read())

# Second file (dashboard_ui.py) - keep everything
with open(input_files[1], 'r') as f:
    contents.append(f.read())

# Third file (dashboard_ui_part2.py) - remove indentation from the beginning
with open(input_files[2], 'r') as f:
    lines = f.readlines()
    # Check if the first line starts with spaces or tabs
    if lines and lines[0].startswith((' ', '\t')):
        # Find the indentation level (count leading spaces)
        indent = len(lines[0]) - len(lines[0].lstrip())
        # Remove this indentation from all lines
        contents.append(''.join(line[indent:] if line.startswith(' ' * indent) else line for line in lines))
    else:
        contents.append(''.join(lines))

# Fourth file (dashboard_ui_part3.py) - remove indentation from the beginning
with open(input_files[3], 'r') as f:
    lines = f.readlines()
    # Check if the first line starts with spaces or tabs
    if lines and lines[0].startswith((' ', '\t')):
        # Find the indentation level (count leading spaces)
        indent = len(lines[0]) - len(lines[0].lstrip())
        # Remove this indentation from all lines
        contents.append(''.join(line[indent:] if line.startswith(' ' * indent) else line for line in lines))
    else:
        contents.append(''.join(lines))

# Write the combined content to the output file
with open(output_file, 'w') as f:
    # Add a shebang line and imports at the top
    f.write("#!/usr/bin/env python3\n")
    f.write('"""\nEnhanced Quantum Trade AI Dashboard\nA modern and interactive GUI for visualizing trading performance\n"""\n\n')
    f.write("import tkinter as tk\n")
    f.write("from tkinter import ttk, messagebox, PhotoImage\n")
    f.write("import threading\n")
    f.write("import random\n")
    f.write("import time\n")
    f.write("import datetime\n")
    f.write("import sys\n")
    f.write("import os\n")
    f.write("import json\n")
    f.write("from collections import deque\n")
    f.write("from matplotlib.figure import Figure\n")
    f.write("from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg\n")
    f.write("import matplotlib.pyplot as plt\n")
    f.write("from matplotlib import style\n")
    f.write("import matplotlib\n")
    f.write("import numpy as np\n\n")
    
    # Add the data_manager content
    f.write("# Part 1: Data Manager\n")
    f.write(contents[0])
    f.write("\n\n")
    
    # Add the dashboard UI content
    f.write("# Part 2: Dashboard UI\n")
    f.write(contents[1])
    f.write("\n\n")
    
    # Add the dashboard UI part 2 content with proper indentation
    f.write("# Part 3: Dashboard UI - Portfolio and Trades\n")
    f.write(contents[2])
    f.write("\n\n")
    
    # Add the dashboard UI part 3 content with proper indentation
    f.write("# Part 4: Dashboard UI - Strategies and Main\n")
    f.write(contents[3])
    f.write("\n\n")
    
    # Add a main section at the bottom
    f.write("# If this file is run directly, start the dashboard\n")
    f.write("if __name__ == \"__main__\":\n")
    f.write("    try:\n")
    f.write("        main()\n")
    f.write("    except Exception as e:\n")
    f.write("        print(f\"Error starting dashboard: {e}\")\n")

print(f"Combined dashboard files into {output_file}")
print("You can now run the dashboard with: python quantum_dashboard.py")
