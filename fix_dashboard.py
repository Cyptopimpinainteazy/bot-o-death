#!/usr/bin/env python3
"""
Quick fix for the dashboard indentation issues
"""
import re

# Read the original file
with open('quantum_dashboard.py', 'r') as f:
    content = f.read()

# Fix common indentation issues
# 1. Make sure all method definitions are properly indented
content = re.sub(r'\ndef (main\(\)):', r'\ndef \1:', content)

# 2. Make sure docstrings and function bodies are properly indented
content = re.sub(r'def main\(\):\n"""', r'def main():\n    """', content)
content = re.sub(r'"""Main function to run the dashboard"""\n# Set', r'"""Main function to run the dashboard"""\n    # Set', content)

# 3. Fix any other indentation issues
content = re.sub(r'# Set up the root window\nroot', r'    # Set up the root window\n    root', content)
content = re.sub(r'# Create the dashboard\napp', r'    # Create the dashboard\n    app', content)
content = re.sub(r'# Set up closing handler\nroot\.protocol', r'    # Set up closing handler\n    root.protocol', content)
content = re.sub(r'# Start the main loop\nroot\.mainloop', r'    # Start the main loop\n    root.mainloop', content)

# Write the fixed content back to a new file
with open('quantum_dashboard_fixed.py', 'w') as f:
    f.write(content)

print("Fixed dashboard saved to quantum_dashboard_fixed.py")
print("You can run it with: python quantum_dashboard_fixed.py")
