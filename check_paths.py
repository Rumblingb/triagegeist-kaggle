#!/usr/bin/env python3
"""Kernel to check available directories and data."""
import os

# List /kaggle/input
print("Listing /kaggle/input:")
for root, dirs, files in os.walk('/kaggle/input'):
    for d in dirs:
        print(f"  DIR: {os.path.join(root, d)}")
    for f in files:
        print(f"  FILE: {os.path.join(root, f)}")
    if root.count(os.sep) > 3:  # Don't go too deep
        break

print("\nIf no files found, trying alternatives:")
# Check /kaggle/working
print("\n/kaggle/working:", os.listdir('/kaggle/working') if os.path.isdir('/kaggle/working') else "N/A")
print("/kaggle/src:", os.listdir('/kaggle/src') if os.path.isdir('/kaggle/src') else "N/A")
