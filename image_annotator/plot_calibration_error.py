import numpy as np
import csv
import matplotlib.pyplot as plt
import eefdataset

# Import the csv as a list of dictionaries
def load_annotations(file_path):
    annotations = []
    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(row)
    return annotations

