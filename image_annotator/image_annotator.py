import os
import csv
import random
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageAnnotator:
    def __init__(self, root, image_folder, output_file="annotations_test_20251023-1007.csv",
                 max_images=500):
        self.root = root
        self.image_folder = image_folder
        self.output_file = output_file
        self.max_images = max_images

        # Collect images
        self.image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.image_files.sort()  # assume filenames ≈ temporal order

        # Load existing annotations so we can skip them
        self.processed_files = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.processed_files.add(row["filename"])

        # Skip annotated images
        self.image_files = [
            f for f in self.image_files 
            if f not in self.processed_files
        ]

        # Now subsample only remaining (unannotated) images
        if len(self.image_files) > self.max_images:
            n = self.max_images
            total = len(self.image_files)
            indices = [
                int(round(i * (total - 1) / (n - 1)))
                for i in range(n)
            ]

            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            self.image_files = [self.image_files[i] for i in unique_indices]

        # Shuffle to avoid similar frames back-to-back
        random.shuffle(self.image_files)

        self.index = 0

        # Keep track of last click position in percentage
        self.last_x_percent = None
        self.last_y_percent = None

        # UI setup
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        self.save_button = tk.Button(button_frame, text="Save & Next", command=self.save_and_next, state="disabled")
        self.save_button.pack(side="left", padx=5)

        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo, state="disabled")
        self.undo_button.pack(side="left", padx=5)

        # Track click
        self.click_x = None
        self.click_y = None
        self.marker = None

        # Load first image
        self.load_image()

        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Return>", lambda event: self.save_and_next())

        # Initialize output file if missing
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w", newline="") as f:
                f.write("filename,x_percent,y_percent\n")

    def load_image(self):
        if self.index >= len(self.image_files):
            self.canvas.delete("all")
            self.canvas.create_text(200, 200, text="All images processed!", font=("Arial", 16))
            self.save_button.config(state="disabled")
            self.undo_button.config(state="disabled")
            return

        image_path = os.path.join(self.image_folder, self.image_files[self.index])
        self.img = Image.open(image_path)
        self.tk_img = ImageTk.PhotoImage(self.img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(width=self.img.width, height=self.img.height)

        self.click_x = None
        self.click_y = None
        self.marker = None
        self.save_button.config(state="disabled")

        # Enable undo if not on the first image
        self.undo_button.config(state="normal" if self.index > 0 else "disabled")

        # If previous annotation exists, use it as an initial guess
        if self.last_x_percent is not None and self.last_y_percent is not None:
            self.click_x = int(self.last_x_percent * self.img.width)
            self.click_y = int(self.last_y_percent * self.img.height)
            self.draw_marker()
            self.save_button.config(state="normal")  # allow immediate save if desired

    def draw_marker(self):
        """Helper to draw the marker at the current click_x, click_y."""
        if self.marker:
            self.canvas.delete(self.marker)
        r = 5
        self.marker = self.canvas.create_oval(
            self.click_x - r, self.click_y - r,
            self.click_x + r, self.click_y + r,
            outline="red", width=2
        )

    def on_click(self, event):
        self.click_x, self.click_y = event.x, event.y
        self.draw_marker()
        self.save_button.config(state="normal")

    def save_and_next(self):
        if self.click_x is None or self.click_y is None or self.index >= len(self.image_files):
            return

        filename = self.image_files[self.index]
        x_percent = self.click_x / self.img.width
        y_percent = self.click_y / self.img.height

        # Store for next image
        self.last_x_percent = x_percent
        self.last_y_percent = y_percent

        with open(self.output_file, "a", newline="") as f:
            f.write(f"{filename},{x_percent:.4f},{y_percent:.4f}\n")

        # Move to next image
        self.index += 1
        self.load_image()

    def undo(self):
        if self.index == 0:
            return

        with open(self.output_file, "r") as f:
            lines = f.readlines()

        if len(lines) > 1:
            lines = lines[:-1]
            with open(self.output_file, "w") as f:
                f.writelines(lines)

        self.index -= 1
        self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Annotator")

    default_dir = os.path.expanduser("/home/wilah/datasets/heshan_october_grapple_data/20251023-1007/rgb/left/")
    folder = filedialog.askdirectory(title="Select Image Folder", initialdir=default_dir)
    if folder:
        app = ImageAnnotator(root, folder)
        root.mainloop()
    else:
        print("No folder selected.")
