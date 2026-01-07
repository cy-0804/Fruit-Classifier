import os

# Set environment variable to use legacy Keras (Keras 2) within TensorFlow
# This is required to load older models (h5) in newer TensorFlow versions
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
import numpy as np

# Use tensorflow.keras which respects the TF_USE_LEGACY_KERAS flag
from tensorflow.keras.models import load_model

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load model and labels
# Load model and labels
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "keras_model.h5")
labels_path = os.path.join(script_dir, "labels.txt")

model = load_model(model_path, compile=False) 
class_names = open(labels_path, "r").readlines()


# Setup customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class FruitExpertGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🍇 Fruit Freshness Expert System")

        # Make window scalable and responsive
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{min(900, screen_width)}x{min(1050, screen_height)}")
        self.configure(bg="#1e1e2f")

        # Title
        ctk.CTkLabel(self, text="🍓 Upload a Fruit Image", font=("Comic Sans MS", 24, "bold"), text_color="#00ffae").pack(pady=20)

        # Upload Button
        ctk.CTkButton(self, text="📤 Upload Image", command=self.upload_image).pack(pady=10)

        # Image display
        self.image_panel = tk.Label(self, bg="#1e1e2f")
        self.image_panel.pack(pady=10)

        # Toggle options
        ctk.CTkLabel(self, text="🔍 Fruit Physical Observations", font=("Arial", 18, "bold")).pack(pady=5)

        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(pady=5)

        self.peel = ctk.CTkCheckBox(self.options_frame, text="Peel is wrinkled")
        self.smell = ctk.CTkCheckBox(self.options_frame, text="Smells sour")
        self.color = ctk.CTkCheckBox(self.options_frame, text="Color is brownish")
        self.firm = ctk.CTkCheckBox(self.options_frame, text="Feels too soft")
        self.mold = ctk.CTkCheckBox(self.options_frame, text="Visible mold or spots")

        for cb in [self.peel, self.smell, self.color, self.firm, self.mold]:
            cb.pack(anchor="w", padx=20)

        # Predict Button
        ctk.CTkButton(self, text="🔮 Predict and Suggest", command=self.predict_and_suggest).pack(pady=15)

        # Results
        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 18))
        self.result_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)

        self.conf_label = ctk.CTkLabel(self, text="", font=("Arial", 14))
        self.conf_label.pack(pady=5)

        self.advice_label = ctk.CTkLabel(self, text="", font=("Arial", 16, "italic"), wraplength=750, justify="center", text_color="#ffde59")
        self.advice_label.pack(pady=30)

        self.image_path = None

    def upload_image(self):
        self.image_path = filedialog.askopenfilename()
        if not self.image_path:
            return

        img = Image.open(self.image_path).convert("RGB")
        img_for_display = img.copy()
        img_for_display.thumbnail((300, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_for_display)

        self.image_panel.config(image=photo)
        self.image_panel.image = photo

    def predict_and_suggest(self):
        if not self.image_path:
            self.result_label.configure(text="⚠️ Please upload an image first.")
            return

        img = Image.open(self.image_path).convert("RGB")
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        class_clean = class_name[2:] if class_name.startswith("0 ") or class_name.startswith("1 ") else class_name
        confidence_percent = confidence_score * 100

        # Adjust confidence based on observations
        toggles = sum([
            self.peel.get(),
            self.smell.get(),
            self.color.get(),
            self.firm.get(),
            self.mold.get()
        ])

        if "fresh" in class_clean.lower():
            confidence_percent -= toggles * 5  # Penalize fresh confidence if bad signs
        elif "rotten" in class_clean.lower():
            confidence_percent += toggles * 3  # Boost rotten confidence if bad signs

        # Clamp to 0–100
        confidence_percent = max(0, min(100, confidence_percent))
        confidence_score = confidence_percent / 100

        # Show result
        self.result_label.configure(text=f"🍏 Class: {class_clean}")
        self.progress_bar.set(confidence_score)
        self.conf_label.configure(text=f"Confidence: {confidence_percent:.2f}%")

        # Get dynamic advice
        final_advice = self.get_combined_advice(class_clean.lower())
        self.advice_label.configure(text=final_advice)

    def get_combined_advice(self, label):
        issues = []

        if self.peel.get():
            issues.append("wrinkled peel")
        if self.smell.get():
            issues.append("sour smell")
        if self.color.get():
            issues.append("brownish color")
        if self.firm.get():
            issues.append("soft texture")
        if self.mold.get():
            issues.append("visible mold")

        issue_count = len(issues)

        if issue_count == 0:
            if "fresh" in label:
                return "✅ This fruit looks fresh and shows no physical spoilage.\nFeel free to enjoy it or store it."
            elif "overripe" in label:
                return "⚠️ It looks overripe but shows no spoilage signs.\nUse it soon in cooking or smoothies."
            else:
                return "ℹ️ No signs of spoilage were detected, but it appears visually rotten.\nInspect before consumption."

        elif issue_count == 1:
            issue = issues[0]
            return f"⚠️ Slight concern detected: {issue}.\nConsider using the fruit soon or inspect further."

        elif issue_count == 2:
            return f"⚠️ Two spoilage signs detected: {', '.join(issues)}.\nUse with caution or repurpose it quickly."

        elif issue_count >= 3:
            if "rotten" in label:
                return f"❌ Classified as rotten with spoilage signs: {', '.join(issues)}.\nPlease discard this fruit."
            elif "fresh" in label:
                return f"⚠️ Classified as fresh, but multiple spoilage signs detected: {', '.join(issues)}.\nBetter to avoid eating it."
            else:
                return f"❌ This fruit is likely bad based on both appearance and signs: {', '.join(issues)}.\nDo not consume."

        return "ℹ️ Unable to confidently assess. Please check manually."

if __name__ == "__main__":
    app = FruitExpertGUI()
    app.mainloop()
