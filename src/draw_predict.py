import argparse
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
from PIL import Image, ImageDraw

from predict import PREDICT_TRANSFORM, load_model

class DrawPredictApp:
    def __init__(self, model_path: str, canvas_size: int = 320, brush_size: int = 8):
        self.model_path = model_path
        self.canvas_size = canvas_size
        self.brush_size = max(2, min(6, int(brush_size)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.classes = load_model(model_path, self.device)

        self.root = tk.Tk()
        self.root.title("Chinese Character Draw Predictor")
        self.root.geometry("460x560")
        self.root.resizable(False, False)

        self.status_var = tk.StringVar(value="Draw a single character, then click Predict")
        self.result_var = tk.StringVar(value="Prediction: -")
        self.topk_var = tk.StringVar(value="Top 3: -")

        self._build_ui()
        self._init_drawing_state()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Draw on black board", font=("Segoe UI", 14, "bold"))
        title.pack(anchor=tk.W, pady=(0, 8))

        self.canvas = tk.Canvas(
            frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#444444",
            cursor="crosshair",
        )
        self.canvas.pack(anchor=tk.CENTER, pady=(0, 10))

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(0, 10))

        clear_btn = ttk.Button(button_row, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT)

        predict_btn = ttk.Button(button_row, text="Predict", command=self.predict_canvas)
        predict_btn.pack(side=tk.LEFT, padx=8)

        size_row = ttk.Frame(frame)
        size_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(size_row, text="Pen size").pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=self.brush_size)
        self.brush_label_var = tk.StringVar(value=str(self.brush_size))
        brush_scale = ttk.Scale(
            size_row,
            from_=2,
            to=6,
            orient=tk.HORIZONTAL,
            command=self.on_brush_change,
        )
        brush_scale.set(self.brush_size)
        brush_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Label(size_row, textvariable=self.brush_label_var, width=3).pack(side=tk.LEFT)

        ttk.Label(frame, textvariable=self.result_var, font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(frame, textvariable=self.topk_var, font=("Segoe UI", 10)).pack(anchor=tk.W, pady=(4, 8))
        ttk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 9)).pack(anchor=tk.W)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def _init_drawing_state(self):
        self.pil_img = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self._init_drawing_state()
        self.result_var.set("Prediction: -")
        self.topk_var.set("Top 3: -")
        self.status_var.set("Canvas cleared")

    def on_press(self, event):
        self.last_x, self.last_y = event.x, event.y
        self._draw_dot(event.x, event.y)

    def on_drag(self, event):
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = event.x, event.y

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=self.brush_size,
            capstyle=tk.ROUND,
            smooth=False,
        )
        self.pil_draw.line(
            (self.last_x, self.last_y, event.x, event.y),
            fill=255,
            width=self.brush_size,
        )

        self.last_x, self.last_y = event.x, event.y

    def on_release(self, _event):
        self.last_x = None
        self.last_y = None

    def _draw_dot(self, x, y):
        r = max(1, self.brush_size // 2)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.pil_draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def on_brush_change(self, value):
        self.brush_size = max(2, min(6, int(float(value))))
        self.brush_label_var.set(str(self.brush_size))

    def _prepare_model_input(self) -> torch.Tensor:
        arr = np.array(self.pil_img)
        ys, xs = np.where(arr > 15)

        if ys.size > 0 and xs.size > 0:
            pad = 14
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(arr.shape[1], int(xs.max()) + pad + 1)
            y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
            cropped = self.pil_img.crop((x0, y0, x1, y1))
        else:
            cropped = self.pil_img

        w, h = cropped.size
        side = max(w, h)
        centered = Image.new("L", (side, side), 0)
        centered.paste(cropped, ((side - w) // 2, (side - h) // 2))
        resized = centered.resize((64, 64), Image.LANCZOS)

        return PREDICT_TRANSFORM(resized).unsqueeze(0).to(self.device)

    def predict_canvas(self):
        tensor = self._prepare_model_input()

        with torch.inference_mode():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            top_probs, top_indices = torch.topk(probs, k=min(3, len(self.classes)), dim=1)

        pred_char = self.classes[pred_idx.item()]
        conf_pct = confidence.item() * 100

        self.result_var.set(f"Prediction: {pred_char} ({conf_pct:.1f}%)")

        tops = []
        for idx, prob in zip(top_indices[0], top_probs[0]):
            tops.append(f"{self.classes[idx.item()]} {prob.item() * 100:.1f}%")
        self.topk_var.set("Top 3: " + " | ".join(tops))

        if conf_pct < 60:
            self.status_var.set("Low confidence. Try centering and drawing thicker strokes.")
        else:
            self.status_var.set("Prediction complete")

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Popup drawing board for character prediction")
    parser.add_argument("--model", default="models/cnn.pth", help="Path to model checkpoint")
    parser.add_argument("--canvas-size", type=int, default=320, help="Canvas width/height in pixels")
    parser.add_argument("--brush", type=int, default=4, help="Brush size (2-6)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    app = DrawPredictApp(model_path=args.model, canvas_size=args.canvas_size, brush_size=args.brush)
    app.run()

if __name__ == "__main__":
    main()
