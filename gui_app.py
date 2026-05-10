import os
import sys
import threading
import io
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ESTÉTICA (BONEFLOW 180° STYLE) ---
COLOR_BG = "#0f111a"  # Azul muy oscuro
COLOR_PANEL = "#161925"
COLOR_ACCENT_BLUE = "#00d2ff" # Azul neón
COLOR_ACCENT_PINK = "#ff007f" # Fucsia neón
COLOR_TEXT = "#e0e0e0"

ctk.set_appearance_mode("Dark")

# Importaciones locales del pipeline
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.neural_manifold.inference import predict_volume_from_dicom
from src.neural_manifold.segment_pde import process_and_save_dl_mesh

def render_latex_to_ctk_image(latex_str, text_color="#00d2ff", font_size=14, width=300, height=80):
    """Renderiza una fórmula LaTeX a una imagen transparente usando Matplotlib."""
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.patch.set_alpha(0.0)
    
    # Renderizamos la matemática nativa de matplotlib
    ax.text(0.5, 0.5, f"${latex_str}$", size=font_size, color=text_color, 
            ha='center', va='center', usetex=False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    
    pil_image = Image.open(buf)
    return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(pil_image.width, pil_image.height))

class BoneFlowApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("BoneFlow AI | Computational Biomechanics Engine")
        self.geometry("1400x850")
        self.configure(fg_color=COLOR_BG)

        # Configuración del grid principal (3 Columnas, 2 Filas)
        self.grid_columnconfigure(0, weight=1) # Left Control
        self.grid_columnconfigure(1, weight=3) # Center Viewport
        self.grid_columnconfigure(2, weight=1) # Right Terminal
        self.grid_rowconfigure(1, weight=1)

        # ==========================================
        # TOP HEADER (Branding & Mathematics)
        # ==========================================
        self.header_frame = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=0, height=100)
        self.header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.header_frame.grid_columnconfigure(1, weight=1)
        self.header_frame.grid_columnconfigure(2, weight=1)
        
        # Logo Left
        logo_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=30, pady=15, sticky="w")
        ctk.CTkLabel(logo_frame, text="BONEFLOW AI", font=ctk.CTkFont(size=28, weight="bold", family="Orbitron"), text_color=COLOR_ACCENT_BLUE).pack(anchor="w")
        ctk.CTkLabel(logo_frame, text="v2.0 Neural Manifold Engine", font=ctk.CTkFont(size=12, slant="italic"), text_color="#5a5c6a").pack(anchor="w")

        # Math Center 1: Focal Loss
        try:
            eq1_img = render_latex_to_ctk_image(r"\mathcal{L}_{Total} = \mathcal{L}_{Dice} + \alpha(1-p)^\gamma \log(p)", text_color=COLOR_ACCENT_PINK, font_size=16)
            self.eq1_label = ctk.CTkLabel(self.header_frame, text="", image=eq1_img)
            self.eq1_label.grid(row=0, column=1, pady=15, sticky="e", padx=20)
        except Exception as e:
            print(f"Error rendering eq1: {e}")

        # Math Center 2: Navier Cauchy
        try:
            eq2_img = render_latex_to_ctk_image(r"\partial_k \sigma_{kj} + f_j = 0", text_color=COLOR_TEXT, font_size=16)
            self.eq2_label = ctk.CTkLabel(self.header_frame, text="", image=eq2_img)
            self.eq2_label.grid(row=0, column=2, pady=15, sticky="w", padx=20)
        except Exception as e:
            print(f"Error rendering eq2: {e}")

        # ==========================================
        # LEFT COLUMN (CONTROLS & STATUS)
        # ==========================================
        self.left_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=20)
        self.left_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.left_frame, text="SYSTEM CONTROLS", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_ACCENT_BLUE).pack(pady=(0,20), anchor="w")

        self.btn_upload = self.create_control_button("1. LOAD DICOM TENSOR", self.select_dicom)
        self.btn_export_dir = self.create_control_button("2. SET EXPORT TARGET", self.select_export_dir)
        self.btn_process = self.create_control_button("3. INITIALIZE SOLVER", self.start_pipeline_thread, state="disabled", fg_color="#1f2335", text_color=COLOR_ACCENT_PINK)

        # Status Indicators
        status_box = ctk.CTkFrame(self.left_frame, fg_color=COLOR_PANEL, corner_radius=10)
        status_box.pack(pady=40, fill="x", ipady=10)
        
        self.status_indicator = ctk.CTkLabel(status_box, text="● KERNEL STANDBY", text_color=COLOR_ACCENT_BLUE, font=ctk.CTkFont(size=12, weight="bold"))
        self.status_indicator.pack(pady=(15, 5))
        
        self.progress_bar = ctk.CTkProgressBar(status_box, height=6, progress_color=COLOR_ACCENT_PINK, fg_color="#1a1c26")
        self.progress_bar.pack(padx=20, pady=10, fill="x")
        self.progress_bar.set(0)

        # Export Target display
        self.export_path_label = ctk.CTkLabel(self.left_frame, text="TARGET: [ DEFAULT ]", font=ctk.CTkFont(family="Consolas", size=10), text_color="#a0a0a0", wraplength=250)
        self.export_path_label.pack(side="bottom", anchor="w", pady=10)

        # ==========================================
        # CENTER COLUMN (3D VIEWPORT)
        # ==========================================
        self.center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.center_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=20)
        self.center_frame.grid_columnconfigure(0, weight=1)
        self.center_frame.grid_rowconfigure(0, weight=1)

        self.viewport_frame = ctk.CTkFrame(self.center_frame, fg_color="#05060a", corner_radius=15, border_width=1, border_color="#2a2c3a")
        self.viewport_frame.grid(row=0, column=0, sticky="nsew")
        
        self.viewport_label = ctk.CTkLabel(self.viewport_frame, text="[ 3D MANIFOLD RENDERER ]\n\nAwaiting Tensor Input...", 
                                         text_color="#3a3c4a", font=ctk.CTkFont(family="Consolas", size=18))
        self.viewport_label.place(relx=0.5, rely=0.5, anchor="center")

        # ==========================================
        # RIGHT COLUMN (TELEMETRY TERMINAL)
        # ==========================================
        self.right_frame = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=10, border_width=1, border_color="#2a2c3a")
        self.right_frame.grid(row=1, column=2, sticky="nsew", padx=(10, 20), pady=20)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

        term_header = ctk.CTkLabel(self.right_frame, text="TELEMETRY / LOGS", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT)
        term_header.grid(row=0, column=0, sticky="w", padx=15, pady=15)

        self.log_textbox = ctk.CTkTextbox(self.right_frame, fg_color="#0a0c12", text_color="#00d2ff", font=("Consolas", 11), corner_radius=5)
        self.log_textbox.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.log_textbox.insert("0.0", "> BONEFLOW AI KERNEL BOOT SEQUENCE...\n> SYSTEM READY.\n")

        # Variables de estado
        self.dicom_path = ""
        self.custom_export_dir = "" 
        self.model_path = os.path.join("data", "03_models", "unet_bone_topology_ep37.pth")

    def create_control_button(self, text, command, state="normal", fg_color="transparent", text_color=COLOR_TEXT):
        btn = ctk.CTkButton(self.left_frame, text=text, command=command, state=state,
                           anchor="w", fg_color=fg_color, text_color=text_color,
                           hover_color="#2a2c3a", height=50, font=ctk.CTkFont(size=13, weight="bold"),
                           border_width=1, border_color="#2a2c3a")
        btn.pack(pady=10, fill="x")
        return btn

    def log(self, message):
        self.log_textbox.insert("end", f"> {message}\n")
        self.log_textbox.see("end")

    def select_dicom(self):
        path = filedialog.askdirectory(title="Select DICOM Directory")
        if path:
            self.dicom_path = path
            self.log(f"Tensor Volume linked: {os.path.basename(path)}")
            self.btn_process.configure(state="normal")
            self.status_indicator.configure(text=f"● TENSOR LOADED", text_color=COLOR_ACCENT_BLUE)

    def select_export_dir(self):
        path = filedialog.askdirectory(title="Select Export Destination")
        if path:
            self.custom_export_dir = path
            self.export_path_label.configure(text=f"TARGET:\n{path}", text_color=COLOR_ACCENT_PINK)
            self.log(f"Export route mapped to: {path}")

    def start_pipeline_thread(self):
        self.btn_process.configure(state="disabled")
        thread = threading.Thread(target=self.run_full_pipeline)
        thread.daemon = True
        thread.start()

    def run_full_pipeline(self):
        try:
            self.log("INITIATING STOCHASTIC SOLVER...")
            self.progress_bar.set(0.1)
            
            self.log("PHASE 1: ATTENTION-RESUNET3D INFERENCE...")
            self.status_indicator.configure(text="● COMPUTING PROBABILITIES", text_color=COLOR_ACCENT_PINK)
            
            if not os.path.exists(self.model_path): 
                self.log(f"[!] WARNING: Model not found at {self.model_path}")
                raise FileNotFoundError("Topology weights missing.")

            prob_volume = predict_volume_from_dicom(
                dicom_dir=self.dicom_path,
                model_path=self.model_path,
                device_str="cpu"
            )
            
            self.progress_bar.set(0.6)
            self.log("PHASE 1 COMPLETE. PROBABILISTIC DOMAIN GENERATED.")

            self.log("PHASE 2: WATERTIGHT MANIFOLD RECONSTRUCTION...")
            self.status_indicator.configure(text="● ASSEMBLING FEM MESH", text_color=COLOR_ACCENT_BLUE)
            
            if self.custom_export_dir:
                output_dir = os.path.join(self.custom_export_dir, os.path.basename(self.dicom_path) + "_FEM")
            else:
                output_dir = os.path.join("data", "02_processed", os.path.basename(self.dicom_path))
            
            process_and_save_dl_mesh(prob_volume, self.dicom_path, output_dir)
            
            self.progress_bar.set(1.0)
            self.log(f"PIPELINE SEQUENCE COMPLETE. Mesh exported to:\n{output_dir}")
            self.status_indicator.configure(text="● SOLVER TERMINATED", text_color="#00ff88")
            messagebox.showinfo("Success", f"Bone Manifold computed and exported to:\n{output_dir}")
            
        except Exception as e:
            self.log(f"CRITICAL SYSTEM ERROR: {str(e)}")
            self.status_indicator.configure(text="● KERNEL PANIC", text_color="red")
            messagebox.showerror("System Error", str(e))
        finally:
            self.btn_process.configure(state="normal")

if __name__ == "__main__":
    app = BoneFlowApp()
    app.mainloop()
