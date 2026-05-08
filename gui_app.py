import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np

# --- CONFIGURACIÓN ESTÉTICA (MOCKUP STYLE) ---
COLOR_BG = "#0f111a"  # Azul muy oscuro (Deep Space)
COLOR_SIDEBAR = "#161925"
COLOR_ACCENT_BLUE = "#00d2ff" # Azul neón
COLOR_ACCENT_PINK = "#ff007f" # Fucsia neón
COLOR_TEXT = "#e0e0e0"

ctk.set_appearance_mode("Dark")

# Importaciones locales del pipeline
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.neural_manifold.inference import predict_volume_from_dicom
from src.neural_manifold.segment_pde import process_and_save_dl_mesh

class BoneFlowApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("BoneFlow AI | Medical Computing Platform")
        self.geometry("1200x800")
        self.configure(fg_color=COLOR_BG)

        # Configuración del grid principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR (Look del Mockup) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="BONEFLOW AI", 
                                      font=ctk.CTkFont(size=22, weight="bold", family="Orbitron"))
        self.logo_label.pack(pady=(40, 60), padx=20)

        # Botones de la Sidebar con "iconos" de texto
        self.btn_upload = self.create_sidebar_button("📂  UPLOAD DICOM", self.select_dicom)
        self.btn_process = self.create_sidebar_button("🧠  PROCESS BONE", self.start_pipeline_thread, state="disabled")
        self.btn_view = self.create_sidebar_button("👁️  3D VIEW", None, state="disabled")
        self.btn_export = self.create_sidebar_button("📥  EXPORT FEM", None, state="disabled")

        self.status_indicator = ctk.CTkLabel(self.sidebar_frame, text="● SYSTEM READY", 
                                            text_color=COLOR_ACCENT_BLUE, font=ctk.CTkFont(size=10, weight="bold"))
        self.status_indicator.pack(side="bottom", pady=20)

        # --- MAIN PANEL ---
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=30, pady=30)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1) # El visor 3D toma el centro

        # Header con info del paciente
        self.header_label = ctk.CTkLabel(self.main_container, text="DASHBOARD / ANALYTICS", 
                                       font=ctk.CTkFont(size=14, weight="bold"), text_color=COLOR_ACCENT_BLUE)
        self.header_label.grid(row=0, column=0, sticky="nw", pady=(0, 20))

        # --- VISUALIZADOR 3D (ZONA CENTRAL) ---
        self.viewport_frame = ctk.CTkFrame(self.main_container, fg_color="#05060a", corner_radius=15, border_width=1, border_color="#1a1c26")
        self.viewport_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 20))
        
        self.viewport_label = ctk.CTkLabel(self.viewport_frame, text="3D VIEWPORT READY\nWaiting for processing...", 
                                         text_color="#3a3c4a", font=ctk.CTkFont(size=16))
        self.viewport_label.place(relx=0.5, rely=0.5, anchor="center")

        # --- CONSOLA Y PROGRESO (ZONA INFERIOR) ---
        self.bottom_frame = ctk.CTkFrame(self.main_container, fg_color=COLOR_SIDEBAR, height=200, corner_radius=15)
        self.bottom_frame.grid(row=2, column=0, sticky="ew")
        self.bottom_frame.grid_propagate(False)
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.log_textbox = ctk.CTkTextbox(self.bottom_frame, fg_color="transparent", text_color="#a0a0a0", font=("Consolas", 11))
        self.log_textbox.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="nsew")
        self.log_textbox.insert("0.0", "> Welcome to BoneFlow AI Platform. Initialize scan to begin analysis.\n")

        self.progress_bar = ctk.CTkProgressBar(self.bottom_frame, height=4, progress_color=COLOR_ACCENT_PINK, fg_color="#1a1c26")
        self.progress_bar.grid(row=1, column=0, padx=20, pady=15, sticky="ew")
        self.progress_bar.set(0)

        self.footer_status = ctk.CTkLabel(self.bottom_frame, text="IDLE", text_color="#5a5c6a", font=ctk.CTkFont(size=10))
        self.footer_status.grid(row=2, column=0, pady=(0, 10))

        # Variables de estado
        self.dicom_path = ""
        self.model_path = os.path.join("data", "03_models", "unet_bone_topology_ep16.pth")

    def create_sidebar_button(self, text, command, state="normal"):
        btn = ctk.CTkButton(self.sidebar_frame, text=text, command=command, state=state,
                           anchor="w", fg_color="transparent", text_color=COLOR_TEXT,
                           hover_color="#1f2335", height=45, font=ctk.CTkFont(size=12, weight="bold"))
        btn.pack(pady=5, padx=10, fill="x")
        return btn

    def log(self, message):
        self.log_textbox.insert("end", f"> {message}\n")
        self.log_textbox.see("end")

    def select_dicom(self):
        path = filedialog.askdirectory()
        if path:
            self.dicom_path = path
            self.log(f"Volume loaded: {os.path.basename(path)}")
            self.btn_process.configure(state="normal", text_color=COLOR_ACCENT_PINK)
            self.footer_status.configure(text=f"READY: {os.path.basename(path)}", text_color=COLOR_ACCENT_BLUE)

    def start_pipeline_thread(self):
        self.btn_process.configure(state="disabled")
        thread = threading.Thread(target=self.run_full_pipeline)
        thread.daemon = True
        thread.start()

    def run_full_pipeline(self):
        try:
            self.log("INITIATING SCAN SEQUENCE...")
            self.progress_bar.set(0.1)
            
            self.log("PHASE 1: NEURAL MANIFOLD INFERENCE...")
            self.footer_status.configure(text="PROCESSING AI...", text_color=COLOR_ACCENT_PINK)
            
            if not os.path.exists(self.model_path): raise FileNotFoundError("Weights missing.")

            prob_volume = predict_volume_from_dicom(
                dicom_dir=self.dicom_path,
                model_path=self.model_path,
                device_str="cpu"
            )
            
            self.progress_bar.set(0.6)
            self.log("PHASE 1 COMPLETE. MAPPING GEOMETRY...")

            self.log("PHASE 2: TOPOLOGICAL REPAIR & SMOOTHING...")
            self.footer_status.configure(text="GENERATING UNIFIED MESH...", text_color=COLOR_ACCENT_BLUE)
            
            output_dir = os.path.join("data", "02_processed", os.path.basename(self.dicom_path))
            
            # Ahora pasamos el volumen de probabilidad directamente para que segment_pde lo suavice
            process_and_save_dl_mesh(prob_volume, self.dicom_path, output_dir)
            
            self.progress_bar.set(1.0)
            self.log("PIPELINE SEQUENCE COMPLETE.")
            self.footer_status.configure(text="FINISHED", text_color="green")
            messagebox.showinfo("Success", f"Bone Manifold exported to:\n{output_dir}")
            
            self.btn_export.configure(state="normal", text_color=COLOR_ACCENT_BLUE)
            self.btn_view.configure(state="normal", text_color=COLOR_ACCENT_PINK)
            
        except Exception as e:
            self.log(f"CRITICAL SYSTEM ERROR: {str(e)}")
            self.footer_status.configure(text="FAILURE", text_color="red")
            messagebox.showerror("System Error", str(e))
        finally:
            self.btn_process.configure(state="normal")

if __name__ == "__main__":
    app = BoneFlowApp()
    app.mainloop()
