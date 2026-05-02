import trimesh
import numpy as np

def export_multipart_stl(input_path, output_path, top_frac=0.10, bot_frac=0.10):
    mesh = trimesh.load(input_path)
    centroids = mesh.triangles_center
    z_coords = centroids[:, 2]
    
    z_max = np.max(z_coords)
    z_min = np.min(z_coords)
    L_z = z_max - z_min
    
    z_neu = z_max - top_frac * L_z
    z_dir = z_min + bot_frac * L_z
    
    mask_N = z_coords > z_neu
    mask_D = z_coords < z_dir
    mask_body = ~(mask_N | mask_D)
    
    masks = [
        ("Gamma_Neumann", mask_N),
        ("Gamma_Dirichlet", mask_D),
        ("Gamma_Body", mask_body)
    ]
    
    with open(output_path, 'w') as f:
        for name, mask in masks:
            f.write(f"solid {name}\n")
            normals = mesh.face_normals[mask]
            faces = mesh.faces[mask]
            
            for n, face in zip(normals, faces):
                v0, v1, v2 = mesh.vertices[face]
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {name}\n")