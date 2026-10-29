import streamlit as st
import tempfile
import os
import numpy as np
import io
import matplotlib.pyplot as plt
import pandas as pd

from ase.io import read
from ase.visualize.plot import plot_atoms
from ase.eos import EquationOfState

# --- PySCF IMPORTS ---
from pyscf import gto, dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import dft as pbc_dft

st.set_page_config(page_title="Babel - DFT Calculator", layout="wide")

st.title("⚛️ Babel: Quantum Chemistry & Solid-State Calculator")
st.markdown("Calculate electronic properties, structure visualizations, and equations of state.(Dr. Raad Haleoot. Mustansiriyah University)")

# Detect if running on Streamlit Cloud (Web Environment)
IS_CLOUD = os.getenv("SERVER_SOFTWARE", "").startswith("streamlit") or "STREAMLIT_SERVER_PORT" in os.environ

# --- SIDEBAR: PARAMETERS & WORKFLOW ---
st.sidebar.header("1. Upload Structure")
uploaded_file = st.sidebar.file_uploader("Upload File (xyz, cif, poscar)", type=['xyz', 'cif', 'poscar'])

st.sidebar.header("2. DFT Parameters")
functional = st.sidebar.selectbox("Exchange-Correlation", ["lda", "pbe", "b3lyp"])

# Basis sets selection
st.sidebar.markdown("**Basis Sets**")
basis_solid = st.sidebar.selectbox("For Solids (Periodic)", ["gth-szv", "gth-dzv", "gth-dzvp"], help="GTH basis sets are required for periodic systems.")
basis_mol = st.sidebar.selectbox("For Molecules (Gas Phase)", ["sto-3g", "3-21g", "6-31g"])

spin = st.sidebar.number_input("Spin (Unpaired Electrons)", min_value=0, value=0)

# --- K-POINT GRID CONTROL ---
st.sidebar.header("3. K-Point Grid (Solids)")

if IS_CLOUD:
    st.sidebar.warning("🌐 Web Mode: K-mesh is limited to 1x1x1 or 2x2x2 to prevent server memory crashes.")
    k_choice = st.sidebar.selectbox("Select K-Mesh", ["1x1x1 (Gamma Point)", "2x2x2 (Coarse Mesh)"])
    if "1x1x1" in k_choice:
        kpts_grid = [1, 1, 1]
    else:
        kpts_grid = [2, 2, 2]
else:
    st.sidebar.info("💻 Local Mode (WSL): Custom K-grid enabled.")
    col_k1, col_k2, col_k3 = st.sidebar.columns(3)
    kx = col_k1.number_input("kx", min_value=1, value=1)
    ky = col_k2.number_input("ky", min_value=1, value=1)
    kz = col_k3.number_input("kz", min_value=1, value=1)
    kpts_grid = [int(kx), int(ky), int(kz)]

st.sidebar.header("4. Choose Workflow")
workflow = st.sidebar.radio("What would you like to calculate?", [
    "👀 1. Visualize Structure",
    "⚡ 2. Single Point (Energy & Gap)",
    "📈 3. Density of States (DOS)",
    "💎 4. Cell Optimization (Energy vs Volume)"
])

# --- HELPER FUNCTIONS ---
def get_image_download_link(fig, filename="plot.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return st.download_button(label=f"🖼️ Download Plot ({filename})", data=buf, file_name=filename, mime="image/png")

def get_csv_download_link(dataframe, filename="data.csv"):
    csv = dataframe.to_csv(index=False).encode('utf-8')
    return st.download_button(label=f"💾 Download Data ({filename})", data=csv, file_name=filename, mime="text/csv")

def build_pbc_cell(atoms, basis):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    atom_str = "".join([f"{sym} {pos[0]} {pos[1]} {pos[2]}\n" for sym, pos in zip(symbols, positions)])
    
    cell = pbc_gto.Cell()
    cell.atom = atom_str
    cell.a = atoms.get_cell()
    cell.basis = basis
    cell.pseudo = 'gth-pade'
    cell.spin = spin
    cell.build()
    return cell

def run_dft(cell_or_mol, is_periodic, kpts_grid):
    if is_periodic:
        if kpts_grid == [1, 1, 1]:
            mf = pbc_dft.RKS(cell_or_mol)
        else:
            kpts = cell_or_mol.make_kpts(kpts_grid)
            mf = pbc_dft.KRKS(cell_or_mol, kpts)
    else:
        mf = dft.RKS(cell_or_mol)
        
    mf.xc = functional
    e_tot = mf.kernel()
    return mf, e_tot

# --- MAIN APP LOGIC ---
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        atoms = read(tmp_file_path)
        is_periodic = any(atoms.pbc)
        hartree_to_ev = 27.2114
        
        active_basis = basis_solid if is_periodic else basis_mol
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Formula:** {atoms.get_chemical_formula()} | **Atoms:** {len(atoms)}")
            st.write(f"**System:** {'Periodic Solid' if is_periodic else 'Molecule'}")
        with col2:
            st.write(f"**Active Basis Set:** `{active_basis}`")
            if is_periodic:
                st.write(f"**Selected K-Points:** {kpts_grid[0]}x{kpts_grid[1]}x{kpts_grid[2]}")
        st.divider()

        # ---------------------------------------------------------
        # WORKFLOW 1: VISUALIZE ONLY
        # ---------------------------------------------------------
        if "1" in workflow:
            st.subheader("👀 Structure Visualization")
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_atoms(atoms, ax, radii=0.5, rotation=('10x,45y,0z'))
            ax.set_axis_off()
            st.pyplot(fig)
            get_image_download_link(fig, "structure.png")

        # ---------------------------------------------------------
        # WORKFLOW 2: SINGLE POINT (Energy + Gap)
        # ---------------------------------------------------------
        elif "2" in workflow:
            if st.button("🚀 Run Calculation"):
                with st.spinner(f"Calculating... (Grid: {kpts_grid[0]}x{kpts_grid[1]}x{kpts_grid[2]})"):
                    if is_periodic:
                        sys_obj = build_pbc_cell(atoms, active_basis)
                    else:
                        atom_str = "".join([f"{s} {p[0]} {p[1]} {p[2]}\n" for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())])
                        sys_obj = gto.Mole(atom=atom_str, basis=active_basis, spin=spin).build()
                    
                    mf, tot_e = run_dft(sys_obj, is_periodic, kpts_grid)
                    
                    if is_periodic and kpts_grid != [1, 1, 1]:
                        mo_e = np.concatenate(mf.mo_energy)
                        mo_o = np.concatenate(mf.mo_occ)
                    else:
                        mo_e = mf.mo_energy[0] if (is_periodic and mf.mo_energy.ndim > 1) else mf.mo_energy
                        mo_o = mf.mo_occ[0] if (is_periodic and mf.mo_occ.ndim > 1) else mf.mo_occ
                    
                    homo_idx = np.where(mo_o > 0)[0][-1]
                    lumo_idx = np.where(mo_o == 0)[0][0]
                    gap_ev = (mo_e[lumo_idx] - mo_e[homo_idx]) * hartree_to_ev
                    
                    st.success("✅ Complete!")
                    st.metric("Total Energy", f"{tot_e:.4f} Hartree")
                    st.metric("Fundamental Gap", f"{gap_ev:.3f} eV")

        # ---------------------------------------------------------
        # WORKFLOW 3: DENSITY OF STATES (Solids only)
        # ---------------------------------------------------------
        elif "3" in workflow:
            if not is_periodic:
                st.error("DOS requires a periodic crystal.")
            else:
                if st.button("📈 Calculate DOS"):
                    with st.spinner("Running DOS calculation..."):
                        cell = build_pbc_cell(atoms, active_basis)
                        mf, _ = run_dft(cell, is_periodic, kpts_grid)
                        
                        if kpts_grid != [1, 1, 1]:
                            energies_ev = np.concatenate(mf.mo_energy) * hartree_to_ev
                        else:
                            energies_ev = mf.mo_energy.flatten() * hartree_to_ev
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(energies_ev, bins=60, color='blue', alpha=0.7, edgecolor='black')
                        ax.set_title(f"Density of States (Grid: {kpts_grid[0]}x{kpts_grid[1]}x{kpts_grid[2]})")
                        ax.set_xlabel("Energy (eV)")
                        ax.set_ylabel("States")
                        ax.axvline(x=0, color='red', linestyle='--', label="Fermi Level (approx)")
                        ax.legend()
                        
                        st.pyplot(fig)
                        get_image_download_link(fig, "DOS_plot.png")

        # ---------------------------------------------------------
        # WORKFLOW 4: EOS (Energy vs Volume)
        # ---------------------------------------------------------
        elif "4" in workflow:
            if not is_periodic:
                st.error("Volume optimization requires a periodic crystal.")
            else:
                if st.button("💎 Run Volume Optimization (EOS)"):
                    with st.spinner("Calculating 5 volume points..."):
                        volumes = []
                        energies = []
                        strains = [0.96, 0.98, 1.00, 1.02, 1.04]
                        progress_bar = st.progress(0)
                        
                        for i, strain in enumerate(strains):
                            scaled_atoms = atoms.copy()
                            scaled_atoms.set_cell(atoms.cell * strain, scale_atoms=True)
                            
                            cell = build_pbc_cell(scaled_atoms, active_basis)
                            mf, e_hartree = run_dft(cell, is_periodic, kpts_grid)
                            
                            volumes.append(scaled_atoms.get_volume())
                            energies.append(e_hartree * hartree_to_ev)
                            progress_bar.progress((i + 1) / len(strains))

                        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
                        v0, e0, B = eos.fit()
                        B_GPa = B * 160.21766 
                        
                        st.success("✅ Volume Optimization Complete!")
                        
                        df_eos = pd.DataFrame({
                            "Volume (Å³)": np.round(volumes, 4),
                            "Energy (eV)": np.round(energies, 4)
                        })
                        
                        st.subheader("📊 Energy vs. Volume Data")
                        st.dataframe(df_eos, use_container_width=True)
                        get_csv_download_link(df_eos, "Energy_Volume_Data.csv")
                        
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Optimized Volume", f"{v0:.2f} Å³")
                        col2.metric("Bulk Modulus", f"{B_GPa:.2f} GPa")
                        
                        st.subheader("📈 Equation of State Curve")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        eos.plot(ax=ax)
                        ax.set_title("Energy vs. Volume")
                        ax.set_ylabel("Energy (eV)")
                        ax.set_xlabel("Volume (Å³)")
                        st.pyplot(fig)
                        get_image_download_link(fig, "EOS_plot.png")

    except Exception as e:
        st.error(f"Calculation Error: {e}")
    finally:
        os.remove(tmp_file_path)
