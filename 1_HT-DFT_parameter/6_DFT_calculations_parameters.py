import os
import shutil
import numpy as np
from ase.db import connect
from ase.io import read, write
from ase.calculators.vasp import Vasp

def check_convergence(outcar_path):
    with open(outcar_path, 'r') as file:
        content = file.read()
        if "reached required accuracy - stopping structural energy minimisation" in content:
            return True
    return False

def get_kpoints(a, b, c, resolution, model_name):
    k_a = round(2 * np.pi / (a * resolution))
    k_b = round(2 * np.pi / (b * resolution))
    k_c = round(2 * np.pi / (c * resolution))
    if k_c == 0:
        with open('error_1.txt', 'a') as f:
            f.write(f"Adjusted k_c to 1 for model {model_name} with lattice constants a={a}, b={b}, c={c} - k_c computed as 0, skipping this model.\n")
    return k_a, k_b, k_c

def get_lattice_constants_from_POSCAR(filename):
    structure = read(filename)
    a, b, c = structure.cell.lengths()
    return a, b, c

db = connect('initial_db.db')
dos_db = connect('final_opitmized_db.db')
nonconversed_db = connect('nonconversed.db')  
error_db = connect('error.db')

vasp_params = {
    'encut' : 520, 
    'ibrion': 2, 
    'ismear': 0,
    'sigma': 0.05, 
    'xc': 'PBE',
    'setups': {'Ba':'_sv','Ca':'_pv','W': '','Zn':'', 'Y':'_sv', 'Zr':'_sv','Sr':'_sv','Nb':'_pv'}, 
    'npar': 4, 
    'istart': 0, 
    'potim' : 0.1,
    'lreal' : False, 
    'algo' : 'All', 
    'lcharg': False, 
    'lwave': False, 
    'nelm': 500, 
    'nsw': 200,   
    'ediffg' : -0.02,
    'ispin' : 2,
    'ivdw' : 12,
    'prec' : 'Accurate',
    'lasph' : True,
    'addgrid' : True,
    'prec': 'Accurate',
    'amix': 0.1,
    'bmix': 0.00001,
    'maxmix' : 80,
    'amin' : 0.01,
    'isym' : 0,
    'symprec' : 1e-5
}

resolution = 0.2

for row in db.select():
    atoms = row.toatoms()
    atoms.pbc = True
    model_name = row.formula

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    is_existing_model = dos_db.count(formula=model_name) != 0

    if not is_existing_model:
        try:
            write('POSCAR', atoms)
            a, b, c = get_lattice_constants_from_POSCAR('POSCAR')
            k_a, k_b, k_c = get_kpoints(a, b, c, resolution, model_name)
            
            if k_c == 0:
                with open('error_1.txt', 'a') as f:
                    f.write(f"Skipping model {model_name} due to k_c=0\n")
                continue

            vasp_params['kpts'] = (k_a, k_b, k_c)
            calc = Vasp(**vasp_params)
            atoms.set_calculator(calc)
            energy = atoms.get_potential_energy()

            outcar_path = os.path.join(os.getcwd(), "OUTCAR")
            if check_convergence(outcar_path):
                dos_db.write(atoms, model=model_name)
            else:
                with open('error_2.txt', 'a') as f:
                    f.write(f"{model_name}: Did not reach the required accuracy.\n")
                if nonconversed_db.count(model=model_name) == 0:
                    nonconversed_db.write(atoms, model=model_name)

        except Exception as e:
            print(f"Error occurred with {model_name}: {str(e)}")
            with open('error.txt', 'a') as f:
                f.write(f"{model_name}: {str(e)}\n")
            if error_db.count(model=model_name) == 0:
                error_db.write(atoms, model=model_name)
            continue

        finally:
            output_files = [
                'CHG', 'CHGCAR', 'CONTCAR', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'INCAR',
                'KPOINTS', 'OSZICAR', 'OUTCAR', 'PCDAT', 'POSCAR', 'POTCAR', 'REPORT',
                'vasp.out', 'vasprun.xml', 'WAVECAR', 'XDATCAR']

            for filename in output_files:
                src = os.path.join(os.getcwd(), filename)
                dest = os.path.join(model_name, filename)
                if os.path.exists(src):
                    shutil.move(src, dest)
    else:
        print(f"Skipping task for {model_name} as it's already in the optimized database.")
