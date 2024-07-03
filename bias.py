import argparse
import json
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB.Selection import unfold_entities

def load_structure(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure("complex", pdb_file)

def get_binding_sites(heatmap_file, pbind_cutoff):
    df = pd.read_csv(heatmap_file)
    print("Columns in the heatmap file:", df.columns)

    required_cols = ['chain', 'residue', 'p(bind)']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Required columns not found. Available columns: {df.columns}")

    binding_sites = df[df['p(bind)'] > pbind_cutoff]
    sites = [f"{row['chain']}{row['residue']}" for _, row in binding_sites.iterrows()]
    print(f"Found {len(sites)} binding sites: {sites}")
    return sites, binding_sites

def find_nearby_residues(structure, binding_sites, target_chain, binder_chain, distance_cutoff):
    nearby_residues = {}
    binding_site_residues = []
    binder_residues = {}
    distances = {}  # To store distances between binding sites and binder residues

    print(f"Analyzing chain {binder_chain} (binder):")
    for chain in structure[0]:
        if chain.id == binder_chain:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard amino acid
                    if 'CA' in residue:  # Check for alpha carbon
                        binder_residues[residue.id[1]] = residue
                        print(f"  Residue {residue.id[1]}: {residue.resname}")
                    else:
                        print(f"  Skipping residue {residue.id[1]} ({residue.resname}): No CA atom")
                else:
                    print(f"  Skipping non-standard residue: {residue.id}")
        elif chain.id == target_chain:
            for residue in chain:
                if residue.id[0] == ' ' and f"{chain.id}{residue.id[1]}" in binding_sites:
                    binding_site_residues.append(residue)

    print(f"\nFound {len(binding_site_residues)} binding site residues in the structure")
    print(f"Found {len(binder_residues)} binder residues in the structure")
    binder_residue_numbers = sorted(binder_residues.keys())
    print(f"Binder residue range: {binder_residue_numbers[0]} to {binder_residue_numbers[-1]} (with gaps)")

    for binding_site in binding_site_residues:
        binding_site_id = f"{binding_site.parent.id}{binding_site.id[1]}"
        nearby = {}
        binding_site_atoms = [atom for atom in binding_site.get_atoms() if atom.name in ['N', 'CA', 'C', 'O']]
        
        for binder_residue_number, binder_residue in binder_residues.items():
            binder_atoms = [atom for atom in binder_residue.get_atoms() if atom.name in ['N', 'CA', 'C', 'O']]
            
            min_distance = float('inf')
            for binding_atom in binding_site_atoms:
                for binder_atom in binder_atoms:
                    distance = binding_atom - binder_atom
                    if distance < min_distance:
                        min_distance = distance
            
            if min_distance <= distance_cutoff:
                binder_residue_id = f"{binder_chain}{binder_residue_number}"
                nearby[binder_residue_id] = min_distance
                
                if binder_residue_id not in distances or min_distance < distances[binder_residue_id][1]:
                    distances[binder_residue_id] = (binding_site_id, min_distance)

        nearby_sorted = sorted(nearby.items(), key=lambda x: binder_residue_numbers.index(int(x[0][1:])))
        nearby_residues[binding_site_id] = nearby_sorted

    print(f"Found nearby residues for {len(nearby_residues)} binding sites")
    for binding_site, nearby in nearby_residues.items():
        print(f"  Binding site {binding_site}: {len(nearby)} nearby residues")
        print(f"    Nearby residues: {[res for res, _ in nearby]}")
    
    return nearby_residues, distances

def scale_probabilities(probs):
    probs = {aa: float(prob) for aa, prob in probs.items()}
    min_prob, max_prob = min(probs.values()), max(probs.values())
    if min_prob == max_prob:
        return {aa: 0 for aa in probs}
    return {aa: -10 + 20 * (prob - min_prob) / (max_prob - min_prob) for aa, prob in probs.items()}

def create_mpnn_bias(nearby_residues, distances, binding_sites_df, output_file):
    mpnn_bias = {}
    
    print("Nearby residues:")
    for binding_site, nearby in nearby_residues.items():
        print(f"  Binding site {binding_site}: {[res for res, _ in nearby]}")
    
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    for binding_site, nearby in nearby_residues.items():
        binding_site_data = binding_sites_df[
            (binding_sites_df['chain'] == binding_site[0]) & 
            (binding_sites_df['residue'] == int(binding_site[1:]))
        ]
        
        if not binding_site_data.empty:
            binding_site_probs = {aa: binding_site_data[aa].values[0] for aa in aa_list}
            binding_site_bias = scale_probabilities(binding_site_probs)
            for residue, _ in nearby:
                if distances[residue][0] == binding_site:
                    mpnn_bias[residue] = binding_site_bias
                    print(f"Transferred bias from {binding_site} to {residue}")
        else:
            print(f"Warning: Binding site {binding_site} not found in heatmap data")

    print(f"Created MPNN bias for {len(mpnn_bias)} residues")
    with open(output_file, 'w') as f:
        json.dump(mpnn_bias, f, indent=2)

def main(args):
    structure = load_structure(args.pdb_file)
    binding_sites, binding_sites_df = get_binding_sites(args.heatmap_file, args.pbind_cutoff)
    nearby_residues, distances = find_nearby_residues(structure, binding_sites, args.target_chain, args.binder_chain, args.distance_cutoff)

    create_mpnn_bias(nearby_residues, distances, binding_sites_df, args.output_file)

    print(f"MPNN bias JSON file saved as {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MPNN bias JSON from PDB and heatmap data")
    parser.add_argument("--pdb_file", required=True, help="Input PDB file containing target and binder")
    parser.add_argument("--heatmap_file", required=True, help="Input CSV file containing heatmap data")
    parser.add_argument("--target_chain", required=True, help="Chain ID of the target protein")
    parser.add_argument("--binder_chain", required=True, help="Chain ID of the binder")
    parser.add_argument("--distance_cutoff", type=float, default=10.0, help="Distance cutoff in Angstroms")
    parser.add_argument("--pbind_cutoff", type=float, default=0.5, help="p(bind) cutoff for considering binding sites")
    parser.add_argument("--output_file", required=True, help="Output JSON file for MPNN bias data")
    
    args = parser.parse_args()
    main(args)

# Example usage:
# python bias.py --pdb_file ./6KFH.pdb --heatmap_file ./results/heatmap_data.csv --target_chain A --binder_chain B --distance_cutoff 12.0 --pbind_cutoff 0.5 --output_file ./results/bias.json