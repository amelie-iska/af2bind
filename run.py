import os
import time
import argparse
import numpy as np
import pandas as pd
import py3Dmol
import jax
import pickle
import copy
import subprocess
import seaborn as sns
from scipy.special import expit as sigmoid
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import protein

import matplotlib.pyplot as plt
import plotly.express as px

aa_order = {v:k for k,v in residue_constants.restype_order.items()}

def download_weights():
    if not os.path.isdir("params"):
        print("Downloading AlphaFold2 params...")
        os.makedirs("params", exist_ok=True)
        os.system("wget -qnc https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar")
        os.system("tar -xf alphafold_params_2021-07-14.tar -C params")
        os.system("rm alphafold_params_2021-07-14.tar")

    if not os.path.isdir("af2bind_params"):
        print("Downloading AF2BIND params...")
        os.makedirs("af2bind_params", exist_ok=True)
        os.system("wget -qnc https://github.com/sokrypton/af2bind/raw/main/attempt_7_2k_lam0-03.zip")
        os.system("unzip attempt_7_2k_lam0-03.zip -d af2bind_params")
        os.system("rm attempt_7_2k_lam0-03.zip")

def get_pdb(pdb_code=""):
    if pdb_code is None or pdb_code == "":
        raise ValueError("PDB code or file path is required")
    elif os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"
    else:
        os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb")
        return f"AF-{pdb_code}-F1-model_v4.pdb"

def af2bind(outputs, mask_sidechains=True, seed=0):
    pair_A = outputs["representations"]["pair"][:-20,-20:]
    pair_B = outputs["representations"]["pair"][-20:,:-20].swapaxes(0,1)
    pair_A = pair_A.reshape(pair_A.shape[0],-1)
    pair_B = pair_B.reshape(pair_B.shape[0],-1)
    x = np.concatenate([pair_A,pair_B],-1)

    if mask_sidechains:
        model_type = f"split_nosc_pair_A_split_nosc_pair_B_{seed}"
    else:
        model_type = f"split_pair_A_split_pair_B_{seed}"
    with open(f"af2bind_params/attempt_7_2k_lam0-03/{model_type}.pickle","rb") as handle:
        params_ = pickle.load(handle)
    params_ = dict(**params_["~"], **params_["linear"])
    p = jax.tree_map(lambda x:np.asarray(x), params_)

    x = (x - p["mean"]) / p["std"]
    x = (x * p["w"][:,0]) + (p["b"] / x.shape[-1])
    p_bind_aa = x.reshape(x.shape[0],2,20,-1).sum((1,3))
    p_bind = sigmoid(p_bind_aa.sum(-1))
    return {"p_bind":p_bind, "p_bind_aa":p_bind_aa}

def generate_pdb_file(af_model, pred_bind, output_dir, show_ligand, use_native_coordinates):
    preds_adj = pred_bind.copy()
    L = af_model._target_len
    aux = copy.deepcopy(af_model.aux["all"])
    aux["plddt"][:,:L] = preds_adj
    if not show_ligand:
        aux["atom_mask"][:,L:] = 0
    x = {k:[] for k in ["aatype", "residue_index", "atom_positions", "atom_mask", "b_factors"]}
    asym_id = []
    for i in range(af_model._target_len):
        for k in ["aatype","atom_mask"]: x[k].append(aux[k][0,i])
        if use_native_coordinates:
            x["atom_positions"].append(af_model._pdb["batch"]["all_atom_positions"][i])
        else:
            x["atom_positions"].append(aux["atom_positions"][0,i])
        x["residue_index"].append(af_model._pdb["idx"]["residue"][i])
        x["b_factors"].append(x["atom_mask"][-1] * aux["plddt"][0,i] * 100.0)
        asym_id.append(af_model._pdb["idx"]["chain"][i])
    x = {k:np.array(v) for k,v in x.items()}

    (n,resnum_) = (0,None)
    pdb_lines = []
    for line in protein.to_pdb(protein.Protein(**x)).splitlines():
        if line[:4] == "ATOM":
            resnum = int(line[22:22+5])
            if resnum_ is None: resnum_ = resnum
            if resnum != resnum_:
                n += 1
                resnum_ = resnum
            pdb_lines.append("%s%s%4i%s" % (line[:21],asym_id[n],resnum,line[26:]))
    with open(os.path.join(output_dir, "output.pdb"), "w") as handle:
        handle.write("\n".join(pdb_lines))

def generate_plots(pred_bind, pred_bind_aa, af_model, args):
    # Plot p(bind) distribution
    plt.figure(figsize=(10, 6))
    plt.hist(pred_bind, bins=50)
    plt.title("Distribution of p(bind)")
    plt.xlabel("p(bind)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.output_dir, "pbind_distribution.png"))
    plt.close()

    # Activation analysis
    blosum_map = list("CSTAGPDEQNHRKMILVWYF")
    cs_label_list = list("ACDEFGHIKLMNPQRSTVWY")
    indices_A_Y_mapping = np.array([cs_label_list.index(letter) for letter in blosum_map])
    pred_bind_aa_blosum = pred_bind_aa[:,indices_A_Y_mapping]
    filt = pred_bind > args.pbind_cutoff
    pred_bind_aa_blosum = pred_bind_aa_blosum[filt]
    res_labels = np.array(af_model._pdb["idx"]["residue"])[filt]
    chain_labels = np.array(af_model._pdb["idx"]["chain"])[filt]

    fig = px.imshow(pred_bind_aa_blosum.T,
                    labels=dict(x="positions", y="amino acids", color="pref"),
                    y=blosum_map,
                    x=[f"{y}_{x}" for x,y in zip(res_labels,chain_labels)],
                    zmin=-1,
                    zmax=1,
                    template="simple_white",
                    color_continuous_scale=["red", "white", "blue"])
    fig.write_html(os.path.join(args.output_dir, "activation_analysis.html"))

def generate_heatmap(pred_bind_aa, pred_bind, af_model, output_dir, pbind_cutoff):
    # Define the order of amino acids
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Create a DataFrame from pred_bind_aa
    df = pd.DataFrame(pred_bind_aa, columns=aa_order)
    
    # Add position information
    df['position'] = range(1, len(df) + 1)
    df['chain'] = [af_model._pdb["idx"]["chain"][i] for i in range(len(df))]
    df['residue'] = [af_model._pdb["idx"]["residue"][i] for i in range(len(df))]
    
    # Add p(bind) values
    df['p(bind)'] = pred_bind
    
    # Melt the DataFrame to long format
    df_melted = df.melt(id_vars=['position', 'chain', 'residue', 'p(bind)'], 
                        var_name='amino_acid', 
                        value_name='aa_prob')
    
    # Save the data as CSV
    pivot_df = df_melted.pivot(index=['position', 'chain', 'residue', 'p(bind)'], columns='amino_acid', values='aa_prob')
    pivot_df.reset_index().to_csv(os.path.join(output_dir, 'heatmap_data.csv'), index=False)

    # Create the heatmap
    plt.figure(figsize=(20, 10))
    heatmap_df = df_melted.pivot(index='amino_acid', columns='position', values='aa_prob')
    heatmap = sns.heatmap(heatmap_df, cmap='YlOrRd', cbar_kws={'label': 'aa_prob'})
    
    # Add black bounding boxes for cells above the threshold
    for i, amino_acid in enumerate(aa_order):
        for j in range(len(df)):
            if heatmap_df.iloc[i, j] > pbind_cutoff:
                heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    plt.title('Heatmap of aa_prob values for each amino acid at each position')
    plt.xlabel('Residue Position')
    plt.ylabel('Amino Acid')
    
    # Adjust x-axis labels to show chain and residue information
    x_labels = [f"{c}:{r}" for c, r in zip(df['chain'], df['residue'])]
    plt.xticks(range(0, len(x_labels), max(1, len(x_labels)//20)), 
               [x_labels[i] for i in range(0, len(x_labels), max(1, len(x_labels)//20))], 
               rotation=90)
    
    # Save the heatmap
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pbind_heatmap.png'), dpi=300)
    plt.close()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Download weights if not present
    download_weights()

    pdb_filename = get_pdb(args.input)

    clear_mem()
    af_model = mk_afdesign_model(protocol="binder", debug=True)
    af_model.prep_inputs(pdb_filename=pdb_filename,
                         chain=args.target_chain,
                         binder_len=20,
                         rm_target_sc=args.mask_sidechains,
                         rm_target_seq=args.mask_sequence)

    r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
    af_model._inputs["residue_index"][-20:] = r_idx.flatten()

    af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
    af_model.predict(verbose=False)

    o = af2bind(af_model.aux["debug"]["outputs"],
                mask_sidechains=args.mask_sidechains)
    pred_bind = o["p_bind"].copy()
    pred_bind_aa = o["p_bind_aa"].copy()

    labels = ["chain", "resi", "resn", "p(bind)"]
    data = []
    for i in range(af_model._target_len):
        c = af_model._pdb["idx"]["chain"][i]
        r = af_model._pdb["idx"]["residue"][i]
        a = aa_order.get(af_model._pdb["batch"]["aatype"][i], "X")
        p = pred_bind[i]
        data.append([c, r, a, p])

    df = pd.DataFrame(data, columns=labels)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)

    # Generate heatmap
    generate_heatmap(pred_bind_aa, pred_bind, af_model, args.output_dir, args.pbind_cutoff)

    generate_pdb_file(af_model, pred_bind, args.output_dir, args.show_ligand, args.use_native_coordinates)
    generate_plots(pred_bind, pred_bind_aa, af_model, args)

    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AF2BIND analysis")
    parser.add_argument("--input", required=True, help="Input PDB file or UniProt ID")
    parser.add_argument("--output_dir", required=True, help="Output directory for results and plots")
    parser.add_argument("--target_chain", default="A", help="Target chain (default: A)")
    parser.add_argument("--mask_sidechains", action="store_true", help="Mask sidechains")
    parser.add_argument("--mask_sequence", action="store_true", help="Mask sequence")
    parser.add_argument("--show_ligand", action="store_true", help="Show ligand in output PDB")
    parser.add_argument("--use_native_coordinates", action="store_true", help="Use native coordinates")
    parser.add_argument("--pbind_cutoff", type=float, default=0.5, help="p(bind) cutoff for activation analysis")
    
    args = parser.parse_args()
    main(args)

# Example usage:
# python run.py --input 6KFH --output_dir ./results --target_chain A --mask_sidechains --pbind_cutoff 0.5