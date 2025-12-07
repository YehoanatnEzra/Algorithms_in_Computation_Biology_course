#!/usr/bin/env python3
import sys
import gzip
import argparse
import math
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
# Metric imports
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score

# Constants
HIDDEN_STATES = ['C', 'N']
OBSERVATIONS = ['A', 'T', 'G', 'C', 'N']
SMALL_NUM = -1e9 # Represents log(0)

def load_fasta_content(path):
    """ Reads fasta file (gz or plain) into a dict. """
    data = {}
    # Check if zipped
    open_func = gzip.open if path.endswith(".gz") else open
    
    try:
        with open_func(path, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                # Use upper case to be safe
                data[record.id] = str(record.seq).upper()
    except IOError as err:
        print(f"Failed to read file {path}: {err}")
        return {}
        
    return data

def process_data(seq_path, lbl_path):
    """ Loads sequences and labels, adding RC for data augmentation.
        Note: preserved from original but not used for train/test split to avoid leakage.
    """
    seq_map = load_fasta_content(seq_path)
    lbl_map = load_fasta_content(lbl_path)

    if seq_map.keys() != lbl_map.keys():
        raise ValueError("Keys in sequence and label files do not match!")

    dataset = []
    keys = list(seq_map.keys())
    
    for k in keys:
        s = seq_map[k]
        l = lbl_map[k]
        
        # Add normal
        dataset.append((s, l))
        
        # Add Reverse Complement
        l_rc = l[::-1] 
        s_rc = str(Seq(s).reverse_complement())
        dataset.append((s_rc, l_rc))

    return dataset

def calc_hmm_params(dataset):
    """ Calculates Log-Probabilities for HMM parameters. """
    
    # Init with 1 for Laplace smoothing
    trans_counts = {s1: {s2: 1 for s2 in HIDDEN_STATES} for s1 in HIDDEN_STATES}
    emit_counts = {s: {n: 1 for n in OBSERVATIONS} for s in HIDDEN_STATES}
    start_counts = {s: 1 for s in HIDDEN_STATES}

    for seq, lbl in dataset:
        # Start
        if not seq or not lbl:
            continue
        if lbl[0] in start_counts:
            start_counts[lbl[0]] += 1
            
        # Transitions and Emissions
        length = len(seq)
        for i in range(length):
            curr_s = lbl[i]
            curr_n = seq[i]
            
            if curr_s not in HIDDEN_STATES: continue
            
            # Count emit
            nuc = curr_n if curr_n in OBSERVATIONS else 'N'
            emit_counts[curr_s][nuc] += 1
            
            # Count trans
            if i > 0:
                prev_s = lbl[i-1]
                if prev_s in HIDDEN_STATES:
                    trans_counts[prev_s][curr_s] += 1

    # Normalize and convert to LOG space
    total_start = sum(start_counts.values())
    pi = {k: math.log(v/total_start) for k,v in start_counts.items()}
    
    a_mat = {}
    for s in HIDDEN_STATES:
        row_sum = sum(trans_counts[s].values())
        a_mat[s] = {k: math.log(v/row_sum) for k,v in trans_counts[s].items()}
        
    b_mat = {}
    for s in HIDDEN_STATES:
        row_sum = sum(emit_counts[s].values())
        b_mat[s] = {k: math.log(v/row_sum) for k,v in emit_counts[s].items()}
        
    return pi, a_mat, b_mat

def viterbi_decode(seq, pi, a, b):
    """ Viterbi implementation using Log-Space (addition instead of multiplication). """
    L = len(seq)
    S = len(HIDDEN_STATES)
    
    # Tables
    v_table = np.zeros((S, L))
    backptr = np.zeros((S, L), dtype=int)
    
    # Init t=0
    first_n = seq[0] if seq[0] in OBSERVATIONS else 'N'
    for i in range(S):
        state = HIDDEN_STATES[i]
        # Log add
        v_table[i, 0] = pi[state] + b[state].get(first_n, SMALL_NUM)
        
    # Forward step
    for t in range(1, L):
        nuc = seq[t]
        safe_nuc = nuc if nuc in OBSERVATIONS else 'N'
        
        for curr_idx in range(S):
            curr_state = HIDDEN_STATES[curr_idx]
            emit_score = b[curr_state].get(safe_nuc, SMALL_NUM)
            
            best_score = -float('inf')
            best_prev = 0
            
            for prev_idx in range(S):
                prev_state = HIDDEN_STATES[prev_idx]
                
                # prev_score + trans + emit
                score = v_table[prev_idx, t-1] + a[prev_state][curr_state] + emit_score
                
                if score > best_score:
                    best_score = score
                    best_prev = prev_idx
            
            v_table[curr_idx, t] = best_score
            backptr[curr_idx, t] = best_prev
            
    # Backtrack
    path_idx = [0] * L
    path_idx[-1] = np.argmax(v_table[:, -1])
    
    for t in range(L-2, -1, -1):
        path_idx[t] = backptr[path_idx[t+1], t+1]
        
    return "".join([HIDDEN_STATES[i] for i in path_idx])

def predict_labels(model, seq):
    pi, a, b = model
    return viterbi_decode(seq, pi, a, b)

def write_results(model, in_file, out_file):
    seqs = load_fasta_content(in_file)
    with gzip.open(out_file, "wt") as out:
        for sid, s in seqs.items():
            pred = predict_labels(model, s)
            out.write(f">{sid}\n{pred}\n")

def show_table(mat, title, rows, cols):
    print(f"\n[{title} (Log-Prob)]")
    header = "      " + "  ".join([f"{c:>8}" for c in cols])
    print(header)
    for r in rows:
        line = f"{r:<4} |"
        for c in cols:
            val = mat[r][c]
            line += f"{val:10.4f}"
        print(line)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CpG HMM Predictor (minimal fixes)")
    p.add_argument("--fasta_path", type=str, required=True)
    p.add_argument("--lbl_path", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--evaluate", action="store_true", help="Run evaluation on held-out set")
    args = p.parse_args()

    # Load maps directly using provided args
    print("Loading fasta maps...")
    seq_map = load_fasta_content(args.fasta_path)
    lbl_map = load_fasta_content(args.lbl_path)

    if not seq_map or not lbl_map:
        # fallback to original names behaviour (try without .gz)
        print("Failed to load via provided paths or files empty. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Build train/test split on keys BEFORE augmentation to avoid leakage.
    import random
    random.seed(42)
    keys = list(seq_map.keys())
    random.shuffle(keys)
    cutoff = int(len(keys) * 0.8)
    train_keys = keys[:cutoff]
    test_keys = keys[cutoff:]

    # Create train_set: augment only train (reverse complement)
    train_set = []
    for k in train_keys:
        s = seq_map[k]
        l = lbl_map[k]
        train_set.append((s, l))
        # augmentation: reverse-complement sequence and reverse label
        s_rc = str(Seq(s).reverse_complement())
        l_rc = l[::-1]
        train_set.append((s_rc, l_rc))

    # Create test_set: no augmentation
    test_set = []
    for k in test_keys:
        test_set.append((seq_map[k], lbl_map[k]))

    print("Training model...")
    pi, a, b = calc_hmm_params(train_set)
    model = (pi, a, b)

    if args.output_file and args.evaluate:
        show_table(a, "Transitions", HIDDEN_STATES, HIDDEN_STATES)
        show_table(b, "Emissions", HIDDEN_STATES, OBSERVATIONS)
        
        print("\nRunning evaluation...")
        yt, yp = [], []
        for seq, lbl in test_set:
            pred = predict_labels(model, seq)
            yt.extend(list(lbl))
            yp.extend(list(pred))
            
        print(f"Accuracy: {accuracy_score(yt, yp):.4f}")
        print(f"Balanced Acc: {balanced_accuracy_score(yt, yp):.4f}")
        print(f"F1 Score: {f1_score(yt, yp, average='weighted'):.4f}")
        
    else:
        # retrain on full dataset (train+test) like original code did
        print("Re-training on full dataset...")
        full_data = []
        for k in seq_map.keys():
            s = seq_map[k]
            l = lbl_map[k]
            full_data.append((s, l))
            s_rc = str(Seq(s).reverse_complement())
            l_rc = l[::-1]
            full_data.append((s_rc, l_rc))
        
        pi_full, a_full, b_full = calc_hmm_params(full_data)
        final_model = (pi_full, a_full, b_full)
        
        print(f"Writing to {args.output_file}...")
        write_results(final_model, args.fasta_path, args.output_file)
        print("Done.")
