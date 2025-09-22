#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
TM-score calculation for peptide structure evaluation
TM-score ranges from 0 to 1, where 1 indicates identical structures
"""
import os
import subprocess
import tempfile
import numpy as np
import warnings
from typing import Tuple, Optional


def compute_tmscore_python(ref_coords: np.ndarray, pred_coords: np.ndarray, 
                          ref_seq: str = None, pred_seq: str = None) -> float:
    """
    Pure Python implementation of TM-score calculation
    
    Args:
        ref_coords: Reference coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3] 
        ref_seq: Reference sequence (optional)
        pred_seq: Predicted sequence (optional)
        
    Returns:
        TM-score value (0-1)
    """
    if len(ref_coords) != len(pred_coords):
        warnings.warn("Coordinate arrays have different lengths")
        min_len = min(len(ref_coords), len(pred_coords))
        ref_coords = ref_coords[:min_len]
        pred_coords = pred_coords[:min_len]
    
    n = len(ref_coords)
    if n == 0:
        return 0.0
    
    # Normalize sequences for TM-score calculation
    L_target = len(ref_coords)  # Target length (reference)
    
    # TM-score normalization factor
    if L_target <= 21:
        d0 = 0.5
    else:
        d0 = 1.24 * (L_target - 15)**(1.0/3.0) - 1.8
    
    # Align structures using Kabsch algorithm
    aligned_pred = kabsch_align(ref_coords, pred_coords)
    
    # Calculate TM-score
    distances = np.linalg.norm(ref_coords - aligned_pred, axis=1)
    d_cut = d0
    
    # TM-score formula
    tm_score = 0.0
    for i in range(n):
        tm_score += 1.0 / (1.0 + (distances[i] / d_cut)**2)
    
    tm_score = tm_score / L_target
    
    return tm_score


def kabsch_align(ref_coords: np.ndarray, pred_coords: np.ndarray) -> np.ndarray:
    """
    Align predicted coordinates to reference using Kabsch algorithm
    
    Args:
        ref_coords: Reference coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3]
        
    Returns:
        Aligned predicted coordinates [N, 3]
    """
    # Center coordinates
    ref_center = np.mean(ref_coords, axis=0)
    pred_center = np.mean(pred_coords, axis=0)
    
    ref_centered = ref_coords - ref_center
    pred_centered = pred_coords - pred_center
    
    # Compute covariance matrix
    H = pred_centered.T @ ref_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and translation
    aligned_pred = (pred_coords - pred_center) @ R.T + ref_center
    
    return aligned_pred


def write_pdb_coords(coords: np.ndarray, sequence: str, filename: str, chain_id: str = 'A'):
    """
    Write coordinates to PDB file for external TM-score calculation
    
    Args:
        coords: Coordinates [N, 3]
        sequence: Amino acid sequence
        filename: Output PDB filename
        chain_id: Chain identifier
    """
    from data.format import VOCAB
    
    with open(filename, 'w') as f:
        f.write("HEADER    PEPTIDE STRUCTURE\n")
        f.write("MODEL        1\n")
        
        for i, (coord, aa) in enumerate(zip(coords, sequence)):
            if aa in VOCAB.abrv_to_symbol:
                aa_name = VOCAB.symbol_to_abrv(aa)
            else:
                aa_name = aa if len(aa) == 3 else 'ALA'
                
            f.write(f"ATOM  {i+1:5d}  CA  {aa_name} {chain_id}{i+1:4d}    "
                   f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           C\n")
        
        f.write("ENDMDL\n")
        f.write("END\n")


def compute_tmscore_external(ref_coords: np.ndarray, pred_coords: np.ndarray,
                           ref_seq: str, pred_seq: str,
                           tmscore_path: str = None) -> float:
    """
    Compute TM-score using external TM-score program
    
    Args:
        ref_coords: Reference coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3]
        ref_seq: Reference sequence
        pred_seq: Predicted sequence  
        tmscore_path: Path to TM-score executable
        
    Returns:
        TM-score value (0-1)
    """
    if tmscore_path is None:
        # Try common paths
        possible_paths = [
            'TMscore',
            '/usr/local/bin/TMscore',
            './TMscore',
            '~/bin/TMscore'
        ]
        
        tmscore_path = None
        for path in possible_paths:
            if os.path.exists(os.path.expanduser(path)):
                tmscore_path = os.path.expanduser(path)
                break
                
        if tmscore_path is None:
            warnings.warn("TM-score executable not found, using Python implementation")
            return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)
    
    # Create temporary PDB files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as ref_file:
        ref_pdb = ref_file.name
        write_pdb_coords(ref_coords, ref_seq, ref_pdb, 'A')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as pred_file:
        pred_pdb = pred_file.name
        write_pdb_coords(pred_coords, pred_seq, pred_pdb, 'B')
    
    try:
        # Run TM-score
        cmd = [tmscore_path, pred_pdb, ref_pdb]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            warnings.warn(f"TM-score failed: {result.stderr}")
            return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)
        
        # Parse TM-score output
        for line in result.stdout.split('\n'):
            if 'TM-score=' in line and 'Chain_1' in line:
                # Extract TM-score from line like: "TM-score= 0.12345 (if normalized by length of Chain_1)"
                tm_score = float(line.split('TM-score=')[1].split()[0])
                return tm_score
                
        warnings.warn("Could not parse TM-score output")
        return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        warnings.warn(f"TM-score external call failed: {e}, using Python implementation")
        return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(ref_pdb)
            os.unlink(pred_pdb)
        except OSError:
            pass


def compute_tmscore(ref_coords: np.ndarray, pred_coords: np.ndarray,
                   ref_seq: str = None, pred_seq: str = None,
                   method: str = 'auto') -> float:
    """
    Compute TM-score between reference and predicted structures
    
    Args:
        ref_coords: Reference coordinates [N, 3] 
        pred_coords: Predicted coordinates [N, 3]
        ref_seq: Reference sequence (required for external method)
        pred_seq: Predicted sequence (required for external method)
        method: 'auto', 'python', or 'external'
        
    Returns:
        TM-score value (0-1)
    """
    if ref_coords.shape[0] == 0 or pred_coords.shape[0] == 0:
        return 0.0
        
    if method == 'python':
        return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)
    elif method == 'external':
        if ref_seq is None or pred_seq is None:
            warnings.warn("Sequences required for external TM-score, falling back to Python")
            return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)
        return compute_tmscore_external(ref_coords, pred_coords, ref_seq, pred_seq)
    else:  # auto
        # Try external first, fall back to Python
        if ref_seq is not None and pred_seq is not None:
            try:
                return compute_tmscore_external(ref_coords, pred_coords, ref_seq, pred_seq)
            except:
                return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)
        else:
            return compute_tmscore_python(ref_coords, pred_coords, ref_seq, pred_seq)


if __name__ == "__main__":
    # Test TM-score calculation
    import numpy as np
    
    # Create test coordinates (identical structures should give TM-score = 1.0)
    ref_coords = np.random.randn(20, 3)
    pred_coords = ref_coords.copy()  # Identical structure
    
    tm_score = compute_tmscore(ref_coords, pred_coords, method='python')
    print(f"TM-score for identical structures: {tm_score:.4f} (should be ~1.0)")
    
    # Test with slightly different structure
    pred_coords_noisy = ref_coords + np.random.randn(20, 3) * 0.5
    tm_score_noisy = compute_tmscore(ref_coords, pred_coords_noisy, method='python')
    print(f"TM-score for noisy structure: {tm_score_noisy:.4f} (should be <1.0)")
    
    print("TM-score implementation test completed!")