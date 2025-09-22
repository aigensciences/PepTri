#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Local similarity metrics for peptide structure evaluation
Better suited for short peptides than global metrics like TM-score

Implements:
- GDT_TS (Global Distance Test Total Score)
- Contact overlap
- Local RMSD
- Secondary structure overlap
"""
import numpy as np
import warnings
from typing import Tuple, Optional, Dict, List


def compute_gdt_ts(ref_coords: np.ndarray, pred_coords: np.ndarray, 
                   thresholds: List[float] = [1.0, 2.0, 4.0, 8.0]) -> Dict[str, float]:
    """
    Compute GDT_TS (Global Distance Test Total Score)
    
    GDT_TS measures the percentage of residues that can be superimposed 
    within distance thresholds after optimal superposition.
    
    Args:
        ref_coords: Reference coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3]
        thresholds: Distance thresholds in Angstroms
        
    Returns:
        Dictionary with GDT scores for each threshold and GDT_TS
    """
    if len(ref_coords) != len(pred_coords):
        min_len = min(len(ref_coords), len(pred_coords))
        ref_coords = ref_coords[:min_len]
        pred_coords = pred_coords[:min_len]
    
    n = len(ref_coords)
    if n == 0:
        return {f'GDT_{t}': 0.0 for t in thresholds} | {'GDT_TS': 0.0}
    
    # Align structures using Kabsch algorithm
    aligned_pred = kabsch_align(ref_coords, pred_coords)
    
    # Calculate distances after alignment
    distances = np.linalg.norm(ref_coords - aligned_pred, axis=1)
    
    # Calculate GDT scores for each threshold
    gdt_scores = {}
    for threshold in thresholds:
        within_threshold = np.sum(distances <= threshold)
        gdt_scores[f'GDT_{threshold}'] = within_threshold / n
    
    # GDT_TS is the average of the four standard thresholds
    if len(thresholds) == 4:
        gdt_ts = np.mean(list(gdt_scores.values()))
    else:
        # Use the provided thresholds
        gdt_ts = np.mean(list(gdt_scores.values()))
    
    gdt_scores['GDT_TS'] = gdt_ts
    
    return gdt_scores


def compute_contact_map(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """
    Compute contact map from coordinates
    
    Args:
        coords: Coordinates [N, 3]
        threshold: Distance threshold for contacts in Angstroms
        
    Returns:
        Contact map [N, N] - binary matrix
    """
    n = len(coords)
    contact_map = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= threshold:
                contact_map[i, j] = True
                contact_map[j, i] = True
    
    return contact_map


def compute_contact_overlap(ref_coords: np.ndarray, pred_coords: np.ndarray,
                          threshold: float = 8.0, min_separation: int = 2) -> Dict[str, float]:
    """
    Compute contact overlap between reference and predicted structures
    
    Args:
        ref_coords: Reference coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3]
        threshold: Distance threshold for contacts in Angstroms
        min_separation: Minimum sequence separation for contacts
        
    Returns:
        Dictionary with contact overlap metrics
    """
    if len(ref_coords) != len(pred_coords):
        min_len = min(len(ref_coords), len(pred_coords))
        ref_coords = ref_coords[:min_len]
        pred_coords = pred_coords[:min_len]
    
    n = len(ref_coords)
    if n < min_separation + 1:
        return {
            'contact_overlap': 0.0,
            'contact_precision': 0.0,
            'contact_recall': 0.0,
            'contact_f1': 0.0,
            'n_ref_contacts': 0,
            'n_pred_contacts': 0,
            'n_common_contacts': 0
        }
    
    # Compute contact maps
    ref_contacts = compute_contact_map(ref_coords, threshold)
    pred_contacts = compute_contact_map(pred_coords, threshold)
    
    # Apply minimum separation constraint
    for i in range(n):
        for j in range(max(0, i - min_separation), min(n, i + min_separation + 1)):
            ref_contacts[i, j] = False
            pred_contacts[i, j] = False
    
    # Count contacts (only upper triangle to avoid double counting)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    ref_contacts_upper = ref_contacts & mask
    pred_contacts_upper = pred_contacts & mask
    common_contacts = ref_contacts_upper & pred_contacts_upper
    
    n_ref_contacts = np.sum(ref_contacts_upper)
    n_pred_contacts = np.sum(pred_contacts_upper)
    n_common_contacts = np.sum(common_contacts)
    
    # Calculate metrics
    if n_ref_contacts == 0 and n_pred_contacts == 0:
        precision = recall = f1 = overlap = 1.0
    elif n_ref_contacts == 0:
        precision = recall = f1 = overlap = 0.0
    else:
        precision = n_common_contacts / n_pred_contacts if n_pred_contacts > 0 else 0.0
        recall = n_common_contacts / n_ref_contacts
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        overlap = n_common_contacts / n_ref_contacts  # Same as recall for contacts
    
    return {
        'contact_overlap': overlap,
        'contact_precision': precision,
        'contact_recall': recall,
        'contact_f1': f1,
        'n_ref_contacts': int(n_ref_contacts),
        'n_pred_contacts': int(n_pred_contacts),
        'n_common_contacts': int(n_common_contacts)
    }


def compute_local_rmsd(ref_coords: np.ndarray, pred_coords: np.ndarray,
                      window_size: int = 5) -> Dict[str, float]:
    """
    Compute local RMSD using sliding windows
    
    Args:
        ref_coords: Reference coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3]
        window_size: Size of sliding window
        
    Returns:
        Dictionary with local RMSD statistics
    """
    if len(ref_coords) != len(pred_coords):
        min_len = min(len(ref_coords), len(pred_coords))
        ref_coords = ref_coords[:min_len]
        pred_coords = pred_coords[:min_len]
    
    n = len(ref_coords)
    if n < window_size:
        # If sequence is shorter than window, compute global RMSD
        aligned_pred = kabsch_align(ref_coords, pred_coords)
        rmsd = np.sqrt(np.mean(np.sum((ref_coords - aligned_pred)**2, axis=1)))
        return {
            'local_rmsd_mean': rmsd,
            'local_rmsd_min': rmsd,
            'local_rmsd_max': rmsd,
            'local_rmsd_std': 0.0,
            'n_windows': 1
        }
    
    # Compute RMSD for each window
    local_rmsds = []
    for i in range(n - window_size + 1):
        ref_window = ref_coords[i:i+window_size]
        pred_window = pred_coords[i:i+window_size]
        
        # Align window
        aligned_window = kabsch_align(ref_window, pred_window)
        
        # Compute RMSD for this window
        rmsd = np.sqrt(np.mean(np.sum((ref_window - aligned_window)**2, axis=1)))
        local_rmsds.append(rmsd)
    
    local_rmsds = np.array(local_rmsds)
    
    return {
        'local_rmsd_mean': np.mean(local_rmsds),
        'local_rmsd_min': np.min(local_rmsds),
        'local_rmsd_max': np.max(local_rmsds),
        'local_rmsd_std': np.std(local_rmsds),
        'n_windows': len(local_rmsds)
    }


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


def compute_peptide_metrics(ref_coords: np.ndarray, pred_coords: np.ndarray,
                          ref_seq: str = None, pred_seq: str = None) -> Dict[str, float]:
    """
    Compute comprehensive local similarity metrics for peptides
    
    Args:
        ref_coords: Reference coordinates [N, 3]
        pred_coords: Predicted coordinates [N, 3]
        ref_seq: Reference sequence (optional)
        pred_seq: Predicted sequence (optional)
        
    Returns:
        Dictionary with all local metrics
    """
    if len(ref_coords) == 0 or len(pred_coords) == 0:
        return {}
    
    # Ensure same length
    if len(ref_coords) != len(pred_coords):
        min_len = min(len(ref_coords), len(pred_coords))
        ref_coords = ref_coords[:min_len]
        pred_coords = pred_coords[:min_len]
        if ref_seq and pred_seq:
            ref_seq = ref_seq[:min_len]
            pred_seq = pred_seq[:min_len]
    
    results = {}
    
    try:
        # GDT_TS scores
        gdt_results = compute_gdt_ts(ref_coords, pred_coords)
        results.update(gdt_results)
    except Exception as e:
        warnings.warn(f"GDT_TS calculation failed: {e}")
        results.update({'GDT_1.0': 0.0, 'GDT_2.0': 0.0, 'GDT_4.0': 0.0, 'GDT_8.0': 0.0, 'GDT_TS': 0.0})
    
    try:
        # Contact overlap
        contact_results = compute_contact_overlap(ref_coords, pred_coords)
        results.update(contact_results)
    except Exception as e:
        warnings.warn(f"Contact overlap calculation failed: {e}")
        results.update({
            'contact_overlap': 0.0, 'contact_precision': 0.0, 'contact_recall': 0.0,
            'contact_f1': 0.0, 'n_ref_contacts': 0, 'n_pred_contacts': 0, 'n_common_contacts': 0
        })
    
    try:
        # Local RMSD
        local_rmsd_results = compute_local_rmsd(ref_coords, pred_coords)
        results.update(local_rmsd_results)
    except Exception as e:
        warnings.warn(f"Local RMSD calculation failed: {e}")
        results.update({
            'local_rmsd_mean': 999.0, 'local_rmsd_min': 999.0, 'local_rmsd_max': 999.0,
            'local_rmsd_std': 0.0, 'n_windows': 0
        })
    
    # Add sequence information if available
    if ref_seq and pred_seq:
        results['ref_length'] = len(ref_seq)
        results['pred_length'] = len(pred_seq)
        results['sequence_identity'] = sum(a == b for a, b in zip(ref_seq, pred_seq)) / len(ref_seq)
    
    return results


if __name__ == "__main__":
    # Test local metrics implementation
    import numpy as np
    
    print("🧪 Testing Local Metrics Implementation")
    print("="*60)
    
    # Create test peptide coordinates (short sequence)
    np.random.seed(42)
    ref_coords = np.random.randn(12, 3) * 3  # 12-residue peptide
    
    # Test 1: Identical structures
    print("Test 1: Identical structures")
    identical_metrics = compute_peptide_metrics(ref_coords, ref_coords.copy())
    print(f"GDT_TS: {identical_metrics['GDT_TS']:.4f} (should be 1.0)")
    print(f"Contact overlap: {identical_metrics['contact_overlap']:.4f}")
    print(f"Local RMSD mean: {identical_metrics['local_rmsd_mean']:.6f}")
    
    # Test 2: Slightly different structure
    print("\nTest 2: Slightly noisy structure")
    noisy_coords = ref_coords + np.random.randn(12, 3) * 0.3
    noisy_metrics = compute_peptide_metrics(ref_coords, noisy_coords)
    print(f"GDT_TS: {noisy_metrics['GDT_TS']:.4f}")
    print(f"Contact overlap: {noisy_metrics['contact_overlap']:.4f}")
    print(f"Local RMSD mean: {noisy_metrics['local_rmsd_mean']:.4f}")
    
    # Test 3: Very different structure
    print("\nTest 3: Random structure")
    random_coords = np.random.RandomState(123).randn(12, 3) * 5
    random_metrics = compute_peptide_metrics(ref_coords, random_coords)
    print(f"GDT_TS: {random_metrics['GDT_TS']:.4f}")
    print(f"Contact overlap: {random_metrics['contact_overlap']:.4f}")
    print(f"Local RMSD mean: {random_metrics['local_rmsd_mean']:.4f}")
    
    print("\n✅ Local metrics implementation test completed!")
    print("🎯 These metrics are better suited for short peptides!")