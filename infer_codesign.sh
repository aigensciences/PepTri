# obtain the binding site, which might also be manually crafted or from other ligands (e.g. small molecule, antibodies)
python -m api.detect_pocket --pdb assets/3IOL_glp1.pdb --target_chains A --ligand_chains B --out assets/3IOL_glp1.json
# sequence-structure codesign with length in [8, 15)
CUDA_VISIBLE_DEVICES=6 python -m api.run_infer \
    --mode codesign \
    --ckpt /home/ubuntu/quang/peptide_design/model_for_git1_______________using_checkgit/exps/codesign_pepbench/LDM_700/version_0/checkpoint/epoch609_step26230.ckpt \
    --pdb assets/3IOL_glp1.pdb \
    --pocket assets/3IOL_glp1.json \
    --out_dir ./output/codesign_glp1_100_run1bdb \
    --length_min 10 \
    --length_max 25 \
    --n_samples 100