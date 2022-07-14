from ipie.propagation.overlap import get_calc_overlap

def compute_walker_overlaps(walker_batch, trial):
    return get_calc_overlap(trial)(walker_batch, trial)
