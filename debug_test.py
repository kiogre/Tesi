#!/usr/bin/env python3
"""
Test harness per verificare la correttezza di solve_optimal_jsp definita in Environment.py

- Confronta OR-Tools (solve_optimal_jsp) con un solver brute-force per istanze piccole.
- Stampa i casi in cui OR-Tools non coincide con il brute-force (o ritorna timeout/-1).
"""
import Neural_Network as nn
import itertools
import math
import torch
from Environment import JSPTrainer  # importa la classe che contiene solve_optimal_jsp
import Generate as gen
# Nota: se solve_optimal_jsp è metodo d'istanza, creiamo un trainer dummy per chiamarlo.

###############################################################################
# Funzioni ausiliarie (brute-force enumerator per trovare il makespan ottimo)
###############################################################################

def sequence_to_schedule(sequence_indices, S_seq):
    """Converte sequenza (lista di indici su S_seq) in schedule Gantt e ritorna makespan."""
    n_machines = int(S_seq[:, 2].max()) + 1
    machine_available_time = [0] * n_machines
    job_completion_times = {}
    for idx in sequence_indices:
        job_id = int(S_seq[idx, 0])
        op_id = int(S_seq[idx, 1])
        machine = int(S_seq[idx, 2])
        proc_time = int(S_seq[idx, 3])
        machine_ready = machine_available_time[machine]
        job_ready = job_completion_times.get((job_id, op_id-1), 0) if op_id > 0 else 0
        start = max(machine_ready, job_ready)
        end = start + proc_time
        machine_available_time[machine] = end
        job_completion_times[(job_id, op_id)] = end
    return max(machine_available_time) if machine_available_time else 0

def valid_permutations_indices(S_seq):
    """
    Genera tutte le permutazioni valide di indici (rispettando le precedenze per job).
    S_seq: N x 4 tensor/list, righe: [job_id, op_id, machine, ptime]
    Ritorna generator di tuple di indici (permute valid).
    """
    S = [tuple(row.tolist()) if hasattr(row, 'tolist') else tuple(row) for row in S_seq]
    nm = len(S)
    indices = list(range(nm))
    # Precompute precedence: per ogni job, lista di op_id e relative indices
    job_to_indices = {}
    for idx, (job, op, *_rest) in enumerate(S):
        job_to_indices.setdefault(int(job), []).append((int(op), idx))
    # For each job sort by op_id, get required order of indices
    job_order = {}
    for job, ops in job_to_indices.items():
        ops_sorted = sorted(ops, key=lambda x: x[0])
        job_order[job] = [idx for (_op, idx) in ops_sorted]

    # Brute-force all permutations but prune invalid by checking precedence as we go.
    # Since nm is small (we will use <= 7), this is feasible.
    for perm in itertools.permutations(indices):
        ok = True
        for job, seq in job_order.items():
            # ensure indices of this job appear in perm in the same relative order
            positions = [perm.index(idx) for idx in seq]
            if positions != sorted(positions):
                ok = False
                break
        if ok:
            yield perm

def brute_force_optimal_makespan(S_seq):
    """Ritorna il makespan ottimo (entire search)."""
    best = math.inf
    best_seq = None
    for perm in valid_permutations_indices(S_seq):
        mspan = sequence_to_schedule(perm, S_seq)
        if mspan < best:
            best = mspan
            best_seq = perm
    if best == math.inf:
        return -1, None
    return int(best), best_seq

###############################################################################
# Test cases (piccole istanze manuali) — puoi aggiungerne altre
###############################################################################
def make_instance_from_list(ops_list):
    """ops_list: lista di [job_id, op_id, machine, ptime] -> torch tensor"""
    return torch.tensor(ops_list, dtype=torch.long)

def run_single_test(instance_tensor, trainer, verbose=True):
    """Confronta OR-Tools solve_optimal_jsp con brute-force per una singola istanza."""
    # Brute-force optimal
    bf_opt, bf_seq = brute_force_optimal_makespan(instance_tensor.numpy())
    # OR-Tools call
    or_opt = trainer.solve_optimal_jsp(instance_tensor)
    if verbose:
        print("Istanza (rows: job,op,machine,ptime):")
        for r in instance_tensor.tolist():
            print(" ", r)
        print(f"Brute-force optimal makespan: {bf_opt}")
        print(f"OR-Tools returned: {or_opt}")
        if bf_seq is not None:
            print("Esempio sequenza ottima (indici):", bf_seq)
            print("Makespan della seq ottima (ricontrollo):",
                  sequence_to_schedule(bf_seq, instance_tensor.numpy()))
    return bf_opt, or_opt

def run_all_tests():
    d_model = 128

    encoder = nn.Lion17Encoder(d_model=d_model)
    decoder = nn.Lion17Decoder(d_model=d_model)
    trainer = JSPTrainer(encoder, decoder) # encoder/decoder non usati qui, serve l'istanza
    print("ATTENZIONE: JSPTrainer richiede encoder/decoder, li impostiamo a None ma abbiamo bisogno solo del metodo OR-Tools.\n")

    tests = []

    # Test 1: 2 jobs x 2 macchine — semplice
    # Job 0: op0 on M0 t=3, op1 on M1 t=2
    # Job 1: op0 on M1 t=2, op1 on M0 t=1
    inst1 = make_instance_from_list([
        [0, 0, 0, 3],
        [0, 1, 1, 2],
        [1, 0, 1, 2],
        [1, 1, 0, 1],
    ])
    tests.append(("2x2_simple", inst1))

    # Test 2: 2 jobs x 3 macchine (each job 3 ops) — small but more permutations
    inst2 = make_instance_from_list([
        [0,0,0,2],
        [0,1,1,3],
        [0,2,2,2],
        [1,0,1,1],
        [1,1,2,4],
        [1,2,0,2],
    ])
    tests.append(("2x3_chain", inst2))

    # Test 3: 3 jobs x 2 macchine (3 jobs * 2 ops = 6 ops)
    inst3 = make_instance_from_list([
        [0,0,0,2],
        [0,1,1,1],
        [1,0,1,2],
        [1,1,0,2],
        [2,0,0,1],
        [2,1,1,3],
    ])
    tests.append(("3x2_var", inst3))

    # Run tests
    all_ok = True
    for name, inst in tests:
        print("\n" + "="*60)
        print("ESECUZIONE TEST:", name)
        bf_opt, or_opt = run_single_test(inst, trainer, verbose=True)
        if or_opt != bf_opt:
            print(">>> DISCREPANZA RILEVATA:", name)
            all_ok = False
        else:
            print("OK: OR-Tools corrisponde al brute-force.")

    # Random small tests (optional): genera istanze con <= 6 ops e confronta
    import random
    random.seed(0)
    for i in range(5):
        # genera n_jobs in [2,3], n_ops per job in [2,3] ma tot_ops <= 7
        n_jobs = random.choice([2,3])
        n_ops = random.choice([2,3])
        ops = []
        for job in range(n_jobs):
            # machines permutation (ciclico o random)
            machines = list(range(min(3, n_ops)))
            random.shuffle(machines)
            for op in range(n_ops):
                machine = machines[op % len(machines)]
                ptime = random.randint(1,4)
                ops.append([job, op, machine, ptime])
        inst_rand = make_instance_from_list(ops)
        print("\n" + "-"*50)
        print(f"Random test {i+1} - ops count {len(ops)}")
        bf_opt, or_opt = run_single_test(inst_rand, trainer, verbose=True)
        if or_opt != bf_opt:
            print(">>> DISCREPANZA in random test", i+1)
            all_ok = False
        else:
            print("OK: coincide.")

    print("\n" + "="*60)
    if all_ok:
        print("TUTTI I TEST SONO PASSATI: OR-Tools concorda con il brute-force nelle istanze testate.")
    else:
        print("ALCUNE DISCREPANZE SONO STATE RILEVATE: controlla l'implementazione di solve_optimal_jsp o il timeout OR-Tools.")
    print("Fine test.")

if __name__ == "__main__":
    # Genera istanza JSP
    
    # run_all_test()

    dataset = gen.generate_dataset(1, 3, 3)  # 1 istanza 3x3
    S_seq = dataset[0]
    print(S_seq)

    # Costruisci grafo
    G, node_features = gen.build_jsp_graph(S_seq)

    print(f"Nodi: {G.number_of_nodes()}")
    print(f"Archi: {G.number_of_edges()}")
    print(f"\nNode features: {node_features}")

    # Visualizza
    gen.visualize_jsp_graph(G, node_features)

    # Converti per PyTorch Geometric
    data, node_mapping = gen.get_pytorch_geometric_data(G, node_features)
    print(f"\nPyG Data:")
    print(f"  x shape: {data.x.shape}")
    print(f"  edge_index shape: {data.edge_index.shape}")
