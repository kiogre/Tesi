from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

####################
# Modificare grafo #
####################

def generate_dataset(size, n_jobs, n_machines, return_graphs = False):
    """Genera dataset di istanze JSP casuali usando metodo Taillard"""
    import numpy as np
    
    dataset = []
    
    for _ in tqdm(range(size), total=size, desc="Creazione di un dataset"):
        operations = []
        
        for job_id in range(n_jobs):
            # Ordine macchine random per ogni job (come Taillard)
            machines = list(range(n_machines))
            np.random.shuffle(machines)
            
            for op_id, machine in enumerate(machines):
                # Tempo di processing casuale
                processing_time = np.random.randint(1, 100)
                operations.append([job_id, op_id, machine, processing_time])
        
        # Converte in tensor
        instance_tensor = torch.tensor(operations, dtype=torch.long)
        if return_graphs:
            # Pre-calcola grafo
            G, node_features = build_jsp_graph(instance_tensor)
            pyg_data, node_mapping = get_pytorch_geometric_data(G, node_features)
            dataset.append((instance_tensor, pyg_data))
        else:
            dataset.append(instance_tensor)
    
    return dataset

def load_orlib_instance(filename: str, instance_name: str):
    """
    Estrae un'istanza specifica (es. "abz5" o "ft06") da un file jobshop1.txt.
    
    Restituisce:
      - S_seq: torch.tensor [n_jobs * n_machines, 4]
               colonne: [job_id, op_id, machine, ptime]
      - (n_jobs, n_machines)
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # trova "instance <nome>"
    start_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith("instance") and instance_name.lower() in line.lower():
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Istanza {instance_name} non trovata in {filename}")

    # cerca la prima riga dopo che contenga due interi = n_jobs n_machines
    n_jobs = n_machines = None
    desc_idx = start_idx + 1
    while desc_idx < len(lines):
        parts = lines[desc_idx].split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            n_jobs, n_machines = map(int, parts)
            break
        desc_idx += 1
    if n_jobs is None:
        raise ValueError(f"Formato non valido per {instance_name} in {filename}")

    # leggi le n_jobs righe successive
    ops = []
    for job_id in range(n_jobs):
        parts = list(map(int, lines[desc_idx + 1 + job_id].split()))
        for op_id in range(n_machines):
            machine = parts[2 * op_id]   # già 0-based nel tuo file
            ptime   = parts[2 * op_id + 1]
            ops.append([job_id, op_id, machine, ptime])

    S_seq = torch.tensor(ops, dtype=torch.long)
    return S_seq, (n_jobs, n_machines)

def build_jsp_graph(S_seq):
    """
    Costruisce grafo da istanza JSP con features complete
    
    Args:
        S_seq: tensor (seq_len, 4) con [job_id, op_id, machine, processing_time]
    
    Returns:
        G: NetworkX DiGraph con nodi operazioni e macchine
        node_features: dict con features dei nodi (ora vettori multi-dimensionali)
    """
    if isinstance(S_seq, torch.Tensor):
        S_seq = S_seq.cpu().numpy()
    
    G = nx.DiGraph()
    node_features = {}
    
    # Calcola info globali per normalizzazione
    max_proc_time = max(row[3] for row in S_seq)
    min_proc_time = min(row[3] for row in S_seq)
    
    # Calcola numero operazioni per job
    job_num_ops = {}
    for job_id, op_id, _, _ in S_seq:
        job_id = int(job_id)
        if job_id not in job_num_ops:
            job_num_ops[job_id] = 0
        job_num_ops[job_id] = max(job_num_ops[job_id], int(op_id) + 1)
    
    # Dizionari
    op_nodes = {}
    machine_nodes = {}
    
    # 1. Crea nodi operazioni con features
    for idx, (job_id, op_id, machine, proc_time) in enumerate(S_seq):
        job_id, op_id, machine = int(job_id), int(op_id), int(machine)
        proc_time = int(proc_time)
        
        op_node = f"J{job_id}_O{op_id}"
        op_nodes[(job_id, op_id)] = op_node
        
        # Calcola features
        total_ops_in_job = job_num_ops[job_id]
        remaining_ops = total_ops_in_job - op_id - 1  # Operazioni dopo questa
        
        # Normalizza processing time [0, 1]
        if max_proc_time > min_proc_time:
            normalized_proc_time = (proc_time - min_proc_time) / (max_proc_time - min_proc_time)
        else:
            normalized_proc_time = 0.5
        
        
        # Features vector
        features = [
            op_id,                    # 0: quale operazione è (0, 1, 2, ...)
            normalized_proc_time,     # 1: processing time normalizzato
            remaining_ops,            # 2: quante operazioni mancano
            1,                        # 3: tipologia del nodo (1 = operazione)
        ]
        
        G.add_node(op_node, 
                   node_type='operation',
                   job_id=job_id,
                   op_id=op_id,
                   machine=machine,
                   processing_time=proc_time)
        
        node_features[op_node] = features
    
    # 2. Crea nodi macchine
    machines = set(int(row[2]) for row in S_seq)
    for machine_id in machines:
        machine_node = f"M{machine_id}"
        machine_nodes[machine_id] = machine_node
        
        # Calcola workload della macchina
        machine_workload = sum(int(row[3]) for row in S_seq if int(row[2]) == machine_id)
        num_ops_on_machine = sum(1 for row in S_seq if int(row[2]) == machine_id)
        
        # Features macchina (diverse da operazioni)
        features = [
            -1,                              # 0: flag "è macchina"
            machine_workload / 1000.0,       # 1: carico totale normalizzato
            num_ops_on_machine,              # 2: numero operazioni
            0,                               # 3: tipologia del nodo (0 = macchina)
        ]
        
        G.add_node(machine_node,
                   node_type='machine',
                   machine_id=machine_id)
        
        node_features[machine_node] = features
    
    # 3. Archi precedenza
    for (job_id, op_id), op_node in op_nodes.items():
        next_op = (job_id, op_id + 1)
        if next_op in op_nodes:
            next_op_node = op_nodes[next_op]
            G.add_edge(op_node, next_op_node, edge_type='precedence')
    
    # 4. Archi bidirezionali operazione <-> macchina
    for idx, (job_id, op_id, machine, proc_time) in enumerate(S_seq):
        job_id, op_id, machine = int(job_id), int(op_id), int(machine)
        
        op_node = op_nodes[(job_id, op_id)]
        machine_node = machine_nodes[machine]
        
        G.add_edge(op_node, machine_node, edge_type='uses_machine')
        G.add_edge(machine_node, op_node, edge_type='used_by')
    
    return G, node_features

def visualize_jsp_graph(G, node_features=None, figsize=(14, 10)):
    """
    Visualizza il grafo JSP
    
    Args:
        G: NetworkX DiGraph
        node_features: dict con features dei nodi (opzionale)
        figsize: dimensioni figura
    """
    plt.figure(figsize=figsize)
    
    # Separa nodi per tipo
    operation_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'operation']
    machine_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'machine']
    
    # Layout: operazioni a sinistra, macchine a destra
    pos = {}
    
    # Organizza operazioni per job
    jobs = {}
    for node in operation_nodes:
        job_id = G.nodes[node]['job_id']
        if job_id not in jobs:
            jobs[job_id] = []
        jobs[job_id].append(node)
    
    # Posiziona operazioni (layout per job)
    y_offset = 0
    for job_id in sorted(jobs.keys()):
        job_ops = sorted(jobs[job_id], key=lambda n: G.nodes[n]['op_id'])
        for i, node in enumerate(job_ops):
            pos[node] = (i * 2, -y_offset)
        y_offset += 1.5
    
    # Posiziona macchine (colonna a destra)
    max_x = max(p[0] for p in pos.values()) + 3
    for i, machine_node in enumerate(sorted(machine_nodes)):
        pos[machine_node] = (max_x, -i * 1.5)
    
    # Separa archi per tipo
    precedence_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'precedence']
    machine_edges = [(u, v) for u, v, d in G.edges(data=True) 
                     if d.get('edge_type') in ['uses_machine', 'used_by']]
    
    # Disegna nodi
    nx.draw_networkx_nodes(G, pos, nodelist=operation_nodes,
                          node_color='lightblue', node_size=800,
                          label='Operations')
    nx.draw_networkx_nodes(G, pos, nodelist=machine_nodes,
                          node_color='lightcoral', node_size=1000,
                          node_shape='s', label='Machines')
    
    # Disegna archi precedenza (operazione -> operazione)
    nx.draw_networkx_edges(G, pos, edgelist=precedence_edges,
                          edge_color='blue', width=2, alpha=0.6,
                          arrows=True, arrowsize=20,
                          arrowstyle='->', connectionstyle='arc3,rad=0.1',
                          label='Precedence')
    
    # Disegna archi macchina (operazione <-> macchina)
    nx.draw_networkx_edges(G, pos, edgelist=machine_edges,
                          edge_color='gray', width=1, alpha=0.3,
                          arrows=False, style='dashed',
                          label='Machine assignment')
    
    # Labels
    labels = {}
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'operation':
            job_id = G.nodes[node]['job_id']
            op_id = G.nodes[node]['op_id']
            labels[node] = f"J{job_id}\nO{op_id}"
        else:
            machine_id = G.nodes[node]['machine_id']
            labels[node] = f"M{machine_id}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    
    plt.title("Job Shop Problem Graph\n(Blue arrows: precedence, Gray dashed: machine assignment)", 
              fontsize=14)
    plt.legend(loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_pytorch_geometric_data(G, node_features):
    """Converte in PyG con features multi-dimensionali"""
    from torch_geometric.data import Data
    
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Node features ora sono vettori
    x = torch.tensor([node_features[node] for node in G.nodes()], 
                     dtype=torch.float32)  # (num_nodes, 6)
    
    # Edge index
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    data = Data(x=x, edge_index=edge_index)
    data.node_ids = list(G.nodes())  # Aggiungi gli ID dei nodi
    
    return data, node_to_idx