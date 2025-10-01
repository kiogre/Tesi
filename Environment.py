import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

#################################
# Modificare calculate makespan #
#################################

##############################################
# Implementare funzione per optimal solution #
##############################################


class JSPTrainer:
    def __init__(self, encoder, decoder, device, lr=1e-4):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), 
            lr=lr
        )
        self.baseline = None
        self.device = device
    
    def train_epoch(self, dataset, batch_size=32):
        total_loss = 0
        num_batches = len(dataset) // batch_size
        
        for i in tqdm(range(num_batches), total=num_batches):
            batch = dataset[i*batch_size:(i+1)*batch_size]
            batch_tensor = torch.stack([inst.detach().clone() for inst in batch]).to(self.device)
            
            # Forward pass
            encoder_outputs = self.encoder(batch_tensor)
            sequences, log_probs = self.decoder(encoder_outputs, batch_tensor)
            
            # Calcola rewards (negative makespan)
            rewards = []
            for b in range(batch_size):
                makespan = self.calculate_makespan_from_sequence(sequences[b], batch_tensor[b])
                rewards.append(-makespan)  # Negativo perché vogliamo minimizzare
            
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
            # Baseline (media mobile)
            if self.baseline is None:
                self.baseline = rewards.mean()
            else:
                self.baseline = 0.9 * self.baseline + 0.1 * rewards.mean()
            
            # REINFORCE loss
            advantage = rewards - self.baseline
            advantage = advantage.to(self.device)
            loss = -(log_probs.sum(dim=1) * advantage).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                0.5
            )
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches

    def indices_to_schedule(output_indices, S_seq):
        """Converte indici in scheduling effettivo"""
        schedule = []
        
        for idx in output_indices:
            operation = S_seq[idx]  # [job_id, op_id, machine, time]
            schedule.append(operation)
        
        return schedule

    def schedule_to_gantt(schedule):
        """Converte in Gantt chart con tempi effettivi"""
        machine_times = {}  # Quando ogni macchina si libera
        job_times = {}      # Quando ogni job finisce ultima operazione
        
        gantt = []
        
        for op in schedule:
            job_id, op_id, machine, processing_time = op
            
            # Trova quando può iniziare
            machine_available = machine_times.get(machine, 0)
            job_available = job_times.get((job_id, op_id-1), 0) if op_id > 0 else 0
            
            start_time = max(machine_available, job_available)
            end_time = start_time + processing_time
            
            gantt.append({
                'job': job_id,
                'operation': op_id, 
                'machine': machine,
                'start': start_time,
                'end': end_time
            })
            
            # Aggiorna disponibilità
            machine_times[machine] = end_time
            job_times[(job_id, op_id)] = end_time
        
        return gantt
    
    def evaluate_model(self, test_dataset, n_samples=100):
        """Valuta performance del modello su test set"""
        if isinstance(test_dataset, torch.Tensor) and test_dataset.dim() == 2:
            test_dataset = [test_dataset]

        self.encoder.eval()
        self.decoder.eval()
        
        total_makespan = 0
        total_gap_from_optimal = 0
        
        with torch.no_grad():
            for i, instance in enumerate(test_dataset[:n_samples]):
                # Forward pass
                instance_batch = instance.unsqueeze(0).to(self.device)
                encoder_outputs = self.encoder(instance_batch)
                sequences, _ = self.decoder(encoder_outputs, instance_batch, training=False)
                
                # Calcola makespan
                makespan = self.calculate_makespan_from_sequence(
                    sequences[0], instance_batch[0]
                )
                total_makespan += makespan
                
                # Confronta con soluzione ottimale (se disponibile)
                optimal_makespan = self.solve_optimal_jsp(instance)
                if optimal_makespan > 0:
                    gap = (makespan - optimal_makespan) / optimal_makespan
                    total_gap_from_optimal += gap
        
        avg_makespan = total_makespan / n_samples
        avg_gap = total_gap_from_optimal / n_samples
        
        self.encoder.train()
        self.decoder.train()
        
        return {
            'avg_makespan': avg_makespan,
            'avg_gap_from_optimal': avg_gap * 100  # Percentuale
        }

    def calculate_makespan_from_sequence(self, sequence_indices, S_seq):
        """Converte sequenza di indici in makespan"""
        n_machines = int(S_seq[:, 2].max()) + 1
        machine_available_time = [0] * n_machines
        job_completion_times = {}
        
        for idx in sequence_indices:
            job_id = int(S_seq[idx, 0])
            op_id = int(S_seq[idx, 1])
            machine = int(S_seq[idx, 2])
            processing_time = int(S_seq[idx, 3])
            
            # Trova quando può iniziare
            machine_ready = machine_available_time[machine]
            job_ready = job_completion_times.get((job_id, op_id-1), 0) if op_id > 0 else 0
            
            start_time = max(machine_ready, job_ready)
            end_time = start_time + processing_time
            
            # Aggiorna tempi
            machine_available_time[machine] = end_time
            job_completion_times[(job_id, op_id)] = end_time
        
        # Makespan = max tempo di completamento
        return max(machine_available_time)

    def solve_optimal_jsp(self, instance):
        """Risolve JSP ottimalmente con OR-Tools"""
        try:
            from ortools.sat.python import cp_model
            
            # Estrai info dall'istanza
            S_seq = instance.numpy()
            n_jobs = int(S_seq[:, 0].max()) + 1
            n_machines = int(S_seq[:, 2].max()) + 1
            
            # Crea il modello
            model = cp_model.CpModel()
            
            # Variabili: start_time per ogni operazione
            start_times = {}
            end_times = {}
            
            for i, op in enumerate(S_seq):
                job_id, op_id, machine, proc_time = op
                start_times[(job_id, op_id)] = model.NewIntVar(0, 10000, f'start_{job_id}_{op_id}')
                end_times[(job_id, op_id)] = model.NewIntVar(0, 10000, f'end_{job_id}_{op_id}')
                
                # Constraint: end = start + processing_time
                model.Add(end_times[(job_id, op_id)] == start_times[(job_id, op_id)] + proc_time)
            
            # Precedence constraints (stesso job)
            for job_id in range(n_jobs):
                job_ops = [op for op in S_seq if op[0] == job_id]
                job_ops = sorted(job_ops, key=lambda x: x[1])  # Ordina per op_id
                
                for i in range(len(job_ops) - 1):
                    curr_op = (job_ops[i][0], job_ops[i][1])
                    next_op = (job_ops[i+1][0], job_ops[i+1][1])
                    model.Add(start_times[next_op] >= end_times[curr_op])
            
            # Resource constraints (stessa macchina)
            machine_ops = {}
            for op in S_seq:
                job_id, op_id, machine, proc_time = op
                if machine not in machine_ops:
                    machine_ops[machine] = []
                machine_ops[machine].append((job_id, op_id))
            
            for machine, ops in machine_ops.items():
                for i in range(len(ops)):
                    for j in range(i + 1, len(ops)):
                        op1, op2 = ops[i], ops[j]
                        
                        # Either op1 before op2 OR op2 before op1
                        b = model.NewBoolVar(f'order_{machine}_{i}_{j}')
                        model.Add(end_times[op1] <= start_times[op2]).OnlyEnforceIf(b)
                        model.Add(end_times[op2] <= start_times[op1]).OnlyEnforceIf(b.Not())
            
            # Objective: minimize makespan
            makespan = model.NewIntVar(0, 10000, 'makespan')
            for job_id in range(n_jobs):
                last_op_of_job = max([(op[0], op[1]) for op in S_seq if op[0] == job_id], 
                                key=lambda x: x[1])
                model.Add(makespan >= end_times[last_op_of_job])
            
            model.Minimize(makespan)
            
            # Solve
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 60.0  # 1 minuto timeout
            
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                return solver.Value(makespan)
            else:
                return -1
        except Exception as e:
            print(f"OR-Tools error: {e}")
            return -1
            
    def plot_gantt(self, schedule):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        for task in schedule:
            job_id = task['job']
            machine = task['machine']
            start = task['start'] 
            duration = task['end'] - task['start']
            
            rect = patches.Rectangle((start, machine), duration, 0.8,
                                facecolor=colors[job_id % len(colors)],
                                alpha=0.7)
            ax.add_patch(rect)
            
            # Label
            ax.text(start + duration/2, machine + 0.4, 
                    f'J{job_id}', ha='center', va='center')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('Gantt Chart')
        plt.show()

    def compare_with_baselines(self, test_instances):
        results = {'neural': [], 'spt': [], 'optimal': []}
        
        for instance in test_instances:
            # Neural network solution
            neural_makespan = self.evaluate_model(instance) # Questo qui valuta la rete neurale
            results['neural'].append(neural_makespan)
            
            # SPT baseline
            spt_schedule = spt_dispatching_rule(instance)
            spt_makespan = self.calculate_makespan_from_sequence(spt_schedule)
            results['spt'].append(spt_makespan)
            
            # Optimal (se possibile con OR-Tools)
            optimal_makespan = self.solve_optimal_jsp(instance)
            results['optimal'].append(optimal_makespan)
        
        return results
            

def spt_dispatching_rule(operations):
    """Shortest Processing Time first"""
    return sorted(operations, key=lambda x: x[3])