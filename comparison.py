import Neural_Network as nn
import Environment as env
import Generate as gen
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def load_and_evaluate_checkpoints(checkpoint_dir, graph, n_jobs=6, n_machines=6, d_model=128, device='cpu', n_samples=100):
    # Genera dataset di validazione
    print(f"Generazione dataset di validazione per graph={graph}...")
    val_dataset = gen.generate_dataset(1000, n_jobs, n_machines, return_graphs=graph)
    
    # Inizializza modello
    if graph:
        encoder = nn.GCNEncoder(n_conv=3, d_model=d_model, n_features=3).to(device)
    else:
        encoder = nn.Lion17Encoder(d_model=d_model).to(device)
    decoder = nn.Lion17Decoder(d_model=d_model).to(device)
    trainer = env.JSPTrainer(encoder, decoder, device, graph=graph)
    
    # Liste per salvare le metriche
    epochs = []
    val_makespans = []
    val_gaps = []
    
    # Cerca i checkpoint nella directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.pth')[0]))  # Estrai il numero dell'epoca
    
    for checkpoint_file in tqdm(checkpoint_files, desc=f"Valutazione checkpoint per graph={graph}"):
        # Estrai il numero dell'epoca dal nome del file
        epoch = int(checkpoint_file.split('_')[2].split('.pth')[0])
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        # Carica il checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        
        # Valuta il modello
        performance = trainer.evaluate_model(val_dataset, n_samples=n_samples)
        
        # Salva le metriche
        epochs.append(epoch)
        val_makespans.append(performance['avg_makespan'])
        val_gaps.append(performance['avg_gap_from_optimal'])
    
    return epochs, val_makespans, val_gaps

def plot_comparison(gcn_metrics, lion_metrics, checkpoint_dir_gcn, checkpoint_dir_lion):
    gcn_epochs, gcn_makespans, gcn_gaps = gcn_metrics
    lion_epochs, lion_makespans, lion_gaps = lion_metrics
    
    plt.figure(figsize=(10, 5))
    
    # Plot Validation Makespan
    plt.subplot(1, 2, 1)
    plt.plot(gcn_epochs, gcn_makespans, label='GCN (graph=True)', color='blue', marker='o')
    plt.plot(lion_epochs, lion_makespans, label='Lion17 (graph=False)', color='orange', marker='o')
    plt.xlabel('Epoca')
    plt.ylabel('Makespan Medio')
    plt.title('Confronto Makespan di Validazione')
    plt.legend()
    plt.grid(True)
    
    # Plot Validation Gap
    plt.subplot(1, 2, 2)
    plt.plot(gcn_epochs, gcn_gaps, label='GCN (graph=True)', color='blue', marker='o')
    plt.plot(lion_epochs, lion_gaps, label='Lion17 (graph=False)', color='orange', marker='o')
    plt.xlabel('Epoca')
    plt.ylabel('Gap % da Ottimale')
    plt.title('Confronto Gap da Ottimale')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

def main():
    device = 'cpu'  # Usa CPU come nel tuo codice originale
    checkpoint_dir_gcn = './checkpoint_GCN_CPU/'  # Directory per GCN (graph=True)
    checkpoint_dir_lion = './checkpoint_CPU/'  # Directory per Lion17 (graph=False)
    n_jobs = 6
    n_machines = 6
    d_model = 128
    n_samples = 100  # Numero di istanze per la valutazione
    
    print(f"Utilizzo del dispositivo: {device}")
    
    # Carica e valuta i checkpoint per GCNEncoder (graph=True)
    print("Valutazione checkpoint per GCNEncoder (graph=True)...")
    gcn_metrics = load_and_evaluate_checkpoints(checkpoint_dir_gcn, graph=True, 
                                               n_jobs=n_jobs, n_machines=n_machines, 
                                               d_model=d_model, device=device, n_samples=n_samples)
    
    # Carica e valuta i checkpoint per Lion17Encoder (graph=False)
    print("\nValutazione checkpoint per Lion17Encoder (graph=False)...")
    lion_metrics = load_and_evaluate_checkpoints(checkpoint_dir_lion, graph=False, 
                                                n_jobs=n_jobs, n_machines=n_machines, 
                                                d_model=d_model, device=device, n_samples=n_samples)
    
    # Visualizza il confronto
    print("\nVisualizzazione del confronto delle performance...")
    plot_comparison(gcn_metrics, lion_metrics, checkpoint_dir_gcn, checkpoint_dir_lion)
    
    # Stampa un riepilogo finale
    print("\nRiepilogo finale:")
    if gcn_metrics[0]:  # Controlla se ci sono epoche
        print(f"GCNEncoder (graph=True) all'epoca {gcn_metrics[0][-1]}:")
        print(f"  Ultimo Makespan Medio: {gcn_metrics[1][-1]:.2f}")
        print(f"  Ultimo Gap da Ottimale: {gcn_metrics[2][-1]:.2f}%")
    else:
        print("Nessun checkpoint trovato per GCNEncoder (graph=True)")
    if lion_metrics[0]:
        print(f"Lion17Encoder (graph=False) all'epoca {lion_metrics[0][-1]}:")
        print(f"  Ultimo Makespan Medio: {lion_metrics[1][-1]:.2f}")
        print(f"  Ultimo Gap da Ottimale: {lion_metrics[2][-1]:.2f}%")
    else:
        print("Nessun checkpoint trovato per Lion17Encoder (graph=False)")

if __name__ == "__main__":
    main()