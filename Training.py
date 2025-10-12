import Neural_Network as nn
import Environment as env
import Generate as gen
import torch
import os

####################
# Modificare grafo #
####################

def main(graph=False, n_jobs=6, n_machines=6, n_epochs=150, save_every = 10, batch_size=512, resume_from_epoch=None, directory = './checkpoint_GCN_CPU/', lr = 1e-4, checkpoint_dir='./checkpoint_GCN_CPU_different_lr', GAT = False):
    device = 'cpu'  # Usa CPU come nel codice originale
    print(f"Utilizzo del dispositivo: {device}")

    # Genera dataset
    print("Generazione dataset...")
    train_dataset = gen.generate_dataset(10000, n_jobs, n_machines, return_graphs=graph)
    val_dataset = gen.generate_dataset(1000, n_jobs, n_machines, return_graphs=graph)
    
    # Inizializza modello
    d_model = 128
    if graph:
        if GAT:
            encoder = nn.GATGCNEncoderDropout(n_gcn_conv=2, d_model=d_model, n_features=4, n_heads = 4).to(device)
        else:
            encoder = nn.GCNEncoderDropout(n_conv=3, d_model=d_model, n_features=4).to(device)
    else:
        encoder = nn.Lion17Encoder(d_model=d_model).to(device)
    decoder = nn.Lion17Decoder(d_model=d_model).to(device)
    
    # Crea il trainer
    trainer = env.JSPTrainer(encoder, decoder, device, graph=graph, lr=lr)
    
    # Carica checkpoint se specificato
    start_epoch = 0
    if resume_from_epoch is not None:
        checkpoint_path = f'{directory}checkpoint_epoch_{resume_from_epoch}.pth'
        if os.path.exists(checkpoint_path):
            print(f"Caricamento checkpoint da {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            start_epoch = resume_from_epoch
            print(f"Ripresa del training dall'epoca {start_epoch + 1}")
        else:
            print(f"Checkpoint {checkpoint_path} non trovato. Inizio training da zero.")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Training loop
    for epoch in range(start_epoch, n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        train_loss = trainer.train_epoch(train_dataset, batch_size)
        val_performance = trainer.evaluate_model(val_dataset, n_samples=100)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Average Makespan: {val_performance['avg_makespan']:.2f}")
        print(f"Val Gap from Optimal: {val_performance['avg_gap_from_optimal']:.2f}%")
        
        # Salva checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth')
            print(f"Checkpoint salvato: {checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    # main(graph=True, resume_from_epoch=None, directory='./checkpoint_GCN_CPU_4_features/', checkpoint_dir='./checkpoint_GCN_CPU_4_features')
    # main(graph=True, GAT = True, resume_from_epoch=None, directory='./checkpoint_GATGCN_CPU_gelu_1e-5/', checkpoint_dir='./checkpoint_GATGCN_CPU_gelu_1e-5', lr = 1e-5)
    # main(graph=True, GAT = True, resume_from_epoch=None, directory='./checkpoint_GATGCN_CPU_gelu_1e-6/', checkpoint_dir='./checkpoint_GATGCN_CPU_gelu_1e-6', lr = 1e-6)
    # main(graph=True, GAT = True, resume_from_epoch=None, directory='./checkpoint_GATGCN_CPU_gelu_5e-6/', checkpoint_dir='./checkpoint_GATGCN_CPU_gelu_5e-6', lr = 5e-6)
    # main(graph=False, resume_from_epoch=None, directory='./checkpoint_CPU/', checkpoint_dir='./checkpoint_CPU')
    main(graph = True, resume_from_epoch=None, directory="./checkpoint_GCN_Dropout/", checkpoint_dir="./checkpoint_GCN_Dropout", GAT = False)
    main(graph=True, GAT = True, resume_from_epoch=None, directory='./checkpoint_GATGCN_CPU_gelu_1e-5_Dropout/', checkpoint_dir='./checkpoint_GATGCN_CPU_gelu_1e-5_Dropout', lr = 1e-5)