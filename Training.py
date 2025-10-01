import Neural_Network as nn
import Environment as env
import Generate as gen
import torch

####################
# Modificare grafo #
####################

def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # A quanto pare i dati che metto sono
    # molto piccoli e quindi non conviene usare la GPU
    device = 'cpu'
    print(f"Using device: {device}")

    # Hyperparameters
    n_jobs = 6
    n_machines = 6
    d_model = 128
    n_epochs = 50
    batch_size = 512
    
    # Dataset
    print("Generating dataset...")
    train_dataset = gen.generate_dataset(10000, n_jobs, n_machines)
    val_dataset = gen.generate_dataset(1000, n_jobs, n_machines)
    
    # Model
    encoder = nn.Lion17Encoder(d_model=d_model).to(device)
    decoder = nn.Lion17Decoder(d_model=d_model).to(device)
    trainer = env.JSPTrainer(encoder, decoder, device)
    
    # Training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        
        train_loss = trainer.train_epoch(train_dataset, batch_size)
        val_performance = trainer.evaluate_model(val_dataset)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Average Makespan: {val_performance['avg_makespan']:.2f}")
        print(f"Val Gap from Optimal: {val_performance['avg_gap_from_optimal']:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
            }, f'checkpoint_epoch_{epoch}.pth')


if __name__ == "__main__":
    # main()
    S_seq, (n_jobs, n_machines) = gen.load_orlib_instance("jobshop1.txt", "la31")
    print(f"Istanza ft06: {n_jobs} jobs x {n_machines} machines")
    print(S_seq)
    d_model = 128
    device = 'cpu'

    encoder = nn.Lion17Encoder(d_model=d_model).to(device)
    decoder = nn.Lion17Decoder(d_model=d_model).to(device)
    trainer = env.JSPTrainer(encoder, decoder, device)

    val_performance = trainer.evaluate_model(S_seq)
    print(f"Val Average Makespan: {val_performance['avg_makespan']:.2f}")
    print(f"Val Gap from Optimal: {val_performance['avg_gap_from_optimal']:.2f}%")
