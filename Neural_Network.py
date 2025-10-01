import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
import torch.nn.functional as F

###########################################
# Dovrei mettere dei batchnorm nella GCN? #
###########################################


class Lion17Encoder(nn.Module):
    def __init__(self, d_model = 128, n_heads = 8, n_layers = 3):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.job_op_embedding = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, d_model)
        )

        self.machine_embedding = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, d_model)
        )

        self.combined = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, d_model)
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in range(n_layers)]
        )


    def forward(self, S_seq):
        batch_size, seq_len, _ = S_seq.shape

        job_op = S_seq[:, :, :2].float()
        machine = S_seq[:, :, 2:].float()

        job_op = job_op.view(-1, 2)
        machine = machine.view(-1, 2)

        job_op = F.relu(self.job_op_embedding(job_op))
        machine = F.relu(self.machine_embedding(machine))

        job_op = job_op.reshape(batch_size, seq_len, -1)
        machine = machine.reshape(batch_size, seq_len, -1)

        combined = job_op + machine

        combined = combined.view(-1, self.d_model)

        X = self.combined(combined)
        X = X.view(batch_size, seq_len, -1)

        for attention in self.attentions:
            # Multi-head attention 
            attn_output, _ = attention(X, X, X)  # Query, Key, Value

            X = attn_output
        
        return X
    


class Lion17Decoder(nn.Module):
    def __init__(self, d_model = 128):
        super().__init__()
        
        self.d_model = d_model

        self.LSTM_layer = nn.LSTM(d_model, d_model, batch_first=True)

        self.pointer_attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, encoder_outputs, S_seq, training = True):
        # encoder_outputs: (batch_size, seq_len, d_model)
        # S_seq: (batch_size, seq_len, 4) per costruire maschere

        device = encoder_outputs.device
        
        batch_size, seq_len, d_model = encoder_outputs.shape
        
        # Inizializza decoder
        decoder_hidden = encoder_outputs.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        h_0 = torch.zeros(1, batch_size, d_model).to(device)  # <-- .to(device)
        c_0 = torch.zeros(1, batch_size, d_model).to(device)  # <-- .to(device)

        
        # Inizializza maschere
        M_sched = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        M_mask = self.init_precedence_mask(S_seq).to(device)  # <-- .to(device)
        
        output_sequence = []
        log_probs = []
        
        for step in range(seq_len):
            # LSTM step
            decoder_output, (h_0, c_0) = self.LSTM_layer(decoder_hidden, (h_0, c_0))
            query = decoder_output.squeeze(1)  # (batch, d_model)
            
            # Calcola attention scores
            scores = self.calculate_pointer_scores(query, encoder_outputs)
            
            # Applica masking
            masked_scores = self.apply_masks(scores, M_sched, M_mask)
            
            # Softmax
            probs = F.softmax(masked_scores, dim=-1)
            log_prob = F.log_softmax(masked_scores, dim=-1)
            
            # Sampling o greedy
            if training:
                chosen_idx = torch.multinomial(probs, 1).squeeze(-1)
            else:
                chosen_idx = torch.argmax(probs, dim=-1)
            
            output_sequence.append(chosen_idx)
            log_probs.append(log_prob.gather(1, chosen_idx.unsqueeze(1)))
            
            # Aggiorna stato
            M_sched = self.update_scheduled_mask(M_sched, chosen_idx)
            M_mask = self.update_precedence_mask(M_mask, chosen_idx, S_seq)
            
            # Prepara per prossimo step
            chosen_embeddings = encoder_outputs.gather(1, chosen_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, d_model))
            decoder_hidden = chosen_embeddings
        
        return torch.stack(output_sequence, dim=1), torch.cat(log_probs, dim=1)

    def calculate_pointer_scores(self, query, encoder_outputs):
        # Come Vinyals et al. 2015
        batch_size, seq_len, d_model = encoder_outputs.shape
        
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([query_expanded, encoder_outputs], dim=-1)
        
        scores = self.pointer_attention(combined).squeeze(-1)
        return scores

    def init_precedence_mask(self, S_seq):
        """
        Definition 3 dal paper: M_mask_kp è true iff l > j,
        dove j è l'indice della prossima operazione del job i-esimo
        """
        device = S_seq.device
        batch_size, seq_len, _ = S_seq.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                job_id = int(S_seq[b, i, 0])
                op_id = int(S_seq[b, i, 1])
                
                # Se non è la prima operazione del job, inizialmente è mascherata
                if op_id > 0:
                    mask[b, i] = True
        
        return mask
    
    def apply_masks(self, scores, M_sched, M_mask):
        """
        Equation (10) dal paper:
        mask(u_t_p|o'_0,...,o'_{t-1}) = {-∞ if M_sched_kp OR M_mask_kp, u_t_p otherwise}
        """
        combined_mask = M_sched | M_mask
        masked_scores = scores.masked_fill(combined_mask, float('-inf'))
        return masked_scores
    
    def update_scheduled_mask(self, M_sched, chosen_indices):
        """
        Definition 2: M_sched_kp diventa true quando l'operazione p è schedulata
        """
        batch_size = chosen_indices.size(0)
        new_mask = M_sched.clone()
        
        for b in range(batch_size):
            chosen_idx = chosen_indices[b].item()
            new_mask[b, chosen_idx] = True
        
        return new_mask
    
    def update_precedence_mask(self, M_mask, chosen_indices, S_seq):
        """
        Dal paper: M_sched_kp ← true, M_mask_kp+1 ← false
        Quando schedulo un'operazione, sblocco la prossima del stesso job
        """
        batch_size = chosen_indices.size(0)
        new_mask = M_mask.clone()
        
        for b in range(batch_size):
            chosen_idx = chosen_indices[b].item()
            
            # Operazione appena schedulata
            job_id = int(S_seq[b, chosen_idx, 0])
            op_id = int(S_seq[b, chosen_idx, 1])
            
            # Sblocca la prossima operazione dello stesso job
            for i in range(S_seq.size(1)):
                if (int(S_seq[b, i, 0]) == job_id and 
                    int(S_seq[b, i, 1]) == op_id + 1):
                    new_mask[b, i] = False
                    break
        
        return new_mask
    

class GCNEncoder(nn.Module):
    def __init__(self, n_conv = 3, d_model = 128, n_features = 4):
        super().__init__()
        self.n_conv = n_conv
        self.d_model = d_model
        self.n_features = n_features

        self.first_layer = GCNConv(self.n_features, self.d_model, add_self_loops=True)

        self.conv_layers = nn.ModuleList([GCNConv(self.d_model, self.d_model, add_self_loops=True) for _ in range(1, self.n_conv)])

    def forward(self, x, edge_index, batch_size = None, seq_len = None, change = False):

        x = self.first_layer(x, edge_index)

        for convolution in self.conv_layers:
            x = F.relu(x)
            x = convolution(x)

        if batch_size is not None and seq_len is not None and change:
            x = x.view(batch_size, seq_len, self.d_model)
            
        return x

# Possibile classe in futuro
'''
class GCNDecoder(nn.Module):
    def __init__(self, d_model = 128):
        super().__init__()
        
        self.d_model = d_model

        self.LSTM_layer = nn.LSTM(d_model, d_model, batch_first=True)

        self.pointer_attention = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, encoder_outputs, S_seq, training = True):
        # encoder_outputs: (batch_size, seq_len, d_model)
        # S_seq: (batch_size, seq_len, 4) per costruire maschere

        device = encoder_outputs.device
        
        batch_size, seq_len, d_model = encoder_outputs.shape
        
        # Inizializza decoder
        decoder_hidden = encoder_outputs.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        h_0 = torch.zeros(1, batch_size, d_model).to(device)  # <-- .to(device)
        c_0 = torch.zeros(1, batch_size, d_model).to(device)  # <-- .to(device)

        
        # Inizializza maschere
        M_sched = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        M_mask = self.init_precedence_mask(S_seq).to(device)  # <-- .to(device)
        
        output_sequence = []
        log_probs = []
        
        for step in range(seq_len):
            # LSTM step
            decoder_output, (h_0, c_0) = self.LSTM_layer(decoder_hidden, (h_0, c_0))
            query = decoder_output.squeeze(1)  # (batch, d_model)
            
            # Calcola attention scores
            scores = self.calculate_pointer_scores(query, encoder_outputs)
            
            # Applica masking
            masked_scores = self.apply_masks(scores, M_sched, M_mask)
            
            # Softmax
            probs = F.softmax(masked_scores, dim=-1)
            log_prob = F.log_softmax(masked_scores, dim=-1)
            
            # Sampling o greedy
            if training:
                chosen_idx = torch.multinomial(probs, 1).squeeze(-1)
            else:
                chosen_idx = torch.argmax(probs, dim=-1)
            
            output_sequence.append(chosen_idx)
            log_probs.append(log_prob.gather(1, chosen_idx.unsqueeze(1)))
            
            # Aggiorna stato
            M_sched = self.update_scheduled_mask(M_sched, chosen_idx)
            M_mask = self.update_precedence_mask(M_mask, chosen_idx, S_seq)
            
            # Prepara per prossimo step
            chosen_embeddings = encoder_outputs.gather(1, chosen_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, d_model))
            decoder_hidden = chosen_embeddings
        
        return torch.stack(output_sequence, dim=1), torch.cat(log_probs, dim=1)

    def calculate_pointer_scores(self, query, encoder_outputs):
        # Come Vinyals et al. 2015
        batch_size, seq_len, d_model = encoder_outputs.shape
        
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([query_expanded, encoder_outputs], dim=-1)
        
        scores = self.pointer_attention(combined).squeeze(-1)
        return scores

    def init_precedence_mask(self, S_seq):
        """
        Definition 3 dal paper: M_mask_kp è true iff l > j,
        dove j è l'indice della prossima operazione del job i-esimo
        """
        device = S_seq.device
        batch_size, seq_len, _ = S_seq.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                job_id = int(S_seq[b, i, 0])
                op_id = int(S_seq[b, i, 1])
                
                # Se non è la prima operazione del job, inizialmente è mascherata
                if op_id > 0:
                    mask[b, i] = True
        
        return mask
    
    def apply_masks(self, scores, M_sched, M_mask):
        """
        Equation (10) dal paper:
        mask(u_t_p|o'_0,...,o'_{t-1}) = {-∞ if M_sched_kp OR M_mask_kp, u_t_p otherwise}
        """
        combined_mask = M_sched | M_mask
        masked_scores = scores.masked_fill(combined_mask, float('-inf'))
        return masked_scores
    
    def update_scheduled_mask(self, M_sched, chosen_indices):
        """
        Definition 2: M_sched_kp diventa true quando l'operazione p è schedulata
        """
        batch_size = chosen_indices.size(0)
        new_mask = M_sched.clone()
        
        for b in range(batch_size):
            chosen_idx = chosen_indices[b].item()
            new_mask[b, chosen_idx] = True
        
        return new_mask
    
    def update_precedence_mask(self, M_mask, chosen_indices, S_seq):
        """
        Dal paper: M_sched_kp ← true, M_mask_kp+1 ← false
        Quando schedulo un'operazione, sblocco la prossima del stesso job
        """
        batch_size = chosen_indices.size(0)
        new_mask = M_mask.clone()
        
        for b in range(batch_size):
            chosen_idx = chosen_indices[b].item()
            
            # Operazione appena schedulata
            job_id = int(S_seq[b, chosen_idx, 0])
            op_id = int(S_seq[b, chosen_idx, 1])
            
            # Sblocca la prossima operazione dello stesso job
            for i in range(S_seq.size(1)):
                if (int(S_seq[b, i, 0]) == job_id and 
                    int(S_seq[b, i, 1]) == op_id + 1):
                    new_mask[b, i] = False
                    break
        
        return new_mask
'''