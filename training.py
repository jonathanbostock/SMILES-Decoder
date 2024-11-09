### Jonathan Bostock 2024-11-09
import torch

from utils import SMILESDecoder, SMILESTokenizer, get_dataloader, device

def main():

    model = SMILESDecoder(
        vocab_size=512,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    )

    dataloader = get_dataloader(
        csv_path="data/allmolgen_pretrain_data_train.csv",
        tokenizer=SMILESTokenizer(),
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            split_positions = batch['split_position'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(input_ids, split_positions, attention_mask)
            
            # Calculate loss (shift logits/labels for next-token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()