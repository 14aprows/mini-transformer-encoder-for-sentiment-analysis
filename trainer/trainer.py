import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.best_acc = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        loop = tqdm(self.train_loader, leave=True)
        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / len(self.train_loader))

        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                preds = torch.argmax(out, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)
        
        return 100 * correct / total
    
    def fit(self, epochs=5):
        for epoch in range(1, epochs+1):
            train_loss = self.train_epoch()
            val_acc = self.validate()

            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print('Model saved!')