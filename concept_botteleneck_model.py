import numpy as np
import torch
from tqdm import tqdm


class CBMTrainer:
    def __init__(self, cbm_model):
        self.cbm_model = cbm_model
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

        self.optimizer = torch.optim.Adam(self.cbm_model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def train(self, concept_scores, labels, batch_size=16, epochs=20):
        """
        Trains the concept bottleneck model using the provided concept scores and labels.
        """

        print("Training concept bottleneck ...")
        print(f"Linear projection: ({concept_scores.shape[1]}) --> ({len(self.cbm_model.output_classes)}).")

        # Tensors
        tensor_X = torch.from_numpy(concept_scores).float()
        tensor_y = torch.from_numpy(labels)
        if tensor_y.ndim > 1:
            tensor_y = tensor_y.argmax(dim=1)
        tensor_y = tensor_y.long()
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.model = self.cbm_model.bottleneck
        self.model.to(self.device)

        with torch.no_grad():
            init_loss = self.criterion(self.model(tensor_X.to(self.device)), tensor_y.to(self.device))
            print(f'Initial loss: {init_loss:.3f}')
        curr_loss = init_loss

        self.init_result = init_loss
        self.model.train()

        for epoch in range(epochs):
            e_loss, num_of_batches = 0, 0

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                num_of_batches += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                e_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            e_loss /= num_of_batches
            
            print(f'Epoch number, loss: {epoch}, {e_loss:.3f}')

            with torch.no_grad():
                curr_loss = self.criterion(self.model(tensor_X.to(self.device)), tensor_y.to(self.device))
                print(f'Full-batch loss: {curr_loss:.3f}')

            self.scheduler.step()

        print(f'Final loss = {curr_loss:.3f}')

    def eval(self, concept_scores, labels, batch_size=16):
        """
        Evaluates the concept bottleneck model on the provided concept scores and labels.
        """
        tensor_X = torch.from_numpy(concept_scores).float()
        tensor_y = torch.from_numpy(labels)
        if tensor_y.ndim > 1:
            tensor_y = tensor_y.argmax(dim=1)
        tensor_y = tensor_y.long()

        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        self.model = self.cbm_model.bottleneck
        self.model.to(self.device)
        self.model.eval()

        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                batch_size_curr = targets.shape[0]
                total_loss += loss.item() * batch_size_curr
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_samples += batch_size_curr

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f'Eval loss: {avg_loss:.3f}')
        print(f'Eval acc: {accuracy:.3f}')
        return avg_loss, accuracy
