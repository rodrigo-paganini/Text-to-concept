"""
Classes to define and train a Concept Bottleneck Model for video classification, using the TextToConcept framework to obtain concept scores from video features.

Authors: E. Cabalé, H. Naranjo, R. Paganini
"""
import numpy as np
import torch
from tqdm import tqdm

class ConceptBottleneckModel(torch.nn.Module):
    def __init__(self, concept_list, output_classes, mtype):
        """Loads the CLIP model for zero-shot classification.

        Args:
            mtype (str): model type
        """
        super(ConceptBottleneckModel, self).__init__()
        self.output_classes = output_classes
        self.concept_list = concept_list
        self.concept_embeddings = None
        self.bottleneck = torch.nn.Linear(len(concept_list), len(output_classes))
        self.mtype = mtype
        self.has_normalizer = False

    def embed_concepts(self, clip_model, templates=None):
        if templates is not None:
            concept_list = [[t.format(c) for t in templates] for c in self.concept_list]
        else:
            concept_list = [[c] for c in self.concept_list]

        tokens = [clip_model.tokenize(c) for c in concept_list]
        device = next(clip_model.parameters()).device
        concept_embeddings = [clip_model.encode_text(token.to(device)).detach().float() for token in tokens]
        self.concept_embeddings = torch.stack(concept_embeddings).mean(dim=1)
        return self.concept_embeddings

    def get_concept_concept_scores(self, vision_features):
        if self.concept_embeddings is None:
            raise ValueError("Concept embeddings not computed. Call embed_concepts(clip_model) first.")

        vision_features = vision_features / np.linalg.norm(vision_features, axis=-1, keepdims=True)
        concept_embeddings = self.concept_embeddings
        concept_embeddings = concept_embeddings.numpy() / np.linalg.norm(concept_embeddings.numpy(), axis=-1, keepdims=True)
        scores = vision_features @ concept_embeddings.T
        return scores

    def predict(self, vision_features):  # TODO make sure this doesn't backprop to the model
        concept_scores = self.get_concept_concept_scores(vision_features)
        output_scores = self.bottleneck(concept_scores)
        return output_scores

    def get_concept_scores_from_loader(self, loader, vision_model):
        all_concept_scores, all_video_names = [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                videos, _, video_names = batch
                videos = videos.to(next(self.parameters()).device)
                if vision_model.has_normalizer:
                    videos = vision_model.get_normalizer(videos)
                vision_features = vision_model.forward_features(videos)
                concept_scores = self.get_concept_concept_scores(vision_features)
                all_concept_scores.append(concept_scores.cpu())
                all_video_names.extend(video_names)
        return torch.cat(all_concept_scores, dim=0).numpy(), all_video_names


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

    def eval(self, concept_scores, labels, batch_size=16, return_preds=False):
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
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                preds = outputs.argmax(dim=1)

                batch_size_curr = targets.shape[0]
                total_loss += loss.item() * batch_size_curr
                total_correct += (preds == targets).sum().item()
                total_samples += batch_size_curr

                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        print(f'Eval loss: {avg_loss:.3f}')
        print(f'Eval acc: {accuracy:.3f}')

        if return_preds:
            return avg_loss, accuracy, all_labels, all_preds
        return avg_loss, accuracy
