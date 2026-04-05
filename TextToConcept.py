from typing import Any, List
import torch
from torchvision import datasets, transforms, models
import torchvision
from tqdm import tqdm
import numpy as np
from LinearAligner import LinearAligner
import clip
import scipy
import os
import ViCLIP

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORMATION = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])
CLIP_IMAGENET_TRANSFORMATION = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

class ClipZeroShotForImages(torch.nn.Module):
    def __init__(self, mtype):
        """Loads the CLIP model for zero-shot classification.

        Args:
            mtype (str): model type
        """
        super(ClipZeroShotForImages, self).__init__()
        self.clip_model, self.clip_preprocess = clip.load(mtype)
        self.to_pil = transforms.ToPILImage()
        self.mtype = mtype
        self.has_normalizer = False
        
    def forward_features(self, img):
        image_features = self.clip_model.encode_image(img)
        return image_features
    
    def encode_text(self, tokens):
        return self.clip_model.encode_text(tokens)

    def tokenize(self, texts: List[str]):
        return self.clip_model.tokenize(texts)


class ClipZeroShotForVideos(torch.nn.Module):
    def __init__(self, mtype):
        """Loads the CLIP model for zero-shot classification.

        Args:
            mtype (str): model type
        """
        super(ClipZeroShotForVideos, self).__init__()
        self.tokenizer = ViCLIP.SimpleTokenizer()
        self.clip_model = ViCLIP.ViCLIP(self.tokenizer)

        # self.clip_model, self.clip_preprocess = clip.load(mtype)
        self.to_pil = transforms.ToPILImage()
        self.mtype = mtype
        self.has_normalizer = False

    def forward_features(self, vid):
        video_feats = self.clip_model.get_vid_features(vid)
        return video_feats

    def encode_text(self, tokens):
        return self.clip_model.encode_text(tokens)

    def tokenize(self, texts: List[str]):
        return self.clip_model.tokenize(texts)


class ZeroShotClassifier:
    """
    Zero-shot classifier using concept weights and trained aligner.
    """
    def __init__(self, model, aligner: LinearAligner, zeroshot_weights: torch.Tensor):
        """
        Args:
            model (nn.Module): The (vision) model for forward feature extraction.
            aligner (LinearAligner): The linear aligner for representation alignment.
            zeroshot_weights (torch.Tensor): The zero-shot weights for classification.
        """
        self.model = model
        self.aligner = aligner
        self.zeroshot_weights = zeroshot_weights.float()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    # this functions returns logits.
    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            reps = self.model.forward_features(x.to(self.device)).flatten(1)
            aligned_reps = self.aligner.get_aligned_representation(reps)
            aligned_reps /= aligned_reps.norm(dim=-1, keepdim=True)
            return aligned_reps @ self.zeroshot_weights.T
        
class TextToConcept:
    # model.forward_features(), model.get_normalizer() should be implemented.
    def __init__(self, model, model_name, input_type = "images") -> None:
        """
        Args:
            model: Vision model with methods/attributes:
                - forward_features(x): returns feature representations for input x.
                - get_normalizer(x): returns normalized input x (if model has normalizer).
                - has_normalizer: boolean indicating if the model has a normalizer.

            model_name (str): Name of the model (for saving reps).

            input_type (Literal['images', 'video']): type of input data.
        """
        self.model = model
        self.model_name = model_name
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_type = input_type
        if self.input_type.lower() == "image":
            self.clip_model = ClipZeroShotForImages('ViT-B/16')
        elif self.input_type.lower() == "video":
            self.clip_model = ClipZeroShotForVideos('ViT-B/16')
        else:
            raise ValueError(f"Input types must be one of: ['image', 'video']. Got: {input_type}")

        self.model.eval().to(self.device)
        self.clip_model.eval().to(self.device)
        self.saved_dsets = {}
    
        
    def save_reps(self, path_to_model, path_to_clip_model):
        """Save the representations to disk."""
        print(f'Saving representations')
        np.save(path_to_model, self.reps_model)
        np.save(path_to_clip_model, self.reps_clip)    
    
    
    def load_reps(self, path_to_model, path_to_clip_model):
        print(f'Loading representations ...')
        self.reps_model = np.load(path_to_model)
        self.reps_clip = np.load(path_to_clip_model)
    
    
    def load_linear_aligner(self, path_to_load):
        self.linear_aligner = LinearAligner()
        self.linear_aligner.load_W(path_to_load)
    
    
    def save_linear_aligner(self, path_to_save):
        self.linear_aligner.save_W(path_to_save)
      
        
    def train_linear_aligner(self, D, save_reps=False, load_reps=False, path_to_model=None, path_to_clip_model=None, epochs=5):
        """Train the linear aligner."""
        if load_reps:
            self.load_reps(path_to_model, path_to_clip_model)
        else:
            print(f'Obtaining representations ...')
            self.reps_model = self.obtain_ftrs(self.model, D)
            self.reps_clip = self.obtain_ftrs(self.clip_model, D)

        if save_reps:
            self.save_reps(path_to_model, path_to_clip_model)
            
        self.linear_aligner = LinearAligner()
        self.linear_aligner.train(self.reps_model, self.reps_clip, epochs=epochs, target_variance=4.5,)
        
        
    def get_zeroshot_weights(self, classes, prompts):
        """Gets prompt features through clip."""
        zeroshot_weights = []
        for c in classes:
            tokens = self.clip_model.tokenize([prompt.format(c) for prompt in prompts])
            c_vecs = self.clip_model.encode_text(tokens.to(self.device))
            c_vec = c_vecs.mean(0)
            c_vec /= c_vec.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(c_vec)
        
        return torch.stack(zeroshot_weights)
    
    
    def get_zero_shot_classifier(self, classes, prompts=['a photo of {}.']):
        """Instances Zero-Shot Classifier with the set of given prompts."""
        return ZeroShotClassifier(self.model, self.linear_aligner, self.get_zeroshot_weights(classes, prompts))
    
    
    def search(self, dset, dset_name, prompts=['a photo of a dog']):    
        """
        Searches in a dataset for elements that are similar to a prompt.
        If provided with multiple prompts, feature representations are averaged.
        """
        tokens = self.clip_model.tokenize(prompts)
        vecs = self.clip_model.encode_text(tokens.to(self.device))
        vec = vecs.detach().mean(0).float().unsqueeze(0)
        vec /= vec.norm(dim=-1, keepdim=True)
        sims = self.get_similarity(dset, dset_name, self.model.has_normalizer, vec)[:, 0]
        return np.argsort(-1 * sims), sims
    
    
    def search_with_encoded_concepts(self, dset, dset_name, vec):
        """
        Searches in a dataset for elements that are similar to a text embedding (vec).
        """
        sims = self.get_similarity(dset, dset_name, self.model.has_normalizer, vec.to(self.device))[:, 0]
        return np.argsort(-1 * sims), sims


    def get_similarity(self, dset, dset_name, do_normalization, vecs: torch.Tensor):
        """
        Gets the similarity each dataset element has with a mean text representations (vecs).
        Outputs a numpy array of shape: [N_batches, batch_size, 1]  TODO verify

        TODO list append efficiency could be improved.
        """
        reps, labels = self.get_dataset_reps(dset, dset_name, do_normalization)
        N = reps.shape[0]
        batch_size = 100
        
        all_sims = []
        with torch.no_grad():    
            for i in range(0, N, batch_size): 
                aligned_reps = self.linear_aligner.get_aligned_representation(torch.from_numpy(reps[i: i+batch_size]).to(self.device))
                aligned_reps /= aligned_reps.norm(dim=-1, keepdim=True)
                sims = aligned_reps @ vecs.T  # 
                sims = sims.detach().cpu().numpy()
                all_sims.append(sims)
            
        return np.vstack(all_sims)


    def get_dataset_reps(self, dset, dset_name, do_normalization):
        """Gets a dataset (vision) model representations from saved run or reruns them."""
        if dset_name in self.saved_dsets:
            path_to_reps, path_to_labels = self.saved_dsets[dset_name]
            return np.load(path_to_reps), np.load(path_to_labels)
        
        loader = torch.utils.data.DataLoader(dset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True) 
        all_reps, all_labels = [], []
        with torch.no_grad():
            for data in tqdm(loader):
                imgs, labels = data[0], data[1]
                if do_normalization:
                    imgs = self.model.get_normalizer(imgs).to(self.device)
                else:
                    imgs = imgs.to(self.device)
                
                reps = self.model.forward_features(imgs).flatten(1)
                
                all_reps.append(reps.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
        
        
        all_reps = np.vstack(all_reps)
        all_labels = np.hstack(all_labels)
        
        self.saved_dsets[dset_name] = (self._get_path_to_reps(dset_name), self._get_path_to_labels(dset_name), )
        os.makedirs(f'datasets/{self.model_name}/', exist_ok=True)
        
        np.save(self._get_path_to_reps(dset_name), all_reps)
        np.save(self._get_path_to_labels(dset_name), all_labels)
        
        return all_reps, all_labels
        
        
    def _get_path_to_labels(self, dset_name):
        return f'datasets/{self.model_name}/{dset_name}_labels.npy'
    
    def _get_path_to_reps(self, dset_name):
        return f'datasets/{self.model_name}/{dset_name}_reps.npy'

    
    def encode_text(self, list_of_prompts):
        """Returns text embeddings for a list of prompts."""
        all_vecs = []
        batch_size = 64
        with torch.no_grad():
            for prompts in list_of_prompts:
                tokens = self.clip_model.tokenize(prompts)
                M = tokens.shape[0]
                curr_vecs = []
                
                for i in range(0, M, batch_size):
                    vecs = self.clip_model.encode_text(tokens[i: i + batch_size].to(self.device)).detach().cpu()
                    curr_vecs.append(vecs)
                
                vecs = torch.vstack(curr_vecs)
                
                vec = vecs.mean(0).float()
                vec /= vec.norm(dim=-1, keepdim=True)
                all_vecs.append(vec)
        
        return torch.stack(all_vecs).to(self.device)
        

    def detect_drift(self, dset1, dset_name1, dset2, dset_name2, prompts):
        """Detect drift between two datasets by shift in similarity distributions (t-test)."""
        vecs = self.encode_text([prompts])
        sims1 = self.get_similarity(dset1, dset_name1, self.model.has_normalizer, vecs)
        sims2 = self.get_similarity(dset2, dset_name2, self.model.has_normalizer, vecs)
        
        stats, p_value = scipy.stats.ttest_ind(sims1[:, 0], sims2[:, 0])
            
        return [stats, p_value], sims1, sims2
        
        
    def concept_logic(self, dset, dset_name, list_of_prompts, signs, scales):
        """
        Retrieves dataset elements that satisfy a logical combination of concepts (prompts).

        Rule: threshold = sim_mean + sign * scale * sim_std

        Args:
            dset: dataset to search in.
            dset_name: name of the dataset (for loading/saving reps).
            list_of_prompts: list of prompts, each representing a concept.
            signs: list of signs (1 or -1) indicating whether to keep elements above or below the threshold for each concept.
            scales: list of scales to apply to the standard deviation when calculating thresholds for each concept.
        """
        vecs = self.encode_text(list_of_prompts)
        sims = self.get_similarity(dset, dset_name, self.model.has_normalizer, vecs)
        means = np.mean(sims, axis=0)
        stds = np.std(sims, axis=0)
        
        ths = means + np.array(signs) * np.array(scales) * stds
        retrieved = np.arange(sims.shape[0])
        for j in range(len(signs)):
            if retrieved.shape[0] == 0:
                break
            
            sim_to_concept = sims[retrieved, j]
            if signs[j] == -1:
                retrieved = retrieved[np.where(sim_to_concept < ths[j])[0]]
            else:
                retrieved = retrieved[np.where(sim_to_concept > ths[j])[0]]
        
        return retrieved, sims
        
        
    def obtain_ftrs(self, model, dset):
        """Instance a dataloader and obtain feature representations for the given model."""
        loader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True) 
        return self.obtain_reps_given_loader(model, loader)
    
    
    def obtain_reps_given_loader(self, model, loader):
        """
        Run model forward_features on the given dataloader and obtain feature representations.
        
        TODO: we can try to torch this us, use .pth instead of .npy
        """
        all_reps = []
        for imgs, _ in tqdm(loader):
            if model.has_normalizer:
                imgs = model.get_normalizer(imgs)
            
            imgs = imgs.to(self.device)
                
            reps = model.forward_features(imgs).flatten(1)
            reps = [x.detach().cpu().numpy() for x in reps]
            
            all_reps.extend(reps)
            
        all_reps = np.stack(all_reps)
        return all_reps


