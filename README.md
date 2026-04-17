# Text-to-Concept for Video inputs

This repository extends the [original work](https://github.com/k1rezaei/Text-to-concept) to extend their work to video inputs, while maintaining the original image input capabilities. With respect to the original, these are our main contributions:

- [TextToConcept.py](TextToConcept.py): was refactored to add video support, without removing the original image support. ClipZeroShot was decoupled into ClipZeroShotForImages and ClipZeroShotFor videos to handle both input types, and the vision model is now instantiable from outside the TextToConcept model. In addition, cache of input data names and labels are stored together with representations for fast and reliable reproduction.

- [train_video_aligner.py](train_video_aligner.py): was created to decouple video alignment and representation caching from the notebook experiments for fast experimentation.

- [video_utils.py](video_utils.py): was created to add utilities for video processing, video dataset creation and video loading.

- Notebooks [xai_example_zeroshot_video.ipynb](xai_example_zeroshot_video.ipynb), [xai_example_search_concept_logic_video.ipynb](xai_example_search_concept_logic_video.ipynb) and [xai_concept_botteleneck_video.ipynb](xai_concept_botteleneck_video.ipynb) were created to perform and document our experiments.

Work by E. Cabalé, H. Naranjo and R. Paganini for the XAI course of Gianni Franchi and Mathieu Fontaine, for the MVA master's at ENS Paris-Saclay.


_Original README:_
# Text-To-Concept (and Back) via Cross-Model Alignment

This repository provides the PyTorch implementation of the following paper:
+ [Text-To-Concept (and Back) via Cross-Model Alignment](https://arxiv.org/abs/2305.06386)


## Paper
**Text-To-Concept (and Back) via Cross-Model Alignment**  
##### Mazda Moayeri*, Keivan Rezaei*, Maziar Sanjabi, and Soheil Feizi  
International Conference on Machine Learning (**_ICML_**), 2023.  
open access: [arXiv](https://arxiv.org/abs/2305.06386)

Please see [guideline.md](guideline.md) for technical details.
two examples of using `TextToConcept` framework can be found in [example_search_concept_logic notebook](example_search_concept_logic.ipynb), [example_zeroshot notebook](example_zeroshot.ipynb), and [example drift notebook](example_drift.ipynb).

## Citation 
If you find our work useful, please cite: 
```
@article{moayeri2023text,
  title={Text-To-Concept (and Back) via Cross-Model Alignment},
  author={Moayeri, Mazda and Rezaei, Keivan and Sanjabi, Maziar and Feizi, Soheil},
  journal={arXiv preprint arXiv:2305.06386},
  year={2023}
}
```





