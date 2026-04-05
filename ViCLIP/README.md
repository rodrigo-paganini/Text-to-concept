---
license: mit
extra_gated_prompt: "You agree to not use the model to conduct experiments that cause harm to human subjects."
extra_gated_fields:
 Name: text
 Company/Organization: text
 Country: text
 E-Mail: text
---
**ViCLIP: a simple video CLIP for transferrable video-text representation**


Built upon <a href="https://github.com/openai/CLIP">CLIP</a>, we make a simple video-text pretraining baseline ViCLIP. It consists of a video encoder (ViT) and a text encoder, as given below. Both modules are initialized from the corresponding CLIP components. We update the native attention in the video encoder to spatiotemporal attention while maintaining other design elements. For efficient learning, we apply masking to videos in pre-training.

## Model Description

- **Repository:** [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)
- **Paper:** [2307.06942](https://arxiv.org/abs/2307.06942)
- **Point of Contact:** mailto:[InternVideo Group](gvx-sh@pjlab.org.cn)


## Data & Model Zoo

<details>
<summary> Pretrained Data & Model </summary>
<br>
<div>

|      Model      |   Training Data   |                                               Descriptions                                                |
| :-----------------: | :----------------------: | :---------------------------------------------------------------------------------------------------: |
| ViCLIP-L-14 \[[ckpt](https://huggingface.co/OpenGVLab/ViCLIP)\] | InternVid-10M-FLT \[[HuggingFace](https://huggingface.co/datasets/OpenGVLab/InternVid)\] |    |
</div>
</details>

## Citation

If you find this work useful for your research, please consider citing InternVid. Your acknowledgement would greatly help us in continuing to contribute resources to the research community.

```
@article{wang2023internvid,
  title={InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation},
  author={Wang, Yi and He, Yinan and Li, Yizhuo and Li, Kunchang and Yu, Jiashuo and Ma, Xin and Chen, Xinyuan and Wang, Yaohui and Luo, Ping and Liu, Ziwei and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2307.06942},
  year={2023}
}

@article{wang2022internvideo,
  title={InternVideo: General Video Foundation Models via Generative and Discriminative Learning},
  author={Wang, Yi and Li, Kunchang and Li, Yizhuo and He, Yinan and Huang, Bingkun and Zhao, Zhiyu and Zhang, Hongjie and Xu, Jilan and Liu, Yi and Wang, Zun and Xing, Sen and Chen, Guo and Pan, Junting and Yu, Jiashuo and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2212.03191},
  year={2022}
}
```
