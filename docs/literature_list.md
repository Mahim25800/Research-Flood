# PX1 paper: literature list

91 sources, grouped by topic. Every entry was checked against a live source (publisher, proceedings, arXiv or an institutional repository) on 9 Nov 2026. Each entry says what the work is and where it fits in our paper. BibTeX keys match `references.bib`; the same table is in `references.csv`.

## Where our paper sits (what the search found)

- **Conformal prediction for flood maps** is rare. Konidakis et al. (IGARSS 2025) build pixel-level prediction sets for optical Sentinel-2 flood maps without domain shift; Li, Matgen and Chini (TGRS 2026) use conformal risk control on SAR flood maps for out-of-distribution abstention and set-valued predictions. Neither bounds the miss rate across flood events or combines the guarantee with test-time adaptation. General conformal segmentation (Mossina et al. 2024) treats images as units. We calibrate a miss-rate (FNR) guarantee with *flood events* as the exchangeable unit (cf. Dunn et al. 2023) across three SAR datasets.
- **Test-time adaptation for SAR flood mapping already exists**: Chen et al. (2026, JGR: Machine Learning and Computation) adapt SAR inundation models to each new coastal flood with pseudo-labels and EMA updates. Our self-training baseline follows that recipe. What we found no precedent for is checking the pseudo-labels against radar and terrain physics, combining this with AdaBN, and attaching a flood-level miss-rate guarantee. TTA also exists in general remote sensing (Wang et al. 2024; Liang et al. 2024), and events-as-domains generalisation for floods (Turkmenli et al. 2025).
- **HAND and terrain** are used as post-hoc masks or exclusion maps (Twele et al. 2016; Zhao et al. 2021; Misra et al. 2025), as Bayesian priors (Tupas et al. 2023) or as model inputs (Tian et al. 2026). We use radar physics plus HAND/slope to correct pseudo-labels *during* label-free adaptation.
- **TTA can fail under real shift** (Niu et al. 2023). Our MMFlood result (TENT and AdaBN hurt under the guarantee; the physics check prevents it) adds evidence from Earth observation.
- These are search findings, not proof of absence. Re-check the newest 2025–2026 papers just before submission.

## Contents

- A. Flood datasets and benchmarks (8)
- B. Classical SAR flood mapping and operational systems (12)
- C. Terrain priors: HAND, exclusion maps and topographic constraints (8)
- D. Deep learning for SAR / satellite flood mapping (8)
- E. Label-scarce, self-supervised and domain-shift work on flood mapping (5)
- F. Physics-guided / knowledge-guided machine learning (3)
- G. Test-time adaptation and batch-norm adaptation (10)
- H. Self-training, pseudo-labels and unsupervised domain adaptation (7)
- I. Conformal prediction and distribution-free risk control (12)
- J. Uncertainty and calibration in deep learning and flood mapping (4)
- K. Backbones, training and libraries (6)
- L. Statistics and research practice (4)
- M. Data sources and application context (4)

## A. Flood datasets and benchmarks

**1. Bonafilia et al. (2020).** Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. [link](https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html)  
What it is: Sentinel-1/2 chips from 11 flood events with 446 hand-labelled 512x512 chips plus weak labels; the standard benchmark for SAR flood segmentation.  
How we use it: Our training data (source domain) and the leave-one-event-out (LOEO) evaluation.  
Key: `bonafilia2020sen1floods11`

**2. Bountos et al. (2024).** Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood mapping. *Advances in Neural Information Processing Systems 37 (NeurIPS), Datasets and Benchmarks Track*. [link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/43612b0662cb6a4986edf859fd6ebafe-Abstract.html)  
What it is: Large multi-temporal Sentinel-1 GRD/SLC flood dataset from 43 events with expert flood annotations and DEM.  
How we use it: Cross-dataset test set (26 events); development set for choosing the primary method (pre-registered rule).  
Key: `bountos2024kurosiwo`

**3. Montello et al. (2022).** MMFlood: A Multimodal Dataset for Flood Delineation From Satellite Imagery. *IEEE Access*, vol. 10, pp. 96774–96787. [doi:10.1109/ACCESS.2022.3205419](https://doi.org/10.1109/ACCESS.2022.3205419)  
What it is: Sentinel-1 images, DEM and hydrography for 95 Copernicus EMS flood activations in 42 countries (2014-2021), with rasterised EMS flood masks; data on Zenodo (record 6534637).  
How we use it: Pre-registered final (untouched) test set: 30 activations after overlap exclusion.  
Key: `montello2022mmflood`

**4. Mateo-Garcia et al. (2021).** Towards global flood mapping onboard low cost satellites with machine learning. *Scientific Reports*, vol. 11, art. 7249. [doi:10.1038/s41598-021-86650-z](https://doi.org/10.1038/s41598-021-86650-z)  
What it is: WorldFloods dataset (Sentinel-2, many CEMS events) and lightweight CNNs for on-board flood mapping.  
How we use it: Related dataset; shows generalisation to new floods as the core difficulty.  
Key: `mateogarcia2021worldfloods`

**5. Zhao et al. (2024).** UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. [link](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.html)  
What it is: SLC-based benchmark separating open-area and urban floods; shows urban flooding is hard for intensity-only models.  
How we use it: Limitations: urban floods are a known failure mode of dark-water rules.  
Key: `zhao2024urbansarfloods`

**6. Hänsch et al. (2022).** SpaceNet 8 - The Detection of Flooded Roads and Buildings. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. [link](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Hansch_SpaceNet_8_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.html)  
What it is: VHR optical benchmark for flooded roads and buildings.  
How we use it: Related-work breadth (flood benchmarks beyond Sentinel-1).  
Key: `hansch2022spacenet8`

**7. Rambour et al. (2020).** SEN12-FLOOD: a SAR and Multispectral Dataset for Flood Detection. *IEEE Dataport*. [link](https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection)  
What it is: Sentinel-1/2 time series labelled for flood presence (image-level).  
How we use it: Related dataset; image-level rather than pixel-level labels.  
Key: `rambour2020sen12flood`

**8. Drakonakis et al. (2022).** OmbriaNet—Supervised Flood Mapping via Convolutional Neural Networks Using Multitemporal Sentinel-1 and Sentinel-2 Data Fusion. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 15. [link](https://doaj.org/article/1a0f358ab0624724a4120b8fcaca14ec)  
What it is: OMBRIA bitemporal flood dataset (EMS events) and the OmbriaNet change-detection CNN.  
How we use it: Related dataset and bitemporal baseline.  
Key: `drakonakis2022ombrianet`

## B. Classical SAR flood mapping and operational systems

**9. Otsu (1979).** A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66. [doi:10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)  
What it is: Histogram threshold maximising between-class variance.  
How we use it: Per-flood adaptive dark-water threshold in our physics rule; Otsu-only baseline.  
Key: `otsu1979threshold`

**10. Martinis et al. (2009).** Towards operational near real-time flood detection using a split-based automatic thresholding procedure on high resolution TerraSAR-X data. *Natural Hazards and Earth System Sciences*, vol. 9, no. 2, pp. 303–314. [doi:10.5194/nhess-9-303-2009](https://doi.org/10.5194/nhess-9-303-2009)  
What it is: Tile-based (split) automatic thresholding for SAR flood detection, the basis of later operational chains.  
How we use it: Classical thresholding lineage; motivates per-scene thresholds instead of fixed dB cut-offs.  
Key: `martinis2009nrt`

**11. Twele et al. (2016).** Sentinel-1-based flood mapping: a fully automated processing chain. *International Journal of Remote Sensing*, vol. 37, no. 13, pp. 2990–3004. [doi:10.1080/01431161.2016.1192304](https://doi.org/10.1080/01431161.2016.1192304)  
What it is: Operational DLR chain: automatic thresholding, fuzzy-logic refinement with HAND/slope, region growing.  
How we use it: Precedent for combining radar thresholds with HAND and slope; we move this logic inside adaptation.  
Key: `twele2016s1chain`

**12. Chini et al. (2017).** A Hierarchical Split-Based Approach for Parametric Thresholding of SAR Images: Flood Inundation as a Test Case. *IEEE Transactions on Geoscience and Remote Sensing*, vol. 55, no. 12, pp. 6975–6988. [doi:10.1109/TGRS.2017.2737664](https://doi.org/10.1109/TGRS.2017.2737664)  
What it is: Hierarchical tiling to find bimodal tiles and fit parametric water/non-water distributions.  
How we use it: Classical adaptive thresholding comparison point.  
Key: `chini2017hsbt`

**13. Giustarini et al. (2013).** A Change Detection Approach to Flood Mapping in Urban Areas Using TerraSAR-X. *IEEE Transactions on Geoscience and Remote Sensing*, vol. 51, no. 4, pp. 2417–2430. [doi:10.1109/TGRS.2012.2210901](https://doi.org/10.1109/TGRS.2012.2210901)  
What it is: Change detection with a pre-flood reference image, including urban areas.  
How we use it: Change detection as an alternative to single-date mapping; our setting is single post-flood image.  
Key: `giustarini2013change`

**14. Giustarini et al. (2016).** Probabilistic Flood Mapping Using Synthetic Aperture Radar Data. *IEEE Transactions on Geoscience and Remote Sensing*, vol. 54, no. 12, pp. 6958–6969. [doi:10.1109/TGRS.2016.2592951](https://doi.org/10.1109/TGRS.2016.2592951)  
What it is: Bayesian per-pixel flood probability from SAR backscatter distributions.  
How we use it: Prior work on probabilistic flood maps; ours adds a guarantee on the miss rate.  
Key: `giustarini2016probabilistic`

**15. Landuyt et al. (2019).** Flood Mapping Based on Synthetic Aperture Radar: An Assessment of Established Approaches. *IEEE Transactions on Geoscience and Remote Sensing*, vol. 57, no. 2, pp. 722–739. [doi:10.1109/TGRS.2018.2860054](https://doi.org/10.1109/TGRS.2018.2860054)  
What it is: Systematic comparison of thresholding, change detection and classification approaches for SAR flood mapping.  
How we use it: Background on how established methods compare; motivates event-level evaluation.  
Key: `landuyt2019assessment`

**16. Clement et al. (2018).** Multi-temporal synthetic aperture radar flood mapping using change detection. *Journal of Flood Risk Management*, vol. 11, no. 2, pp. 152–168. [doi:10.1111/jfr3.12303](https://doi.org/10.1111/jfr3.12303)  
What it is: Sentinel-1 multi-temporal change detection for flood extent.  
How we use it: Change-detection background.  
Key: `clement2018multitemporal`

**17. Tsyganskaya et al. (2018).** SAR-based detection of flooded vegetation – a review of characteristics and approaches. *International Journal of Remote Sensing*, vol. 39, no. 8, pp. 2255–2293. [doi:10.1080/01431161.2017.1420938](https://doi.org/10.1080/01431161.2017.1420938)  
What it is: Review of double-bounce and other signatures of flooded vegetation in SAR.  
How we use it: Limitations: dark-water rules miss flooded vegetation (bright, not dark).  
Key: `tsyganskaya2018vegetation`

**18. Salamon et al. (2021).** The New, Systematic Global Flood Monitoring Product of the Copernicus Emergency Management Service. *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*. [link](https://dblp1.uni-trier.de/rec/conf/igarss/SalamonMRCB0MCB21.html)  
What it is: Introduces CEMS Global Flood Monitoring (GFM): every Sentinel-1 scene processed by an ensemble of three algorithms.  
How we use it: Operational context; GFM uses HAND/exclusion masks, which our method uses during adaptation.  
Key: `salamon2021gfm`

**19. Krullikowski et al. (2023).** Estimating ensemble likelihoods for the Sentinel-1 based Global Flood Monitoring product of the Copernicus Emergency Management Service. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 16. [link](https://arxiv.org/abs/2304.12488)  
What it is: Ensemble likelihood (uncertainty) layer for the GFM product.  
How we use it: Operational uncertainty; contrast with our conformal miss-rate guarantee.  
Key: `krullikowski2023gfmensemble`

**20. Misra et al. (2025).** Mapping global floods with 10 years of satellite radar data. *Nature Communications*, vol. 16, art. 5762. [doi:10.1038/s41467-025-60973-1](https://doi.org/10.1038/s41467-025-60973-1)  
What it is: Change-detection CNN on Sentinel-1 pairs run over 2014-2024; DEM slope masks and auxiliary layers remove false positives.  
How we use it: State of the art in global SAR flood mapping; uses terrain only as a post-hoc mask.  
Key: `misra2025globalfloods`

## C. Terrain priors: HAND, exclusion maps and topographic constraints

**21. Rennó et al. (2008).** HAND, a new terrain descriptor using SRTM-DEM: Mapping terra-firme rainforest environments in Amazonia. *Remote Sensing of Environment*, vol. 112, no. 9, pp. 3469–3481. [doi:10.1016/j.rse.2008.03.018](https://doi.org/10.1016/j.rse.2008.03.018)  
What it is: Introduces Height Above the Nearest Drainage (HAND).  
How we use it: Defines the HAND terrain descriptor used in our physics check (HAND < 5 m).  
Key: `renno2008hand`

**22. Nobre et al. (2011).** Height Above the Nearest Drainage – a hydrologically relevant new terrain model. *Journal of Hydrology*, vol. 404, no. 1–2, pp. 13–29. [doi:10.1016/j.jhydrol.2011.03.051](https://doi.org/10.1016/j.jhydrol.2011.03.051)  
What it is: Shows HAND relates to soil water conditions and hydrology.  
How we use it: Hydrological justification for HAND as a flood plausibility prior.  
Key: `nobre2011hand`

**23. Alaska Satellite Facility & HydroSAR team (2022).** Global 30m Height Above Nearest Drainage (HAND) derived from the Copernicus GLO-30 DEM. *Registry of Open Data on AWS*. [link](https://registry.opendata.aws/glo-30-hand/)  
What it is: Global 30 m HAND tiles computed from the Copernicus GLO-30 DEM, distributed as COGs on AWS.  
How we use it: Our HAND source for all three datasets.  
Key: `glo30hand2022`

**24. Zhao et al. (2021).** Deriving exclusion maps from C-band SAR time-series in support of floodwater mapping. *Remote Sensing of Environment*, vol. 265, art. 112668. [doi:10.1016/j.rse.2021.112668](https://doi.org/10.1016/j.rse.2021.112668)  
What it is: Exclusion maps marking where SAR cannot see floodwater (shadow, permanent low backscatter, urban), used in GFM.  
How we use it: Closest operational analogue to our 'physics check'; we use such knowledge to filter pseudo-labels rather than outputs.  
Key: `zhao2021exclusion`

**25. Tupas et al. (2023).** Improving Sentinel-1 Flood Maps Using a Topographic Index as Prior in Bayesian Inference. *Water*, vol. 15, no. 23, art. 4034. [doi:10.3390/w15234034](https://doi.org/10.3390/w15234034)  
What it is: Replaces the hard HAND exclusion mask with a HAND-based prior in the TU Wien Bayesian flood algorithm; fewer misses.  
How we use it: Soft use of HAND as a prior; supports our choice of HAND as a plausibility signal.  
Key: `tupas2023handprior`

**26. Tian et al. (2026).** DeepSAR Flood Mapper: global flood mapping on Google Earth Engine cloud platform using MLP deep learning model with Sentinel-1 SAR imagery and HAND topographic data. *GIScience & Remote Sensing*, vol. 63, no. 1, art. 2612306. [doi:10.1080/15481603.2025.2612306](https://doi.org/10.1080/15481603.2025.2612306)  
What it is: Deep model taking Sentinel-1 plus HAND as input features, deployed on GEE.  
How we use it: HAND as a model input (supervised); ours uses HAND label-free at test time.  
Key: `tian2026deepsar`

**27. Hansen et al. (2025).** Flood mapping using Sentinel-1 imagery with topographical and hydrological contextualization: Case study from Ribe, Denmark. *International Journal of Applied Earth Observation and Geoinformation*, vol. 143, art. 104816. [doi:10.1016/j.jag.2025.104816](https://doi.org/10.1016/j.jag.2025.104816)  
What it is: Local thresholding refined with terrain/drainage context and fuzzy logic.  
How we use it: Recent rule-based terrain refinement; complementary to our learning-based use.  
Key: `hansen2025ribe`

**28. Sami et al. (2024).** EvaNet: Elevation-Guided Flood Extent Mapping on Earth Imagery. *Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 1200–1208. [doi:10.24963/ijcai.2024/133](https://doi.org/10.24963/ijcai.2024/133)  
What it is: Segmentation network with an elevation-guided (gravity) loss so lower neighbours of flooded pixels are also flooded.  
How we use it: Physics/terrain constraint used in training; ours applies terrain checks label-free at test time.  
Key: `sami2024evanet`

## D. Deep learning for SAR / satellite flood mapping

**29. Bentivoglio et al. (2022).** Deep learning methods for flood mapping: a review of existing applications and future research directions. *Hydrology and Earth System Sciences*, vol. 26, no. 16, pp. 4345–4378. [doi:10.5194/hess-26-4345-2022](https://doi.org/10.5194/hess-26-4345-2022)  
What it is: Review of DL for flood inundation, susceptibility and hazard mapping; flags generalisation and uncertainty as open problems.  
How we use it: Introduction: the generalisation and uncertainty gaps we address.  
Key: `bentivoglio2022review`

**30. Lee & Li (2025).** Geospatial Artificial Intelligence for Satellite-Based Flood Extent Mapping: Concepts, Advances, and Future Perspectives. *arXiv preprint arXiv:2504.02214*. [link](https://arxiv.org/abs/2504.02214)  
What it is: Recent review of GeoAI for flood extent mapping, including uncertainty and change detection.  
How we use it: Up-to-date review for the introduction.  
Key: `lee2025geoaiflood`

**31. Nemni et al. (2020).** Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery. *Remote Sensing*, vol. 12, no. 16, art. 2532. [doi:10.3390/rs12162532](https://doi.org/10.3390/rs12162532)  
What it is: UNOSAT U-Net/XNet for operational rapid flood segmentation from Sentinel-1.  
How we use it: Operational DL precedent (UN rapid mapping).  
Key: `nemni2020unosat`

**32. Bai et al. (2021).** Enhancement of Detecting Permanent Water and Temporary Water in Flood Disasters by Fusing Sentinel-1 and Sentinel-2 Imagery Using Deep Learning Algorithms: Demonstration of Sen1Floods11 Benchmark Datasets. *Remote Sensing*, vol. 13, no. 11, art. 2220. [doi:10.3390/rs13112220](https://doi.org/10.3390/rs13112220)  
What it is: BASNet-based S1/S2 fusion on Sen1Floods11.  
How we use it: Sen1Floods11 baselines.  
Key: `bai2021basnet`

**33. Konapala et al. (2021).** Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning. *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 180, pp. 163–173. [doi:10.1016/j.isprsjprs.2021.08.016](https://doi.org/10.1016/j.isprsjprs.2021.08.016)  
What it is: Compares S1, S2 and combinations (with elevation) for U-Net flood mapping on Sen1Floods11.  
How we use it: Sen1Floods11 baselines; effect of adding elevation inputs (our V1 variant).  
Key: `konapala2021s1s2`

**34. Helleis et al. (2022).** Sentinel-1-Based Water and Flood Mapping: Benchmarking Convolutional Neural Networks Against an Operational Rule-Based Processing Chain. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 15. [link](https://elib.dlr.de/187304)  
What it is: CNNs vs DLR's operational rule-based chain on Sentinel-1 water/flood mapping.  
How we use it: Shows CNNs vs thresholds trade-offs; supports combining both.  
Key: `helleis2022benchmark`

**35. Jakubik et al. (2023).** Foundation Models for Generalist Geospatial Artificial Intelligence. *arXiv preprint arXiv:2310.18660*. [link](https://arxiv.org/abs/2310.18660)  
What it is: Prithvi geospatial foundation model; fine-tuned on Sen1Floods11 among other tasks.  
How we use it: Stronger-backbone context; future work with foundation models.  
Key: `jakubik2023prithvi`

**36. Garg et al. (2023).** Cross Modal Distillation for Flood Extent Mapping. *arXiv preprint arXiv:2302.08180*. [link](https://arxiv.org/abs/2302.08180)  
What it is: Optical teacher distils into a SAR student using unlabelled data; +6.5% IoU on Sen1Floods11.  
How we use it: Label-efficient flood mapping; ours needs no target labels or optical data.  
Key: `garg2023crossmodal`

## E. Label-scarce, self-supervised and domain-shift work on flood mapping

**37. Akiva et al. (2021).** H2O-Net: Self-Supervised Flood Segmentation via Adversarial Domain Adaptation and Label Refinement. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*. [link](https://openaccess.thecvf.com/content/WACV2021/papers/Akiva_H2O-Net_Self-Supervised_Flood_Segmentation_via_Adversarial_Domain_Adaptation_and_Label_WACV_2021_paper.pdf)  
What it is: Adversarial domain adaptation plus label refinement for flood segmentation without pixel labels.  
How we use it: Closest prior on label refinement for floods; ours is test-time, per event, and physics-checked.  
Key: `akiva2021h2onet`

**38. Paul & Ganju (2021).** Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning. *NeurIPS 2021 Workshop on Tackling Climate Change with Machine Learning*. [link](https://www.climatechange.ai/papers/neurips2021/1)  
What it is: Iterative pseudo-labelling with U-Net ensembles and CRF on the ETCI 2021 data.  
How we use it: Pseudo-labelling for SAR floods (offline, semi-supervised) vs our per-event test-time version.  
Key: `paul2021semisup`

**39. Yadav et al. (2022).** Unsupervised Flood Detection on SAR Time Series. *arXiv preprint arXiv:2212.03675*. [link](https://arxiv.org/abs/2212.03675)  
What it is: Unsupervised probabilistic change detection on SAR time series with reconstruction and contrastive learning.  
How we use it: Label-free flood mapping alternative (needs time series; ours uses one post-event image).  
Key: `yadav2022unsupts`

**40. Turkmenli et al. (2025).** A comparative domain generalization study for SAR image-based flood segmentation. *Joint Urban Remote Sensing Event (JURSE)*. [doi:10.1109/JURSE60372.2025.11075977](https://doi.org/10.1109/JURSE60372.2025.11075977)  
What it is: Treats each Sen1Floods11 event as a source domain and compares domain-generalisation methods (~3% IoU gain).  
How we use it: Directly related: event-as-domain view of flood generalisation; ours adapts at test time instead.  
Key: `turkmenli2025dgflood`

**41. Chen et al. (2026).** Improving Cross-Event Generalization for SAR-Based Coastal Inundation Mapping via Test-Time Domain Adaptation. *Journal of Geophysical Research: Machine Learning and Computation*, vol. 3, no. 4, art. e2025JH001145. [doi:10.1029/2025JH001145](https://doi.org/10.1029/2025JH001145)  
What it is: CIM-TTDA: adapts a pretrained SAR water-segmentation model to each new coastal flood with pseudo-labels and EMA (teacher-student) updates; +0.15 mean IoU on 3 cyclone floods (8 scenes).  
How we use it: Closest prior work: test-time adaptation of SAR flood models across events. Our 'self-training' baseline follows this recipe; we add physics-checked pseudo-labels, AdaBN, a flood-level miss-rate guarantee and a 3-dataset, pre-registered evaluation.  
Key: `chen2026cimttda`

## F. Physics-guided / knowledge-guided machine learning

**42. Karpatne et al. (2017).** Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data. *IEEE Transactions on Knowledge and Data Engineering*, vol. 29, no. 10, pp. 2318–2331. [doi:10.1109/TKDE.2017.2720168](https://doi.org/10.1109/TKDE.2017.2720168)  
What it is: Framework for using scientific knowledge to constrain/guide data-driven models.  
How we use it: Positions our physics check as theory-guided learning.  
Key: `karpatne2017tgds`

**43. Willard et al. (2022).** Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems. *ACM Computing Surveys*, vol. 55, no. 4, art. 66. [doi:10.1145/3514228](https://doi.org/10.1145/3514228)  
What it is: Survey of physics-guided losses, architectures, initialisation and hybrid models.  
How we use it: Related-work taxonomy; ours is knowledge-guided pseudo-label correction (a less common category).  
Key: `willard2022integrating`

**44. Reichstein et al. (2019).** Deep learning and process understanding for data-driven Earth system science. *Nature*, vol. 566, pp. 195–204. [doi:10.1038/s41586-019-0912-1](https://doi.org/10.1038/s41586-019-0912-1)  
What it is: Argues for hybrid physical/deep-learning models in Earth science.  
How we use it: Motivation for physics-aware deep learning in Earth observation.  
Key: `reichstein2019deep`

## G. Test-time adaptation and batch-norm adaptation

**45. Wang et al. (2021).** Tent: Fully Test-Time Adaptation by Entropy Minimization. *International Conference on Learning Representations (ICLR)*. [link](https://arxiv.org/abs/2006.10726)  
What it is: Adapts batch-norm affine parameters at test time by minimising prediction entropy.  
How we use it: Baseline (TENT); H1 compares against it.  
Key: `wang2021tent`

**46. Li et al. (2018).** Adaptive Batch Normalization for practical domain adaptation. *Pattern Recognition*, vol. 80, pp. 109–117. [doi:10.1016/j.patcog.2018.03.005](https://doi.org/10.1016/j.patcog.2018.03.005)  
What it is: Re-estimates batch-norm statistics on target data (AdaBN).  
How we use it: First stage of our method and baseline (H2).  
Key: `li2018adabn`

**47. Schneider et al. (2020).** Improving robustness against common corruptions by covariate shift adaptation. *Advances in Neural Information Processing Systems 33 (NeurIPS)*. [link](https://papers.nips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html)  
What it is: Shows simple BN statistic adaptation greatly improves robustness to shifts.  
How we use it: Supports BN re-estimation as a strong, cheap baseline.  
Key: `schneider2020covariate`

**48. Sun et al. (2020).** Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. *International Conference on Machine Learning (ICML)*, PMLR 119. [link](https://arxiv.org/abs/1909.13231)  
What it is: Test-time training using a self-supervised auxiliary task.  
How we use it: TTA background; requires training-time changes, unlike ours.  
Key: `sun2020ttt`

**49. Wang et al. (2022).** Continual Test-Time Domain Adaptation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7201–7211. [link](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)  
What it is: Mean-teacher pseudo-labels with augmentation averaging and stochastic restore for continual TTA.  
How we use it: Teacher-student TTA closest in mechanics to our self-training stage.  
Key: `wang2022cotta`

**50. Niu et al. (2022).** Efficient Test-Time Model Adaptation without Forgetting. *International Conference on Machine Learning (ICML)*, PMLR 162. [link](https://proceedings.mlr.press/v162/niu22a.html)  
What it is: Filters unreliable/redundant samples and regularises to avoid forgetting during TTA.  
How we use it: Reliability filtering in TTA; ours filters with physics instead of entropy.  
Key: `niu2022eata`

**51. Niu et al. (2023).** Towards Stable Test-time Adaptation in Dynamic Wild World. *International Conference on Learning Representations (ICLR)*. [link](https://arxiv.org/abs/2302.12400)  
What it is: Shows TTA can collapse in the wild and proposes sharpness-aware reliable entropy minimisation.  
How we use it: Evidence that TTA can harm; supports our 'safe adaptation' framing.  
Key: `niu2023sar`

**52. Liang et al. (2025).** A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts. *International Journal of Computer Vision*, vol. 133, pp. 31–64. [doi:10.1007/s11263-024-02181-w](https://doi.org/10.1007/s11263-024-02181-w)  
What it is: Survey of test-time domain, batch and instance adaptation.  
How we use it: TTA taxonomy for related work.  
Key: `liang2025ttasurvey`

**53. Wang et al. (2024).** Exploring Uncertainty-Based Self-Prompt for Test-Time Adaptation Semantic Segmentation in Remote Sensing Images. *Remote Sensing*, vol. 16, no. 7, art. 1239. [doi:10.3390/rs16071239](https://doi.org/10.3390/rs16071239)  
What it is: Uncertainty-guided teacher-student TTA for remote-sensing segmentation.  
How we use it: TTA in remote sensing; no physics or guarantees.  
Key: `wang2024rsselfprompt`

**54. Liang et al. (2024).** Low Saturation Confidence Distribution-based Test-Time Adaptation for Cross-Domain Remote Sensing Image Classification. *arXiv preprint arXiv:2408.16265*. [link](https://arxiv.org/abs/2408.16265)  
What it is: TTA losses for class imbalance and low-confidence samples in remote-sensing classification.  
How we use it: TTA in remote sensing (classification).  
Key: `liang2024lscdtta`

## H. Self-training, pseudo-labels and unsupervised domain adaptation

**55. Lee (2013).** Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. *ICML 2013 Workshop on Challenges in Representation Learning*. [link](https://storage.googleapis.com/kaggle-forum-message-attachments/8924/pseudo_label_final.pdf)  
What it is: Train on the model's own confident predictions as labels.  
How we use it: Origin of pseudo-labelling.  
Key: `lee2013pseudolabel`

**56. Tarvainen & Valpola (2017).** Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. *Advances in Neural Information Processing Systems 30 (NeurIPS)*. [link](https://arxiv.org/abs/1703.01780)  
What it is: EMA teacher provides stable targets for the student.  
How we use it: Our teacher-student (EMA) self-training stage.  
Key: `tarvainen2017meanteacher`

**57. Xie et al. (2020).** Self-Training With Noisy Student Improves ImageNet Classification. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10687–10698. [link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf)  
What it is: Large-scale self-training with noise/augmentation on the student.  
How we use it: Self-training background; augmentation in our crops.  
Key: `xie2020noisystudent`

**58. Sohn et al. (2020).** FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. *Advances in Neural Information Processing Systems 33 (NeurIPS)*. [link](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html)  
What it is: Confidence-thresholded pseudo-labels with weak/strong augmentation.  
How we use it: Justifies our confidence thresholds (p>0.9 / p<0.1) for pseudo-labels.  
Key: `sohn2020fixmatch`

**59. Zou et al. (2018).** Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training. *European Conference on Computer Vision (ECCV)*. [link](https://openaccess.thecvf.com/content_ECCV_2018/html/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.html)  
What it is: Class-balanced self-training for segmentation domain adaptation.  
How we use it: Self-training UDA for segmentation; class imbalance (water is rare) motivates our 'add' step.  
Key: `zou2018cbst`

**60. Hoyer et al. (2022).** DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9924–9935. [link](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html)  
What it is: Strong UDA segmentation recipe with teacher-student self-training.  
How we use it: State-of-the-art UDA self-training (needs source data; ours is source-free).  
Key: `hoyer2022daformer`

**61. Tuia et al. (2016).** Domain Adaptation for the Classification of Remote Sensing Data: An Overview of Recent Advances. *IEEE Geoscience and Remote Sensing Magazine*, vol. 4, no. 2, pp. 41–57. [doi:10.1109/MGRS.2016.2548504](https://doi.org/10.1109/MGRS.2016.2548504)  
What it is: Overview of domain adaptation in remote sensing.  
How we use it: Domain shift is a long-standing remote-sensing problem.  
Key: `tuia2016darsreview`

## I. Conformal prediction and distribution-free risk control

**62. Vovk et al. (2022).** Algorithmic Learning in a Random World. *Springer*, 2nd edition. [doi:10.1007/978-3-031-06649-8](https://doi.org/10.1007/978-3-031-06649-8)  
What it is: Foundational book on conformal prediction (1st edition 2005).  
How we use it: Conformal prediction foundations.  
Key: `vovk2022alrw`

**63. Lei et al. (2018).** Distribution-Free Predictive Inference for Regression. *Journal of the American Statistical Association*, vol. 113, no. 523, pp. 1094–1111. [doi:10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116)  
What it is: Split conformal inference and its guarantees.  
How we use it: Split-conformal background.  
Key: `lei2018distfree`

**64. Angelopoulos & Bates (2023).** Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494–591. [doi:10.1561/2200000101](https://doi.org/10.1561/2200000101)  
What it is: Accessible introduction to conformal prediction and risk control (arXiv title: A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification).  
How we use it: Method section reference for conformal ideas.  
Key: `angelopoulos2023gentle`

**65. Bates et al. (2021).** Distribution-free, Risk-controlling Prediction Sets. *Journal of the ACM*, vol. 68, no. 6, art. 43. [doi:10.1145/3478535](https://doi.org/10.1145/3478535)  
What it is: High-probability control of general risks, including segmentation examples.  
How we use it: Alternative (high-probability) guarantee; discussion of stronger per-flood guarantees.  
Key: `bates2021rcps`

**66. Angelopoulos et al. (2024).** Conformal Risk Control. *International Conference on Learning Representations (ICLR)*. [link](https://arxiv.org/abs/2208.02814)  
What it is: Controls the expected value of any monotone loss (e.g. false negative rate) with a simple calibration rule.  
How we use it: Core of our miss-rate guarantee (FNR control with floods as exchangeable units).  
Key: `angelopoulos2024crc`

**67. Angelopoulos et al. (2021).** Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. *arXiv preprint arXiv:2110.01052*. [link](https://arxiv.org/abs/2110.01052)  
What it is: Multiple-testing framework for risk control over many parameters.  
How we use it: Extension path for controlling several risks (miss rate and false alarms) at once.  
Key: `angelopoulos2021ltt`

**68. Tibshirani et al. (2019).** Conformal Prediction Under Covariate Shift. *Advances in Neural Information Processing Systems 32 (NeurIPS)*. [link](https://papers.nips.cc/paper/8522-conformal-prediction-under-covariate-shift)  
What it is: Weighted conformal prediction when test inputs are shifted.  
How we use it: Discussion: when floods are not exchangeable (cross-dataset shift).  
Key: `tibshirani2019covshift`

**69. Barber et al. (2023).** Conformal prediction beyond exchangeability. *The Annals of Statistics*, vol. 51, no. 2, pp. 816–845. [doi:10.1214/23-AOS2276](https://doi.org/10.1214/23-AOS2276)  
What it is: Coverage bounds when exchangeability fails, with weighting.  
How we use it: Limitations: guarantee degrades gracefully under distribution drift.  
Key: `barber2023beyond`

**70. Dunn et al. (2023).** Distribution-Free Prediction Sets for Two-Layer Hierarchical Models. *Journal of the American Statistical Association*, vol. 118, no. 544, pp. 2491–2502. [doi:10.1080/01621459.2022.2060112](https://doi.org/10.1080/01621459.2022.2060112)  
What it is: Conformal prediction when data come in groups (exchangeable groups, dependent members).  
How we use it: Theory for treating floods (not pixels) as the exchangeable unit, as we do.  
Key: `dunn2023hierarchical`

**71. Mossina et al. (2024).** Conformal Semantic Image Segmentation: Post-hoc Quantification of Predictive Uncertainty. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. [link](https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/html/Mossina_Conformal_Semantic_Image_Segmentation_Post-hoc_Quantification_of_Predictive_Uncertainty_CVPRW_2024_paper.html)  
What it is: Conformal risk control for multi-class semantic segmentation (images as units).  
How we use it: Closest conformal-segmentation work; ours calibrates across flood events under domain shift.  
Key: `mossina2024conformalseg`

**72. Konidakis et al. (2025).** Uncertainty-Aware Flood Segmentation from Sentinel 2 Observations with Conformal Prediction. *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*. [link](https://users.ics.forth.gr/~tsakalid/PAPERS/CNFRS/2025-IGARSS-Konidakis.pdf)  
What it is: Inductive CP and CV+ prediction sets for OmbriaNet flood segmentation on Sentinel-2 (OMBRIA).  
How we use it: Only conformal flood-segmentation paper found: optical, pixel-level sets, no domain shift or SAR; key contrast for our novelty.  
Key: `konidakis2025cpflood`

**73. Li et al. (2026).** Quantifying and Communicating Uncertainty in SAR-Based Flood Mapping via Density-Aware Neural Networks and Conformal Risk Control. *IEEE Transactions on Geoscience and Remote Sensing*, vol. 64, pp. 1-20. [doi:10.1109/TGRS.2026.3661208](https://doi.org/10.1109/TGRS.2026.3661208)  
What it is: Density-aware network separates model-knowledge and data uncertainty in SAR flood maps; conformal risk control sets out-of-distribution abstention and set-valued predictions (found in the final novelty check, 10 Nov 2026).  
How we use it: Closest SAR work using conformal risk control; ours differs by calibrating a flood-level miss-rate bound across events and combining it with label-free test-time adaptation.  
Key: `li2026crcflood`

## J. Uncertainty and calibration in deep learning and flood mapping

**74. Guo et al. (2017).** On Calibration of Modern Neural Networks. *International Conference on Machine Learning (ICML)*, PMLR 70. [link](https://proceedings.mlr.press/v70/guo17a)  
What it is: Modern networks are over-confident; temperature scaling.  
How we use it: Why a fixed 0.5 threshold cannot be trusted under shift.  
Key: `guo2017calibration`

**75. Gal & Ghahramani (2016).** Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *International Conference on Machine Learning (ICML)*, PMLR 48. [link](https://proceedings.mlr.press/v48/gal16.html)  
What it is: MC dropout for uncertainty.  
How we use it: Heuristic uncertainty methods (no guarantees) for contrast.  
Key: `gal2016dropout`

**76. Lakshminarayanan et al. (2017).** Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems 30 (NeurIPS)*. [link](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)  
What it is: Deep ensembles for uncertainty.  
How we use it: Heuristic uncertainty for contrast; our 3 model seeds are a small ensemble.  
Key: `lakshminarayanan2017ensembles`

**77. Chen et al. (2026).** Integrating predictive uncertainty into satellite-based flood mapping. *EGUsphere preprint (under review for Natural Hazards and Earth System Sciences)*, egusphere-2026-4233. [link](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4233/)  
What it is: Evidential deep learning vs ensembles and MC dropout for flood mapping on WorldFloods.  
How we use it: Very recent flood-uncertainty work without guarantees; contrast with conformal control.  
Key: `chen2026uncertaintyflood`

## K. Backbones, training and libraries

**78. Ronneberger et al. (2015).** U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS 9351, pp. 234–241. [doi:10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)  
What it is: Encoder-decoder with skip connections for segmentation.  
How we use it: Our first backbone (U-Net from scratch).  
Key: `ronneberger2015unet`

**79. He et al. (2016).** Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)  
What it is: Residual networks (ResNet).  
How we use it: ResNet-34 encoder of our second backbone.  
Key: `he2016resnet`

**80. Ioffe & Szegedy (2015).** Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *International Conference on Machine Learning (ICML)*, PMLR 37, pp. 448–456. [link](https://proceedings.mlr.press/v37/ioffe15)  
What it is: Batch normalisation.  
How we use it: BN layers are what AdaBN and TENT adapt.  
Key: `ioffe2015bn`

**81. Deng et al. (2009).** ImageNet: A large-scale hierarchical image database. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 248–255. [doi:10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)  
What it is: ImageNet dataset.  
How we use it: Pre-training of the ResNet-34 encoder.  
Key: `deng2009imagenet`

**82. Iakubovskii (2019).** Segmentation Models Pytorch. *GitHub repository*. [link](https://github.com/qubvel-org/segmentation_models.pytorch)  
What it is: Library of segmentation architectures with pretrained encoders.  
How we use it: Implementation of the ResNet-34 U-Net.  
Key: `iakubovskii2019smp`

**83. Loshchilov & Hutter (2019).** Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*. [link](https://arxiv.org/abs/1711.05101)  
What it is: AdamW optimiser.  
How we use it: Optimiser used for training.  
Key: `loshchilov2019adamw`

## L. Statistics and research practice

**84. Wilcoxon (1945).** Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83. [doi:10.2307/3001968](https://doi.org/10.2307/3001968)  
What it is: Signed-rank test.  
How we use it: Our paired one-sided tests over events.  
Key: `wilcoxon1945`

**85. Holm (1979).** A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70. [link](https://bibbase.org/network/publication/holm-asimplesequentiallyrejectivemultipletestprocedure-1979)  
What it is: Holm step-down correction for multiple tests.  
How we use it: Correction across H1-H3.  
Key: `holm1979`

**86. Demšar (2006).** Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, vol. 7, pp. 1–30. [link](https://www.jmlr.org/papers/v7/demsar06a.html)  
What it is: Recommends Wilcoxon signed-rank for comparing methods over multiple datasets.  
How we use it: Justifies Wilcoxon with events as units.  
Key: `demsar2006`

**87. Nosek et al. (2018).** The preregistration revolution. *Proceedings of the National Academy of Sciences*, vol. 115, no. 11, pp. 2600–2606. [doi:10.1073/pnas.1708274114](https://doi.org/10.1073/pnas.1708274114)  
What it is: Argues for preregistering hypotheses and analysis plans.  
How we use it: Justifies our pre-registered MMFlood test.  
Key: `nosek2018prereg`

## M. Data sources and application context

**88. Torres et al. (2012).** GMES Sentinel-1 mission. *Remote Sensing of Environment*, vol. 120, pp. 9–24. [doi:10.1016/j.rse.2011.05.028](https://doi.org/10.1016/j.rse.2011.05.028)  
What it is: Sentinel-1 C-band SAR mission description.  
How we use it: Data source description.  
Key: `torres2012sentinel1`

**89. European Space Agency (2024).** Copernicus DEM Product Handbook (GLO-30). *Copernicus Data Space Ecosystem*. [link](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)  
What it is: Copernicus GLO-30 DEM documentation.  
How we use it: Elevation and slope inputs; source of GLO-30 HAND.  
Key: `copdem2024handbook`

**90. Tellman et al. (2021).** Satellite imaging reveals increased proportion of population exposed to floods. *Nature*, vol. 596, pp. 80–86. [doi:10.1038/s41586-021-03695-w](https://doi.org/10.1038/s41586-021-03695-w)  
What it is: Global Flood Database: 913 events (2000-2018) show rising population exposure.  
How we use it: Introduction: why rapid, reliable flood maps matter.  
Key: `tellman2021exposure`

**91. Islam et al. (2026).** Quantifying Flash Flood Inundation and Assessing Damage Using Satellite Earth Observations: The Case of 2022 Flash Flood in Bangladesh. *Geomatics, Natural Hazards and Risk*, vol. 17, no. 1. [doi:10.1080/19475705.2026.2614729](https://doi.org/10.1080/19475705.2026.2614729)  
What it is: Sentinel-1 mapping of the 2022 Sylhet flash flood (>1,600 km2 inundated) and damage.  
How we use it: Optional Bangladesh case study context and comparison.  
Key: `islam2026sylhet`
