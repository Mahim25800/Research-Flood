# PX1 paper reference list. Every entry was checked against a live web source during this session (link in "url").
# Fields: key, cat, authors, year, title, venue, details, doi, url, type, summary, use
R = []
def add(**k): R.append(k)

CATS = {
 "A": "Flood datasets and benchmarks",
 "B": "Classical SAR flood mapping and operational systems",
 "C": "Terrain priors: HAND, exclusion maps and topographic constraints",
 "D": "Deep learning for SAR / satellite flood mapping",
 "E": "Label-scarce, self-supervised and domain-shift work on flood mapping",
 "F": "Physics-guided / knowledge-guided machine learning",
 "G": "Test-time adaptation and batch-norm adaptation",
 "H": "Self-training, pseudo-labels and unsupervised domain adaptation",
 "I": "Conformal prediction and distribution-free risk control",
 "J": "Uncertainty and calibration in deep learning and flood mapping",
 "K": "Backbones, training and libraries",
 "L": "Statistics and research practice",
 "M": "Data sources and application context",
}

# ---------------- A. datasets
add(key="bonafilia2020sen1floods11", cat="A", type="inproceedings", authors="Bonafilia, Derrick and Tellman, Beth and Anderson, Tyler and Issenberg, Erica", year=2020,
    title="Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1", venue="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)", details="", doi="",
    url="https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html",
    summary="Sentinel-1/2 chips from 11 flood events with 446 hand-labelled 512x512 chips plus weak labels; the standard benchmark for SAR flood segmentation.",
    use="Our training data (source domain) and the leave-one-event-out (LOEO) evaluation.")
add(key="bountos2024kurosiwo", cat="A", type="inproceedings", authors="Bountos, Nikolaos Ioannis and others", year=2024,
    title="Kuro Siwo: 33 billion m$^2$ under the water. A global multi-temporal satellite dataset for rapid flood mapping", venue="Advances in Neural Information Processing Systems 37 (NeurIPS), Datasets and Benchmarks Track", details="", doi="",
    url="https://proceedings.neurips.cc/paper_files/paper/2024/hash/43612b0662cb6a4986edf859fd6ebafe-Abstract.html",
    summary="Large multi-temporal Sentinel-1 GRD/SLC flood dataset from 43 events with expert flood annotations and DEM.",
    use="Cross-dataset test set (26 events); development set for choosing the primary method (pre-registered rule).")
add(key="montello2022mmflood", cat="A", type="article", authors="Montello, Fabio and Arnaudo, Edoardo and Rossi, Claudio", year=2022,
    title="MMFlood: A Multimodal Dataset for Flood Delineation From Satellite Imagery", venue="IEEE Access", details="vol. 10, pp. 96774--96787", doi="10.1109/ACCESS.2022.3205419",
    url="https://iris.polito.it/handle/11583/2971238",
    summary="Sentinel-1 images, DEM and hydrography for 95 Copernicus EMS flood activations in 42 countries (2014-2021), with rasterised EMS flood masks; data on Zenodo (record 6534637).",
    use="Pre-registered final (untouched) test set: 30 activations after overlap exclusion.")
add(key="mateogarcia2021worldfloods", cat="A", type="article", authors="Mateo-Garcia, Gonzalo and Veitch-Michaelis, Joshua and Smith, Lewis and Oprea, Silviu Vlad and Schumann, Guy and Gal, Yarin and Baydin, Atilim Gunes and Backes, Dietmar", year=2021,
    title="Towards global flood mapping onboard low cost satellites with machine learning", venue="Scientific Reports", details="vol. 11, art. 7249", doi="10.1038/s41598-021-86650-z",
    url="https://oatml.cs.ox.ac.uk/publications/202103_MateoGarcia2021.html",
    summary="WorldFloods dataset (Sentinel-2, many CEMS events) and lightweight CNNs for on-board flood mapping.",
    use="Related dataset; shows generalisation to new floods as the core difficulty.")
add(key="zhao2024urbansarfloods", cat="A", type="inproceedings", authors="Zhao, Jie and Xiong, Zhitong and Zhu, Xiao Xiang", year=2024,
    title="UrbanSARFloods: Sentinel-1 SLC-Based Benchmark Dataset for Urban and Open-Area Flood Mapping", venue="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)", details="", doi="",
    url="https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Zhao_UrbanSARFloods_Sentinel-1_SLC-Based_Benchmark_Dataset_for_Urban_and_Open-Area_Flood_CVPRW_2024_paper.html",
    summary="SLC-based benchmark separating open-area and urban floods; shows urban flooding is hard for intensity-only models.",
    use="Limitations: urban floods are a known failure mode of dark-water rules.")
add(key="hansch2022spacenet8", cat="A", type="inproceedings", authors="H{\\\"a}nsch, Ronny and others", year=2022,
    title="SpaceNet 8 - The Detection of Flooded Roads and Buildings", venue="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)", details="", doi="",
    url="https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Hansch_SpaceNet_8_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.html",
    summary="VHR optical benchmark for flooded roads and buildings.",
    use="Related-work breadth (flood benchmarks beyond Sentinel-1).")
add(key="rambour2020sen12flood", cat="A", type="misc", authors="Rambour, Cl{\\'e}ment and Audebert, Nicolas and Koeniguer, Elise and Le Saux, Bertrand and Crucianu, Michel and Datcu, Mihai", year=2020,
    title="SEN12-FLOOD: a SAR and Multispectral Dataset for Flood Detection", venue="IEEE Dataport", details="", doi="",
    url="https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection",
    summary="Sentinel-1/2 time series labelled for flood presence (image-level).",
    use="Related dataset; image-level rather than pixel-level labels.")
add(key="drakonakis2022ombrianet", cat="A", type="article", authors="Drakonakis, Georgios I. and Tsagkatakis, Grigorios and Fotiadou, Konstantina and Tsakalides, Panagiotis", year=2022,
    title="OmbriaNet---Supervised Flood Mapping via Convolutional Neural Networks Using Multitemporal Sentinel-1 and Sentinel-2 Data Fusion", venue="IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing", details="vol. 15", doi="",
    url="https://doaj.org/article/1a0f358ab0624724a4120b8fcaca14ec",
    summary="OMBRIA bitemporal flood dataset (EMS events) and the OmbriaNet change-detection CNN.",
    use="Related dataset and bitemporal baseline.")

# ---------------- B. classical SAR flood mapping
add(key="otsu1979threshold", cat="B", type="article", authors="Otsu, Nobuyuki", year=1979,
    title="A Threshold Selection Method from Gray-Level Histograms", venue="IEEE Transactions on Systems, Man, and Cybernetics", details="vol. 9, no. 1, pp. 62--66", doi="10.1109/TSMC.1979.4310076",
    url="https://www.scilit.net/publications/e8d46ae32d16168862672c390a1ef80a",
    summary="Histogram threshold maximising between-class variance.",
    use="Per-flood adaptive dark-water threshold in our physics rule; Otsu-only baseline.")
add(key="martinis2009nrt", cat="B", type="article", authors="Martinis, Sandro and Twele, Andr{\\'e} and Voigt, Stefan", year=2009,
    title="Towards operational near real-time flood detection using a split-based automatic thresholding procedure on high resolution TerraSAR-X data", venue="Natural Hazards and Earth System Sciences", details="vol. 9, no. 2, pp. 303--314", doi="10.5194/nhess-9-303-2009",
    url="https://nhess.copernicus.org/articles/9/303/2009/",
    summary="Tile-based (split) automatic thresholding for SAR flood detection, the basis of later operational chains.",
    use="Classical thresholding lineage; motivates per-scene thresholds instead of fixed dB cut-offs.")
add(key="twele2016s1chain", cat="B", type="article", authors="Twele, Andr{\\'e} and Cao, Wenxi and Plank, Simon and Martinis, Sandro", year=2016,
    title="Sentinel-1-based flood mapping: a fully automated processing chain", venue="International Journal of Remote Sensing", details="vol. 37, no. 13, pp. 2990--3004", doi="10.1080/01431161.2016.1192304",
    url="https://elib.dlr.de/102476",
    summary="Operational DLR chain: automatic thresholding, fuzzy-logic refinement with HAND/slope, region growing.",
    use="Precedent for combining radar thresholds with HAND and slope; we move this logic inside adaptation.")
add(key="chini2017hsbt", cat="B", type="article", authors="Chini, Marco and Hostache, Renaud and Giustarini, Laura and Matgen, Patrick", year=2017,
    title="A Hierarchical Split-Based Approach for Parametric Thresholding of SAR Images: Flood Inundation as a Test Case", venue="IEEE Transactions on Geoscience and Remote Sensing", details="vol. 55, no. 12, pp. 6975--6988", doi="10.1109/TGRS.2017.2737664",
    url="https://www.list.lu/en/environment/scientific-publications/scientific-publications-detail/a-hierarchical-split-based-approach-for-parametric-thresholding-of-sar-images-flood-inundation-as-a",
    summary="Hierarchical tiling to find bimodal tiles and fit parametric water/non-water distributions.",
    use="Classical adaptive thresholding comparison point.")
add(key="giustarini2013change", cat="B", type="article", authors="Giustarini, Laura and Hostache, Renaud and Matgen, Patrick and Schumann, Guy J.-P. and Bates, Paul D. and Mason, David C.", year=2013,
    title="A Change Detection Approach to Flood Mapping in Urban Areas Using TerraSAR-X", venue="IEEE Transactions on Geoscience and Remote Sensing", details="vol. 51, no. 4, pp. 2417--2430", doi="10.1109/TGRS.2012.2210901",
    url="https://centaur.reading.ac.uk/29866",
    summary="Change detection with a pre-flood reference image, including urban areas.",
    use="Change detection as an alternative to single-date mapping; our setting is single post-flood image.")
add(key="giustarini2016probabilistic", cat="B", type="article", authors="Giustarini, Laura and Hostache, Renaud and Kavetski, Dmitri and Chini, Marco and Corato, Giovanni and Schlaffer, Stefan and Matgen, Patrick", year=2016,
    title="Probabilistic Flood Mapping Using Synthetic Aperture Radar Data", venue="IEEE Transactions on Geoscience and Remote Sensing", details="vol. 54, no. 12, pp. 6958--6969", doi="10.1109/TGRS.2016.2592951",
    url="https://www.list.lu/fr/recherche/environment/publications-scientifiques/scientific-publications-detail/probabilistic-flood-mapping-using-synthetic-aperture-radar-data",
    summary="Bayesian per-pixel flood probability from SAR backscatter distributions.",
    use="Prior work on probabilistic flood maps; ours adds a guarantee on the miss rate.")
add(key="landuyt2019assessment", cat="B", type="article", authors="Landuyt, Lisa and Van Wesemael, Alexandra and Schumann, Guy J.-P. and Hostache, Renaud and Verhoest, Niko E. C. and Van Coillie, Frieke M. B.", year=2019,
    title="Flood Mapping Based on Synthetic Aperture Radar: An Assessment of Established Approaches", venue="IEEE Transactions on Geoscience and Remote Sensing", details="vol. 57, no. 2, pp. 722--739", doi="10.1109/TGRS.2018.2860054",
    url="https://dblp.dagstuhl.de/rec/journals/tgrs/LanduytWSHVC19.html",
    summary="Systematic comparison of thresholding, change detection and classification approaches for SAR flood mapping.",
    use="Background on how established methods compare; motivates event-level evaluation.")
add(key="clement2018multitemporal", cat="B", type="article", authors="Clement, M. A. and Kilsby, C. G. and Moore, P.", year=2018,
    title="Multi-temporal synthetic aperture radar flood mapping using change detection", venue="Journal of Flood Risk Management", details="vol. 11, no. 2, pp. 152--168", doi="10.1111/jfr3.12303",
    url="https://www.peeref.com/works/4635733",
    summary="Sentinel-1 multi-temporal change detection for flood extent.",
    use="Change-detection background.")
add(key="tsyganskaya2018vegetation", cat="B", type="article", authors="Tsyganskaya, Viktoriya and Martinis, Sandro and Marzahn, Philip and Ludwig, Ralf", year=2018,
    title="SAR-based detection of flooded vegetation -- a review of characteristics and approaches", venue="International Journal of Remote Sensing", details="vol. 39, no. 8, pp. 2255--2293", doi="10.1080/01431161.2017.1420938",
    url="https://www.peeref.com/works/5411218",
    summary="Review of double-bounce and other signatures of flooded vegetation in SAR.",
    use="Limitations: dark-water rules miss flooded vegetation (bright, not dark).")
add(key="salamon2021gfm", cat="B", type="inproceedings", authors="Salamon, Peter and others", year=2021,
    title="The New, Systematic Global Flood Monitoring Product of the Copernicus Emergency Management Service", venue="IEEE International Geoscience and Remote Sensing Symposium (IGARSS)", details="", doi="",
    url="https://dblp1.uni-trier.de/rec/conf/igarss/SalamonMRCB0MCB21.html",
    summary="Introduces CEMS Global Flood Monitoring (GFM): every Sentinel-1 scene processed by an ensemble of three algorithms.",
    use="Operational context; GFM uses HAND/exclusion masks, which our method uses during adaptation.")
add(key="krullikowski2023gfmensemble", cat="B", type="article", authors="Krullikowski, Christian and others", year=2023,
    title="Estimating ensemble likelihoods for the Sentinel-1 based Global Flood Monitoring product of the Copernicus Emergency Management Service", venue="IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing", details="vol. 16", doi="",
    url="https://arxiv.org/abs/2304.12488",
    summary="Ensemble likelihood (uncertainty) layer for the GFM product.",
    use="Operational uncertainty; contrast with our conformal miss-rate guarantee.")
add(key="misra2025globalfloods", cat="B", type="article", authors="Misra, Amit and White, Kevin and Fobi Nsutezo, Simone and others", year=2025,
    title="Mapping global floods with 10 years of satellite radar data", venue="Nature Communications", details="vol. 16, art. 5762", doi="10.1038/s41467-025-60973-1",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12216028",
    summary="Change-detection CNN on Sentinel-1 pairs run over 2014-2024; DEM slope masks and auxiliary layers remove false positives.",
    use="State of the art in global SAR flood mapping; uses terrain only as a post-hoc mask.")

# ---------------- C. terrain priors
add(key="renno2008hand", cat="C", type="article", authors="Renn{\\'o}, Camilo Daleles and Nobre, Antonio Donato and Cuartas, Luz Adriana and Soares, Jo{\\~a}o Vianei and Hodnett, Martin G. and Tomasella, Javier and Waterloo, Maarten J.", year=2008,
    title="HAND, a new terrain descriptor using SRTM-DEM: Mapping terra-firme rainforest environments in Amazonia", venue="Remote Sensing of Environment", details="vol. 112, no. 9, pp. 3469--3481", doi="10.1016/j.rse.2008.03.018",
    url="https://repositorio.inpa.gov.br/handle/1/18530",
    summary="Introduces Height Above the Nearest Drainage (HAND).",
    use="Defines the HAND terrain descriptor used in our physics check (HAND < 5 m).")
add(key="nobre2011hand", cat="C", type="article", authors="Nobre, A. D. and Cuartas, L. A. and Hodnett, M. and Renn{\\'o}, C. D. and Rodrigues, G. and Silveira, A. and Waterloo, M. and Saleska, S.", year=2011,
    title="Height Above the Nearest Drainage -- a hydrologically relevant new terrain model", venue="Journal of Hydrology", details="vol. 404, no. 1--2, pp. 13--29", doi="10.1016/j.jhydrol.2011.03.051",
    url="https://repositorio.inpa.gov.br/handle/1/18178",
    summary="Shows HAND relates to soil water conditions and hydrology.",
    use="Hydrological justification for HAND as a flood plausibility prior.")
add(key="glo30hand2022", cat="C", type="misc", authors="{Alaska Satellite Facility} and {HydroSAR team}", year=2022,
    title="Global 30m Height Above Nearest Drainage (HAND) derived from the Copernicus GLO-30 DEM", venue="Registry of Open Data on AWS", details="", doi="",
    url="https://registry.opendata.aws/glo-30-hand/",
    summary="Global 30 m HAND tiles computed from the Copernicus GLO-30 DEM, distributed as COGs on AWS.",
    use="Our HAND source for all three datasets.")
add(key="zhao2021exclusion", cat="C", type="article", authors="Zhao, Jie and Pelich, Ramona and Hostache, Renaud and Matgen, Patrick and Wagner, Wolfgang and Chini, Marco", year=2021,
    title="Deriving exclusion maps from C-band SAR time-series in support of floodwater mapping", venue="Remote Sensing of Environment", details="vol. 265, art. 112668", doi="10.1016/j.rse.2021.112668",
    url="https://repositum.tuwien.at/handle/20.500.12708/18262",
    summary="Exclusion maps marking where SAR cannot see floodwater (shadow, permanent low backscatter, urban), used in GFM.",
    use="Closest operational analogue to our 'physics check'; we use such knowledge to filter pseudo-labels rather than outputs.")
add(key="tupas2023handprior", cat="C", type="article", authors="Tupas, Mark Edwin and Roth, Florian and Bauer-Marschallinger, Bernhard and Wagner, Wolfgang", year=2023,
    title="Improving Sentinel-1 Flood Maps Using a Topographic Index as Prior in Bayesian Inference", venue="Water", details="vol. 15, no. 23, art. 4034", doi="10.3390/w15234034",
    url="https://www.mdpi.com/2073-4441/15/23/4034",
    summary="Replaces the hard HAND exclusion mask with a HAND-based prior in the TU Wien Bayesian flood algorithm; fewer misses.",
    use="Soft use of HAND as a prior; supports our choice of HAND as a plausibility signal.")
add(key="tian2026deepsar", cat="C", type="article", authors="Tian, Dan and Wang, Lei and Liu, Hongxing and Cohen, Sagy and Shu, Song", year=2026,
    title="DeepSAR Flood Mapper: global flood mapping on Google Earth Engine cloud platform using MLP deep learning model with Sentinel-1 SAR imagery and HAND topographic data", venue="GIScience \\& Remote Sensing", details="vol. 63, no. 1, art. 2612306", doi="10.1080/15481603.2025.2612306",
    url="https://cris.bgu.ac.il/en/publications/deepsar-flood-mapper-global-flood-mapping-on-google-earth-engine--2/",
    summary="Deep model taking Sentinel-1 plus HAND as input features, deployed on GEE.",
    use="HAND as a model input (supervised); ours uses HAND label-free at test time.")
add(key="hansen2025ribe", cat="C", type="article", authors="Hansen, Mark and Vejby, Jacob and Koch, Julian", year=2025,
    title="Flood mapping using Sentinel-1 imagery with topographical and hydrological contextualization: Case study from Ribe, Denmark", venue="International Journal of Applied Earth Observation and Geoinformation", details="vol. 143, art. 104816", doi="10.1016/j.jag.2025.104816",
    url="https://www.sciencedirect.com/science/article/pii/S1569843225004637",
    summary="Local thresholding refined with terrain/drainage context and fuzzy logic.",
    use="Recent rule-based terrain refinement; complementary to our learning-based use.")

# ---------------- D. deep learning flood mapping
add(key="bentivoglio2022review", cat="D", type="article", authors="Bentivoglio, Roberto and Isufi, Elvin and Jonkman, Sebastian Nicolaas and Taormina, Riccardo", year=2022,
    title="Deep learning methods for flood mapping: a review of existing applications and future research directions", venue="Hydrology and Earth System Sciences", details="vol. 26, no. 16, pp. 4345--4378", doi="10.5194/hess-26-4345-2022",
    url="https://hess.copernicus.org/articles/26/4345/2022/",
    summary="Review of DL for flood inundation, susceptibility and hazard mapping; flags generalisation and uncertainty as open problems.",
    use="Introduction: the generalisation and uncertainty gaps we address.")
add(key="lee2025geoaiflood", cat="D", type="misc", authors="Lee, Hyunho and Li, Wenwen", year=2025,
    title="Geospatial Artificial Intelligence for Satellite-Based Flood Extent Mapping: Concepts, Advances, and Future Perspectives", venue="arXiv preprint arXiv:2504.02214", details="", doi="",
    url="https://arxiv.org/abs/2504.02214",
    summary="Recent review of GeoAI for flood extent mapping, including uncertainty and change detection.",
    use="Up-to-date review for the introduction.")
add(key="nemni2020unosat", cat="D", type="article", authors="Nemni, Edoardo and Bullock, Joseph and Belabbes, Samir and Bromley, Lars", year=2020,
    title="Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery", venue="Remote Sensing", details="vol. 12, no. 16, art. 2532", doi="10.3390/rs12162532",
    url="https://durham-repository.worktribe.com/output/1263837/fully-convolutional-neural-network-for-rapid-flood-segmentation-in-synthetic-aperture-radar-imagery",
    summary="UNOSAT U-Net/XNet for operational rapid flood segmentation from Sentinel-1.",
    use="Operational DL precedent (UN rapid mapping).")
add(key="bai2021basnet", cat="D", type="article", authors="Bai, Yanbing and others", year=2021,
    title="Enhancement of Detecting Permanent Water and Temporary Water in Flood Disasters by Fusing Sentinel-1 and Sentinel-2 Imagery Using Deep Learning Algorithms: Demonstration of Sen1Floods11 Benchmark Datasets", venue="Remote Sensing", details="vol. 13, no. 11, art. 2220", doi="10.3390/rs13112220",
    url="https://doaj.org/article/9fd78fd0687a4424b4fca82c4c65da33",
    summary="BASNet-based S1/S2 fusion on Sen1Floods11.",
    use="Sen1Floods11 baselines.")
add(key="konapala2021s1s2", cat="D", type="article", authors="Konapala, Goutam and Kumar, Sujay V. and Ahmad, Shahryar Khalique", year=2021,
    title="Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning", venue="ISPRS Journal of Photogrammetry and Remote Sensing", details="vol. 180, pp. 163--173", doi="10.1016/j.isprsjprs.2021.08.016",
    url="https://ntrs.nasa.gov/citations/20220000062",
    summary="Compares S1, S2 and combinations (with elevation) for U-Net flood mapping on Sen1Floods11.",
    use="Sen1Floods11 baselines; effect of adding elevation inputs (our V1 variant).")
add(key="helleis2022benchmark", cat="D", type="article", authors="Helleis, Max and Wieland, Marc and Krullikowski, Christian and Martinis, Sandro and Plank, Simon", year=2022,
    title="Sentinel-1-Based Water and Flood Mapping: Benchmarking Convolutional Neural Networks Against an Operational Rule-Based Processing Chain", venue="IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing", details="vol. 15", doi="",
    url="https://elib.dlr.de/187304",
    summary="CNNs vs DLR's operational rule-based chain on Sentinel-1 water/flood mapping.",
    use="Shows CNNs vs thresholds trade-offs; supports combining both.")
add(key="jakubik2023prithvi", cat="D", type="misc", authors="Jakubik, Johannes and others", year=2023,
    title="Foundation Models for Generalist Geospatial Artificial Intelligence", venue="arXiv preprint arXiv:2310.18660", details="", doi="",
    url="https://arxiv.org/abs/2310.18660",
    summary="Prithvi geospatial foundation model; fine-tuned on Sen1Floods11 among other tasks.",
    use="Stronger-backbone context; future work with foundation models.")
add(key="garg2023crossmodal", cat="D", type="misc", authors="Garg, Shubhika and Feinstein, Ben and Timnat, Shahar and Batchu, Vishal and Dror, Gideon and Rosenthal, Adi Gerzi and Gulshan, Varun", year=2023,
    title="Cross Modal Distillation for Flood Extent Mapping", venue="arXiv preprint arXiv:2302.08180", details="", doi="",
    url="https://arxiv.org/abs/2302.08180",
    summary="Optical teacher distils into a SAR student using unlabelled data; +6.5% IoU on Sen1Floods11.",
    use="Label-efficient flood mapping; ours needs no target labels or optical data.")

# ---------------- E. label-scarce / domain shift on flood mapping
add(key="akiva2021h2onet", cat="E", type="inproceedings", authors="Akiva, Peri and Purri, Matthew and Dana, Kristin and Tellman, Beth and Anderson, Tyler", year=2021,
    title="H2O-Net: Self-Supervised Flood Segmentation via Adversarial Domain Adaptation and Label Refinement", venue="Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)", details="", doi="",
    url="https://openaccess.thecvf.com/content/WACV2021/papers/Akiva_H2O-Net_Self-Supervised_Flood_Segmentation_via_Adversarial_Domain_Adaptation_and_Label_WACV_2021_paper.pdf",
    summary="Adversarial domain adaptation plus label refinement for flood segmentation without pixel labels.",
    use="Closest prior on label refinement for floods; ours is test-time, per event, and physics-checked.")
add(key="paul2021semisup", cat="E", type="inproceedings", authors="Paul, Sayak and Ganju, Siddha", year=2021,
    title="Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning", venue="NeurIPS 2021 Workshop on Tackling Climate Change with Machine Learning", details="", doi="",
    url="https://www.climatechange.ai/papers/neurips2021/1",
    summary="Iterative pseudo-labelling with U-Net ensembles and CRF on the ETCI 2021 data.",
    use="Pseudo-labelling for SAR floods (offline, semi-supervised) vs our per-event test-time version.")
add(key="yadav2022unsupts", cat="E", type="misc", authors="Yadav, Ritu and Nascetti, Andrea and Azizpour, Hossein and Ban, Yifang", year=2022,
    title="Unsupervised Flood Detection on SAR Time Series", venue="arXiv preprint arXiv:2212.03675", details="", doi="",
    url="https://arxiv.org/abs/2212.03675",
    summary="Unsupervised probabilistic change detection on SAR time series with reconstruction and contrastive learning.",
    use="Label-free flood mapping alternative (needs time series; ours uses one post-event image).")
add(key="turkmenli2025dgflood", cat="E", type="inproceedings", authors="Turkmenli, Ilter and Kayabol, Koray and Aptoula, Erchan", year=2025,
    title="A comparative domain generalization study for SAR image-based flood segmentation", venue="Joint Urban Remote Sensing Event (JURSE)", details="", doi="10.1109/JURSE60372.2025.11075977",
    url="https://research.sabanciuniv.edu/id/eprint/52161/",
    summary="Treats each Sen1Floods11 event as a source domain and compares domain-generalisation methods (~3% IoU gain).",
    use="Directly related: event-as-domain view of flood generalisation; ours adapts at test time instead.")

# ---------------- F. physics-guided ML
add(key="karpatne2017tgds", cat="F", type="article", authors="Karpatne, Anuj and Atluri, Gowtham and Faghmous, James H. and Steinbach, Michael and Banerjee, Arindam and Ganguly, Auroop and Shekhar, Shashi and Samatova, Nagiza and Kumar, Vipin", year=2017,
    title="Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data", venue="IEEE Transactions on Knowledge and Data Engineering", details="vol. 29, no. 10, pp. 2318--2331", doi="10.1109/TKDE.2017.2720168",
    url="https://experts.umn.edu/en/publications/theory-guided-data-science-a-new-paradigm-for-scientific-discover/",
    summary="Framework for using scientific knowledge to constrain/guide data-driven models.",
    use="Positions our physics check as theory-guided learning.")
add(key="willard2022integrating", cat="F", type="article", authors="Willard, Jared and Jia, Xiaowei and Xu, Shaoming and Steinbach, Michael and Kumar, Vipin", year=2022,
    title="Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems", venue="ACM Computing Surveys", details="vol. 55, no. 4, art. 66", doi="10.1145/3514228",
    url="https://arxiv.org/abs/2003.04919",
    summary="Survey of physics-guided losses, architectures, initialisation and hybrid models.",
    use="Related-work taxonomy; ours is knowledge-guided pseudo-label correction (a less common category).")
add(key="reichstein2019deep", cat="F", type="article", authors="Reichstein, Markus and Camps-Valls, Gustau and Stevens, Bjorn and Jung, Martin and Denzler, Joachim and Carvalhais, Nuno and Prabhat", year=2019,
    title="Deep learning and process understanding for data-driven Earth system science", venue="Nature", details="vol. 566, pp. 195--204", doi="10.1038/s41586-019-0912-1",
    url="https://pub.inf-cv.uni-jena.de/bib/reichstein2019deep",
    summary="Argues for hybrid physical/deep-learning models in Earth science.",
    use="Motivation for physics-aware deep learning in Earth observation.")

# ---------------- G. test-time adaptation
add(key="wang2021tent", cat="G", type="inproceedings", authors="Wang, Dequan and Shelhamer, Evan and Liu, Shaoteng and Olshausen, Bruno and Darrell, Trevor", year=2021,
    title="Tent: Fully Test-Time Adaptation by Entropy Minimization", venue="International Conference on Learning Representations (ICLR)", details="", doi="",
    url="https://arxiv.org/abs/2006.10726",
    summary="Adapts batch-norm affine parameters at test time by minimising prediction entropy.",
    use="Baseline (TENT); H1 compares against it.")
add(key="li2018adabn", cat="G", type="article", authors="Li, Yanghao and Wang, Naiyan and Shi, Jianping and Hou, Xiaodi and Liu, Jiaying", year=2018,
    title="Adaptive Batch Normalization for practical domain adaptation", venue="Pattern Recognition", details="vol. 80, pp. 109--117", doi="10.1016/j.patcog.2018.03.005",
    url="https://www.icst.pku.edu.cn/struct/Projects/AdaBN.html",
    summary="Re-estimates batch-norm statistics on target data (AdaBN).",
    use="First stage of our method and baseline (H2).")
add(key="schneider2020covariate", cat="G", type="inproceedings", authors="Schneider, Steffen and Rusak, Evgenia and Eck, Luisa and Bringmann, Oliver and Brendel, Wieland and Bethge, Matthias", year=2020,
    title="Improving robustness against common corruptions by covariate shift adaptation", venue="Advances in Neural Information Processing Systems 33 (NeurIPS)", details="", doi="",
    url="https://papers.nips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html",
    summary="Shows simple BN statistic adaptation greatly improves robustness to shifts.",
    use="Supports BN re-estimation as a strong, cheap baseline.")
add(key="sun2020ttt", cat="G", type="inproceedings", authors="Sun, Yu and Wang, Xiaolong and Liu, Zhuang and Miller, John and Efros, Alexei and Hardt, Moritz", year=2020,
    title="Test-Time Training with Self-Supervision for Generalization under Distribution Shifts", venue="International Conference on Machine Learning (ICML)", details="PMLR 119", doi="",
    url="https://arxiv.org/abs/1909.13231",
    summary="Test-time training using a self-supervised auxiliary task.",
    use="TTA background; requires training-time changes, unlike ours.")
add(key="wang2022cotta", cat="G", type="inproceedings", authors="Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin", year=2022,
    title="Continual Test-Time Domain Adaptation", venue="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", details="pp. 7201--7211", doi="",
    url="https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html",
    summary="Mean-teacher pseudo-labels with augmentation averaging and stochastic restore for continual TTA.",
    use="Teacher-student TTA closest in mechanics to our self-training stage.")
add(key="niu2022eata", cat="G", type="inproceedings", authors="Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Chen, Yaofo and Zheng, Shijian and Zhao, Peilin and Tan, Mingkui", year=2022,
    title="Efficient Test-Time Model Adaptation without Forgetting", venue="International Conference on Machine Learning (ICML)", details="PMLR 162", doi="",
    url="https://proceedings.mlr.press/v162/niu22a.html",
    summary="Filters unreliable/redundant samples and regularises to avoid forgetting during TTA.",
    use="Reliability filtering in TTA; ours filters with physics instead of entropy.")
add(key="niu2023sar", cat="G", type="inproceedings", authors="Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Wen, Zhiquan and Chen, Yaofo and Zhao, Peilin and Tan, Mingkui", year=2023,
    title="Towards Stable Test-time Adaptation in Dynamic Wild World", venue="International Conference on Learning Representations (ICLR)", details="", doi="",
    url="https://arxiv.org/abs/2302.12400",
    summary="Shows TTA can collapse in the wild and proposes sharpness-aware reliable entropy minimisation.",
    use="Evidence that TTA can harm; supports our 'safe adaptation' framing.")
add(key="liang2025ttasurvey", cat="G", type="article", authors="Liang, Jian and He, Ran and Tan, Tieniu", year=2025,
    title="A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts", venue="International Journal of Computer Vision", details="vol. 133, pp. 31--64", doi="10.1007/s11263-024-02181-w",
    url="https://arxiv.org/abs/2303.15361",
    summary="Survey of test-time domain, batch and instance adaptation.",
    use="TTA taxonomy for related work.")
add(key="wang2024rsselfprompt", cat="G", type="article", authors="Wang, Ziquan and Zhang, Yongsheng and Zhang, Zhenchao and Jiang, Zhipeng and Yu, Ying and Li, Lei and Zhang, Lei", year=2024,
    title="Exploring Uncertainty-Based Self-Prompt for Test-Time Adaptation Semantic Segmentation in Remote Sensing Images", venue="Remote Sensing", details="vol. 16, no. 7, art. 1239", doi="10.3390/rs16071239",
    url="https://doaj.org/article/2941cc0f3aab45898de544d816cce33d",
    summary="Uncertainty-guided teacher-student TTA for remote-sensing segmentation.",
    use="TTA in remote sensing; no physics or guarantees.")
add(key="liang2024lscdtta", cat="G", type="misc", authors="Liang, Yu and Cao, Shilei and Zhang, Xiucheng and Zheng, Juepeng and Huang, Jianxi and Fu, Haohuan", year=2024,
    title="Low Saturation Confidence Distribution-based Test-Time Adaptation for Cross-Domain Remote Sensing Image Classification", venue="arXiv preprint arXiv:2408.16265", details="", doi="",
    url="https://arxiv.org/abs/2408.16265",
    summary="TTA losses for class imbalance and low-confidence samples in remote-sensing classification.",
    use="TTA in remote sensing (classification).")

# ---------------- H. self-training and UDA
add(key="lee2013pseudolabel", cat="H", type="inproceedings", authors="Lee, Dong-Hyun", year=2013,
    title="Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks", venue="ICML 2013 Workshop on Challenges in Representation Learning", details="", doi="",
    url="https://storage.googleapis.com/kaggle-forum-message-attachments/8924/pseudo_label_final.pdf",
    summary="Train on the model's own confident predictions as labels.",
    use="Origin of pseudo-labelling.")
add(key="tarvainen2017meanteacher", cat="H", type="inproceedings", authors="Tarvainen, Antti and Valpola, Harri", year=2017,
    title="Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results", venue="Advances in Neural Information Processing Systems 30 (NeurIPS)", details="", doi="",
    url="https://arxiv.org/abs/1703.01780",
    summary="EMA teacher provides stable targets for the student.",
    use="Our teacher-student (EMA) self-training stage.")
add(key="xie2020noisystudent", cat="H", type="inproceedings", authors="Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V.", year=2020,
    title="Self-Training With Noisy Student Improves ImageNet Classification", venue="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", details="pp. 10687--10698", doi="",
    url="https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf",
    summary="Large-scale self-training with noise/augmentation on the student.",
    use="Self-training background; augmentation in our crops.")
add(key="sohn2020fixmatch", cat="H", type="inproceedings", authors="Sohn, Kihyuk and Berthelot, David and Carlini, Nicholas and Zhang, Zizhao and Zhang, Han and Raffel, Colin A. and Cubuk, Ekin Dogus and Kurakin, Alexey and Li, Chun-Liang", year=2020,
    title="FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence", venue="Advances in Neural Information Processing Systems 33 (NeurIPS)", details="", doi="",
    url="https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html",
    summary="Confidence-thresholded pseudo-labels with weak/strong augmentation.",
    use="Justifies our confidence thresholds (p>0.9 / p<0.1) for pseudo-labels.")
add(key="zou2018cbst", cat="H", type="inproceedings", authors="Zou, Yang and Yu, Zhiding and Kumar, B. V. K. Vijaya and Wang, Jinsong", year=2018,
    title="Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training", venue="European Conference on Computer Vision (ECCV)", details="", doi="",
    url="https://openaccess.thecvf.com/content_ECCV_2018/html/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.html",
    summary="Class-balanced self-training for segmentation domain adaptation.",
    use="Self-training UDA for segmentation; class imbalance (water is rare) motivates our 'add' step.")
add(key="hoyer2022daformer", cat="H", type="inproceedings", authors="Hoyer, Lukas and Dai, Dengxin and Van Gool, Luc", year=2022,
    title="DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation", venue="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", details="pp. 9924--9935", doi="",
    url="https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html",
    summary="Strong UDA segmentation recipe with teacher-student self-training.",
    use="State-of-the-art UDA self-training (needs source data; ours is source-free).")
add(key="tuia2016darsreview", cat="H", type="article", authors="Tuia, Devis and Persello, Claudio and Bruzzone, Lorenzo", year=2016,
    title="Domain Adaptation for the Classification of Remote Sensing Data: An Overview of Recent Advances", venue="IEEE Geoscience and Remote Sensing Magazine", details="vol. 4, no. 2, pp. 41--57", doi="10.1109/MGRS.2016.2548504",
    url="https://infoscience.epfl.ch/record/225237",
    summary="Overview of domain adaptation in remote sensing.",
    use="Domain shift is a long-standing remote-sensing problem.")

# ---------------- I. conformal prediction
add(key="vovk2022alrw", cat="I", type="book", authors="Vovk, Vladimir and Gammerman, Alexander and Shafer, Glenn", year=2022,
    title="Algorithmic Learning in a Random World", venue="Springer", details="2nd edition", doi="10.1007/978-3-031-06649-8",
    url="https://econpapers.repec.org/RePEc:spr:sprbok:978-3-031-06649-8",
    summary="Foundational book on conformal prediction (1st edition 2005).",
    use="Conformal prediction foundations.")
add(key="lei2018distfree", cat="I", type="article", authors="Lei, Jing and G'Sell, Max and Rinaldo, Alessandro and Tibshirani, Ryan J. and Wasserman, Larry", year=2018,
    title="Distribution-Free Predictive Inference for Regression", venue="Journal of the American Statistical Association", details="vol. 113, no. 523, pp. 1094--1111", doi="10.1080/01621459.2017.1307116",
    url="https://ideas.repec.org/a/taf/jnlasa/v113y2018i523p1094-1111.html",
    summary="Split conformal inference and its guarantees.",
    use="Split-conformal background.")
add(key="angelopoulos2023gentle", cat="I", type="article", authors="Angelopoulos, Anastasios N. and Bates, Stephen", year=2023,
    title="Conformal Prediction: A Gentle Introduction", venue="Foundations and Trends in Machine Learning", details="vol. 16, no. 4, pp. 494--591", doi="10.1561/2200000101",
    url="https://arxiv.org/abs/2107.07511",
    summary="Accessible introduction to conformal prediction and risk control (arXiv title: A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification).",
    use="Method section reference for conformal ideas.")
add(key="bates2021rcps", cat="I", type="article", authors="Bates, Stephen and Angelopoulos, Anastasios and Lei, Lihua and Malik, Jitendra and Jordan, Michael", year=2021,
    title="Distribution-free, Risk-controlling Prediction Sets", venue="Journal of the ACM", details="vol. 68, no. 6, art. 43", doi="10.1145/3478535",
    url="https://arxiv.org/abs/2101.02703",
    summary="High-probability control of general risks, including segmentation examples.",
    use="Alternative (high-probability) guarantee; discussion of stronger per-flood guarantees.")
add(key="angelopoulos2024crc", cat="I", type="inproceedings", authors="Angelopoulos, Anastasios N. and Bates, Stephen and Fisch, Adam and Lei, Lihua and Schuster, Tal", year=2024,
    title="Conformal Risk Control", venue="International Conference on Learning Representations (ICLR)", details="", doi="",
    url="https://arxiv.org/abs/2208.02814",
    summary="Controls the expected value of any monotone loss (e.g. false negative rate) with a simple calibration rule.",
    use="Core of our miss-rate guarantee (FNR control with floods as exchangeable units).")
add(key="angelopoulos2021ltt", cat="I", type="article", authors="Angelopoulos, Anastasios N. and Bates, Stephen and Cand{\\`e}s, Emmanuel J. and Jordan, Michael I. and Lei, Lihua", year=2021,
    title="Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control", venue="arXiv preprint arXiv:2110.01052", details="", doi="",
    url="https://arxiv.org/abs/2110.01052",
    summary="Multiple-testing framework for risk control over many parameters.",
    use="Extension path for controlling several risks (miss rate and false alarms) at once.")
add(key="tibshirani2019covshift", cat="I", type="inproceedings", authors="Tibshirani, Ryan J. and Barber, Rina Foygel and Cand{\\`e}s, Emmanuel J. and Ramdas, Aaditya", year=2019,
    title="Conformal Prediction Under Covariate Shift", venue="Advances in Neural Information Processing Systems 32 (NeurIPS)", details="", doi="",
    url="https://papers.nips.cc/paper/8522-conformal-prediction-under-covariate-shift",
    summary="Weighted conformal prediction when test inputs are shifted.",
    use="Discussion: when floods are not exchangeable (cross-dataset shift).")
add(key="barber2023beyond", cat="I", type="article", authors="Barber, Rina Foygel and Cand{\\`e}s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.", year=2023,
    title="Conformal prediction beyond exchangeability", venue="The Annals of Statistics", details="vol. 51, no. 2, pp. 816--845", doi="10.1214/23-AOS2276",
    url="https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.pdf",
    summary="Coverage bounds when exchangeability fails, with weighting.",
    use="Limitations: guarantee degrades gracefully under distribution drift.")
add(key="dunn2023hierarchical", cat="I", type="article", authors="Dunn, Robin and Wasserman, Larry and Ramdas, Aaditya", year=2023,
    title="Distribution-Free Prediction Sets for Two-Layer Hierarchical Models", venue="Journal of the American Statistical Association", details="vol. 118, no. 544, pp. 2491--2502", doi="10.1080/01621459.2022.2060112",
    url="https://arxiv.org/abs/1809.07441",
    summary="Conformal prediction when data come in groups (exchangeable groups, dependent members).",
    use="Theory for treating floods (not pixels) as the exchangeable unit, as we do.")
add(key="mossina2024conformalseg", cat="I", type="inproceedings", authors="Mossina, Luca and Dalmau, Joseba and And{\\'e}ol, L{\\'e}o", year=2024,
    title="Conformal Semantic Image Segmentation: Post-hoc Quantification of Predictive Uncertainty", venue="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)", details="", doi="",
    url="https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/html/Mossina_Conformal_Semantic_Image_Segmentation_Post-hoc_Quantification_of_Predictive_Uncertainty_CVPRW_2024_paper.html",
    summary="Conformal risk control for multi-class semantic segmentation (images as units).",
    use="Closest conformal-segmentation work; ours calibrates across flood events under domain shift.")
add(key="konidakis2025cpflood", cat="I", type="inproceedings", authors="Konidakis, Ioannis and Panayidou, Klea and Tsagkatakis, Grigorios and Tsakalides, Panagiotis", year=2025,
    title="Uncertainty-Aware Flood Segmentation from Sentinel 2 Observations with Conformal Prediction", venue="IEEE International Geoscience and Remote Sensing Symposium (IGARSS)", details="", doi="",
    url="https://users.ics.forth.gr/~tsakalid/PAPERS/CNFRS/2025-IGARSS-Konidakis.pdf",
    summary="Inductive CP and CV+ prediction sets for OmbriaNet flood segmentation on Sentinel-2 (OMBRIA).",
    use="Only conformal flood-segmentation paper found: optical, pixel-level sets, no domain shift or SAR; key contrast for our novelty.")

add(key="li2026crcflood", cat="I", type="article", authors="Li, Yu and Matgen, Patrick and Chini, Marco", year=2026,
    title="Quantifying and Communicating Uncertainty in SAR-Based Flood Mapping via Density-Aware Neural Networks and Conformal Risk Control", venue="IEEE Transactions on Geoscience and Remote Sensing", details="vol. 64, pp. 1-20", doi="10.1109/TGRS.2026.3661208",
    url="https://doi.org/10.1109/TGRS.2026.3661208",
    summary="Density-aware network separates model-knowledge and data uncertainty in SAR flood maps; conformal risk control sets out-of-distribution abstention and set-valued predictions (found in the final novelty check, 10 Nov 2026).",
    use="Closest SAR work using conformal risk control; ours differs by calibrating a flood-level miss-rate bound across events and combining it with label-free test-time adaptation.")

# ---------------- J. uncertainty and calibration
add(key="guo2017calibration", cat="J", type="inproceedings", authors="Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.", year=2017,
    title="On Calibration of Modern Neural Networks", venue="International Conference on Machine Learning (ICML)", details="PMLR 70", doi="",
    url="https://proceedings.mlr.press/v70/guo17a",
    summary="Modern networks are over-confident; temperature scaling.",
    use="Why a fixed 0.5 threshold cannot be trusted under shift.")
add(key="gal2016dropout", cat="J", type="inproceedings", authors="Gal, Yarin and Ghahramani, Zoubin", year=2016,
    title="Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", venue="International Conference on Machine Learning (ICML)", details="PMLR 48", doi="",
    url="https://proceedings.mlr.press/v48/gal16.html",
    summary="MC dropout for uncertainty.",
    use="Heuristic uncertainty methods (no guarantees) for contrast.")
add(key="lakshminarayanan2017ensembles", cat="J", type="inproceedings", authors="Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles", year=2017,
    title="Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", venue="Advances in Neural Information Processing Systems 30 (NeurIPS)", details="", doi="",
    url="https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html",
    summary="Deep ensembles for uncertainty.",
    use="Heuristic uncertainty for contrast; our 3 model seeds are a small ensemble.")
add(key="chen2026uncertaintyflood", cat="J", type="misc", authors="Chen, Chi-Ju and Yang, Chun-Hao and Weng, Wei and Wang, Li-Pen", year=2026,
    title="Integrating predictive uncertainty into satellite-based flood mapping", venue="EGUsphere preprint (under review for Natural Hazards and Earth System Sciences)", details="egusphere-2026-4233", doi="",
    url="https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4233/",
    summary="Evidential deep learning vs ensembles and MC dropout for flood mapping on WorldFloods.",
    use="Very recent flood-uncertainty work without guarantees; contrast with conformal control.")

# ---------------- K. backbones and training
add(key="ronneberger2015unet", cat="K", type="inproceedings", authors="Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas", year=2015,
    title="U-Net: Convolutional Networks for Biomedical Image Segmentation", venue="Medical Image Computing and Computer-Assisted Intervention (MICCAI)", details="LNCS 9351, pp. 234--241", doi="10.1007/978-3-319-24574-4_28",
    url="https://arxiv.org/abs/1505.04597",
    summary="Encoder-decoder with skip connections for segmentation.",
    use="Our first backbone (U-Net from scratch).")
add(key="he2016resnet", cat="K", type="inproceedings", authors="He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian", year=2016,
    title="Deep Residual Learning for Image Recognition", venue="Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)", details="pp. 770--778", doi="10.1109/CVPR.2016.90",
    url="https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
    summary="Residual networks (ResNet).",
    use="ResNet-34 encoder of our second backbone.")
add(key="ioffe2015bn", cat="K", type="inproceedings", authors="Ioffe, Sergey and Szegedy, Christian", year=2015,
    title="Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", venue="International Conference on Machine Learning (ICML)", details="PMLR 37, pp. 448--456", doi="",
    url="https://proceedings.mlr.press/v37/ioffe15",
    summary="Batch normalisation.",
    use="BN layers are what AdaBN and TENT adapt.")
add(key="deng2009imagenet", cat="K", type="inproceedings", authors="Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li", year=2009,
    title="ImageNet: A large-scale hierarchical image database", venue="Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)", details="pp. 248--255", doi="10.1109/CVPR.2009.5206848",
    url="https://mlanthology.org/cvpr/2009/deng2009cvpr-imagenet",
    summary="ImageNet dataset.",
    use="Pre-training of the ResNet-34 encoder.")
add(key="iakubovskii2019smp", cat="K", type="misc", authors="Iakubovskii, Pavel", year=2019,
    title="Segmentation Models Pytorch", venue="GitHub repository", details="", doi="",
    url="https://github.com/qubvel-org/segmentation_models.pytorch",
    summary="Library of segmentation architectures with pretrained encoders.",
    use="Implementation of the ResNet-34 U-Net.")
add(key="loshchilov2019adamw", cat="K", type="inproceedings", authors="Loshchilov, Ilya and Hutter, Frank", year=2019,
    title="Decoupled Weight Decay Regularization", venue="International Conference on Learning Representations (ICLR)", details="", doi="",
    url="https://arxiv.org/abs/1711.05101",
    summary="AdamW optimiser.",
    use="Optimiser used for training.")

# ---------------- L. statistics and practice
add(key="wilcoxon1945", cat="L", type="article", authors="Wilcoxon, Frank", year=1945,
    title="Individual Comparisons by Ranking Methods", venue="Biometrics Bulletin", details="vol. 1, no. 6, pp. 80--83", doi="10.2307/3001968",
    url="https://link.springer.com/doi/10.1007/978-1-4612-4380-9_15",
    summary="Signed-rank test.",
    use="Our paired one-sided tests over events.")
add(key="holm1979", cat="L", type="article", authors="Holm, Sture", year=1979,
    title="A Simple Sequentially Rejective Multiple Test Procedure", venue="Scandinavian Journal of Statistics", details="vol. 6, no. 2, pp. 65--70", doi="",
    url="https://bibbase.org/network/publication/holm-asimplesequentiallyrejectivemultipletestprocedure-1979",
    summary="Holm step-down correction for multiple tests.",
    use="Correction across H1-H3.")
add(key="demsar2006", cat="L", type="article", authors="Dem{\\v{s}}ar, Janez", year=2006,
    title="Statistical Comparisons of Classifiers over Multiple Data Sets", venue="Journal of Machine Learning Research", details="vol. 7, pp. 1--30", doi="",
    url="https://www.jmlr.org/papers/v7/demsar06a.html",
    summary="Recommends Wilcoxon signed-rank for comparing methods over multiple datasets.",
    use="Justifies Wilcoxon with events as units.")
add(key="nosek2018prereg", cat="L", type="article", authors="Nosek, Brian A. and Ebersole, Charles R. and DeHaven, Alexander C. and Mellor, David T.", year=2018,
    title="The preregistration revolution", venue="Proceedings of the National Academy of Sciences", details="vol. 115, no. 11, pp. 2600--2606", doi="10.1073/pnas.1708274114",
    url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5856500",
    summary="Argues for preregistering hypotheses and analysis plans.",
    use="Justifies our pre-registered MMFlood test.")

# ---------------- M. data sources and context
add(key="torres2012sentinel1", cat="M", type="article", authors="Torres, Ramon and others", year=2012,
    title="GMES Sentinel-1 mission", venue="Remote Sensing of Environment", details="vol. 120, pp. 9--24", doi="10.1016/j.rse.2011.05.028",
    url="https://sentiwiki.copernicus.eu/__attachments/1673968/GMES_Sentinel_1_mission.pdf",
    summary="Sentinel-1 C-band SAR mission description.",
    use="Data source description.")
add(key="copdem2024handbook", cat="M", type="manual", authors="{European Space Agency}", year=2024,
    title="Copernicus DEM Product Handbook (GLO-30)", venue="Copernicus Data Space Ecosystem", details="", doi="",
    url="https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
    summary="Copernicus GLO-30 DEM documentation.",
    use="Elevation and slope inputs; source of GLO-30 HAND.")
add(key="tellman2021exposure", cat="M", type="article", authors="Tellman, B. and Sullivan, J. A. and Kuhn, C. and Kettner, A. J. and Doyle, C. S. and Brakenridge, G. R. and Erickson, T. A. and Slayback, D. A.", year=2021,
    title="Satellite imaging reveals increased proportion of population exposed to floods", venue="Nature", details="vol. 596, pp. 80--86", doi="10.1038/s41586-021-03695-w",
    url="https://ideas.repec.org/a/nat/nature/v596y2021i7870d10.1038_s41586-021-03695-w.html",
    summary="Global Flood Database: 913 events (2000-2018) show rising population exposure.",
    use="Introduction: why rapid, reliable flood maps matter.")
add(key="islam2026sylhet", cat="M", type="article", authors="Islam, Md Sariful and Mitra, Juthi Rani and Rahman, Munshi Khaledur", year=2026,
    title="Quantifying Flash Flood Inundation and Assessing Damage Using Satellite Earth Observations: The Case of 2022 Flash Flood in Bangladesh", venue="Geomatics, Natural Hazards and Risk", details="vol. 17, no. 1", doi="10.1080/19475705.2026.2614729",
    url="https://dspace.mit.edu/handle/1721.1/165213",
    summary="Sentinel-1 mapping of the 2022 Sylhet flash flood (>1,600 km2 inundated) and damage.",
    use="Optional Bangladesh case study context and comparison.")

# ---------------- added after re-checking the earlier novelty notes (9 Nov 2026)
add(key="chen2026cimttda", cat="E", type="article", authors="Chen, Wantai and Wang, Chong and Zhao, Zimeng and Li, Xiaofeng", year=2026,
    title="Improving Cross-Event Generalization for SAR-Based Coastal Inundation Mapping via Test-Time Domain Adaptation", venue="Journal of Geophysical Research: Machine Learning and Computation", details="vol. 3, no. 4, art. e2025JH001145", doi="10.1029/2025JH001145",
    url="https://api.crossref.org/works/10.1029/2025JH001145",
    summary="CIM-TTDA: adapts a pretrained SAR water-segmentation model to each new coastal flood with pseudo-labels and EMA (teacher-student) updates; +0.15 mean IoU on 3 cyclone floods (8 scenes).",
    use="Closest prior work: test-time adaptation of SAR flood models across events. Our 'self-training' baseline follows this recipe; we add physics-checked pseudo-labels, AdaBN, a flood-level miss-rate guarantee and a 3-dataset, pre-registered evaluation.")
add(key="sami2024evanet", cat="C", type="inproceedings", authors="Sami, Mirza Tanzim and Yan, Da and Adhikari, Saugat and Yuan, Lyuheng and Han, Jiao and Jiang, Zhe and Khalil, Jalal and Zhou, Yang", year=2024,
    title="EvaNet: Elevation-Guided Flood Extent Mapping on Earth Imagery", venue="Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI)", details="pp. 1200--1208", doi="10.24963/ijcai.2024/133",
    url="https://ijcai.org/proceedings/2024/133",
    summary="Segmentation network with an elevation-guided (gravity) loss so lower neighbours of flooded pixels are also flooded.",
    use="Physics/terrain constraint used in training; ours applies terrain checks label-free at test time.")
