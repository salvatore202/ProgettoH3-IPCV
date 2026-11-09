# Segmentazione Semantica e di Istanza su COCO 2017 e Cityscapes

**Gruppo H3** · Image Processing & Computer Vision 2026

<!-- TODO: aggiungere i nomi dei componenti del gruppo -->

<p align="center">
  <img src="docs/cityscapes_input.jpg" alt="Cityscapes - immagine di input" width="32%">
  <img src="docs/semantic_labels.jpg" alt="Cityscapes - etichette semantiche" width="32%">
  <img src="docs/instance_labels.jpg" alt="Cityscapes - etichette di istanza" width="32%">
  <br>
  <sub>Input (Cityscapes), segmentazione semantica (una classe per pixel) e segmentazione di istanza (una maschera e un box per oggetto).</sub>
</p>

## Panoramica

Il progetto confronta diversi approcci alla **segmentazione semantica** e **di istanza**, dall'addestramento da zero di una rete convoluzionale fino all'uso di modelli di fondazione senza ulteriore training. I dataset utilizzati sono **COCO 2017 (Stuff)** e **Cityscapes**.

Le attività svolte sono:

1. Implementazione e addestramento **da zero** (con *babysitting*) di una **FCN** su COCO.
2. Confronto di diverse **funzioni di loss** e delle loro combinazioni.
3. **Transfer learning**: fine-tuning della FCN su Cityscapes.
4. **Transfer learning**: fine-tuning di **YOLO11** su Cityscapes.
5. **Foundation model**: uso di **Mask2Former** senza retraining.
6. **Valutazione** di tutti i metodi con metriche semantiche e di istanza.
7. **Explainable AI**: *Activation Maximization* per interpretare le feature apprese dalla FCN.

Il codice si trova nel [notebook Kaggle del progetto](https://www.kaggle.com/code/gianmarcoiovino/notebookprogettoh3).

## Dataset

### COCO 2017 (Stuff)

[COCO](https://cocodataset.org/) raccoglie immagini di scene quotidiane. Le annotazioni **COCO-Stuff** ([repository](https://github.com/nightrome/cocostuff)) coprono le immagini di COCO 2017 con **171 classi**: 91 classi *stuff* (regioni amorfe come cielo, erba, neve, sabbia, muri, pavimenti, strade, alberi, acqua) e le 80 classi *thing* di COCO (persone, auto, cani, sedie, ...). Lo split è quello di COCO 2017: 118k immagini di training e 5k di validazione.

<p align="center">
  <img src="docs/coco_stuff_label_maps.jpg" alt="COCO-Stuff - mappe di etichette" width="38%">
  <img src="docs/coco_val2017_cats.jpg" alt="COCO val2017 - gatti su un divano" width="28%">
  <img src="docs/coco_val2017_kitchen.jpg" alt="COCO val2017 - cucina" width="28%">
  <br>
  <sub>A sinistra: mappe di etichette di COCO-Stuff. A destra: due immagini di esempio da COCO val2017.</sub>
</p>

### Cityscapes

[Cityscapes](https://www.cityscapes-dataset.com/) contiene scene urbane di guida autonoma riprese in 50 città a risoluzione 2048×1024.

- **Annotazioni:** 5.000 immagini con annotazioni fini (2.975 train, 500 val, 1.525 test con etichette nascoste) e 20k con annotazioni grossolane.
- **Segmentazione semantica:** 30 classi definite, 19 usate per training e valutazione (road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle).
- **Segmentazione di istanza:** 8 classi *thing* (person, rider, car, truck, bus, train, motorcycle, bicycle), con molti oggetti piccoli e occlusi.

<p align="center">
  <img src="docs/cityscapes_semantic_sample.jpg" alt="Cityscapes - annotazione fine semantica" width="30%">
  <img src="docs/cityscapes_instances_sample.jpg" alt="Cityscapes - annotazione fine di istanza" width="30%">
  <br>
  <sub>Esempi di annotazioni fini di Cityscapes: semantiche (sinistra) e di istanza (destra).</sub>
</p>

## Pipeline

```mermaid
flowchart LR
    A["COCO 2017 Stuff"] --> B["FCN da zero (babysitting)"]
    B --> C["Confronto loss: CE, Jaccard, Dice, Focal, combinazioni"]
    C --> D["Fine-tuning FCN su Cityscapes"]
    B --> J["Activation Maximization"]
    E["YOLO11 pre-addestrato"] --> F["Fine-tuning YOLO11 su Cityscapes"]
    G["Mask2Former pre-addestrato"] --> H["Inferenza senza retraining"]
    D --> I["Valutazione"]
    F --> I
    H --> I
```

## Metodi

### 1. FCN da zero su COCO

Viene implementata e addestrata da zero una [Fully Convolutional Network](https://docs.pytorch.org/vision/main/models/fcn.html) sul dataset COCO 2017 (Stuff). L'addestramento è seguito con la tecnica del *babysitting*, cioè un monitoraggio attento del training: sanity check iniziali, overfit su un piccolo subset, osservazione delle curve di loss su train e validation, tuning degli iperparametri.

### 2. Funzioni di loss

La FCN viene addestrata con diverse loss, singolarmente e in combinazione:

- **Cross-Entropy**, eventualmente pesata per classe e con i pixel non etichettati ignorati: $\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i}\sum_{c} y_{i,c}\log p_{i,c}$
- **Jaccard (IoU) Loss**, che ottimizza direttamente l'IoU: $\mathcal{L}_{J} = 1 - \frac{\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i - \sum_i p_i g_i}$
- **Dice Loss**, robusta alle classi rare: $\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}$
- **Focal Loss**, che si concentra sui pixel difficili: $\mathcal{L}_{F} = -\alpha\,(1-p_t)^{\gamma}\log p_t$
- **Combinazioni**, ad esempio $\mathcal{L} = \lambda_1\mathcal{L}_{CE} + \lambda_2\mathcal{L}_{Dice}$

### 3. Fine-tuning della FCN su Cityscapes

I pesi della FCN addestrata su COCO vengono usati come inizializzazione e adattati a Cityscapes (transfer learning).

### 4. Fine-tuning di YOLO11 su Cityscapes

[YOLO11](https://docs.ultralytics.com/models/yolo11/), nella variante per la segmentazione di istanza, viene ri-addestrato su Cityscapes a partire da pesi pre-addestrati. La loss è multi-task: $L = L_{cls} + L_{box} + L_{mask}$.

### 5. Mask2Former (foundation model)

[Mask2Former](https://github.com/facebookresearch/Mask2Former) viene utilizzato **senza alcun retraining**, applicando direttamente un checkpoint pre-addestrato e valutandone le predizioni con le stesse metriche degli altri metodi.

### 6. Explainable AI: Activation Maximization

Per interpretare la FCN si genera, tramite gradient ascent sull'immagine di ingresso, l'input che massimizza l'attivazione di un filtro o di un layer selezionato:

$$x^{*} = \arg\max_{x}\; a_{l,k}(x) - \lambda\,R(x)$$

dove $a_{l,k}$ è l'attivazione del filtro $k$ nel layer $l$ e $R(x)$ è un termine di regolarizzazione. Le immagini ottenute per diversi layer e filtri vengono analizzate per capire quali strutture catturano (bordi, texture, regioni semantiche).

<!-- TODO: inserire qui le immagini di Activation Maximization -->

## Valutazione

**Segmentazione semantica**

- **IoU** (e mIoU, media sulle classi): $\text{IoU} = \frac{TP}{TP + FP + FN}$. Il mIoU è la metrica di riferimento, perché la pixel accuracy è dominata dalle classi grandi come road e sky.
- **F1 Score** per classe: $F_1 = \frac{2\,TP}{2\,TP + FP + FN}$
- **Matrice di confusione** a livello di pixel tra le classi

**Segmentazione di istanza**

- **Precision**: $\frac{TP}{TP + FP}$ e **Recall**: $\frac{TP}{TP + FN}$, con il match tra predizioni e ground truth basato sull'IoU tra maschere
- **Average Precision (AP)**: area sotto la curva Precision-Recall di una classe
- **mean Average Precision (mAP)**: media dell'AP sulle classi, calcolata a una soglia fissa (AP50) o mediata su più soglie di IoU (@[.5:.95])

## Risultati

<!-- TODO: compilare le tabelle e specificare lo split di valutazione -->

### Segmentazione semantica

| Modello | Training | Loss | mIoU | F1 |
|---|---|---|---|---|
| FCN | COCO (da zero) | Cross-Entropy | TBD | TBD |
| FCN | COCO (da zero) | Jaccard | TBD | TBD |
| FCN | COCO (da zero) | Dice | TBD | TBD |
| FCN | COCO (da zero) | Focal | TBD | TBD |
| FCN | COCO (da zero) | Combinazione | TBD | TBD |
| FCN | Fine-tuning su Cityscapes | TBD | TBD | TBD |
| Mask2Former | Nessuno (zero-shot) | - | TBD | TBD |

### Segmentazione di istanza

| Modello | Training | Precision | Recall | AP | mAP |
|---|---|---|---|---|---|
| YOLO11 | Fine-tuning su Cityscapes | TBD | TBD | TBD | TBD |
| Mask2Former | Nessuno (zero-shot) | TBD | TBD | TBD | TBD |

<!-- TODO: aggiungere esempi qualitativi (immagine, ground truth, predizione) e matrici di confusione -->

## Contenuto della repository

```text
.
├── README.md
└── docs/    # immagini usate in questo README
```

## Riferimenti

- T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
- H. Caesar, J. Uijlings, V. Ferrari, *COCO-Stuff: Thing and Stuff Classes in Context*, CVPR 2018.
- M. Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
- A. Kirillov et al., *Panoptic Segmentation*, CVPR 2019.
- J. Long, E. Shelhamer, T. Darrell, *Fully Convolutional Networks for Semantic Segmentation*, CVPR 2015.
- B. Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation* (Mask2Former), CVPR 2022.
- Ultralytics, *YOLO11*.

## Crediti e licenze

Le immagini nella cartella `docs/` sono estratte dalle slide del corso *IPCV 2026 – Datasets, Tasks, Losses & Metrics*. Provengono dai dataset Cityscapes e COCO (val2017), dal repository COCO-Stuff e dalla figura di Kirillov et al. (*Panoptic Segmentation*, CVPR 2019), e restano soggette ai termini d'uso originali.
