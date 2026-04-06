# LeadSense NJ — Technical Specification

## AI-Driven Lead Contamination Risk Prediction for New Jersey Water Infrastructure

**Project Codename:** LeadSense NJ
**Target Venue:** NJBDA Annual Symposium (Student Poster Session)
**Team Size:** 1–3 undergraduate students
**Estimated Build Time:** 4–6 weeks

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Research Questions](#2-research-questions)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Data Sources & Acquisition](#4-data-sources--acquisition)
5. [Data Preprocessing Pipeline](#5-data-preprocessing-pipeline)
6. [Model 1: Multi-Modal Fusion Network](#6-model-1-multi-modal-fusion-network)
7. [Model 2: Graph Neural Network for Water Network Topology](#7-model-2-graph-neural-network-for-water-network-topology)
8. [Model 3: Uncertainty Quantification Layer](#8-model-3-uncertainty-quantification-layer)
9. [Model 4: Fairness-Constrained Pipe Replacement Optimizer](#9-model-4-fairness-constrained-pipe-replacement-optimizer)
10. [Model 5: Explainability Engine (SHAP)](#10-model-5-explainability-engine-shap)
11. [Model 6: LLM Policy Brief Generator](#11-model-6-llm-policy-brief-generator)
12. [Evaluation & Metrics](#12-evaluation--metrics)
13. [Demo Application](#13-demo-application)
14. [Tech Stack](#14-tech-stack)
15. [Directory Structure](#15-directory-structure)
16. [Risk Register & Mitigations](#16-risk-register--mitigations)
17. [Timeline](#17-timeline)
18. [References](#18-references)

---

## 1. Problem Statement

### 1.1 Background

Lead contamination in drinking water is a silent public health crisis. Lead is a potent neurotoxin — there is **no safe level of lead exposure** (CDC, WHO). Children exposed to lead suffer irreversible cognitive damage, reduced IQ, behavioral disorders, and developmental delays. Adults face kidney damage, cardiovascular disease, and reproductive harm.

### 1.2 Why New Jersey

New Jersey has one of the oldest water infrastructures in the United States:

- **Over 350,000 lead service lines (LSLs)** remain in active use across the state (NJ DEP estimate, 2023).
- Newark spent **$75 million** between 2019–2021 replacing ~23,000 LSLs after a federal emergency.
- NJ passed the **"Get the Lead Out" law (S2024/A4066, 2021)** mandating all LSLs be replaced within 10 years — but utilities lack the tools to prioritize which blocks to dig up first.
- Low-income communities and communities of color are **disproportionately affected** because they tend to live in older housing stock with legacy infrastructure (EPA EJ analysis).

### 1.3 The Gap

Current lead risk assessment is **reactive**: test water after people drink it, find lead, then scramble. There is no widely deployed **predictive system** that tells a utility "Block X has an 87% probability of dangerous lead levels — replace those pipes before anyone gets poisoned."

### 1.4 What LeadSense NJ Does

LeadSense NJ is a **multi-modal, graph-aware, uncertainty-quantified AI system** that predicts lead contamination risk at the census-block level across New Jersey, optimizes pipe replacement order under fairness constraints, and generates explainable policy briefs — all from publicly available data.

---

## 2. Research Questions

| # | Question | AI Method |
|---|---|---|
| RQ1 | Can multi-modal deep learning (tabular + satellite + temporal) predict census-block-level lead risk more accurately than single-modality baselines? | Multi-modal fusion network |
| RQ2 | Does modeling the water distribution network as a graph improve prediction over treating locations independently? | Graph Neural Network |
| RQ3 | How well-calibrated are the model's uncertainty estimates, and do they correlate with actual risk? | MC Dropout / Bayesian NN |
| RQ4 | Can we allocate pipe replacement budgets in a way that is both cost-efficient and demographically equitable? | Fairness-constrained optimization |
| RQ5 | What are the dominant drivers of lead contamination risk in NJ, and do they vary geographically? | SHAP explainability |

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                             │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌─────────┐ │
│  │ NJ DEP   │ │ Census   │ │ Sentinel-2│ │ EPA      │ │ NJ GIS  │ │
│  │ Water    │ │ ACS      │ │ Satellite │ │ SDWIS    │ │ Pipe    │ │
│  │ Quality  │ │ Demo-    │ │ Imagery   │ │ Violation│ │ Network │ │
│  │ Data     │ │ graphics │ │           │ │ Records  │ │ Maps    │ │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬────┘ │
│       │            │             │             │            │       │
└───────┼────────────┼─────────────┼─────────────┼────────────┼───────┘
        │            │             │             │            │
        ▼            ▼             ▼             ▼            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ Geo-spatial  │ │ Temporal     │ │ Image Tile   │                │
│  │ Join & Align │ │ Resampling   │ │ Extraction   │                │
│  │ (Census Block│ │ (Quarterly)  │ │ (per Block)  │                │
│  │  GEOID key)  │ │              │ │              │                │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                │
│         │                │                │                         │
└─────────┼────────────────┼────────────────┼─────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MULTI-MODAL FUSION MODEL                          │
│                                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                    │
│  │ Tabular    │  │ Temporal   │  │ Vision     │                    │
│  │ Encoder    │  │ Encoder    │  │ Encoder    │                    │
│  │ (MLP)      │  │ (LSTM/     │  │ (ResNet-18 │                    │
│  │            │  │  Trans.)   │  │  pretrained│                    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                    │
│        │               │               │                            │
│        └───────────┬───┴───────────────┘                            │
│                    ▼                                                 │
│           ┌────────────────┐                                        │
│           │ Cross-Attention│                                        │
│           │ Fusion Layer   │                                        │
│           └───────┬────────┘                                        │
│                   │                                                  │
│                   ▼                                                  │
│           ┌────────────────┐                                        │
│           │ Fused Feature  │                                        │
│           │ Vector (256-d) │                                        │
│           └───────┬────────┘                                        │
│                   │                                                  │
└───────────────────┼─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              GRAPH NEURAL NETWORK (Water Topology)                  │
│                                                                     │
│  Nodes = census blocks (with fused feature vectors)                 │
│  Edges = pipe connections / adjacency in water network              │
│                                                                     │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                │
│  │ GraphSAGE  │───▶│ GraphSAGE  │───▶│ GraphSAGE  │                │
│  │ Layer 1    │    │ Layer 2    │    │ Layer 3    │                │
│  │ (256→128)  │    │ (128→64)   │    │ (64→32)    │                │
│  └────────────┘    └────────────┘    └────────────┘                │
│                                            │                        │
│                                            ▼                        │
│                                   ┌────────────────┐               │
│                                   │ MC Dropout      │               │
│                                   │ Head (T=50)     │               │
│                                   │ → μ(risk), σ²   │               │
│                                   └───────┬────────┘               │
│                                           │                         │
└───────────────────────────────────────────┼─────────────────────────┘
                                            │
                    ┌───────────────────────┬┘
                    ▼                       ▼
┌────────────────────────┐  ┌─────────────────────────────────────────┐
│  SHAP EXPLAINABILITY   │  │  FAIRNESS-CONSTRAINED OPTIMIZER         │
│                        │  │                                         │
│  Per-block feature     │  │  Input: risk scores, replacement costs, │
│  attribution scores    │  │         demographic data, budget B      │
│  → top risk drivers    │  │                                         │
│  → geographic patterns │  │  Objective: maximize total risk reduced │
│                        │  │  Subject to:                            │
│                        │  │    - Σ cost_i · x_i ≤ B                 │
│                        │  │    - demographic parity constraint      │
│                        │  │    - min coverage per county            │
│                        │  │                                         │
│                        │  │  Output: ordered replacement schedule   │
└───────────┬────────────┘  └──────────────────┬──────────────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM POLICY BRIEF GENERATOR                       │
│                                                                     │
│  Input: risk score, uncertainty, SHAP values, fairness metrics,     │
│         replacement schedule, demographic context                   │
│                                                                     │
│  Output: plain-English per-municipality policy brief                │
│          "Camden Block Group 340070023001 — Risk: HIGH (0.87 ±0.04)│
│           Primary drivers: housing pre-1950 (SHAP +0.31),          │
│           low pH water chemistry (SHAP +0.22)..."                  │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DEMO APPLICATION                      │
│                                                                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ Interactive  │ │ Risk Detail  │ │ Fairness     │ │ Policy    │ │
│  │ NJ Map      │ │ Panel        │ │ Dashboard    │ │ Brief     │ │
│  │ (Folium/    │ │ (per block)  │ │ (equity      │ │ Viewer    │ │
│  │  Plotly)    │ │              │ │  metrics)    │ │           │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Sources & Acquisition

### 4.1 NJ DEP Safe Drinking Water Act (SDWA) Data

| Field | Detail |
|---|---|
| **Source** | NJ Department of Environmental Protection, Division of Water Supply & Geoscience |
| **URL** | https://www.nj.gov/dep/watersupply/ and https://sdwis.epa.gov/ (federal mirror) |
| **What it contains** | Water system violations, lead & copper rule (LCR) sampling results, 90th percentile lead levels per public water system (PWS), action level exceedances |
| **Granularity** | Per PWS (a PWS can serve one town or many). Individual sample-level data available via OPRA request or EPA SDWIS federal database |
| **Key columns** | `PWS_ID`, `violation_type`, `contaminant_code` (1030 = lead), `compliance_period`, `sample_result_ppb`, `90th_percentile_ppb`, `action_level_exceedance` (boolean) |
| **Format** | CSV / Excel downloads, or EPA SDWIS REST API |
| **Volume** | ~600 public water systems in NJ, each with quarterly sampling records going back 20+ years |
| **Access method** | Direct download from EPA SDWIS Violation Report or NJ DEP NJEMS portal. For sample-level data, submit OPRA request or use EPA's `SDWIS/FED` API endpoint: `https://data.epa.gov/efservice/VIOLATION/PRIMACY_AGENCY_CODE/NJ/CSV` |
| **Preprocessing needed** | Parse PWS service area boundaries to map violations to census blocks (see Section 5.2) |

### 4.2 US Census American Community Survey (ACS)

| Field | Detail |
|---|---|
| **Source** | US Census Bureau, ACS 5-Year Estimates |
| **URL** | https://data.census.gov/ and API: https://api.census.gov/data/ |
| **What it contains** | Demographics per census block group: median income, race/ethnicity, housing age (year built), % owner-occupied, % children under 6, poverty rate, education level |
| **Granularity** | Census block group (~600-3,000 people per group). NJ has ~6,320 block groups |
| **Key tables** | `B25034` (year structure built), `B19013` (median household income), `B02001` (race), `B17001` (poverty), `B09001` (children under 18) |
| **Critical variable** | **Year structure built (B25034)** — houses built before 1986 likely have lead solder; before 1950 likely have lead service lines. This is the single strongest predictor of lead risk |
| **API example** | `https://api.census.gov/data/2022/acs/acs5?get=B25034_001E,B25034_010E,B25034_011E&for=block%20group:*&in=state:34` |
| **Format** | JSON via API, or CSV bulk download |
| **Volume** | ~6,320 rows (one per NJ block group) × ~30 feature columns |

### 4.3 Sentinel-2 Satellite Imagery

| Field | Detail |
|---|---|
| **Source** | European Space Agency (ESA) Copernicus Open Access Hub / Microsoft Planetary Computer |
| **URL** | https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a |
| **What it contains** | Multispectral satellite imagery at 10m ground resolution (visible bands) and 20m (vegetation/SWIR bands) |
| **Why it matters** | Older neighborhoods have visually distinct characteristics: denser building footprints, older roof materials (tar/asphalt vs. modern), narrower streets, less tree canopy, more impervious surface. A CNN can learn these as proxy indicators for infrastructure age, which correlates with lead pipe prevalence |
| **Bands used** | B02 (Blue, 10m), B03 (Green, 10m), B04 (Red, 10m), B08 (NIR, 10m) — 4-channel input to CNN |
| **Tile size** | 256×256 pixels at 10m/px = 2.56 km × 2.56 km per tile. One tile covers ~1-4 census block groups depending on urban density |
| **Access method** | Microsoft Planetary Computer STAC API. Free, no login required for Sentinel-2 L2A (atmospherically corrected). Use `pystac-client` + `planetary-computer` Python packages |
| **Volume** | ~6,320 tiles (one per block group centroid), each 256×256×4 = 262 KB. Total: ~1.6 GB |
| **Cloud filtering** | Select scenes with <10% cloud cover. Use `s2:nodata_pixel_percentage` and `eo:cloud_cover` STAC properties |
| **Temporal** | Use most recent cloud-free composite (median pixel value over 3-month window) to eliminate transient objects |

### 4.4 EPA SDWIS Federal Violation Database

| Field | Detail |
|---|---|
| **Source** | EPA Safe Drinking Water Information System (SDWIS/FED) |
| **URL** | https://www.epa.gov/enviro/sdwis-search and https://data.epa.gov/efservice/ |
| **What it contains** | Every drinking water violation in the US since 1993. Includes violation type (MCL = Maximum Contaminant Level, TT = Treatment Technique, MR = Monitoring/Reporting), contaminant, severity, return-to-compliance dates |
| **Key filter** | `PRIMACY_AGENCY_CODE = NJ`, `CONTAMINANT_CODE = 1030` (lead) or `1040` (copper) |
| **Temporal depth** | 1993–present. Critical for building the temporal feature encoder — we need violation *trajectories*, not just snapshots |
| **Format** | REST API returning CSV/JSON. Rate limited to 10,000 rows per request; paginate with `&rows=0:9999` |

### 4.5 NJ Water Utility GIS Pipe Maps

| Field | Detail |
|---|---|
| **Source** | Individual NJ water utilities (varies by municipality). Some publish via NJ Geographic Information Network (NJGIN) |
| **URL** | https://njgin.nj.gov/ and individual utility GIS portals |
| **What it contains** | Pipe network geometry: pipe segments with material type (lead, copper, galvanized, PVC), diameter, installation year, service connections |
| **Availability** | **Inconsistent.** Newark, Jersey City, and Trenton have published pipe inventories due to LSL replacement mandates. Many smaller utilities have not. This is a known data gap |
| **Fallback strategy** | Where pipe maps are unavailable, approximate the water network topology from: (a) road centerline networks (NJ DOT), since pipes typically follow roads; (b) OpenStreetMap road graph; (c) spatial adjacency (Queen contiguity) between census block groups. See Section 7.5 for details |
| **Format** | Shapefiles (.shp), GeoJSON, or ArcGIS Feature Service REST endpoints |

### 4.6 EPA Toxic Release Inventory (TRI) & Superfund Sites

| Field | Detail |
|---|---|
| **Source** | EPA TRI and CERCLIS (Superfund) databases |
| **URL** | https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files and https://www.epa.gov/enviro/sems-search |
| **What it contains** | Locations and chemical releases of industrial facilities. Proximity to lead-emitting facilities (smelters, battery recyclers, paint manufacturers) is a known risk factor for soil and water contamination |
| **Key fields** | Facility lat/lon, chemical released (`LEAD COMPOUNDS`), quantity released (lbs/year), media (air/water/land) |
| **Derived feature** | For each census block group: distance to nearest lead-emitting TRI facility, cumulative lead released within 5 km radius |

### 4.7 NOAA / NJ State Climatological Data

| Field | Detail |
|---|---|
| **Source** | NOAA Climate Data Online |
| **URL** | https://www.ncdc.noaa.gov/cdo-web/ |
| **What it contains** | Temperature, precipitation. Relevant because cold temperatures increase lead leaching (ice expansion cracks pipes, low-flow winter conditions increase contact time) |
| **Derived features** | Mean winter temperature, freeze-thaw cycle count, annual precipitation |

### 4.8 Summary: Feature Matrix

| Source | Features | Type | Count |
|---|---|---|---|
| Census ACS | Housing age distribution, income, race, poverty, children %, education | Tabular | ~20 |
| EPA SDWIS | Historical violation count, 90th percentile lead, years since last violation, compliance rate | Tabular + Temporal | ~8 |
| NJ DEP Water Quality | pH, alkalinity, chlorine residual, hardness, turbidity | Tabular + Temporal | ~6 |
| Sentinel-2 | 4-band satellite tile per block group | Image (256×256×4) | 1 tile |
| TRI/Superfund | Distance to facilities, cumulative releases | Tabular | ~4 |
| Climate | Winter temp, freeze-thaw cycles, precipitation | Tabular | ~3 |
| Pipe Network | Pipe age, material, diameter, network position (if available) | Graph | ~5 |
| **Total** | | | **~46 features + image + graph** |

---

## 5. Data Preprocessing Pipeline

### 5.1 Spatial Alignment

**All data must be aligned to a common geographic unit.** We use **Census Block Group (CBG)** as the spatial key, identified by the 12-digit GEOID (e.g., `340070023001` = NJ, Camden County, Tract 002300, Block Group 1).

**Steps:**

1. **Census data** is already at CBG level — direct join on GEOID.
2. **Water quality/violations** are at the PWS (Public Water System) level. Each PWS serves a geographic area. Map PWS → CBG using:
   - NJ DEP publishes PWS service area boundaries as shapefiles.
   - Spatial join: for each CBG centroid, find which PWS service area polygon contains it.
   - If a CBG overlaps multiple PWS areas, assign the PWS whose area covers the majority of the CBG polygon (area-weighted).
3. **Satellite tiles** are extracted per CBG centroid. For each CBG:
   - Get centroid lat/lon from Census TIGER/Line shapefiles.
   - Query Planetary Computer STAC API for the Sentinel-2 tile containing that point.
   - Crop a 256×256 pixel window centered on the centroid.
4. **TRI/Superfund** facilities have point coordinates. Compute distance features using `geopandas` `distance()` or `sklearn.neighbors.BallTree` for efficiency.
5. **Climate data** is at weather station level. Interpolate to CBG centroids using inverse distance weighting (IDW) from the nearest 3 stations.

### 5.2 Temporal Alignment

Align all time-varying data to **quarterly intervals** (Q1–Q4):

- Water quality samples: aggregate to quarterly 90th percentile and mean.
- Violations: binary indicator per quarter (any violation in this quarter? y/n) + cumulative count.
- Climate: quarterly mean temperature, quarterly precipitation total.
- Satellite: use the most recent annual composite (one image per block group per year).

**Lookback window:** 8 quarters (2 years) of temporal features fed into the temporal encoder.

### 5.3 Missing Data Strategy

| Data Source | Expected Missingness | Handling |
|---|---|---|
| Census ACS | <1% (well-covered) | Rare; impute with county median if any |
| Water quality samples | ~15-25% of CBGs have no direct sample (served by larger PWS) | Assign parent PWS value to all child CBGs |
| Satellite imagery | ~5% (persistent cloud cover) | Use nearest temporal composite; flag as imputed |
| Pipe network | ~60-70% of CBGs lack detailed pipe maps | Use fallback graph construction (Section 7.5) |
| TRI distances | 0% (complete for all CBGs) | N/A |

### 5.4 Target Variable Construction

**The label we are predicting:** Binary classification — does this CBG have **elevated lead risk** (1) or not (0)?

**Definition of "elevated lead risk":**

A CBG is labeled **positive** (risk = 1) if ANY of the following are true:
- The PWS serving it has had a **Lead and Copper Rule action level exceedance** (90th percentile > 15 ppb) in the past 5 years.
- The PWS serving it has had **any individual sample > 15 ppb** in the past 3 years.
- The CBG's **median housing year built is before 1950** AND the PWS serving it has a **water chemistry profile associated with lead leaching** (pH < 7.0 OR alkalinity < 30 mg/L).

**Why this composite label:** Using only violation records would miss CBGs where lead is present but utilities haven't been caught yet (reporting bias). The housing-age + water-chemistry rule captures **latent risk** based on known lead science — acidic, soft water leaches more lead from pipes.

**Label balance estimate:** ~20-30% positive based on NJ housing stock age distribution. Manageable class imbalance; addressed with weighted loss or SMOTE if needed.

### 5.5 Train/Test Split Strategy

**Spatial cross-validation** — NOT random split. Random splitting would leak spatial autocorrelation (neighboring blocks have similar risk).

Method: **Spatial K-Fold (K=5)** using `sklearn_extra` or manual implementation:
1. Cluster all 6,320 CBGs into 5 spatial clusters using K-Means on (lat, lon).
2. Each fold holds out one spatial cluster as test, trains on the other 4.
3. Report mean ± std of all metrics across 5 folds.

This ensures the model is evaluated on **geographically unseen regions**, which is the real deployment scenario.

---

## 6. Model 1: Multi-Modal Fusion Network

### 6.1 Purpose

Combine three heterogeneous data modalities (tabular features, temporal sequences, satellite imagery) into a single unified representation per census block group.

### 6.2 Tabular Encoder

**Input:** 41 scalar features per CBG (demographics, water chemistry, TRI distances, climate).

**Architecture:**

```
Input (41)
  → BatchNorm1d(41)
  → Linear(41, 128) → ReLU → Dropout(0.3)
  → Linear(128, 128) → ReLU → Dropout(0.3)
  → Linear(128, 64)
  → Output: tabular embedding (64-d)
```

**Design decisions:**
- BatchNorm first to handle features on wildly different scales (income in thousands vs. pH 6-9) without manual normalization. The network learns its own normalization.
- Two hidden layers are sufficient for tabular data of this width. Deeper networks overfit on <10K samples.
- Dropout 0.3 for regularization — also reused for MC Dropout uncertainty later (Section 8).

### 6.3 Temporal Encoder

**Input:** Sequence of 8 quarterly observations × 14 temporal features = tensor of shape `(8, 14)`.

Temporal features per quarter:
- 90th percentile lead level (ppb)
- Mean lead level (ppb)
- Number of samples taken
- Any violation (binary)
- Cumulative violation count
- pH (mean)
- Alkalinity (mean)
- Chlorine residual (mean)
- Hardness (mean)
- Turbidity (mean)
- Mean temperature
- Total precipitation
- Freeze-thaw cycle count
- Days below freezing

**Architecture (LSTM variant):**

```
Input (8 timesteps × 14 features)
  → LSTM(input_size=14, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
  → Take final hidden state: (64-d)
  → Linear(64, 64)
  → Output: temporal embedding (64-d)
```

**Alternative (Transformer variant, recommended if compute allows):**

```
Input (8 × 14)
  → Linear(14, 64) — project each timestep to 64-d
  → PositionalEncoding(max_len=8)
  → TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.3) × 2 layers
  → Mean pool across timesteps
  → Output: temporal embedding (64-d)
```

**Why both options:** LSTM is simpler, faster, and works well for sequences this short. Transformer is more expressive and handles missing timesteps more gracefully (masking), but is overkill for 8 steps. Start with LSTM, upgrade to Transformer if time permits.

### 6.4 Vision Encoder

**Input:** Sentinel-2 satellite tile, shape `(4, 256, 256)` — 4 channels (B, G, R, NIR).

**Architecture:**

```
Input (4, 256, 256)
  → Modify ResNet-18 first conv layer: Conv2d(4, 64, 7, stride=2, padding=3)
    (standard ResNet expects 3 channels; we replace the first layer to accept 4)
  → ResNet-18 backbone (pretrained on ImageNet, fine-tuned)
    → Remove final FC layer
    → Global Average Pooling
  → Output: 512-d feature vector
  → Linear(512, 128) → ReLU → Dropout(0.3)
  → Linear(128, 64)
  → Output: vision embedding (64-d)
```

**Design decisions:**
- **Why ResNet-18?** Small enough to fine-tune on a laptop GPU (11M params). ResNet-50 would be better but may be too heavy for a student team without a beefy GPU.
- **Why pretrained on ImageNet?** Even though satellite images =/= natural photos, low-level features (edges, textures, spatial frequencies) transfer well. This is empirically validated in remote sensing literature (Neumann et al., 2019).
- **4-channel modification:** We initialize the new Conv2d layer by copying the pretrained RGB weights for channels 0-2 and initializing channel 3 (NIR) as the mean of the RGB weights. This preserves the pretrained knowledge.
- **NIR channel value:** Near-infrared reveals vegetation health and impervious surface boundaries much better than visible light alone. Vegetation correlates with newer, better-maintained neighborhoods.

### 6.5 Cross-Attention Fusion

**Input:** Three embeddings — `tab_emb (64-d)`, `temp_emb (64-d)`, `vis_emb (64-d)`.

**Why not just concatenate?** Concatenation treats all modalities as equally important everywhere. Cross-attention lets the model learn which modality to pay attention to for each sample. For a rural CBG with no water quality data, the model can upweight satellite features. For an urban CBG with dense testing history, it can upweight temporal features.

**Architecture:**

```
Stack embeddings: (3, 64) — treat as sequence of 3 tokens

  → MultiheadAttention(embed_dim=64, num_heads=4)
    Q = K = V = stacked embeddings
    (self-attention across modalities)
  → LayerNorm → Feed-forward (64 → 128 → 64) → LayerNorm
  → Flatten: (3 × 64) = 192-d
  → Linear(192, 256) → ReLU → Dropout(0.3)
  → Output: fused feature vector (256-d)
```

**Interpretation:** After fusion, the 256-d vector encodes information from all three modalities, weighted by learned cross-modal attention. This vector is the input to the GNN.

### 6.6 Loss Function (Fusion Model Standalone)

If training the fusion model alone (without GNN):

```python
loss = BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]))
```

Where `class_weight = n_negative / n_positive` to handle class imbalance.

---

## 7. Model 2: Graph Neural Network for Water Network Topology

### 7.1 Purpose

Water contamination doesn't respect census block boundaries. Lead leaches from pipes, and **where you are in the distribution network matters**:
- Blocks at the **end of distribution lines** (dead-ends) have lower water flow, longer contact time with pipes, and higher lead levels.
- Blocks **downstream** from a contaminated segment inherit risk.
- Blocks near **treatment plants** benefit from fresher corrosion control chemicals.

A GNN captures these topological relationships that a standard tabular/image model cannot.

### 7.2 Graph Construction

**Nodes:** Each census block group is a node (N ≈ 6,320 for NJ).
**Node features:** The 256-d fused feature vector from the multi-modal fusion model.
**Edges:** Connections representing water network adjacency.

### 7.3 Edge Construction (When Pipe Data Is Available)

For municipalities with published pipe GIS data (Newark, Jersey City, Trenton, etc.):

1. Load pipe segment geometries (lines) from shapefiles.
2. Build a pipe network graph: intersections = graph nodes, pipe segments = graph edges.
3. For each census block group, identify all pipe segments within its boundary.
4. Two CBGs share an edge if a pipe segment crosses both their boundaries.
5. **Edge weight** = inverse of pipe segment length (shorter pipes = stronger connection = more contamination transfer).

### 7.4 Edge Construction (Approximation When Pipe Data Is Unavailable)

For ~60-70% of NJ where pipe maps don't exist:

**Method: Road-Network Proxy**

Rationale: Water mains in urban/suburban NJ are almost always laid along road rights-of-way. The road network is a strong proxy for the pipe network.

1. Download NJ road centerlines from NJGIN (https://njgin.nj.gov/) or OpenStreetMap via `osmnx`.
2. Build road network graph using `osmnx.graph_from_polygon(cbg_polygon)`.
3. Two CBGs share an edge if a road segment connects them.
4. Edge weight = inverse of road distance between CBG centroids.

**Fallback: Spatial Adjacency**

If road network processing is too slow:

1. Use Queen contiguity: two CBGs share an edge if their polygons share any boundary point.
2. Build adjacency using `libpysal.weights.Queen.from_dataframe(gdf)`.
3. Edge weight = 1 / euclidean_distance(centroid_i, centroid_j).

### 7.5 Graph Neural Network Architecture

**Framework:** PyTorch Geometric (`torch_geometric`)

```
Input: Graph G = (V, E)
  V = 6,320 nodes, each with 256-d feature vector
  E = ~25,000 edges (estimated; avg degree ~8)

Layer 1: SAGEConv(256, 128, aggr='mean')
  → ReLU → Dropout(0.3)
  Each node aggregates its neighbors' features (1-hop)

Layer 2: SAGEConv(128, 64, aggr='mean')
  → ReLU → Dropout(0.3)
  Each node now has 2-hop neighborhood info

Layer 3: SAGEConv(64, 32, aggr='mean')
  → ReLU → Dropout(0.3)
  Each node now has 3-hop neighborhood info
  (3 hops ≈ 3 pipe segments away — meaningful contamination radius)

Prediction Head:
  → Linear(32, 16) → ReLU → Dropout(0.3)
  → Linear(16, 1) → Sigmoid
  → Output: P(lead_risk) per node
```

**Why GraphSAGE (SAGEConv)?**
- GraphSAGE uses **sampling-based aggregation**, which scales to large graphs. GCN requires full-graph Laplacian computation, which is memory-heavy.
- SAGEConv supports **inductive learning** — it can generalize to unseen nodes (new CBGs or new municipalities) without retraining.
- Mean aggregation is robust and interpretable.

### 7.6 Training Procedure

```python
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]))

# Spatial CV: train on 4 spatial folds, validate on 1
for epoch in range(200):
    model.train()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # Validate
    model.eval()
    val_auc = roc_auc_score(data.y[val_mask], out[val_mask])
    scheduler.step(val_auc)

    # Early stopping: patience=15 epochs
```

**End-to-end training:** The fusion model and GNN are trained jointly — gradients flow from the GNN loss back through the fusion encoders. This lets the satellite encoder learn features that are useful *in the context of network topology*, not just in isolation.

---

## 8. Model 3: Uncertainty Quantification Layer

### 8.1 Purpose

A risk prediction without uncertainty is irresponsible. If the model says "Block X has 72% lead risk" but the confidence interval is ±30%, that prediction is useless for policy. Judges who understand statistics will immediately ask "how confident are you?" — we need an answer.

### 8.2 Method: Monte Carlo Dropout (MC Dropout)

MC Dropout (Gal & Ghahramani, 2016) is the simplest, most practical uncertainty quantification method for deep learning:

1. Keep dropout **enabled** at inference time (normally dropout is disabled at test time).
2. Run the same input through the model **T = 50 times**, each time with different dropout masks.
3. Collect T predictions: `p_1, p_2, ..., p_T`.
4. **Mean prediction:** `μ = (1/T) Σ p_t` — this is the risk score.
5. **Predictive uncertainty:** `σ² = (1/T) Σ (p_t - μ)²` — this is the model's confidence.

### 8.3 Implementation

```python
def predict_with_uncertainty(model, data, T=50):
    model.train()  # keep dropout ON

    predictions = []
    for _ in range(T):
        with torch.no_grad():
            pred = torch.sigmoid(model(data.x, data.edge_index))
        predictions.append(pred)

    predictions = torch.stack(predictions)  # (T, N)
    mean = predictions.mean(dim=0)          # (N,)
    variance = predictions.var(dim=0)       # (N,)

    return mean, variance
```

### 8.4 Uncertainty Decomposition

Total uncertainty can be split into:
- **Aleatoric uncertainty** (data noise — irreducible): Some blocks genuinely have ambiguous risk due to conflicting signals (old housing but good water chemistry). Captured by training with a heteroscedastic loss that predicts both μ and σ per sample.
- **Epistemic uncertainty** (model ignorance — reducible with more data): The model is unsure because it hasn't seen enough similar blocks. Captured by MC Dropout variance.

For the poster, reporting **total uncertainty** (σ from MC Dropout) is sufficient. Decomposition is a stretch goal.

### 8.5 Calibration Check

A well-calibrated model should satisfy: "Of all blocks where we predict 80% risk, approximately 80% should actually have lead issues."

**Calibration metric:** Expected Calibration Error (ECE).

```python
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted = calibration_curve(y_true, y_pred_mean, n_bins=10)
ECE = np.mean(np.abs(fraction_of_positives - mean_predicted))
```

**If poorly calibrated:** Apply Platt scaling (logistic regression on validation predictions) or temperature scaling.

**Target:** ECE < 0.05.

---

## 9. Model 4: Fairness-Constrained Pipe Replacement Optimizer

### 9.1 Purpose

This is the **policy layer** that turns predictions into action. Given a fixed pipe replacement budget, which blocks should be prioritized? A naive approach (replace highest-risk first) tends to concentrate resources in a few high-risk urban areas while ignoring smaller, lower-income towns that also have serious risk. The fairness constraint prevents this.

### 9.2 Formal Problem Definition

**Decision variable:** `x_i ∈ {0, 1}` for each CBG `i` — replace pipes in this block (1) or not (0).

**Objective — maximize total risk reduction:**

```
maximize  Σ_i  risk_score_i × x_i
```

**Subject to:**

1. **Budget constraint:**
   ```
   Σ_i  cost_i × x_i  ≤  B
   ```
   Where `cost_i` = estimated pipe replacement cost for CBG `i` (based on number of service connections × average cost per connection, ~$10,000 per LSL in NJ).

2. **Demographic parity constraint:**
   ```
   | (Σ_i x_i × minority_pop_i / Σ_i x_i × pop_i) - statewide_minority_rate |  ≤  ε
   ```
   The demographic composition of served blocks must be within ε of the statewide average. This prevents the optimizer from systematically skipping minority neighborhoods.

3. **Minimum county coverage:**
   ```
   For each county c:  Σ_{i ∈ c}  x_i  ≥  min_coverage_c
   ```
   Every county gets at least some replacements, preventing total neglect of rural counties.

### 9.3 Solution Method

This is a **constrained binary optimization problem** (variant of the knapsack problem). Solvable with:

**Option A: Integer Linear Programming (ILP)**
```python
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# Or use PuLP for cleaner syntax:
import pulp

prob = pulp.LpProblem("pipe_replacement", pulp.LpMaximize)
x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(N)]

# Objective
prob += pulp.lpSum(risk[i] * x[i] for i in range(N))

# Budget constraint
prob += pulp.lpSum(cost[i] * x[i] for i in range(N)) <= B

# Demographic parity
prob += pulp.lpSum(x[i] * minority_pop[i] for i in range(N)) >= \
        (statewide_rate - epsilon) * pulp.lpSum(x[i] * pop[i] for i in range(N))
prob += pulp.lpSum(x[i] * minority_pop[i] for i in range(N)) <= \
        (statewide_rate + epsilon) * pulp.lpSum(x[i] * pop[i] for i in range(N))

# County minimums
for county_indices in counties.values():
    prob += pulp.lpSum(x[i] for i in county_indices) >= min_per_county

prob.solve(pulp.PULP_CBC_CMD(timeLimit=60))
```

**Option B: Greedy Heuristic (Faster, Good Enough)**

If ILP is too slow for the demo:
1. Sort blocks by `risk_score_i / cost_i` (risk-per-dollar efficiency).
2. Greedily select blocks until budget is exhausted.
3. After each selection, check fairness constraint. If violated, skip to next block.
4. O(N log N) time.

### 9.4 Demo Parameters

| Parameter | Default | Adjustable in Demo |
|---|---|---|
| Budget B | $100M (realistic NJ scale) | Slider: $10M–$500M |
| Fairness threshold ε | 0.05 (5% tolerance) | Slider: 0.01–0.20 |
| Min coverage per county | 5 blocks | Slider: 1–20 |

### 9.5 Output

- Ordered list of CBGs to replace, with cost and risk reduction per block.
- Total risk reduced, total cost, fairness metric achieved.
- Comparison: "With fairness constraint: reduces risk by 847 units at $98M. Without fairness constraint: reduces risk by 892 units at $98M — but serves 23% fewer minority residents." This comparison is the wow moment for judges.

---

## 10. Model 5: Explainability Engine (SHAP)

### 10.1 Purpose

SHAP (SHapley Additive exPlanations) answers the question every judge will ask: **"Why does the model think this block is high-risk?"**

### 10.2 Implementation

```python
import shap

# For the tabular encoder component:
explainer = shap.DeepExplainer(tabular_encoder, background_data[:100])
shap_values = explainer.shap_values(test_data)

# For the full pipeline (treat as black box):
explainer = shap.KernelExplainer(model.predict, background_data[:100])
shap_values = explainer.shap_values(test_data[:50])
```

**Practical note:** `KernelExplainer` on the full multi-modal model is very slow (~30 min for 50 samples). Two options:
1. Use `DeepExplainer` on just the tabular branch — fast and gives the most interpretable results anyway (satellite SHAP values are hard to explain to judges).
2. Pre-compute SHAP values for all test blocks and cache them. Load at demo time.

### 10.3 Outputs

**Per-block explanation:**
```
Block 340070023001 — Risk Score: 0.87 (σ=0.04)
Top risk drivers:
  +0.31  housing_pct_pre_1950 (value: 72%)
  +0.22  water_ph_mean (value: 6.4)
  +0.14  90th_pctl_lead_ppb (value: 12.8)
  +0.09  poverty_rate (value: 28%)
  -0.08  dist_to_treatment_plant_km (value: 1.2)
```

**Global feature importance:** Aggregate SHAP across all blocks to answer "What drives lead risk in NJ overall?"

**Geographic SHAP maps:** Plot each SHAP feature value on the NJ map. This reveals geographic patterns: "pH is the dominant risk driver in South Jersey, while housing age dominates in North Jersey." This is a genuinely novel research finding.

---

## 11. Model 6: LLM Policy Brief Generator

### 11.1 Purpose

The cherry on top. For any selected CBG, generate a plain-English policy brief that a mayor or utility manager can read and act on.

### 11.2 Implementation

```python
from anthropic import Anthropic

client = Anthropic()

def generate_policy_brief(block_data: dict) -> str:
    prompt = f"""You are an environmental policy analyst specializing in lead contamination
    in water infrastructure. Generate a concise, actionable policy brief for the following
    census block group.

    Block Group: {block_data['geoid']}
    Municipality: {block_data['municipality']}
    County: {block_data['county']}

    Risk Score: {block_data['risk_score']:.2f} (Uncertainty: ±{block_data['uncertainty']:.2f})
    Risk Category: {block_data['risk_category']}

    Top Risk Drivers (SHAP):
    {block_data['shap_summary']}

    Demographics:
    - Population: {block_data['population']}
    - Median Income: ${block_data['median_income']:,}
    - % Children Under 6: {block_data['pct_children_under_6']:.1f}%
    - % Minority: {block_data['pct_minority']:.1f}%
    - % Housing Pre-1950: {block_data['pct_housing_pre_1950']:.1f}%

    Water Quality (Latest Quarter):
    - pH: {block_data['ph']:.1f}
    - 90th Percentile Lead: {block_data['lead_90th']:.1f} ppb
    - Alkalinity: {block_data['alkalinity']:.1f} mg/L

    Replacement Priority Rank: #{block_data['replacement_rank']} of {block_data['total_blocks']}
    Estimated Replacement Cost: ${block_data['replacement_cost']:,}

    Generate a 200-word policy brief with:
    1. A one-sentence risk summary.
    2. The primary contamination drivers and why they matter.
    3. Two specific recommended actions (immediate and long-term).
    4. An environmental justice note if this is a vulnerable community.
    """

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

### 11.3 Offline Fallback

If API access is unavailable during demo (conference Wi-Fi is unreliable):
- Pre-generate briefs for the top 100 highest-risk blocks.
- Cache as JSON files.
- Demo app loads cached briefs by GEOID.
- Show the generation prompt + cached output to demonstrate the capability.

---

## 12. Evaluation & Metrics

### 12.1 Prediction Performance

| Metric | What It Measures | Target |
|---|---|---|
| **AUROC** | Discrimination ability — can the model separate high-risk from low-risk blocks? | ≥ 0.85 |
| **AUPRC** | Precision-recall tradeoff — important because positive class is minority (~25%) | ≥ 0.70 |
| **F1 Score** (threshold 0.5) | Balanced accuracy at decision threshold | ≥ 0.75 |
| **Sensitivity / Recall** | Of all truly high-risk blocks, how many do we catch? Critical for public health — missing a dangerous block is worse than a false alarm | ≥ 0.85 |
| **Specificity** | Of all truly safe blocks, how many do we correctly clear? | ≥ 0.70 |

### 12.2 Uncertainty Quality

| Metric | What It Measures | Target |
|---|---|---|
| **ECE (Expected Calibration Error)** | Are predicted probabilities well-calibrated? | ≤ 0.05 |
| **Uncertainty-Error Correlation** | Do high-uncertainty predictions correspond to high error? (Spearman ρ between σ and |y - ŷ|) | ρ ≥ 0.3 |

### 12.3 Fairness Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **Demographic Parity Difference** | Difference in positive prediction rate between demographic groups | ≤ 0.10 |
| **Equalized Odds Difference** | Difference in TPR and FPR between groups | ≤ 0.10 |
| **Replacement Equity Ratio** | Ratio of replacement spending in minority vs. non-minority areas, normalized by population | 0.9–1.1 (near parity) |

### 12.4 Ablation Study

To prove each component adds value (this is what makes it *research*, not just engineering):

| Model Variant | Components | Expected AUROC |
|---|---|---|
| Baseline: Logistic Regression (tabular only) | LR | ~0.72 |
| MLP (tabular only) | Tab encoder | ~0.76 |
| MLP + Temporal | Tab + LSTM | ~0.79 |
| MLP + Temporal + Satellite | Tab + LSTM + CNN | ~0.82 |
| Full fusion (no GNN) | Fusion model | ~0.84 |
| **Full pipeline (fusion + GNN)** | Everything | **~0.87** |

This table is the centerpiece of the poster. It tells a clear story: each modality and the graph structure add measurable predictive power.

---

## 13. Demo Application

### 13.1 Framework

**Streamlit** — fastest path to a polished, interactive web app for a student team.

### 13.2 Pages / Tabs

**Tab 1: NJ Risk Map**
- Full interactive choropleth map of NJ at the census block group level.
- Color: risk score (green → yellow → red gradient).
- Click any block → popup with risk score, uncertainty, top SHAP drivers.
- Library: `folium` with `streamlit-folium`, or `plotly.express.choropleth_mapbox`.
- Layer toggles: risk score, uncertainty, housing age, water chemistry pH.

**Tab 2: Block Detail View**
- Select a block group by GEOID or click from map.
- Shows:
  - Risk gauge (speedometer-style visualization).
  - Uncertainty band.
  - SHAP waterfall chart (using `shap.plots.waterfall`).
  - Historical water quality trend line (quarterly lead levels).
  - Demographics summary table.
  - LLM-generated policy brief.

**Tab 3: Fairness & Optimization Dashboard**
- Budget slider ($10M–$500M).
- Fairness tolerance slider (ε).
- Run optimizer → shows:
  - Map of selected blocks (highlighted in blue).
  - Total risk reduced, total cost, blocks served.
  - Side-by-side comparison: with vs. without fairness constraint.
  - Bar chart: spending per county, colored by demographic composition.

**Tab 4: Model Performance**
- AUROC / AUPRC curves.
- Confusion matrix.
- Calibration plot.
- Ablation results table.
- Feature importance bar chart (global SHAP).

**Tab 5: About & Methodology**
- Project summary, research questions, data sources.
- Architecture diagram (the ASCII art from Section 3, rendered as an image).
- Team info, acknowledgments, references.

### 13.3 Performance Requirements

| Requirement | Target |
|---|---|
| Map load time | < 3 seconds |
| Block detail load time | < 1 second (SHAP values pre-computed) |
| Optimizer solve time | < 10 seconds for full NJ |
| Policy brief generation | < 5 seconds (or instant if cached) |

All predictions and SHAP values are **pre-computed and stored as parquet/JSON**. The demo app is a viewer, not a real-time inference engine. This ensures it runs smoothly on conference Wi-Fi with a laptop.

---

## 14. Tech Stack

### 14.1 Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `python` | 3.10+ | Runtime |
| `torch` | 2.1+ | Deep learning framework |
| `torch_geometric` | 2.4+ | Graph neural networks (SAGEConv, data loaders) |
| `torchvision` | 0.16+ | ResNet-18 pretrained backbone |
| `scikit-learn` | 1.3+ | Metrics, calibration, preprocessing |
| `pandas` | 2.1+ | Tabular data manipulation |
| `geopandas` | 0.14+ | Geospatial data (shapefiles, spatial joins) |
| `numpy` | 1.24+ | Numerical operations |
| `shap` | 0.43+ | Explainability |
| `streamlit` | 1.28+ | Demo application |
| `streamlit-folium` | 0.15+ | Interactive map embedding |
| `folium` | 0.15+ | Leaflet.js map rendering |
| `plotly` | 5.18+ | Interactive charts |
| `anthropic` | 0.39+ | Claude API for policy briefs |
| `pystac-client` | 0.7+ | Sentinel-2 satellite imagery access |
| `planetary-computer` | 1.0+ | Microsoft Planetary Computer authentication |
| `rasterio` | 1.3+ | Satellite image reading and cropping |
| `osmnx` | 1.7+ | Road network graph extraction from OpenStreetMap |
| `libpysal` | 4.9+ | Spatial weights and adjacency matrices |
| `pulp` | 2.7+ | Integer linear programming solver |
| `requests` | 2.31+ | EPA/Census API calls |
| `pytest` | 7.4+ | Testing |

### 14.2 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU training feasible for this data size) | NVIDIA GPU with 4+ GB VRAM (e.g., GTX 1650) |
| RAM | 8 GB | 16 GB (satellite tiles are memory-hungry) |
| Storage | 5 GB | 10 GB (satellite imagery cache) |
| Training time (CPU) | ~2–4 hours | — |
| Training time (GPU) | — | ~20–40 minutes |

### 14.3 External APIs

| API | Auth | Rate Limit | Cost |
|---|---|---|---|
| Census ACS API | Free API key (instant) | 500 requests/day | Free |
| EPA SDWIS API | None | 10,000 rows/request | Free |
| Microsoft Planetary Computer STAC | None | Generous | Free |
| Anthropic Claude API | API key | Standard tier | ~$0.50 for 100 briefs |

---

## 15. Directory Structure

```
leadsense_nj/
├── README.md                        # Project overview
├── LEADSENSE_NJ_SPEC.md            # This document
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
│
├── data/
│   ├── raw/                         # Downloaded raw data (gitignored)
│   │   ├── census_acs/
│   │   ├── epa_sdwis/
│   │   ├── nj_dep/
│   │   ├── sentinel2_tiles/
│   │   ├── tri_superfund/
│   │   └── nj_road_network/
│   ├── processed/                   # Cleaned, aligned data
│   │   ├── feature_matrix.parquet   # (N_blocks × 46 features)
│   │   ├── temporal_tensor.npy      # (N_blocks × 8 quarters × 14 features)
│   │   ├── satellite_tiles/         # (N_blocks × 256 × 256 × 4) as .npy
│   │   ├── graph_edges.csv          # Edge list for GNN
│   │   ├── labels.csv               # Target variable per block
│   │   └── spatial_folds.csv        # Fold assignments for spatial CV
│   └── cache/                       # Pre-computed artifacts for demo
│       ├── predictions.parquet      # Risk scores + uncertainty
│       ├── shap_values.parquet      # SHAP per block × feature
│       └── policy_briefs.json       # Cached LLM briefs
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── census_loader.py         # Census ACS API client
│   │   ├── epa_loader.py            # EPA SDWIS API client
│   │   ├── satellite_loader.py      # Sentinel-2 tile downloader
│   │   ├── tri_loader.py            # TRI/Superfund data loader
│   │   ├── road_network_loader.py   # OSMnx road graph builder
│   │   ├── preprocessor.py          # Spatial/temporal alignment
│   │   └── label_builder.py         # Target variable construction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tabular_encoder.py       # MLP for tabular features
│   │   ├── temporal_encoder.py      # LSTM/Transformer for time series
│   │   ├── vision_encoder.py        # ResNet-18 for satellite tiles
│   │   ├── fusion.py                # Cross-attention fusion module
│   │   ├── gnn.py                   # GraphSAGE network
│   │   ├── pipeline.py              # End-to-end model (fusion + GNN)
│   │   └── uncertainty.py           # MC Dropout inference wrapper
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── fairness_optimizer.py    # ILP with fairness constraints
│   │   └── greedy_optimizer.py      # Greedy heuristic fallback
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_engine.py           # SHAP value computation
│   │   └── policy_brief.py          # LLM brief generator
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py               # AUROC, AUPRC, F1, ECE, fairness
│       ├── calibration.py           # Calibration curve + Platt scaling
│       └── ablation.py              # Ablation study runner
│
├── app/
│   ├── streamlit_app.py             # Main Streamlit application
│   ├── pages/
│   │   ├── 1_risk_map.py
│   │   ├── 2_block_detail.py
│   │   ├── 3_fairness_dashboard.py
│   │   ├── 4_model_performance.py
│   │   └── 5_about.py
│   └── assets/
│       ├── architecture_diagram.png
│       └── leadsense_logo.png
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_demo_prep.ipynb
│
├── tests/
│   ├── test_data_loaders.py
│   ├── test_preprocessor.py
│   ├── test_models.py
│   ├── test_optimizer.py
│   └── test_metrics.py
│
├── scripts/
│   ├── download_all_data.py         # One-click data acquisition
│   ├── run_preprocessing.py         # Full preprocessing pipeline
│   ├── train.py                     # Model training entrypoint
│   ├── evaluate.py                  # Evaluation + ablation runner
│   ├── generate_cache.py            # Pre-compute demo artifacts
│   └── run_demo.py                  # Launch Streamlit app
│
└── artifacts/
    ├── models/                      # Saved model checkpoints
    │   ├── fusion_model.pt
    │   └── full_pipeline.pt
    ├── figures/                     # Generated plots for poster
    │   ├── auroc_curve.png
    │   ├── calibration_plot.png
    │   ├── ablation_table.png
    │   ├── shap_global.png
    │   └── fairness_comparison.png
    └── reports/
        ├── evaluation_report.md
        └── ablation_results.json
```

---

## 16. Risk Register & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **Pipe network GIS data unavailable for most of NJ** | High | Medium | Road-network proxy (Section 7.4) is pre-planned. Ablation study will quantify how much real pipe data helps vs. proxy. Even without pipe data, the multi-modal fusion model alone is novel |
| 2 | **Satellite imagery adds no predictive value** | Medium | Low | Ablation study will reveal this. If satellite adds <1% AUROC, drop it and focus on tabular + temporal + GNN. The project is still strong without it |
| 3 | **Label construction is noisy (composite label based on heuristics)** | High | Medium | Acknowledge this limitation explicitly in the poster. Perform sensitivity analysis: how do results change under different label definitions? This is intellectually honest and judges respect it |
| 4 | **Class imbalance (25% positive)** | Medium | Medium | Weighted loss function, SMOTE for tabular features, stratified spatial CV. 75/25 split is mild imbalance; not a crisis |
| 5 | **Conference Wi-Fi too slow for live API calls** | High | Low | All predictions pre-cached. LLM briefs pre-generated for top 100 blocks. Demo runs fully offline |
| 6 | **Training too slow on CPU** | Medium | Low | Reduce satellite tile resolution (128×128 instead of 256×256). Use smaller ResNet (ResNet-10). GNN on 6K nodes trains in minutes even on CPU |
| 7 | **EPA API rate limits hit during data collection** | Low | Low | Paginate requests, cache responses locally. One-time download, not real-time |
| 8 | **Spatial autocorrelation inflates metrics** | High | High | Spatial cross-validation (Section 5.5) is specifically designed to prevent this. Standard random CV would be a methodological error — we avoid it by design |
| 9 | **Judges unfamiliar with GNNs** | Medium | Medium | SHAP explanations and the interactive demo make the outputs intuitive even if the judges don't understand the architecture. The poster should lead with the problem and results, not the math |
| 10 | **Ethical concerns about identifying "dangerous" neighborhoods** | Medium | High | Frame carefully: the tool identifies *infrastructure risk*, not *neighborhood quality*. Emphasize that the fairness constraint ensures equitable resource allocation. Include an ethics section on the poster |

---

## 17. Timeline

### Assuming 5-week build (working ~15-20 hrs/week)

| Week | Focus | Deliverables |
|---|---|---|
| **Week 1** | Data acquisition & preprocessing | All raw data downloaded. Feature matrix assembled. Labels constructed. Spatial CV folds assigned |
| **Week 2** | Fusion model development | Tabular encoder, temporal encoder, vision encoder, cross-attention fusion — all implemented and tested individually. Baseline results (tabular-only logistic regression and MLP) |
| **Week 3** | GNN + uncertainty + training | Graph constructed (road proxy). GraphSAGE integrated. MC Dropout implemented. End-to-end training pipeline running. Full ablation study results |
| **Week 4** | Optimization + explainability + LLM | Fairness optimizer working. SHAP values computed. LLM briefs generated. All artifacts cached |
| **Week 5** | Demo app + poster + polish | Streamlit app complete. Poster designed. Dry-run presentations. Edge case testing. README finalized |

---

## 18. References

1. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML 2016*.
2. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). "Inductive Representation Learning on Large Graphs" (GraphSAGE). *NeurIPS 2017*.
3. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP). *NeurIPS 2017*.
4. Neumann, M., et al. (2019). "In-domain representation learning for remote sensing." *arXiv:1911.06721*.
5. NJ DEP. (2021). "New Jersey's Lead Service Line Replacement Initiative." NJ Department of Environmental Protection.
6. EPA. (2023). "Lead and Copper Rule Revisions (LCRR)." EPA 816-F-21-006.
7. Abernethy, J., et al. (2018). "ActiveRemediation: The Search for Lead Pipes in Flint, Michigan." *KDD 2018*. — Directly relevant prior work applying ML to lead pipe prediction in Flint.
8. Luo, R., et al. (2021). "Predicting Lead in Water Using Machine Learning." *Environmental Science & Technology*.
9. Syrgkanis, V., et al. (2019). "Machine Learning Estimation of Heterogeneous Treatment Effects." *Econometrica*. — Relevant for causal fairness framing.
10. Corbett-Davies, S., & Goel, S. (2018). "The Measure and Mismeasure of Fairness: A Critical Review of Fair Machine Learning." *arXiv:1808.00023*. — Framework for fairness constraint selection.

---

## Appendix A: Key Prior Work Comparison

| Paper | Method | Data | Limitation LeadSense NJ Addresses |
|---|---|---|---|
| Abernethy et al. (KDD 2018) | Logistic regression + active learning for Flint, MI | Parcels, water tests | Single city, single modality (tabular only), no graph structure, no fairness constraints |
| Luo et al. (EST 2021) | Random forest on water chemistry features | Water quality data | No spatial modeling, no satellite imagery, no uncertainty quantification |
| EPA LCRR Risk Assessment | Rule-based (pipe material + age thresholds) | Utility self-reports | Not ML-based, no probabilistic output, no explainability, relies on utility honesty |

**LeadSense NJ's contribution:** First system to combine multi-modal deep learning (tabular + satellite + temporal), graph neural networks on water infrastructure topology, calibrated uncertainty, and fairness-constrained optimization for lead risk prediction. Applied to all of NJ (statewide scale), not a single city.

---

*End of specification.*
