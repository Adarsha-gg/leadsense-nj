# Abstract

**RoadPulse AI: Computer Vision for Risk-Aware Road Repair Prioritization**

Road infrastructure degradation creates safety risks, vehicle damage costs, and inefficient maintenance spending. Manual pavement inspection is expensive, slow, and difficult to scale across high-traffic corridors. RoadPulse AI addresses this by combining computer vision and risk analytics into a practical decision-support system for transportation agencies and operations teams.

The system ingests road video (dashcam or survey footage), detects hazards such as potholes and crack patterns, estimates hazard severity, and computes a composite risk score using contextual factors including traffic intensity, weather exposure, school-zone proximity, and equity prioritization settings. Detected hazards are geotagged when GPS telemetry is available and aggregated into segment-level repair candidates. A budget-constrained prioritization module then ranks segments by expected safety impact and projected risk reduction.

RoadPulse AI provides outputs tailored for mixed technical and non-technical stakeholders: annotated visual frames, hazard distribution analytics, timeline trends, interactive risk maps, downloadable repair queues, and a what-if simulation to estimate risk reduction under different repair capacities. The platform is designed to be deployable with lightweight models for rapid prototyping while supporting upgrade to trained YOLO backends for production-scale inference.

This project aligns with accessible and sustainable AI principles by focusing on transparent scoring, operational utility, and data-informed infrastructure maintenance decisions.

