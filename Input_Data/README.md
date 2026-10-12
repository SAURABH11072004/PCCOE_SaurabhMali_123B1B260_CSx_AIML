# Input Data Directory

This directory contains test datasets, baseline specifications, and evaluation benchmarks used by the **AUTOSAR HLD Document Analysis Assistant**:

1. `synthetic_autosar_hld.pdf`:
   - Baseline Revision 1.0 of the High-Level Design specification for an Electric Vehicle Powertrain Domain Controller (PDC) and Body Control Module (BCM).
   - Contains 6 detailed pages covering AUTOSAR layered models, SWCs (`ThrottleControl_SWC`, `BrakeManager_SWC`, `BatteryManagement_SWC`, `ThermalControl_SWC`, `Gateway_SWC`), Port Prototypes, Signal Dictionaries, and Functional Execution Flows.
   - Includes controlled deliberate architectural inconsistencies (e.g. conflicting signal cycle times and undefined interfaces) for automated testing.
   - **Label**: `SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT`.

2. `synthetic_autosar_hld_v2.pdf`:
   - Updated Revision 2.0 specification introducing Cybersecurity HSM integration (`SecurityManager_SWC`), unified signal timing, and removed legacy ports. Used for revision comparison tests.

3. `evaluation_dataset.json`:
   - Benchmark evaluation dataset comprising 20 curated questions with ground-truth expected pages, target sections, expected architectural facts, and negative abstention labels.
