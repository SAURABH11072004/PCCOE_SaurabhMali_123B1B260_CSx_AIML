#!/usr/bin/env python3
"""Synthetic AUTOSAR HLD Document and Evaluation Dataset Generator.
Generates multi-page test PDFs containing realistic AUTOSAR architecture specifications
and controlled inconsistencies for automated RAG, extraction, and comparison testing.
Explicitly labeled: SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT.
"""

import json
from pathlib import Path
import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / "Input_Data"
EVAL_RESULTS_DIR = PROJECT_ROOT / "Evaluation_Results"
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


V1_PAGES = [
    # Page 1
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V1 | Revision: 1.0 | Status: Draft Engineering Specification
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 1: Executive Summary & Architectural Scope
This document specifies the High-Level Design (HLD) for the next-generation Electric Vehicle Powertrain Domain Controller (PDC) and Body Control Module (BCM). The system is developed in accordance with the AUTOSAR Classic Platform Release 4.4.0 specification.

1.1 Purpose and Architectural Objectives
The primary purpose of the Powertrain Domain Controller is to manage vehicle propulsion, torque coordination, high-voltage battery state management, thermal cooling loops, and high-speed gateway communication. The software architecture strictly enforces modular component separation through the AUTOSAR Virtual Functional Bus (VFB) and the Runtime Environment (RTE).

1.2 Layered Architecture Overview
The system follows standard AUTOSAR layered architecture:
- Application Software Layer (ASW): Houses Atomic Software Components (SWCs).
- Runtime Environment (RTE): Facilitates location-independent inter-SWC communication and Basic Software abstraction.
- Basic Software (BSW): Includes Microcontroller Abstraction Layer (MCAL), ECU Abstraction, System Services (EcuM, BswM), and Communication Stack (Com, CanIf, CanSM, PduR).
All interfaces conform to AUTOSAR Sender-Receiver and Client-Server communication paradigms.""",

    # Page 2
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V1 | Revision: 1.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 2: Architectural Overview & AUTOSAR Layered Model
The architecture decomposes vehicle functions into distinct, decoupled software components (SWCs). Inter-component communications are mapped through standardized AUTOSAR Port Prototypes.

2.1 Execution Model
The Powertrain Domain Controller executes on a multi-core 32-bit automotive microcontroller (e.g., TriCore AURIX or RH850). The operating system is an OSEK/VDX compliant AUTOSAR OS supporting static priority-preemptive task scheduling:
- Task_10ms: High-priority periodic task executing powertrain throttle and brake torque loops.
- Task_20ms: Medium-priority task managing battery pack voltage and current monitoring.
- Task_100ms: Low-priority supervisory task coordinating thermal loops and gateway routing.
- Task_Background: Non-time-critical diagnostic and background memory verification.

2.2 Memory Mapping and Safety Isolation
Components assigned to safety-critical functions (ASIL-C / ASIL-D) operate in protected memory partitions with dedicated Hardware Memory Protection Units (MPU). Non-safety body functions operate in separate non-trusted partitions.""",

    # Page 3
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V1 | Revision: 1.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 3: Software Components (SWCs) and Port Definitions
The Application Layer defines five primary software components:

3.1 ThrottleControl_SWC
- Component Type: Atomic Application Software Component (ASIL-D).
- Description: Responsible for calculating requested engine/motor throttle angle based on driver accelerator pedal inputs.
- Provided Ports:
  - PpThrottlePos: Provides calculated throttle command across interface If_ThrottlePosition.
- Required Ports:
  - RpPedalSensor: Receives dual redundant analog accelerator pedal sensor voltages.

3.2 BrakeManager_SWC
- Component Type: Atomic Application Software Component (ASIL-D).
- Description: Coordinates vehicle deceleration, brake-by-wire hydraulic blending, and regenerative torque requests.
- Provided Ports:
  - PpBrakeTorqueRequest: Provides electric motor regenerative torque request via interface If_BrakeTorque.
- Required Ports:
  - RpWheelSpeed: Receives four-wheel speed sensor pulses from the ABS/ESC controller.

3.3 BatteryManagement_SWC
- Component Type: Atomic Application Software Component (ASIL-C).
- Description: Monitors high-voltage battery state of charge (SoC), pack temperature, and high-voltage contactor state.
- Provided Ports:
  - PpBMSStatus: Broadcasts battery pack state and discharge limits via interface If_BatteryState.
- Required Ports:
  - RpCellVoltage: Receives individual lithium-ion cell voltages from internal BMS slave modules.

3.4 ThermalControl_SWC
- Component Type: Sensor/Actuator Software Component (ASIL-B).
- Description: Controls powertrain cooling loop water pumps, refrigerant valves, and front radiator fans.
- Provided Ports:
  - PpFanSpeedCommand: Actuates PWM radiator fan speed.
- Required Ports:
  - RpCoolantTemp: Reads coolant temperature sensors via interface If_ThermalData.
  - RpRefrigerantPressure: Reads high-pressure AC circuit pressure via interface If_RefrigerantPressure.

3.5 Gateway_SWC
- Component Type: Complex Device Driver / Application Gateway SWC (QM).
- Description: Translates and filters signals between CAN-FD Powertrain Bus and Automotive Ethernet Backbone.
- Provided Ports:
  - PpEthTelemetry: Transmits telemetry frames to vehicle telematics unit.""",

    # Page 4
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V1 | Revision: 1.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 4: Software Interfaces and Signal Dictionary
All communications between SWCs use AUTOSAR Sender-Receiver interfaces.

4.1 Interface Definitions:
- Interface If_ThrottlePosition:
  - Sig_ThrottleAngle: float32, range [0.0, 100.0] %, Resolution 0.1 %, Update Rate: 10ms.
  - Sig_ThrottleFaultStatus: uint8, range [0, 3] (0=OK, 1=Degraded, 2=Fault, 3=SensorLoss), Update Rate: 10ms.
- Interface If_BrakeTorque:
  - Sig_RequestedTorque: int16, range [-500, 500] Nm, Resolution 1 Nm, Update Rate: 10ms.
  - Sig_BrakePedalState: boolean (0=Released, 1=Depressed), Update Rate: 10ms.
- Interface If_BatteryState:
  - Sig_StateOfCharge: uint8, range [0, 100] %, Resolution 1 %, Update Rate: 100ms.
  - Sig_PackVoltage: float32, range [0.0, 800.0] V, Resolution 0.05 V, Update Rate: 50ms.
  - Sig_PackCurrent: float32, range [-400.0, 400.0] A, Resolution 0.1 A, Update Rate: 20ms.
- Interface If_ThermalData:
  - Sig_CoolantTemp: int16, range [-40, 150] degC, Resolution 1 degC, Update Rate: 500ms.
  - Sig_PumpSpeedTarget: uint16, range [0, 6000] RPM, Resolution 10 RPM, Update Rate: 100ms.

4.2 Interface Table Summary:
| Interface Name | Data Elements | Data Type | Cycle Time | Safety Level |
| If_ThrottlePosition | Sig_ThrottleAngle, Sig_ThrottleFaultStatus | float32, uint8 | 10ms | ASIL-D |
| If_BrakeTorque | Sig_RequestedTorque, Sig_BrakePedalState | int16, boolean | 10ms | ASIL-D |
| If_BatteryState | Sig_StateOfCharge, Sig_PackVoltage, Sig_PackCurrent | uint8, float32, float32 | 100ms / 50ms | ASIL-C |
| If_ThermalData | Sig_CoolantTemp, Sig_PumpSpeedTarget | int16, uint16 | 500ms / 100ms | ASIL-B |""",

    # Page 5
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V1 | Revision: 1.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 5: Functional Flows and Inter-SWC Communication
The system realizes vehicle operational behavior through coordinated functional execution chains:

5.1 Flow 1: Driver Acceleration Request Flow
1. Driver depresses the accelerator pedal.
2. Accelerator pedal voltage is acquired through MCAL ADC driver by RpPedalSensor port on ThrottleControl_SWC.
3. ThrottleControl_SWC validates dual sensor plausibility and checks for cross-channel plausibility faults.
4. Calculated throttle demand is published as Sig_ThrottleAngle on port PpThrottlePos.
5. The RTE transfers Sig_ThrottleAngle to the inverter actuator manager within 10ms.

5.2 Flow 2: Regenerative Braking Coordination Flow
1. BrakeManager_SWC detects brake pedal application and evaluates vehicle speed from RpWheelSpeed.
2. BrakeManager_SWC queries battery absorption capacity from BatteryManagement_SWC via PpBMSStatus.
3. BrakeManager_SWC calculates maximum permissible regenerative torque and issues torque request to InverterControl_SWC.
4. If battery State of Charge exceeds 95%, regenerative braking is tapered and hydraulic friction braking assumes full deceleration torque.

5.3 Flow 3: Thermal Protection Derating Flow
1. ThermalControl_SWC monitors Sig_CoolantTemp.
2. If coolant temperature exceeds 105 degC, ThermalControl_SWC requests power limit derating.
3. ThrottleControl_SWC clamps maximum allowable throttle angle to 50% to prevent electric motor stator overheating.""",

    # Page 6
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V1 | Revision: 1.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 6: Dependencies, Timing & RTE Interactions
This section specifies inter-module execution dependencies, periodic task mapping, and bus communications.

6.1 Task Scheduling Matrix
- Runnable_CalculateThrottle is mapped to Task_10ms.
- Runnable_EvaluateBrakeTorque is mapped to Task_10ms.
- Runnable_MonitorBatteryState is mapped to Task_20ms.
- Runnable_RegulateThermalLoop is mapped to Task_100ms.

6.2 Communication Timing Specifications
For network transmission on CAN-FD Channel 1, message frames are serialized by the AUTOSAR COM stack.
The transmission period for Sig_ThrottleAngle is configured for 50ms cyclic execution on CAN-FD Channel 1 to conserve bus bandwidth.
(Note: Engineering review required: Compare with Section 4 update rate).

6.3 Fault Diagnostics and Recovery Concept
All hardware sensor anomalies and communication timeouts are logged via the Diagnostic Event Manager (DEM). For detailed diagnostic trouble code (DTC) mapping, see Section 9.4.
Software Component recovery actions follow ISO 26262 ASIL hazard mitigation guidelines."""
]


V2_PAGES = [
    # Page 1
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V2 | Revision: 2.0 | Status: Approved Engineering Baseline
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 1: Executive Summary & Architectural Scope
This document specifies the updated High-Level Design (HLD) Revision 2.0 for the Electric Vehicle Powertrain Domain Controller (PDC) and Body Control Module (BCM). Standard AUTOSAR Classic Platform Release 4.4.0 layered model is followed.

1.1 Purpose and Revision 2.0 Highlights
Revision 2.0 introduces Hardware Security Module (HSM) cybersecurity integration, expands battery fast-charging support, aligns signal timing discrepancies, and removes legacy OBD diagnostics in favor of UDS over Ethernet (DoIP).""",

    # Page 2
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V2 | Revision: 2.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 2: Architectural Overview & AUTOSAR Layered Model
The architecture decomposes vehicle functions into distinct, decoupled software components (SWCs).
Task scheduling and memory protection partitions are updated to incorporate dedicated crypto execution contexts on Core 1.""",

    # Page 3
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V2 | Revision: 2.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 3: Software Components (SWCs) and Port Definitions
The Application Layer defines six primary software components:

3.1 ThrottleControl_SWC
- Provided Ports: PpThrottlePos (interface If_ThrottlePosition).
- Required Ports: RpPedalSensor.

3.2 BrakeManager_SWC
- Provided Ports: PpBrakeTorqueRequest (interface If_BrakeTorque).
- Required Ports: RpWheelSpeed.

3.3 BatteryManagement_SWC
- Provided Ports:
  - PpBMSStatus: Broadcasts battery pack state via interface If_BatteryState.
  - PpFastChargeControl: Manages DC Fast Charging contactors and 800V pre-charge state machine.
- Required Ports: RpCellVoltage.

3.4 ThermalControl_SWC
- Provided Ports: PpFanSpeedCommand.
- Required Ports: RpCoolantTemp (interface If_ThermalData).

3.5 Gateway_SWC
- Central gateway routing between CAN-FD and Automotive Ethernet Backbone.

3.6 SecurityManager_SWC
- Component Type: Security Service Software Component (ASIL-B).
- Description: Manages secure boot authentication, message authentication codes (SecOC), and cryptographic key provisioning.
- Provided Ports:
  - PpCryptoKeyService: Provides cryptographic verification services to application SWCs.""",

    # Page 4
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V2 | Revision: 2.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 4: Software Interfaces and Signal Dictionary
All communications use standardized AUTOSAR Sender-Receiver interfaces.

4.1 Interface Definitions:
- Interface If_ThrottlePosition:
  - Sig_ThrottleAngle: float32, range [0.0, 100.0] %, Update Rate: 10ms (Consistent across all layers).
  - Sig_ThrottleFaultStatus: uint8, range [0, 3], Update Rate: 10ms.
- Interface If_BrakeTorque:
  - Sig_RequestedTorque: int16, range [-500, 500] Nm, Update Rate: 10ms.
  - Sig_BrakePedalState: boolean, Update Rate: 10ms.
- Interface If_BatteryState:
  - Sig_StateOfCharge: uint8, range [0, 100] %, Update Rate: 100ms.
  - Sig_PackVoltage: float32, range [0.0, 800.0] V, Update Rate: 50ms.
  - Sig_PackCurrent: float32, range [-400.0, 400.0] A, Update Rate: 20ms.
- Interface If_ThermalData:
  - Sig_CoolantTemp: int16, range [-40, 150] degC, Update Rate: 500ms.
  - Sig_PumpSpeedTarget: uint16, range [0, 6000] RPM, Update Rate: 100ms.
- Interface If_SecurityVerification:
  - Sig_SecOCAuthStatus: uint8, range [0, 1], Update Rate: 10ms.""",

    # Page 5
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V2 | Revision: 2.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 5: Functional Flows and Inter-SWC Communication
Functional execution chains updated for cybersecurity verification and DC fast-charge coordination.
Flow 1: Driver Acceleration Request Flow (Verified end-to-end within 10ms).
Flow 2: Regenerative Braking Coordination Flow.
Flow 3: Thermal Protection Derating Flow.""",

    # Page 6
    """SYNTHETIC TEST DATA — NOT AN OFFICIAL AUTOSAR DOCUMENT
Document ID: AUTOSAR-HLD-SYNTH-2026-V2 | Revision: 2.0
Title: High-Level Design (HLD) Document — Automotive Powertrain & Body Domain Controller

Section 6: Dependencies, Timing & RTE Interactions
6.1 Timing Synchronization
Transmission period for Sig_ThrottleAngle is unified at 10ms cyclic execution across COM, RTE, and CAN-FD channels, resolving previous ambiguity.

Section 7: Cyber Security Hardware Security Module (HSM) Integration
7.1 Security Architecture
Integrates ISO/SAE 21434 automotive cyber security management system requirements.
Hardware Cryptographic Accelerator handles AES-128 CMAC calculation for all safety-critical CAN-FD frames.
Key storage is securely partitioned within internal HSM secure flash memory."""
]


def create_pdf(pages_text: list, output_path: Path):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page(width=612, height=792)  # Standard Letter / A4
        # Margin: left 50, top 50, right 562, bottom 742
        rect = fitz.Rect(50, 50, 562, 742)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
    doc.save(str(output_path))
    doc.close()
    print(f"Created PDF: {output_path} ({len(pages_text)} pages)")


def generate_evaluation_dataset() -> list:
    return [
        {
            "id": "Q01",
            "type": "factual_lookup",
            "question": "What is the primary function of ThrottleControl_SWC, and what are its provided and required ports?",
            "target_sections": ["Section 3: Software Components (SWCs) and Port Definitions"],
            "expected_page": 3,
            "expected_facts": [
                "ThrottleControl_SWC",
                "PpThrottlePos",
                "RpPedalSensor",
                "electronic throttle",
                "ASIL-D"
            ],
            "should_abstain": False
        },
        {
            "id": "Q02",
            "type": "signal_lookup",
            "question": "What are the data type, range, and update rate of Sig_ThrottleAngle in the interface dictionary?",
            "target_sections": ["Section 4: Software Interfaces and Signal Dictionary"],
            "expected_page": 4,
            "expected_facts": [
                "Sig_ThrottleAngle",
                "float32",
                "0.0, 100.0",
                "10ms"
            ],
            "should_abstain": False
        },
        {
            "id": "Q03",
            "type": "signal_lookup",
            "question": "What signals are contained within the If_BatteryState interface and what are their update rates?",
            "target_sections": ["Section 4: Software Interfaces and Signal Dictionary"],
            "expected_page": 4,
            "expected_facts": [
                "Sig_StateOfCharge",
                "Sig_PackVoltage",
                "Sig_PackCurrent",
                "100ms",
                "50ms",
                "20ms"
            ],
            "should_abstain": False
        },
        {
            "id": "Q04",
            "type": "component_lookup",
            "question": "Which software component controls vehicle deceleration, brake-by-wire hydraulic blending, and regenerative torque requests?",
            "target_sections": ["Section 3: Software Components (SWCs) and Port Definitions"],
            "expected_page": 3,
            "expected_facts": [
                "BrakeManager_SWC",
                "PpBrakeTorqueRequest",
                "RpWheelSpeed"
            ],
            "should_abstain": False
        },
        {
            "id": "Q05",
            "type": "functional_flow",
            "question": "Explain Flow 1: Driver Acceleration Request Flow from accelerator pedal depression to actuator publication.",
            "target_sections": ["Section 5: Functional Flows and Inter-SWC Communication"],
            "expected_page": 5,
            "expected_facts": [
                "Flow 1",
                "RpPedalSensor",
                "ThrottleControl_SWC",
                "Sig_ThrottleAngle",
                "PpThrottlePos",
                "10ms"
            ],
            "should_abstain": False
        },
        {
            "id": "Q06",
            "type": "functional_flow",
            "question": "What happens in Flow 2 if the battery State of Charge exceeds 95% during regenerative braking?",
            "target_sections": ["Section 5: Functional Flows and Inter-SWC Communication"],
            "expected_page": 5,
            "expected_facts": [
                "95%",
                "regenerative braking is tapered",
                "hydraulic friction braking"
            ],
            "should_abstain": False
        },
        {
            "id": "Q07",
            "type": "functional_flow",
            "question": "How does ThermalControl_SWC trigger derating if coolant temperature exceeds 105 degC?",
            "target_sections": ["Section 5: Functional Flows and Inter-SWC Communication"],
            "expected_page": 5,
            "expected_facts": [
                "105 degC",
                "Sig_CoolantTemp",
                "derating",
                "50%"
            ],
            "should_abstain": False
        },
        {
            "id": "Q08",
            "type": "inconsistency_detection",
            "question": "Is there a conflict regarding the transmission period of Sig_ThrottleAngle between Section 4 and Section 6.2?",
            "target_sections": [
                "Section 4: Software Interfaces and Signal Dictionary",
                "Section 6: Dependencies, Timing & RTE Interactions"
            ],
            "expected_page": 6,
            "expected_facts": [
                "10ms in Section 4",
                "50ms in Section 6.2",
                "conflict",
                "inconsistency"
            ],
            "should_abstain": False
        },
        {
            "id": "Q09",
            "type": "issue_lookup",
            "question": "Does ThermalControl_SWC reference any interface that is missing from the interface dictionary?",
            "target_sections": ["Section 3: Software Components (SWCs) and Port Definitions"],
            "expected_page": 3,
            "expected_facts": [
                "RpRefrigerantPressure",
                "If_RefrigerantPressure",
                "missing",
                "not defined"
            ],
            "should_abstain": False
        },
        {
            "id": "Q10",
            "type": "architecture_overview",
            "question": "What AUTOSAR Classic release and operating system standard are specified in the document?",
            "target_sections": [
                "Section 1: Executive Summary & Architectural Scope",
                "Section 2: Architectural Overview & AUTOSAR Layered Model"
            ],
            "expected_page": 1,
            "expected_facts": [
                "AUTOSAR Classic Platform Release 4.4.0",
                "OSEK/VDX",
                "VFB",
                "RTE"
            ],
            "should_abstain": False
        },
        {
            "id": "Q11",
            "type": "safety_lookup",
            "question": "What ASIL safety integrity levels are allocated to ThrottleControl_SWC, BrakeManager_SWC, and BatteryManagement_SWC?",
            "target_sections": ["Section 3: Software Components (SWCs) and Port Definitions"],
            "expected_page": 3,
            "expected_facts": [
                "ASIL-D for ThrottleControl_SWC",
                "ASIL-D for BrakeManager_SWC",
                "ASIL-C for BatteryManagement_SWC"
            ],
            "should_abstain": False
        },
        {
            "id": "Q12",
            "type": "task_scheduling",
            "question": "Which tasks in the scheduling matrix execute Runnable_CalculateThrottle and Runnable_RegulateThermalLoop?",
            "target_sections": ["Section 6: Dependencies, Timing & RTE Interactions"],
            "expected_page": 6,
            "expected_facts": [
                "Task_10ms",
                "Task_100ms",
                "Runnable_CalculateThrottle",
                "Runnable_RegulateThermalLoop"
            ],
            "should_abstain": False
        },
        {
            "id": "Q13",
            "type": "negative_abstention",
            "question": "What is the radar sensor frequency and antenna array configuration for Autonomous Emergency Braking (AEB)?",
            "target_sections": [],
            "expected_page": None,
            "expected_facts": [],
            "should_abstain": True
        },
        {
            "id": "Q14",
            "type": "negative_abstention",
            "question": "What is the chemical composition of the cathode material used in the battery cells?",
            "target_sections": [],
            "expected_page": None,
            "expected_facts": [],
            "should_abstain": True
        },
        {
            "id": "Q15",
            "type": "negative_abstention",
            "question": "Which software component manages the infotainment head-unit Bluetooth pairing state?",
            "target_sections": [],
            "expected_page": None,
            "expected_facts": [],
            "should_abstain": True
        },
        {
            "id": "Q16",
            "type": "component_lookup",
            "question": "What are the ports and role of Gateway_SWC in the architecture?",
            "target_sections": ["Section 3: Software Components (SWCs) and Port Definitions"],
            "expected_page": 3,
            "expected_facts": [
                "Gateway_SWC",
                "PpEthTelemetry",
                "CAN-FD",
                "Automotive Ethernet"
            ],
            "should_abstain": False
        },
        {
            "id": "Q17",
            "type": "signal_lookup",
            "question": "What is the update rate and range of Sig_PumpSpeedTarget in the If_ThermalData interface?",
            "target_sections": ["Section 4: Software Interfaces and Signal Dictionary"],
            "expected_page": 4,
            "expected_facts": [
                "Sig_PumpSpeedTarget",
                "0, 6000",
                "RPM",
                "100ms"
            ],
            "should_abstain": False
        },
        {
            "id": "Q18",
            "type": "cross_reference",
            "question": "Does Section 6.3 refer to any external section for diagnostic trouble code mapping?",
            "target_sections": ["Section 6: Dependencies, Timing & RTE Interactions"],
            "expected_page": 6,
            "expected_facts": [
                "Section 9.4",
                "DEM",
                "Diagnostic Event Manager"
            ],
            "should_abstain": False
        },
        {
            "id": "Q19",
            "type": "negative_abstention",
            "question": "What is the lidar point cloud resolution and laser wavelength used in the vehicle perception system?",
            "target_sections": [],
            "expected_page": None,
            "expected_facts": [],
            "should_abstain": True
        },
        {
            "id": "Q20",
            "type": "flow_inconsistency",
            "question": "Does Flow 2 reference any component that is missing from Section 3's component list?",
            "target_sections": [
                "Section 3: Software Components (SWCs) and Port Definitions",
                "Section 5: Functional Flows and Inter-SWC Communication"
            ],
            "expected_page": 5,
            "expected_facts": [
                "InverterControl_SWC",
                "not declared",
                "missing"
            ],
            "should_abstain": False
        }
    ]


def main():
    print("Generating synthetic AUTOSAR HLD documents...")
    pdf1_path = INPUT_DATA_DIR / "synthetic_autosar_hld.pdf"
    pdf2_path = INPUT_DATA_DIR / "synthetic_autosar_hld_v2.pdf"
    
    create_pdf(V1_PAGES, pdf1_path)
    create_pdf(V2_PAGES, pdf2_path)

    dataset = generate_evaluation_dataset()
    dataset_path = INPUT_DATA_DIR / "evaluation_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Created evaluation dataset: {dataset_path} ({len(dataset)} questions)")

    # Also save to Evaluation_Results for evaluation runner
    eval_q_path = EVAL_RESULTS_DIR / "evaluation_questions.json"
    with open(eval_q_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Synced evaluation questions to {eval_q_path}")


if __name__ == "__main__":
    main()
