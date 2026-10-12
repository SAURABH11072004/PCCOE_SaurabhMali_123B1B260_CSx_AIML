# Architectural Entity Extraction Prompt

## Purpose
This prompt directs the model to extract formal AUTOSAR architectural entities from provided HLD text chunks.

---

## Entity Types
1. **Component**: Software Component (Atomic SWC, Composition, Sensor/Actuator SWC, Complex Device Driver).
2. **Interface**: Sender-Receiver or Client-Server interface definitions.
3. **Port**: P-Port (Provided Port) or R-Port (Required Port) attached to a component.
4. **Signal**: Data elements transmitted across interfaces with data type, range, and cycle time.
5. **Functional Flow**: End-to-end data/control flow across multiple components.
6. **Dependency**: Inter-SWC, RTE, or BSW dependencies.
7. **Requirement / Reference**: Traceable requirement IDs or normative document references.

---

## Prompt Template

```text
You are an AUTOSAR Architecture Parser. Analyze the following architectural text extracted from an HLD document and extract all formal architectural entities.

Output ONLY a valid JSON array of objects with the following schema:
[
  {
    "name": "<Entity Name, e.g. ThrottleControl_SWC, PpThrottlePos, Sig_ThrottleAngle>",
    "type": "<Component | Interface | Port | Signal | Functional Flow | Dependency | Requirement>",
    "description": "<Concise engineering description based solely on text>",
    "source_section": "<Section title or heading>",
    "supporting_excerpt": "<Direct quote from text supporting this extraction>",
    "confidence": 0.95
  }
]

RULES:
- Do not extract speculative entities.
- If an entity is not explicitly mentioned, omit it.
- Ensure the output strictly parses as valid JSON.

INPUT TEXT:
{text_passage}
```
