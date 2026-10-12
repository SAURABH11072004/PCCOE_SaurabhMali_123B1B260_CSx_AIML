#!/usr/bin/env python3
"""Build TechPulse Submission Package.

Organizes the working project at E:\\AUTOSAR HLD into the exact
eight-folder TechPulse submission structure and builds a validated ZIP.

Submission root:
    PCCOE_Saurabh_Mali_123B1B260_PRN_CSx_AIML/
    ├── Synopsis/
    ├── Input_Data/
    ├── Code/
    ├── Model_Prompts_Config/
    ├── Evaluation_Results/
    ├── Documentation/
    ├── Video/
    └── Declarations/

Design rules
------------
- Idempotent: safe to run repeatedly; staging → atomic rename removes the
  previous copy only after the new one is fully built.
- Explicit allowlist: every copied file is named specifically.
- Secrets blocked: .env is never copied; .env.example is in Model_Prompts_Config/.
- No nesting: Code/ does not contain Code/, src/ is not at submission root.
- Missing artifacts: reported clearly, never fabricated.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
SUBMISSION_NAME = "PCCOE_Saurabh_Mali_123B1B260_PRN_CSx_AIML"
SUBMISSION_DIR  = PROJECT_ROOT / SUBMISSION_NAME
ZIP_PATH        = PROJECT_ROOT / f"{SUBMISSION_NAME}.zip"
STAGE_DIR       = PROJECT_ROOT / f"{SUBMISSION_NAME}_staging"

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(msg)


def warn(msg: str):
    print(f"  [WARN] {msg}", file=sys.stderr)


def copy_file(src: Path, dest_dir: Path) -> bool:
    """Copy a single file into dest_dir. Returns True on success."""
    if not src.exists():
        warn(f"Source file not found, skipping: {src.relative_to(PROJECT_ROOT)}")
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)
    return True


def copy_tree(src: Path, dest: Path):
    """Copy a directory tree, excluding caches and bytecode."""
    if not src.exists():
        warn(f"Source directory not found, skipping: {src.relative_to(PROJECT_ROOT)}")
        return
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pyo", ".pytest_cache",
        ".mypy_cache", "*.egg-info",
    )
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=ignore)


def build_staging(stage: Path):
    """Populate the staging directory with all submission content."""
    P = PROJECT_ROOT

    # ── Mandatory subdirectory scaffold ─────────────────────────────────────
    required_subdirs = [
        "Synopsis", "Input_Data", "Code",
        "Model_Prompts_Config", "Evaluation_Results",
        "Documentation", "Video", "Declarations",
    ]
    for sd in required_subdirs:
        (stage / sd).mkdir(parents=True, exist_ok=True)

    # ── Synopsis ─────────────────────────────────────────────────────────────
    log("\n[Synopsis]")
    copy_file(P / "Synopsis" / "Project_Synopsis.md",  stage / "Synopsis")
    copy_file(P / "Synopsis" / "Project_Synopsis.pdf", stage / "Synopsis")

    # ── Input_Data ───────────────────────────────────────────────────────────
    log("\n[Input_Data]")
    copy_file(P / "Input_Data" / "synthetic_autosar_hld.pdf",    stage / "Input_Data")
    copy_file(P / "Input_Data" / "synthetic_autosar_hld_v2.pdf", stage / "Input_Data")
    copy_file(P / "Input_Data" / "evaluation_dataset.json",      stage / "Input_Data")
    copy_file(P / "Input_Data" / "README.md",                    stage / "Input_Data")

    # ── Code ─────────────────────────────────────────────────────────────────
    log("\n[Code]")
    code = stage / "Code"
    copy_file(P / "app.py",          code)
    copy_file(P / "requirements.txt", code)
    copy_file(P / ".gitignore",       code)
    copy_file(P / "conftest.py",      code)
    copy_file(P / "pytest.ini",       code)

    # Source code sub-packages (no __pycache__, no .pyc)
    for pkg in ["src", "tests", "evaluation", "scripts"]:
        log(f"  Copying {pkg}/")
        copy_tree(P / pkg, code / pkg)

    # Code README
    code_readme = """\
# AUTOSAR HLD Document Analysis Assistant — Source Code

## Quick Start
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then add your GOOGLE_API_KEY
streamlit run app.py
```

## Run Tests
```powershell
pytest tests/ -v
```

## Run Evaluation
```powershell
python -m evaluation.run_evaluation
```

## Notes
- `.env.example` with placeholder credentials is in `Model_Prompts_Config/`.
- Embedding model (`all-MiniLM-L6-v2`) downloads automatically on first use.
- ChromaDB data directory is created at runtime (`./chroma_db/`).
"""
    (code / "README.md").write_text(code_readme, encoding="utf-8")
    log("  Written Code/README.md")

    # ── Model_Prompts_Config ─────────────────────────────────────────────────
    log("\n[Model_Prompts_Config]")
    mpc_src = P / "Model_Prompts_Config"
    for name in [
        "model_configuration.md",
        "rag_system_prompt.md",
        "extraction_prompt.md",
        "evaluation_methodology.md",
    ]:
        copy_file(mpc_src / name, stage / "Model_Prompts_Config")
    # .env.example belongs here (model/config credentials template)
    copy_file(P / ".env.example", stage / "Model_Prompts_Config")

    # ── Evaluation_Results ───────────────────────────────────────────────────
    log("\n[Evaluation_Results]")
    er_src = P / "Evaluation_Results"
    for name in [
        "Evaluation_Report.md",
        "answer_results.json",
        "citation_validation.json",
        "evaluation_questions.json",
        "metrics.json",
        "retrieval_results.json",
    ]:
        copy_file(er_src / name, stage / "Evaluation_Results")

    # ── Documentation ────────────────────────────────────────────────────────
    log("\n[Documentation]")
    doc_src = P / "Documentation"
    for name in [
        "Technical_Report.md",
        "Technical_Report.pdf",
        "System_Architecture.md",
        "Installation_Guide.md",
        "User_Guide.md",
        "Testing_Report.md",
        "Known_Limitations.md",
        "Video_Demonstration_Script.md",
    ]:
        copy_file(doc_src / name, stage / "Documentation")

    # ── Video ─────────────────────────────────────────────────────────────────
    log("\n[Video]")
    copy_file(P / "Video" / "README.md", stage / "Video")
    # Copy any actual video files if present
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        for vf in (P / "Video").glob(ext):
            copy_file(vf, stage / "Video")

    # ── Declarations ──────────────────────────────────────────────────────────
    log("\n[Declarations]")
    decl_src = P / "Declarations"
    copy_file(decl_src / "Student_Declaration_Template.md",  stage / "Declarations")
    copy_file(decl_src / "Faculty_Approval_Checklist.md",    stage / "Declarations")

    # ── Submission Checklist (at submission root) ─────────────────────────────
    copy_file(P / "Submission_Checklist.md", stage)

    log(f"\nStaging complete: {stage}")


def verify_structure(d: Path) -> list[str]:
    """Return list of missing required paths within the submission directory."""
    required = [
        "Synopsis/Project_Synopsis.md",
        "Synopsis/Project_Synopsis.pdf",
        "Input_Data/synthetic_autosar_hld.pdf",
        "Input_Data/evaluation_dataset.json",
        "Code/app.py",
        "Code/requirements.txt",
        "Code/src/rag_pipeline.py",
        "Code/tests/__init__.py",
        "Code/evaluation/run_evaluation.py",
        "Model_Prompts_Config/.env.example",
        "Model_Prompts_Config/model_configuration.md",
        "Evaluation_Results/metrics.json",
        "Evaluation_Results/Evaluation_Report.md",
        "Documentation/Technical_Report.md",
        "Documentation/Technical_Report.pdf",
        "Documentation/Installation_Guide.md",
        "Documentation/System_Architecture.md",
        "Declarations/Student_Declaration_Template.md",
        "Video/README.md",
        "Submission_Checklist.md",
    ]
    missing = []
    for rel in required:
        if not (d / rel).exists():
            missing.append(rel)
    return missing


def verify_no_secrets(d: Path):
    """Raise if any real .env (not .env.example) is found."""
    for f in d.rglob(".env"):
        raise RuntimeError(f"SECURITY VIOLATION — real .env found in submission: {f}")
    for f in d.rglob("*.env"):
        if f.name != ".env.example":
            raise RuntimeError(f"SECURITY VIOLATION — secret env file in submission: {f}")


def verify_no_nesting(d: Path):
    """Raise if the submission folder is nested inside itself."""
    nested = d / SUBMISSION_NAME
    if nested.exists():
        raise RuntimeError(f"STRUCTURE VIOLATION — submission nested inside itself: {nested}")
    # Confirm exactly 8 folders at root (+ optional Submission_Checklist.md)
    top_dirs = [p for p in d.iterdir() if p.is_dir()]
    top_names = sorted(p.name for p in top_dirs)
    expected = sorted([
        "Synopsis", "Input_Data", "Code",
        "Model_Prompts_Config", "Evaluation_Results",
        "Documentation", "Video", "Declarations",
    ])
    extra = [n for n in top_names if n not in expected]
    if extra:
        raise RuntimeError(f"STRUCTURE VIOLATION — unexpected top-level dirs: {extra}")


def build_zip(source_dir: Path, zip_path: Path):
    """Create the submission ZIP from the verified submission directory."""
    log(f"\nBuilding ZIP: {zip_path.name}")
    tmp_zip = zip_path.with_suffix(".tmp.zip")
    if tmp_zip.exists():
        tmp_zip.unlink()

    # Paths that must NEVER enter the ZIP
    blocked_names = {"__pycache__", ".pytest_cache", ".mypy_cache",
                     ".venv", "venv", "chroma_db", "chroma_eval_db",
                     "tmp_mock", ".env"}

    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)
            # Prune traversal — modify dirs in-place
            dirs[:] = [
                d for d in dirs
                if d not in blocked_names and not d.startswith(".")
            ]
            for fname in files:
                if fname in blocked_names:
                    continue
                if fname == ".env":          # explicit check
                    continue
                if fname.endswith(".pyc") or fname.endswith(".pyo"):
                    continue
                fpath = root_path / fname
                arcname = str(fpath.relative_to(PROJECT_ROOT))
                zf.write(fpath, arcname=arcname)

    # Atomic replace
    if zip_path.exists():
        zip_path.unlink()
    tmp_zip.rename(zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    log(f"ZIP written: {zip_path} ({size_mb:.2f} MB)")


def validate_zip(zip_path: Path):
    """Open the final ZIP and verify mandatory entries are present."""
    log("\nValidating ZIP contents...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    # Normalize path separators
    names_fwd = {n.replace("\\", "/") for n in names}

    mandatory = [
        f"{SUBMISSION_NAME}/Synopsis/Project_Synopsis.pdf",
        f"{SUBMISSION_NAME}/Input_Data/synthetic_autosar_hld.pdf",
        f"{SUBMISSION_NAME}/Code/app.py",
        f"{SUBMISSION_NAME}/Code/requirements.txt",
        f"{SUBMISSION_NAME}/Code/src/rag_pipeline.py",
        f"{SUBMISSION_NAME}/Model_Prompts_Config/.env.example",
        f"{SUBMISSION_NAME}/Evaluation_Results/metrics.json",
        f"{SUBMISSION_NAME}/Documentation/Technical_Report.pdf",
        f"{SUBMISSION_NAME}/Declarations/Student_Declaration_Template.md",
    ]

    all_ok = True
    for item in mandatory:
        normalized = item.replace("\\", "/")
        found = any(normalized in n for n in names_fwd)
        status = "[OK]   " if found else "[MISS] "
        log(f"  {status} {item}")
        if not found:
            all_ok = False

    # Security check
    for n in names_fwd:
        if n.endswith("/.env") or n.split("/")[-1] == ".env":
            raise RuntimeError(f"SECURITY VIOLATION — .env in ZIP: {n}")
        if "__pycache__" in n:
            raise RuntimeError(f"HYGIENE VIOLATION — __pycache__ in ZIP: {n}")

    log(f"\n  Total ZIP entries: {len(names)}")
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def build_package():
    sep = "=" * 72
    log(sep)
    log("TECHPULSE SUBMISSION PACKAGE BUILDER")
    log(f"Submission name : {SUBMISSION_NAME}")
    log(f"Project root    : {PROJECT_ROOT}")
    log(sep)

    # Step 1 — Build into a staging directory
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    build_staging(STAGE_DIR)

    # Step 2 — Pre-publish verification
    log("\n[Verification]")
    missing = verify_structure(STAGE_DIR)
    verify_no_secrets(STAGE_DIR)
    verify_no_nesting(STAGE_DIR)

    if missing:
        log("\n  Missing required files (not fabricated):")
        for m in missing:
            log(f"    - {m}")
    else:
        log("  All mandatory files present.")

    # Step 3 — Atomic swap: staging → submission dir
    if SUBMISSION_DIR.exists():
        old_backup = PROJECT_ROOT / f"{SUBMISSION_NAME}_prev"
        if old_backup.exists():
            shutil.rmtree(old_backup)
        SUBMISSION_DIR.rename(old_backup)
    STAGE_DIR.rename(SUBMISSION_DIR)
    # Remove backup only after successful rename
    old_backup = PROJECT_ROOT / f"{SUBMISSION_NAME}_prev"
    if old_backup.exists():
        shutil.rmtree(old_backup)
    log(f"\nSubmission directory ready: {SUBMISSION_DIR}")

    # Step 4 — Build and validate ZIP
    build_zip(SUBMISSION_DIR, ZIP_PATH)
    all_ok = validate_zip(ZIP_PATH)

    # Step 5 — Final report
    log(f"\n{sep}")
    if missing:
        log("PACKAGE BUILT — some artifacts are missing (see above).")
        log("Do NOT fabricate replacements; obtain real files before submission.")
    elif all_ok:
        log("ALL CHECKS PASSED — submission package is ready.")
    else:
        log("PACKAGE BUILT — some ZIP entries could not be verified (see above).")
    log(f"Submission dir : {SUBMISSION_DIR}")
    log(f"ZIP archive    : {ZIP_PATH}")
    log(sep)

    # Print final tree (top 2 levels)
    log("\nSubmission folder tree (depth 2):")
    items = sorted(SUBMISSION_DIR.iterdir())
    for i, item in enumerate(items):
        prefix = "+-- "
        if item.is_dir():
            log(f"  {prefix}{item.name}/")
            children = sorted(item.iterdir())
            for j, child in enumerate(children[:12]):
                c_prefix = "|   +-- " if j < len(children) - 1 else "|   `-- "
                log(f"  {c_prefix}{child.name}")
            if len(children) > 12:
                log(f"  |   ... and {len(children) - 12} more")
        else:
            log(f"  {prefix}{item.name}")


if __name__ == "__main__":
    build_package()
