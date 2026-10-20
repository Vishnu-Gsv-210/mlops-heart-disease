"""Validate Kubernetes manifests: parse YAML and check required deployment fields.

Checks deployment/kubernetes/*.yaml when that directory exists; otherwise it
validates whatever deployment/*.yaml manifests exist and says so.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = ROOT / "deployment"
KUBERNETES_DIR = DEPLOYMENT_DIR / "kubernetes"

REQUIRED_TOP_LEVEL = {"apiVersion", "kind", "metadata"}
CONTAINER_REQUIRED = {"name", "image"}


def check_document(doc: dict, path: Path, index: int) -> list[str]:
    """Return a list of human-readable problems for one manifest document."""
    errors: list[str] = []
    label = f"{path.name}[{index}] ({doc.get('kind', 'unknown')})"
    missing = REQUIRED_TOP_LEVEL - set(doc)
    if missing:
        errors.append(f"{label}: missing top-level fields {sorted(missing)}")
        return errors
    if not (doc.get("metadata") or {}).get("name"):
        errors.append(f"{label}: metadata.name is required")
        return errors

    kind = (doc.get("kind") or "").lower()
    spec = doc.get("spec") or {}
    if kind == "deployment":
        if "replicas" not in spec:
            errors.append(f"{label}: spec.replicas is required")
        selector = ((spec.get("selector") or {}).get("matchLabels") or {})
        if not selector:
            errors.append(f"{label}: spec.selector.matchLabels is required")
        template = (spec.get("template") or {})
        containers = (((template.get("spec") or {}).get("containers")) or [])
        if not containers:
            errors.append(f"{label}: spec.template.spec.containers is required")
        for i, container in enumerate(containers):
            missing_container = CONTAINER_REQUIRED - set(container)
            if missing_container:
                errors.append(
                    f"{label}: container[{i}] missing fields {sorted(missing_container)}"
                )
        template_metadata_labels = ((template.get("metadata") or {}).get("labels") or {})
        if selector and not template_metadata_labels:
            errors.append(f"{label}: spec.template.metadata.labels must match the selector")
    elif kind == "service":
        if not (spec.get("selector") or {}):
            errors.append(f"{label}: spec.selector is required")
        ports = spec.get("ports") or []
        if not ports:
            errors.append(f"{label}: spec.ports is required")
        for i, port in enumerate(ports):
            if "port" not in port:
                errors.append(f"{label}: spec.ports[{i}].port is required")
    return errors


def validate_manifest_dir(directory: Path) -> tuple[list[Path], list[str]]:
    """Validate every *.yaml/*.yml file in a directory; return (files, errors)."""
    manifests = sorted(
        p for p in directory.glob("*.yaml")
    ) + sorted(
        p for p in directory.glob("*.yml")
    )
    errors: list[str] = []
    for path in manifests:
        try:
            documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        except yaml.YAMLError as exc:
            errors.append(f"{path.name}: invalid YAML ({exc})")
            continue
        for index, doc in enumerate(documents):
            if doc is None:  # empty document between --- separators
                continue
            if not isinstance(doc, dict):
                errors.append(f"{path.name}[{index}]: document is not a mapping")
                continue
            errors.extend(check_document(doc, path, index))
    return manifests, errors


def main() -> int:
    if KUBERNETES_DIR.is_dir() and any(KUBERNETES_DIR.glob("*.yaml")):
        target = KUBERNETES_DIR
        source = "deployment/kubernetes/"
    else:
        target = DEPLOYMENT_DIR
        source = "deployment/ (kubernetes/ not present yet)"
    manifests, errors = validate_manifest_dir(target)

    print(f"Validating Kubernetes manifests from {source}")
    if not manifests:
        print("No manifest files found; nothing to validate.")
        return 0
    for path in manifests:
        print(f"  checked: {path.relative_to(ROOT)}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"OK: {len(manifests)} manifest file(s) parsed; all required fields present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
