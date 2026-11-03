"""Analysis pipeline for the `main.py` Vector PDF Extractor.

Requires a sibling main.py exposing `extract_pdf(pdf_path)` with the
structure from the previous lesson: pages / vector_paths / commands.

Usage:
    python pipeline.py demo.pdf --rules rules.json --out result.json

Not a semantic CAD-symbol detector. Works with known PDF primitives and
configurable geometric rules; circular paths are heuristic candidates.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pdfplumber
from docx import Document
from pydantic import BaseModel, Field, model_validator
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree

from main import extract_pdf


class Rule(BaseModel):
    name: str = Field(min_length=1)
    kind: Literal["rectangle", "circle"]
    min_short_side_pt: float = Field(ge=0)
    max_short_side_pt: float = Field(gt=0)
    min_long_side_pt: float = Field(ge=0)
    max_long_side_pt: float = Field(gt=0)
    group_gap_pt: float = Field(default=0, ge=0)
    layer: str | None = None

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.min_short_side_pt > self.max_short_side_pt:
            raise ValueError("min_short_side_pt > max_short_side_pt")
        if self.min_long_side_pt > self.max_long_side_pt:
            raise ValueError("min_long_side_pt > max_long_side_pt")
        return self


class RuleSet(BaseModel):
    rules: list[Rule] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_names(self):
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Rule names must be unique")
        return self


class Feature(BaseModel):
    feature_id: str
    rule: str
    page_number: int = Field(ge=1)
    path_id: int = Field(ge=1)
    command_index: int | None
    layer: str | None
    bbox_pt: tuple[float, float, float, float]
    center_pt: tuple[float, float]
    short_side_pt: float = Field(ge=0)
    long_side_pt: float = Field(ge=0)


class Group(BaseModel):
    group_id: str
    rule: str
    count: int = Field(ge=1)
    feature_ids: list[str]
    bbox_pt: tuple[float, float, float, float]


class PageResult(BaseModel):
    page_number: int = Field(ge=1)
    words: list[dict]
    primitive_counts: dict[str, int]
    features: list[Feature]
    groups: list[Group]
    counts_by_rule: dict[str, int]


class PipelineResult(BaseModel):
    source_pdf: str
    coordinate_units: Literal["pt"] = "pt"
    pages: list[PageResult]
    docx: dict | None = None
    email: str | None = None
    report: list[dict] | None = None


def rounded_bounds(geometry) -> tuple[float, float, float, float]:
    return tuple(round(float(v), 2) for v in geometry.bounds)


def rectangle_dimensions(polygon: Polygon) -> tuple[float, float]:
    """Dimensions of the minimum rotated rectangle (handles rotated quads)."""
    rotated = polygon.minimum_rotated_rectangle
    points = list(rotated.exterior.coords)
    sides = [math.dist(points[i], points[i + 1]) for i in range(4)]
    return min(sides), max(sides)


def shapes_from_path(path: dict):
    """Yield (kind, command_index, Shapely geometry) for known primitives."""
    for index, command in enumerate(path["commands"]):
        kind = command["type"]
        if kind == "rectangle":
            shape = box(*command["bbox"])
            if shape.area > 0:
                yield "rectangle", index, shape
        elif kind == "quadrilateral":
            shape = Polygon(command["corners"])
            if shape.is_valid and shape.area > 0:
                yield "rectangle", index, shape

    # Heuristic: four Bézier segments in an approximately square path
    # can be a circle, but it is not guaranteed to be a circle.
    if path.get("circle_candidate", False):
        x0, y0, x1, y1 = path["bbox"]
        width, height = x1 - x0, y1 - y0
        if width > 0 and height > 0:
            center = Point((x0 + x1) / 2, (y0 + y1) / 2)
            yield "circle", None, center.buffer(min(width, height) / 2)


def group_features(features: list[Feature], geometries: dict, rules: list[Rule], page_number: int) -> list[Group]:
    groups: list[Group] = []

    for rule in rules:
        chosen = [feature for feature in features if feature.rule == rule.name]
        if not chosen:
            continue
        shapes = [geometries[f.feature_id] for f in chosen]
        tree = STRtree(shapes)  # Shapely 2 returns indices, not geometries
        parent = list(range(len(shapes)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a

        for i, shape in enumerate(shapes):
            candidate_indices = tree.query(shape.buffer(rule.group_gap_pt))
            for candidate_index in candidate_indices:
                j = int(candidate_index)
                if j <= i or chosen[i].layer != chosen[j].layer:
                    continue
                if shape.distance(shapes[j]) <= rule.group_gap_pt:
                    union(i, j)

        buckets: dict[int, list[int]] = defaultdict(list)
        for i in range(len(chosen)):
            buckets[find(i)].append(i)

        for number, indices in enumerate(buckets.values(), start=1):
            bounds = [shapes[i].bounds for i in indices]
            groups.append(Group(
                group_id=f"p{page_number}:{rule.name}:g{number}",
                rule=rule.name,
                count=len(indices),
                feature_ids=[chosen[i].feature_id for i in indices],
                bbox_pt=(
                    round(min(b[0] for b in bounds), 2),
                    round(min(b[1] for b in bounds), 2),
                    round(max(b[2] for b in bounds), 2),
                    round(max(b[3] for b in bounds), 2),
                ),
            ))

    return groups


def detect_page(page: dict, rules: list[Rule]) -> PageResult:
    page_number = page["page_number"]
    features: list[Feature] = []
    geometries: dict = {}

    for path in page["vector_paths"]:
        for kind, command_index, shape in shapes_from_path(path):
            if kind == "rectangle":
                short, long = rectangle_dimensions(shape)
            else:
                x0, y0, x1, y1 = path["bbox"]
                short, long = sorted([x1 - x0, y1 - y0])

            for rule in rules:
                if rule.kind != kind:
                    continue
                if rule.layer is not None and rule.layer != path["layer"]:
                    continue
                if not (rule.min_short_side_pt <= short <= rule.max_short_side_pt):
                    continue
                if not (rule.min_long_side_pt <= long <= rule.max_long_side_pt):
                    continue

                feature_id = f"p{page_number}:{rule.name}:{path['id']}:{command_index}"
                features.append(Feature(
                    feature_id=feature_id,
                    rule=rule.name,
                    page_number=page_number,
                    path_id=path["id"],
                    command_index=command_index,
                    layer=path["layer"],
                    bbox_pt=rounded_bounds(shape),
                    center_pt=(round(shape.centroid.x, 2), round(shape.centroid.y, 2)),
                    short_side_pt=round(short, 2),
                    long_side_pt=round(long, 2),
                ))
                geometries[feature_id] = shape

    return PageResult(
        page_number=page_number,
        words=page["words"],
        primitive_counts=page["primitive_counts"],
        features=features,
        groups=group_features(features, geometries, rules, page_number),
        counts_by_rule={r.name: sum(f.rule == r.name for f in features) for r in rules},
    )


def read_docx(path: str | Path) -> dict:
    document = Document(path)
    return {
        "paragraphs": [p.text for p in document.paragraphs if p.text.strip()],
        "tables": [
            [[cell.text for cell in row.cells] for row in table.rows]
            for table in document.tables
        ],
    }


def read_report_pdf(path: str | Path) -> list[dict]:
    pages: list[dict] = []
    with pdfplumber.open(path) as document:
        for page_number, page in enumerate(document.pages, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.extract_text() or "",
                "tables": page.extract_tables(),
            })
    return pages


def run_pipeline(
    pdf_path: str | Path,
    rules_path: str | Path,
    *,
    docx: str | Path | None = None,
    email: str | Path | None = None,
    report: str | Path | None = None,
) -> PipelineResult:
    rules = RuleSet.model_validate_json(Path(rules_path).read_text(encoding="utf-8"))
    extracted = extract_pdf(pdf_path)
    return PipelineResult(
        source_pdf=extracted["file"],
        pages=[detect_page(page, rules.rules) for page in extracted["pages"]],
        docx=read_docx(docx) if docx else None,
        email=Path(email).read_text(encoding="utf-8") if email else None,
        report=read_report_pdf(report) if report else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze vector PDF drawings using geometric rules")
    parser.add_argument("pdf", help="Input vector PDF")
    parser.add_argument("--rules", default="rules.json")
    parser.add_argument("--out", default="result.json")
    parser.add_argument("--docx", help="Optional DOCX instructions")
    parser.add_argument("--email", help="Optional UTF-8 email TXT")
    parser.add_argument("--report", help="Optional text-based report PDF")
    args = parser.parse_args()

    result = run_pipeline(args.pdf, args.rules, docx=args.docx, email=args.email, report=args.report)
    Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Zapisano: {args.out}")
    for page in result.pages:
        print(f"Strona {page.page_number}: {page.counts_by_rule}; grupy={len(page.groups)}")


if __name__ == "__main__":
    main()
