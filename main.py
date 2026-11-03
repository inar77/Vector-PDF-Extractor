
import argparse
import json
from collections import Counter
from pathlib import Path

import pymupdf


def xy(point):
    """Zwraca współrzędne punktu."""
    return [
        round(point.x, 2),
        round(point.y, 2)
    ]


def bbox(rect):
    """Zwraca współrzędne prostokąta ograniczającego."""
    return [
        round(value, 2)
        for value in (
            rect.x0,
            rect.y0,
            rect.x1,
            rect.y1
        )
    ]


def serialize_command(item):
    """Konwertuje instrukcję rysowania PDF na JSON."""

    command = item[0]

    if command == "l":
        return {
            "type": "line",
            "start": xy(item[1]),
            "end": xy(item[2])
        }

    if command == "re":
        return {
            "type": "rectangle",
            "bbox": bbox(item[1])
        }

    if command == "qu":
        quad = item[1]

        return {
            "type": "quadrilateral",
            "corners": [
                xy(quad.ul),
                xy(quad.ur),
                xy(quad.lr),
                xy(quad.ll)
            ]
        }

    if command == "c":
        return {
            "type": "bezier",
            "points": [
                xy(point)
                for point in item[1:5]
            ]
        }

    return {
        "type": "unknown",
        "pdf_command": str(command)
    }


def extract_pdf(pdf_path):
    """Odczytuje tekst i geometrię z PDF."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {pdf_path}"
        )

    result = {
        "file": pdf_path.name,
        "pages": []
    }

    with pymupdf.open(pdf_path) as document:

        if not document.is_pdf:
            raise ValueError("Plik musi być dokumentem PDF")

        for page_number, page in enumerate(
            document, start=1
        ):

            # 1. Tekst i współrzędne
            words = []

            for word in page.get_text("words", sort=True):
                words.append({
                    "text": word[4],
                    "bbox": [
                        round(value, 2)
                        for value in word[:4]
                    ]
                })

            paths = []
            counts = Counter()

            # 2. Ekstrakcja geometrii wektorowej
            for path_id, drawing in enumerate(
                page.get_drawings(), start=1
            ):

                commands = [
                    serialize_command(item)
                    for item in drawing["items"]
                ]

                kinds = [
                    command["type"]
                    for command in commands
                ]

                # 3. Liczenie podstawowych figur
                counts["lines"] += kinds.count("line")

                counts["rectangles"] += kinds.count(
                    "rectangle"
                )

                counts["quadrilaterals"] += kinds.count(
                    "quadrilateral"
                )

                counts["bezier_segments"] += kinds.count(
                    "bezier"
                )

                # 4. Wstępne wykrywanie okręgów
                bounds = drawing["rect"]

                width = bounds.width
                height = bounds.height

                circle_candidate = (
                    len(commands) == 4
                    and all(
                        kind == "bezier"
                        for kind in kinds
                    )
                    and width > 0
                    and height > 0
                    and abs(width - height)
                    / max(width, height) <= 0.03
                )

                if circle_candidate:
                    counts["circle_candidates"] += 1

                # 5. Zapis danych geometrycznych
                paths.append({
                    "id": path_id,
                    "layer": drawing.get("layer") or None,
                    "bbox": bbox(bounds),
                    "line_width": drawing.get("width"),
                    "commands": commands,
                    "circle_candidate": circle_candidate
                })

            result["pages"].append({
                "page_number": page_number,
                "size_pt": [
                    round(page.rect.width, 2),
                    round(page.rect.height, 2)
                ],
                "words": words,
                "vector_paths": paths,
                "primitive_counts": {
                    name: counts[name]
                    for name in (
                        "lines",
                        "rectangles",
                        "quadrilaterals",
                        "bezier_segments",
                        "circle_candidates"
                    )
                }
            })

    return result


def create_demo_pdf(output_path):
    """Tworzy testowy rysunek techniczny."""

    document = pymupdf.open()

    page = document.new_page(
        width=600,
        height=400
    )

    page.insert_text(
        (55, 55),
        "CAD EXPORT - SAMPLE DRAWING",
        fontsize=13
    )

    # Cztery prostokątne elementy
    for x in (100, 170, 240, 310):
        page.draw_rect(
            pymupdf.Rect(
                x, 125,
                x + 40, 145
            ),
            width=1
        )

    # Okrąg
    page.draw_circle(
        pymupdf.Point(200, 245),
        18,
        width=1
    )

    # Linia
    page.draw_line(
        (65, 305),
        (400, 305),
        width=1
    )

    document.save(output_path)
    document.close()


def main():
    parser = argparse.ArgumentParser(
        description="Vector PDF to JSON extractor"
    )

    parser.add_argument(
        "pdf",
        nargs="?",
        help="Ścieżka do wektorowego PDF"
    )

    parser.add_argument(
        "--output",
        default="output.json"
    )

    parser.add_argument(
        "--demo",
        action="store_true"
    )

    args = parser.parse_args()

    if args.demo:
        create_demo_pdf("demo.pdf")
        pdf_path = "demo.pdf"

    elif args.pdf:
        pdf_path = args.pdf

    else:
        parser.error("Podaj plik PDF lub --demo")

    result = extract_pdf(pdf_path)

    Path(args.output).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"Zapisano: {args.output}")

    for page in result["pages"]:
        print(
            f"Strona {page['page_number']}: "
            f"{page['primitive_counts']}"
        )


if __name__ == "__main__":
    main()
