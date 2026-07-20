import argparse
from pathlib import Path

from pdf2image import convert_from_path


def convert_pdfs(input_dir: Path, output_dir: Path, dpi: int) -> int:
    """Convert every top-level PDF in ``input_dir`` to page-level PNG files."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")

    converted_pages = 0
    for pdf_path in pdf_paths:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        for page_index, page in enumerate(pages):
            output_path = output_dir / f"{pdf_path.stem}_p{page_index}.png"
            page.save(output_path, format="PNG")
            page.close()
            converted_pages += 1
            print(f"Saved {output_path}")

    if not pdf_paths:
        print(f"No PDF files found in {input_dir}")
    else:
        print(f"Converted {len(pdf_paths)} PDF file(s) into {converted_pages} PNG page(s).")
    return converted_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert each page of a directory of PDFs to PNG images.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing source PDF files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory in which to write PNG pages.")
    parser.add_argument("--dpi", type=int, default=300, help="Rasterisation resolution (default: 300 DPI).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_pdfs(args.input_dir, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
