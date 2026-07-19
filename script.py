from pdf2image import convert_from_path
import os

src_dir, out_dir = "/Users/hasam/Desktop/data", "/Users/hasam/Desktop/data/images"
for fname in os.listdir(src_dir):
    if fname.endswith(".pdf"):
        pages = convert_from_path(os.path.join(src_dir, fname), dpi=300)
        for i, page in enumerate(pages):
            page.save(os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_p{i}.png"))
