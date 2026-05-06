"""
Offline pre-generation of semantic attributes for SADA.

Generates K domain-independent visual attributes per class using Qwen3.5-9B,
and saves the result to data/semantic_attributes.json.

Usage:
    python scripts/generate_attributes.py --dataset EuroSAT --num_attrs 5
    python scripts/generate_attributes.py --dataset all --num_attrs 5
"""

import argparse
import json
import os
import re
import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# Class name lists (mirrors VtT.py:run_lora)
# ---------------------------------------------------------------------------
LABEL_NAMES = {
    "CropDisease": [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
        "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
        "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ],
    "EuroSAT": [
        "Annual Crop Land", "Forest", "Herbaceous Vegetation Land",
        "Highway or Road", "Industrial Buildings", "Pasture Land",
        "Permanent Crop Land", "Residential Buildings", "River", "Sea or Lake",
    ],
    "ISIC": [
        "Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma",
        "Actinic Keratosis", "Benign Keratosis", "Dermatofibroma", "Vascular Lesion",
    ],
    "ChestX": [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumothorax",
    ],
}


_PLACEHOLDER_RE = re.compile(
    r"^(string\d*|str\d*|item\d*|attribute\d*|a|b|c|\.\.\.|<[^>]+>)$",
    re.IGNORECASE,
)

def _is_placeholder(attrs: list) -> bool:
    """Return True if the list looks like template/example strings."""
    return all(_PLACEHOLDER_RE.match(a) for a in attrs)


def parse_json_list(text: str, num_attrs: int, class_name: str) -> list:
    """Extract a JSON list from LLM output; fall back to class_name repeated."""
    # strip any <think>...</think> style blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # strip markdown code fences
    text = re.sub(r"```(?:json)?```?", "", text)
    # find all [...] blocks and try from last to first to skip thinking examples
    candidates = re.findall(r"\[[^\[\]]*\]", text, re.DOTALL)
    for raw in reversed(candidates):
        try:
            attrs = json.loads(raw)
            if not (isinstance(attrs, list) and len(attrs) >= 1):
                continue
            attrs = [str(a).strip() for a in attrs if str(a).strip()]
            if len(attrs) == 0 or _is_placeholder(attrs):
                continue
            while len(attrs) < num_attrs:
                attrs.append(class_name)
            return attrs[:num_attrs]
        except json.JSONDecodeError:
            continue
    print(f"  [warn] JSON parse failed for '{class_name}', using class name as fallback")
    return [class_name] * num_attrs


def generate_attributes(class_names, pipe, num_attrs):
    result = {}
    for cls_name in class_names:
        display = cls_name.replace("_", " ")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a computer vision expert. "
                    "When asked about a class, describe its visual appearance as seen in photographs or medical images. "
                    "Reply with ONLY a JSON array of strings, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"List {num_attrs} distinct visual semantic attributes that describe "
                    f"the appearance of '{display}' in an image. "
                    f"Each attribute should be a short descriptive phrase (e.g. 'dark irregular spots', 'smooth glossy surface'). "
                    "Return ONLY a JSON array of strings."
                ),
            },
        ]
        out = pipe(
            messages,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=False,
            return_full_text=False,
        )
        raw = out[0]["generated_text"]
        # return_full_text=False returns a string (new tokens only)
        if isinstance(raw, list):
            # fallback: full conversation list — take last assistant turn
            last = raw[-1]
            generated = last.get("content") or last.get("text") or str(last)
        else:
            generated = str(raw)
        attrs = parse_json_list(generated, num_attrs, cls_name)
        print(f"  {cls_name}: {attrs}")
        result[cls_name] = attrs
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "EuroSAT", "ISIC", "CropDisease", "ChestX"])
    parser.add_argument("--classes", nargs="+", default=None,
                        help="specific class names to (re-)generate, overrides --dataset")
    parser.add_argument("--num_attrs", type=int, default=5)
    parser.add_argument("--output", default="data/semantic_attributes.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if os.path.exists(args.output):
        with open(args.output) as f:
            all_attrs = json.load(f)
        print(f"Loaded {len(all_attrs)} existing entries from {args.output}")
    else:
        all_attrs = {}

    if args.classes:
        # explicit class list — always regenerate (ignore existing entries)
        to_generate = args.classes
        print(f"Regenerating {len(to_generate)} specified classes: {to_generate}")
    else:
        datasets = list(LABEL_NAMES.keys()) if args.dataset == "all" else [args.dataset]
        all_classes = []
        for ds in datasets:
            all_classes.extend(LABEL_NAMES[ds])
        seen = set()
        unique_classes = [c for c in all_classes if not (c in seen or seen.add(c))]
        to_generate = [c for c in unique_classes if c not in all_attrs]
        print(f"Generating attributes for {len(to_generate)} classes "
              f"(skipping {len(unique_classes) - len(to_generate)} already done)")

    if not to_generate:
        print("Nothing to generate.")
        return

    print(f"Loading model {args.model} ...")
    pipe = pipeline(
        "text-generation",
        model=args.model,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    new_attrs = generate_attributes(to_generate, pipe, args.num_attrs)
    all_attrs.update(new_attrs)

    with open(args.output, "w") as f:
        json.dump(all_attrs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_attrs)} entries to {args.output}")


if __name__ == "__main__":
    main()
