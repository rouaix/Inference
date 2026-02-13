#!/usr/bin/env python3
"""
Extrait le tokenizer depuis les métadonnées d'un fichier GGUF et le sauvegarde
comme tokenizer.json (format HuggingFace tokenizers).

Supporte :
  - BPE   (tokenizer.ggml.model = "gpt2")   → Magistral, Mistral v3+
  - UNIGRAM (tokenizer.ggml.model = "llama") → TinyLlama, Mistral v1/v2
"""

import numpy as np
from pathlib import Path


def _read_field(fields, name):
    """Lit un champ GGUF et retourne la liste de ses valeurs."""
    if name not in fields:
        return None
    field = fields[name]
    values = []
    for idx in field.data:
        val = field.parts[idx]
        if hasattr(val, 'dtype'):
            if val.dtype == np.uint8:
                # uint8 = bytes d'une string GGUF → décoder en UTF-8
                raw = val.tobytes()
                try:
                    values.append(raw.decode('utf-8'))
                except UnicodeDecodeError:
                    values.append(raw.decode('utf-8', errors='replace'))
            elif val.dtype.kind == 'f':
                if val.ndim == 0:
                    values.append(float(val.item()))
                else:
                    values.extend(float(x) for x in val.flat)
            else:
                # int32, int64, uint32… — vrais entiers
                if val.ndim == 0:
                    values.append(int(val.item()))
                else:
                    values.extend(int(x) for x in val.flat)
        elif isinstance(val, (bytes, bytearray)):
            try:
                values.append(val.decode('utf-8'))
            except UnicodeDecodeError:
                values.append(val.decode('utf-8', errors='replace'))
        else:
            values.append(val)
    return values if values else None


def extract_tokenizer(reader, output_dir: Path):
    """
    Extrait le tokenizer depuis les métadonnées GGUF et sauvegarde tokenizer.json.

    Parameters
    ----------
    reader     : gguf.GGUFReader déjà ouvert
    output_dir : dossier de destination (Path)
    """
    fields = reader.fields

    tokens     = _read_field(fields, "tokenizer.ggml.tokens")
    tok_types  = _read_field(fields, "tokenizer.ggml.token_type")
    merges_raw = _read_field(fields, "tokenizer.ggml.merges")
    model_type = _read_field(fields, "tokenizer.ggml.model")
    bos_ids    = _read_field(fields, "tokenizer.ggml.bos_token_id")
    eos_ids    = _read_field(fields, "tokenizer.ggml.eos_token_id")
    unk_ids    = _read_field(fields, "tokenizer.ggml.unknown_token_id")

    if not tokens:
        print("[WARN] tokenizer.ggml.tokens introuvable dans le GGUF")
        return

    tok_type_str = model_type[0] if model_type else "llama"
    bos_id = bos_ids[0] if bos_ids else 1
    eos_id = eos_ids[0] if eos_ids else 2
    unk_id = unk_ids[0] if unk_ids else 0

    vocab = {tok: i for i, tok in enumerate(tokens)}

    try:
        from tokenizers import Tokenizer, AddedToken
        from tokenizers.models import BPE, Unigram
        from tokenizers.pre_tokenizers import ByteLevel, Metaspace
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder, Metaspace as MetaspaceDecoder

        if tok_type_str in ("gpt2", "bpe"):
            # --- BPE (Magistral, Mistral v3+) ---
            merges = []
            if merges_raw:
                for m in merges_raw:
                    parts = m.split(" ", 1)
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))

            model = BPE(
                vocab=vocab,
                merges=merges,
                unk_token=tokens[unk_id] if unk_id < len(tokens) else "<unk>",
            )
            tokenizer = Tokenizer(model)
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            tokenizer.decoder = ByteLevelDecoder()

        else:
            # --- UNIGRAM (TinyLlama, LLaMA, Mistral v1/v2) ---
            scores_raw = _read_field(fields, "tokenizer.ggml.scores")
            scores = scores_raw if scores_raw else [0.0] * len(tokens)
            unigram_vocab = [(tok, float(sc)) for tok, sc in zip(tokens, scores)]
            model = Unigram(vocab=unigram_vocab, unk_id=unk_id, byte_fallback=True)
            tokenizer = Tokenizer(model)
            tokenizer.pre_tokenizer = Metaspace(replacement="\u2581", add_prefix_space=True)
            tokenizer.decoder = MetaspaceDecoder(replacement="\u2581", add_prefix_space=True)

        bos_token = tokens[bos_id] if bos_id < len(tokens) else "<s>"
        eos_token = tokens[eos_id] if eos_id < len(tokens) else "</s>"
        tokenizer.add_special_tokens([
            AddedToken(bos_token, special=True),
            AddedToken(eos_token, special=True),
        ])

        output_path = output_dir / "tokenizer.json"
        tokenizer.save(str(output_path))
        print(f"   [OK] Tokenizer extrait : {output_path.name} ({len(tokens)} tokens, type={tok_type_str})")

    except ImportError:
        print("[WARN] tokenizers (HuggingFace) non disponible — tokenizer non extrait")
        print("       Installez-le avec : pip install tokenizers")


if __name__ == "__main__":
    import argparse
    import gguf

    parser = argparse.ArgumentParser(description="Extrait le tokenizer d'un fichier GGUF")
    parser.add_argument("gguf_file", help="Chemin vers le fichier GGUF")
    parser.add_argument("output_dir", help="Dossier de destination pour tokenizer.json")
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.gguf_file)
    extract_tokenizer(reader, Path(args.output_dir))
