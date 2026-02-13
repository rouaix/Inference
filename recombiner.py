import argparse
import json
import numpy as np
import gguf
from pathlib import Path
from typing import Dict, Any, List

class GGUFRecombiner:
    """
    Reconstruit un fichier GGUF à partir de fragments et d'un manifeste.
    """
    def __init__(self, fragments_dir: str):
        self.fragments_dir = Path(fragments_dir)
        self.manifest_path = self.fragments_dir / "manifest.json"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)

    def recombine(self, output_file: str):
        print(f"📖 Lecture du manifeste : {self.manifest_path}")
        output_path = Path(output_file)

        # 1. Architecture et initialisation
        metadata = self.manifest.get("metadata", {})
        arch = metadata.get("general.architecture", "llama")

        # Hack: si arch est une liste d'entiers (ASCII), on convertit en string
        if isinstance(arch, list) and all(isinstance(x, int) for x in arch):
            try:
                arch = "".join(chr(x) for x in arch)
                print(f"🔧 Correction architecture: {arch}")
            except:
                pass

        print(f"🔨 Initialisation GGUFWriter (arch: {arch})...")
        gw = gguf.GGUFWriter(output_path, arch)

        # 2. Restauration des métadonnées (KV pairs)
        print("📝 Restauration des métadonnées...")
        for key, val in metadata.items():
            if key in ["general.architecture", "GGUF.version"]: continue

            try:
                gw.add_key_value(key, val)
            except Exception as e:
                print(f"⚠️ Impossible d'ajouter la métadonnée {key}: {e}")

        # 3. Préparation des fragments
        print("📦 Analyse des fragments...")
        tensor_groups = {}
        for frag in self.manifest["fragments"]:
            tname = frag.get("tensor_name")
            if not tname:
                print(f"❌ Erreur: fragment {frag['fragment_id']} sans tensor_name. Impossible de reconstruire.")
                continue

            if tname not in tensor_groups:
                tensor_groups[tname] = []
            tensor_groups[tname].append(frag)

        # 4. Reassemblage et ajout des tenseurs
        print(f"🔗 Reassemblage de {len(tensor_groups)} tenseurs...")

        count = 0
        total = len(tensor_groups)

        for tname, frags in tensor_groups.items():
            frags.sort(key=lambda x: x["shard_index"])

            total_shards = frags[0]["total_shards"]
            if len(frags) != total_shards:
                print(f"❌ Tenseur {tname} incomplet ({len(frags)}/{total_shards} shards). Skip.")
                continue

            full_data = bytearray()
            for f in frags:
                f_path = self.fragments_dir / f"{f['fragment_id']}.dat"
                with open(f_path, "rb") as bf:
                    full_data.extend(bf.read())

            dtype_str = frags[0]["dtype"]
            shape = tuple(frags[0]["shape"])
            tensor_type_name = frags[0].get("tensor_type", "F32")

            # Récupération du type GGML pour la réécriture
            try:
                ggml_type = getattr(gguf.GGMLQuantizationType, tensor_type_name)
            except AttributeError:
                ggml_type = None

            try:
                # Analyse du type pour savoir si c'est un format standard (reshape) ou quantifié (raw)
                # "Q" ou "K" dans le nom du type indique souvent une quantification (ex: Q4_K, Q8_0)
                is_quantized = "Q" in tensor_type_name or "K" in tensor_type_name

                # Cas spécifiques : I8, I16, I32, F16, F32, BF16 sont standards
                # uint8 peut être standard pour les masques, mais souvent utilisé pour les données quantifiées raw

                if not is_quantized and ("float" in dtype_str or "int" in dtype_str) and dtype_str != "uint8":
                    # Tenseurs standards (F32, F16, I32...)
                    arr = np.frombuffer(full_data, dtype=dtype_str).reshape(shape)
                    gw.add_tensor(tname, arr)
                else:
                    # Types quantifiés (Q4, Q5, Q6...) ou uint8 brut considéré comme blob
                    arr = np.frombuffer(full_data, dtype=np.uint8)
                    gw.add_tensor(tname, arr, raw_shape=shape, raw_dtype=ggml_type)

                count += 1
                if count % 10 == 0:
                    print(f"   ... {count}/{total} tenseurs ajoutés")

            except Exception as e:
                print(f"❌ Erreur lors de l'ajout du tenseur {tname}: {e}")

        # 5. Ecriture finale
        print("💾 Ecriture du fichier final...")
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
        print(f"✅ Fichier reconstruit : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fragments_dir", help="Directory containing fragments")
    parser.add_argument("--output", default="reconstructed.gguf", help="Output GGUF file")
    args = parser.parse_args()

    rec = GGUFRecombiner(args.fragments_dir)
    rec.recombine(args.output)
