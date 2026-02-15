"""
tests_debug/test_fragment_executor.py
======================================
Validation du Fragment Executor contre la référence LlamaLayer.

Usage
-----
    # Tests unitaires des kernels (pas besoin du modèle)
    .venv\\Scripts\\python.exe tests_debug/test_fragment_executor.py --units-only

    # Tests complets (nécessite un dossier de fragments)
    .venv\\Scripts\\python.exe tests_debug/test_fragment_executor.py \\
        models/Magistral-Small-2509-Q4_K_M_fragments

    # Avec comparaison token par token
    .venv\\Scripts\\python.exe tests_debug/test_fragment_executor.py \\
        models/Magistral-Small-2509-Q4_K_M_fragments --compare-tokens

    # Avec mesure mémoire (nécessite psutil)
    .venv\\Scripts\\python.exe tests_debug/test_fragment_executor.py \\
        models/Magistral-Small-2509-Q4_K_M_fragments --memory
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Ajouter le dossier racine pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.p2p_inference import rms_norm, softmax, swiglu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(name: str):
    print(f"  [OK] {name}")


def _skip(name: str, reason: str):
    print(f"  [SKIP] {name} — {reason}")


def _fail(name: str, msg: str):
    print(f"  [FAIL] {name} - {msg}")
    raise AssertionError(f"{name}: {msg}")


# ---------------------------------------------------------------------------
# 1. Tests unitaires des kernels numba
# ---------------------------------------------------------------------------

def test_kernels_unit():
    """Vérifie chaque kernel numba vs la référence NumPy (atol=1e-5)."""
    print("\n-- Tests unitaires kernels numba --")

    try:
        from inference.kernels_numba import rms_norm_numba, softmax_numba, swiglu_numba, warmup_kernels
    except ImportError:
        _skip("kernels_numba", "module non disponible (numba non installé ?)")
        return

    print("  Warm-up JIT...")
    warmup_kernels()

    # rms_norm
    x = np.random.randn(8, 2048).astype(np.float32)
    w = np.random.randn(2048).astype(np.float32)
    ref = rms_norm(x, w, 1e-5)
    res = rms_norm_numba(x, w, np.float32(1e-5))
    diff = np.max(np.abs(ref - res))
    assert diff <= 1e-5, f"max_diff={diff:.2e}"
    _ok(f"rms_norm_numba  (max_diff={diff:.2e})")

    # softmax
    s = np.random.randn(32, 16, 16).astype(np.float32)
    ref = softmax(s)
    res = softmax_numba(s)
    diff = np.max(np.abs(ref - res))
    assert diff <= 1e-5, f"max_diff={diff:.2e}"
    _ok(f"softmax_numba   (max_diff={diff:.2e})")

    # swiglu
    g = np.random.randn(4, 5632).astype(np.float32)
    ref = swiglu(g)
    res = swiglu_numba(g)
    diff = np.max(np.abs(ref - res))
    assert diff <= 1e-5, f"max_diff={diff:.2e}"
    _ok(f"swiglu_numba    (max_diff={diff:.2e})")


# ---------------------------------------------------------------------------
# 2. Test de dequantize_q8_0 (fragments Q8_0 requis)
# ---------------------------------------------------------------------------

def test_dequantize_q8_0(fragments_dir: str):
    """dequantize_q8_0 numba vs LocalFragmentLoader legacy."""
    print("\n-- Test dequantize_q8_0 --")

    try:
        from inference.kernels_numba import dequantize_q8_0
    except ImportError:
        _skip("dequantize_q8_0", "kernels_numba non disponible")
        return

    from distribution.local import LocalFragmentLoader
    loader = LocalFragmentLoader(fragments_dir)

    # Chercher un tenseur Q8_0 dans le manifest
    q8_tensor = None
    for tname, frags in loader.fragments_map.items():
        if frags[0].get("tensor_type", "") == "Q8_0":
            q8_tensor = tname
            break

    if q8_tensor is None:
        _skip("dequantize_q8_0", "aucun tenseur Q8_0 dans ce modèle (Q4_K_M ?)")
        return

    ref  = loader.load_tensor(q8_tensor)
    frags = loader.fragments_map[q8_tensor]
    raw  = b"".join(loader.load_raw(f["fragment_id"]) for f in frags)
    res  = dequantize_q8_0(raw, tuple(frags[0]["shape"]))
    diff = np.max(np.abs(ref - res))
    assert diff <= 1e-5, f"{q8_tensor}: max_diff={diff:.2e}"
    _ok(f"dequantize_q8_0 '{q8_tensor}' (max_diff={diff:.2e})")


# ---------------------------------------------------------------------------
# 3. Test de sortie de couche — FragmentExecutor vs LlamaLayer
# ---------------------------------------------------------------------------

def test_layer_output(fragments_dir: str):
    """
    Compare FragmentExecutor.forward() vs LlamaLayer.forward() sur la couche 0.
    Tolérance : atol=1e-4 (légères différences float32 entre NumPy et Numba).
    """
    print("\n-- Test layer output (couche 0) --")

    try:
        from inference.fragment_executor import FragmentExecutor
    except ImportError:
        _skip("layer_output", "fragment_executor non disponible")
        return

    from distribution.local import LocalFragmentLoader
    from inference.p2p_inference import P2PInferenceEngine, LlamaLayer, precompute_freqs_cis

    engine = P2PInferenceEngine(fragments_dir)
    loader = LocalFragmentLoader(fragments_dir)
    cfg    = engine.config

    np.random.seed(42)
    x         = np.random.randn(4, cfg.dim).astype(np.float32)
    head_dim  = cfg.dim // cfg.n_heads
    freqs_cis = precompute_freqs_cis(head_dim, cfg.dim * 2, theta=cfg.rope_freq_base)

    # Référence : LlamaLayer
    print("  Calcul référence (LlamaLayer)...")
    ref_layer = LlamaLayer(engine, 0)
    ref_x, ref_k, ref_v = ref_layer.forward(x.copy(), freqs_cis, None, None, start_pos=0)

    # Résultat : FragmentExecutor
    print("  Calcul FragmentExecutor...")
    with FragmentExecutor(loader, 0, cfg) as ex:
        res_x, res_k, res_v = ex.forward(x.copy(), None, None, None, start_pos=0)
        # rope_cache=None → calcul local des fréquences (correct pour le test)

    diff_x = np.max(np.abs(ref_x - res_x))
    diff_k = np.max(np.abs(ref_k - res_k))
    diff_v = np.max(np.abs(ref_v - res_v))

    assert diff_x <= 1e-4, f"activation mismatch: max_diff={diff_x:.2e}"
    assert diff_k <= 1e-4, f"cache_k mismatch: max_diff={diff_k:.2e}"
    assert diff_v <= 1e-4, f"cache_v mismatch: max_diff={diff_v:.2e}"

    _ok(f"FragmentExecutor.forward() ~= LlamaLayer.forward()  "
        f"(x:{diff_x:.2e}, k:{diff_k:.2e}, v:{diff_v:.2e})")


# ---------------------------------------------------------------------------
# 4. Test de génération end-to-end
# ---------------------------------------------------------------------------

def test_generation_tokens(fragments_dir: str, max_tokens: int = 3):
    """
    Génère des tokens avec temperature=0 (greedy déterministe).
    Compare le moteur original (LlamaLayer) et le FragmentExecutor.
    """
    print(f"\n-- Test génération end-to-end ({max_tokens} tokens, greedy) --")

    try:
        import inference.fragment_executor as _fe_mod
    except ImportError:
        _skip("génération", "fragment_executor non disponible")
        return

    import inference.p2p_inference as _p2p

    # Désactiver temporairement le FragmentExecutor (monkey-patch) pour la référence
    orig_fn = _p2p._try_import_fragment_executor
    _p2p._try_import_fragment_executor = lambda: None

    print("  Generation reference (LlamaLayer)...")
    engine_ref = _p2p.P2PInferenceEngine(fragments_dir)
    tokens_ref = engine_ref.generate("Hello", max_tokens=max_tokens, temperature=0.0)

    # Restaurer pour le run FragmentExecutor
    _p2p._try_import_fragment_executor = orig_fn

    print("  Generation FragmentExecutor...")
    engine_new = _p2p.P2PInferenceEngine(fragments_dir)
    tokens_new = engine_new.generate("Hello", max_tokens=max_tokens, temperature=0.0)

    # Restaurer
    _p2p._try_import_fragment_executor = orig_fn

    assert tokens_ref == tokens_new, \
        f"Token mismatch!\n  Ref: {tokens_ref}\n  New: {tokens_new}"
    _ok(f"Génération identique : {tokens_ref}")


# ---------------------------------------------------------------------------
# 5. Test de libération mémoire
# ---------------------------------------------------------------------------

def test_memory_release(fragments_dir: str):
    """
    Vérifie qu'un `with FragmentExecutor(...)` libère bien la mémoire après __exit__.
    Nécessite psutil.
    """
    print("\n-- Test libération mémoire --")

    try:
        import psutil, os
    except ImportError:
        _skip("memory_release", "psutil non disponible")
        return

    try:
        from inference.fragment_executor import FragmentExecutor
    except ImportError:
        _skip("memory_release", "fragment_executor non disponible")
        return

    from distribution.local import LocalFragmentLoader
    from inference.p2p_inference import P2PInferenceEngine

    engine = P2PInferenceEngine(fragments_dir)
    loader = LocalFragmentLoader(fragments_dir)
    proc   = psutil.Process(os.getpid())

    mem_before = proc.memory_info().rss
    with FragmentExecutor(loader, 0, engine.config, track_memory=True) as ex:
        mem_loaded = proc.memory_info().rss
        delta_load = (mem_loaded - mem_before) / 1024 / 1024

    mem_after    = proc.memory_info().rss
    delta_free   = (mem_loaded - mem_after) / 1024 / 1024
    delta_load_v = delta_load

    print(f"  Chargement couche 0 : +{delta_load_v:.1f} MB")
    print(f"  Libération          : -{delta_free:.1f} MB")

    # Seuil bas (5 MB) car même une petite couche doit avoir un poids mesurable
    assert delta_load_v > 1.0, \
        f"Chargement trop faible (+{delta_load_v:.1f} MB) — vérifier que la couche charge bien ses tenseurs"
    assert delta_free > 0.5, \
        f"Libération insuffisante (-{delta_free:.1f} MB) — gc.collect() n'a peut-être pas libéré"

    _ok(f"Memoire liberee apres __exit__ (-{delta_free:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validation du Fragment Executor vs LlamaLayer"
    )
    parser.add_argument(
        "fragments_dir", nargs="?",
        default=None,
        help="Dossier de fragments (ex: models/Magistral-Small-2509-Q4_K_M_fragments)"
    )
    parser.add_argument(
        "--units-only", action="store_true",
        help="Tester uniquement les kernels numba (pas de modèle requis)"
    )
    parser.add_argument(
        "--compare-tokens", action="store_true",
        help="Comparer les tokens générés entre LlamaLayer et FragmentExecutor"
    )
    parser.add_argument(
        "--memory", action="store_true",
        help="Tester la libération mémoire (nécessite psutil)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  test_fragment_executor.py")
    print("=" * 50)

    passed = 0
    failed = 0

    def run(fn, *a):
        nonlocal passed, failed
        try:
            fn(*a)
            passed += 1
        except (AssertionError, Exception) as e:
            failed += 1
            print(f"  [ERREUR] {e}")

    # Kernels unitaires — toujours
    run(test_kernels_unit)

    if args.units_only:
        print(f"\n{'='*50}")
        print(f"  {passed} OK  {failed} FAIL  (--units-only)")
        return

    if args.fragments_dir is None:
        print("\n[INFO] Aucun fragments_dir fourni. Arrêt après tests unitaires.")
        print("       Passer un chemin pour les tests complets.")
        return

    fdir = args.fragments_dir

    run(test_dequantize_q8_0, fdir)
    run(test_layer_output, fdir)

    if args.compare_tokens:
        run(test_generation_tokens, fdir)

    if args.memory:
        run(test_memory_release, fdir)

    print(f"\n{'='*50}")
    status = "OK" if failed == 0 else "FAIL"
    print(f"  {passed} OK  {failed} FAIL  [{status}]")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
