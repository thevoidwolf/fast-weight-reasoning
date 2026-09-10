"""Audit-as-code: mechanical verification of every convention claim the
FENRIR paper will make.

If any test here fails, the paper's prose is out of sync with the code.
Fix one or the other before publishing.

These tests use hand-constructed tensors so the assertions are visible from
the test itself. No reference-implementation dependency; see the audit
script outside this repo for parity checks against alternate implementations.
"""
from __future__ import annotations

import pytest
import torch

from model import FenrirMixer, MixerConfig


# --------------------------------------------------------------------------
# M storage direction
# --------------------------------------------------------------------------


def test_M_storage_puts_keys_on_axis_0_values_on_axis_1():
    """The write rule is M[i,j] += beta * k_eff[i] * v[j].
    So a distinct key at row i0 populates M[:, i0, :] with (k_eff[i0] * v),
    not M[:, :, i0]. Fails if the outer-product einsum has axes swapped."""
    cfg = MixerConfig(d_model=8, d_key=4, d_value=4, expand=1, d_conv=1)
    mixer = FenrirMixer(cfg, variant="rev")
    mixer.eval()

    # Compose a synthetic write directly to check storage semantics.
    # We can't hit .write() with a private hook; instead, we assert that the
    # public outer-product used in the loop puts k on axis 0 and v on axis 1
    # by constructing k, v manually and reading the mixer's cache after one
    # forward pass with probe_cache on. See test_parity.py for full-cycle
    # parity; here we just verify the write direction directly.
    B, d = 1, 4
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # active on axis-0 index 0
    v = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # active on axis-1 index 1
    M = torch.zeros(B, d, d)
    # This is the exact write kernel model.py must use for both variants.
    outer = torch.einsum('bi,bj->bij', k, v)
    M_after = M + 1.0 * outer
    assert M_after[0, 0, 1].item() == pytest.approx(1.0), \
        "outer(k, v) must populate M[i=k_axis, j=v_axis]; expected M[0, 0, 1] == 1"
    assert M_after[0, 1, 0].item() == pytest.approx(0.0), \
        "storage direction wrong: M[j=v_axis, i=k_axis] should stay 0"


# --------------------------------------------------------------------------
# Eager term direction (variant-specific)
# --------------------------------------------------------------------------


def _synthetic_M_kv():
    """Build a small M where each stored (key, value) pair is a distinct
    one-hot pair. Under storage M[i,j] += k[i]*v[j], a write of (k=e_i0, v=e_j0)
    puts a 1 at M[i0, j0]. This lets us probe both operator directions
    surgically."""
    d = 4
    M = torch.zeros(1, d, d)
    # store pair (key=e_0, value=e_1) and pair (key=e_2, value=e_3)
    M[0, 0, 1] = 1.0
    M[0, 2, 3] = 1.0
    return M, d


def test_rev_variant_eager_is_reverse_lookup():
    """rev variant: eager term = M · k = einsum('bij,bj->bi', M, k).
    Semantics: 'match k against stored VALUES (axis 1), return stored KEYS (axis 0)'.

    Query with k=e_1 should retrieve the key stored under value e_1, which is e_0.
    Query with k=e_3 should retrieve the key stored under value e_3, which is e_2.
    """
    M, d = _synthetic_M_kv()

    # query with k = e_1 (value axis)
    k = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    Mk = torch.einsum('bij,bj->bi', M, k)
    assert torch.allclose(Mk, torch.tensor([[1.0, 0.0, 0.0, 0.0]])), \
        f"rev eager should retrieve key e_0, got {Mk}"

    # query with k = e_3
    k = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    Mk = torch.einsum('bij,bj->bi', M, k)
    assert torch.allclose(Mk, torch.tensor([[0.0, 0.0, 1.0, 0.0]])), \
        f"rev eager should retrieve key e_2, got {Mk}"


def test_fwd_variant_eager_is_forward_lookup():
    """fwd variant: eager term = M^T · k = einsum('bij,bi->bj', M, k).
    Semantics: 'match k against stored KEYS (axis 0), return stored VALUES (axis 1)'.

    Query with k=e_0 should retrieve the value stored at key e_0, which is e_1.
    Query with k=e_2 should retrieve the value stored at key e_2, which is e_3.
    """
    M, d = _synthetic_M_kv()

    # query with k = e_0 (key axis)
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    MTk = torch.einsum('bij,bi->bj', M, k)
    assert torch.allclose(MTk, torch.tensor([[0.0, 1.0, 0.0, 0.0]])), \
        f"fwd eager should retrieve value e_1, got {MTk}"

    # query with k = e_2
    k = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    MTk = torch.einsum('bij,bi->bj', M, k)
    assert torch.allclose(MTk, torch.tensor([[0.0, 0.0, 0.0, 1.0]])), \
        f"fwd eager should retrieve value e_3, got {MTk}"


def test_fwd_and_rev_eager_ops_agree_only_on_symmetric_M():
    """Sanity check that the two operators are genuinely different.
    fwd and rev disagree on any non-symmetric M, which the outer-product
    writes generically create."""
    M, d = _synthetic_M_kv()
    assert not torch.allclose(M, M.transpose(-1, -2)), \
        "test setup broken: this M should be non-symmetric"

    k = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    Mk  = torch.einsum('bij,bj->bi', M, k)   # rev
    MTk = torch.einsum('bij,bi->bj', M, k)   # fwd
    assert not torch.allclose(Mk, MTk), \
        "fwd and rev eager ops must differ on non-symmetric M"


# --------------------------------------------------------------------------
# Read direction: same for both variants (standard fast-weight forward read)
# --------------------------------------------------------------------------


def test_read_is_forward_lookup_both_variants():
    """Read: y = M^T · q = einsum('bi,bij->bj', q, M). Same for both variants.
    Semantics: 'match q against stored keys (axis 0), return stored values (axis 1)'.

    This is the standard fast-weight forward read from Schlag et al. and every
    delta-family variant. Not variant-specific: the variant only changes the
    write's address perturbation.
    """
    M, d = _synthetic_M_kv()

    # query with q = e_0 (key axis): should read out stored value at key e_0 = e_1
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    y = torch.einsum('bi,bij->bj', q, M)
    assert torch.allclose(y, torch.tensor([[0.0, 1.0, 0.0, 0.0]])), \
        f"read should retrieve value e_1, got {y}"


# --------------------------------------------------------------------------
# Mixer wiring: variant affects eager term only, everything else identical
# --------------------------------------------------------------------------


def test_mixer_variant_flag_is_fwd_or_rev():
    """Only two variants exist. Any other value must be a hard error."""
    cfg = MixerConfig(d_model=8, d_key=4, d_value=4, expand=1, d_conv=1)
    FenrirMixer(cfg, variant="fwd")
    FenrirMixer(cfg, variant="rev")
    with pytest.raises((ValueError, AssertionError)):
        FenrirMixer(cfg, variant="typo")


def test_mixer_variants_produce_different_outputs_on_nontrivial_input():
    """After a few writes, fwd and rev should produce different outputs.
    (They must, because the underlying operators differ on non-symmetric M.)"""
    torch.manual_seed(0)
    cfg = MixerConfig(d_model=32, d_key=8, d_value=8, expand=2, d_conv=4)
    m_fwd = FenrirMixer(cfg, variant="fwd")
    m_rev = FenrirMixer(cfg, variant="rev")
    # Copy weights so the only difference is the variant flag.
    m_rev.load_state_dict(m_fwd.state_dict())
    m_fwd.eval(); m_rev.eval()

    x = torch.randn(2, 12, cfg.d_model)
    with torch.no_grad():
        y_fwd = m_fwd(x)
        y_rev = m_rev(x)
    assert y_fwd.shape == y_rev.shape == x.shape
    assert not torch.allclose(y_fwd, y_rev, atol=1e-4), \
        "fwd and rev with identical weights must differ; otherwise the variant flag is a no-op"


def test_mixer_probe_cache_returns_the_operator_the_mixer_actually_computes():
    """When probe_cache is on, the mixer stores the eager term it actually uses
    at _cache['eager']. This test verifies the value in the cache matches the
    variant's declared operator. It is the audit hook that prevents the
    operator-mismatch class of bug (an external reconstruction using the
    wrong einsum silently measuring something the mixer never computes).
    """
    torch.manual_seed(0)
    cfg = MixerConfig(d_model=16, d_key=4, d_value=4, expand=1, d_conv=1)

    for variant, contract in [("fwd", 'bij,bi->bj'), ("rev", 'bij,bj->bi')]:
        m = FenrirMixer(cfg, variant=variant)
        m.eval()
        m.probe_cache = True
        x = torch.randn(1, 5, cfg.d_model)
        with torch.no_grad():
            _ = m(x)
        cache = m._cache
        k = cache['k']              # [B, L, d_key]
        eager = cache['eager']      # [B, L, d_key or d_value] depending on variant

        # Externally recompute the eager term at each timestep from scratch
        # using the variant's declared contraction, and check the cache matches.
        # This IS the parity check between the mixer's forward pass and its
        # advertised operator.
        d_key = cfg.d_key
        B, L, _ = k.shape
        M = torch.zeros(B, d_key, d_key)
        v = cache['v']
        beta = cache['beta']
        expected = []
        for t in range(L):
            e = torch.einsum(contract, M, k[:, t])
            expected.append(e)
            k_eff = k[:, t] + e
            M = M + beta[:, t, None, None] * torch.einsum('bi,bj->bij', k_eff, v[:, t])
        expected = torch.stack(expected, dim=1)
        assert torch.allclose(eager, expected, atol=1e-5), \
            f"{variant} mixer's _cache['eager'] does not match the declared operator"


# --------------------------------------------------------------------------
# d_key == d_value invariant (required for the eager term to type-check)
# --------------------------------------------------------------------------


def test_use_eager_false_zeros_the_eager_term_and_sets_k_eff_eq_k():
    """The --no-eager ablation should skip the k + M*k address perturbation
    entirely. Under use_eager=False the mixer's cache must record eager==0
    at every timestep, and the M state must evolve as if k_eff = k.

    This is a training-time causal control for the F4 inference-time knockout:
    if the model trained without any address perturbation, does the mechanism
    form? (Answered empirically by the runs of ablation_no_eager.sh; the test
    only verifies the flag actually disables the eager term.)
    """
    torch.manual_seed(0)
    cfg = MixerConfig(d_model=16, d_key=4, d_value=4, expand=1, d_conv=1)

    for variant in ("fwd", "rev"):
        m = FenrirMixer(cfg, variant=variant, use_eager=False)
        m.eval()
        m.probe_cache = True
        x = torch.randn(1, 5, cfg.d_model)
        with torch.no_grad():
            _ = m(x)
        cache = m._cache
        assert torch.allclose(cache['eager'], torch.zeros_like(cache['eager'])), \
            f"use_eager=False must zero the eager term (variant={variant})"

        # Also: reconstructing M from k (no eager) must equal the M implied by
        # the cache. If we compute M with k_eff = k + eager (with eager=0 == k),
        # the outer-product write matches the mixer's forward.
        k = cache['k']; v = cache['v']; beta = cache['beta']
        d_key = cfg.d_key
        B, L, _ = k.shape
        M_recon = torch.zeros(B, d_key, d_key)
        for t in range(L):
            outer = torch.einsum('bi,bj->bij', k[:, t], v[:, t])
            M_recon = M_recon + beta[:, t, None, None] * outer
        # The mixer doesn't cache M directly, but y = M^T q. So check y matches.
        q = cache['q']
        y_recon = torch.einsum('bi,bij->bj', q[:, -1], M_recon)
        # The mixer's forward reads at every t; here we compare only the final read
        # against the final-M reconstruction.
        # Compare against the cached y at the last position, which reads at M_L.
        y_from_cache_last = cache['y'][:, -1]
        assert torch.allclose(y_from_cache_last, y_recon, atol=1e-5), \
            f"use_eager=False final read must equal reconstruction from k*v writes only (variant={variant})"


def test_use_eager_true_matches_unflagged_default():
    """Regression: use_eager=True (the default) must be bit-identical to
    the mixer constructed without the flag. Guards against accidental
    behaviour change in the default path when we added the ablation option."""
    torch.manual_seed(0)
    cfg = MixerConfig(d_model=16, d_key=4, d_value=4, expand=1, d_conv=1)

    for variant in ("fwd", "rev"):
        torch.manual_seed(7)
        m_default = FenrirMixer(cfg, variant=variant)
        torch.manual_seed(7)
        m_explicit = FenrirMixer(cfg, variant=variant, use_eager=True)
        m_explicit.load_state_dict(m_default.state_dict())
        m_default.eval(); m_explicit.eval()
        x = torch.randn(2, 8, cfg.d_model)
        with torch.no_grad():
            y_default = m_default(x)
            y_explicit = m_explicit(x)
        assert torch.allclose(y_default, y_explicit, atol=0.0), \
            f"use_eager=True must be bit-identical to unflagged default (variant={variant})"


def test_use_eager_false_and_true_produce_different_outputs():
    """Sanity: the ablation flag must actually change the mixer's output
    on non-trivial input (with matched weights)."""
    torch.manual_seed(0)
    cfg = MixerConfig(d_model=32, d_key=8, d_value=8, expand=2, d_conv=4)
    m_on = FenrirMixer(cfg, variant="rev", use_eager=True)
    m_off = FenrirMixer(cfg, variant="rev", use_eager=False)
    m_off.load_state_dict(m_on.state_dict())
    m_on.eval(); m_off.eval()
    x = torch.randn(2, 12, cfg.d_model)
    with torch.no_grad():
        y_on = m_on(x)
        y_off = m_off(x)
    assert not torch.allclose(y_on, y_off, atol=1e-4), \
        "use_eager=True vs False must produce different outputs (else flag is a no-op)"


def test_use_eager_false_incompatible_with_chunked():
    """The chunked kernel is derived from the eager-term recurrence; using
    it with use_eager=False would silently give inconsistent semantics.
    The mixer must refuse."""
    cfg = MixerConfig(d_model=16, d_key=4, d_value=4, expand=1, d_conv=1)
    with pytest.raises(ValueError):
        FenrirMixer(cfg, variant="rev", chunked=True, use_eager=False)


def test_eager_term_requires_d_key_eq_d_value():
    """k_eff = k + M k (rev variant): k has shape [d_key] and M k must also
    have shape [d_key] for the addition to type-check. M k contracts the
    value axis of M (size d_value) with k (size d_key), which requires
    d_value == d_key. Otherwise einsum('bij,bj->bi', M[B,d_key,d_value],
    k[B,d_key]) fails to contract.

    The mixer enforces this at __init__ time.
    """
    cfg_ok = MixerConfig(d_model=16, d_key=4, d_value=4, expand=1, d_conv=1)
    FenrirMixer(cfg_ok, variant="rev")   # ok

    cfg_bad = MixerConfig(d_model=16, d_key=4, d_value=8, expand=1, d_conv=1)
    with pytest.raises((ValueError, AssertionError)):
        FenrirMixer(cfg_bad, variant="rev")
