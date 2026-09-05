# KronVerb expander — planned parameter surface

KronVerb's macro layer deliberately exposes only log-N-ish grouped controls.
A companion expander module (VCV expander mechanism, left/right adjacency)
would expose the raw layers. Everything below is already supported by, or a
small extension of, the existing DSP core in `src/KronVerb.cpp`; kernel
entries are stored complex for exactly this reason.

## Matrix layer
- **Raw kernel angles**: per-level mixing angle θ_l (4 levels), replacing the
  single DIFFUSE macro. Per-level CV = audio-rate matrix modulation with
  structural meaning (level 0 = even/odd partition coupling, level 3 =
  half/half coupling).
- **Full U(2) kernels**: per-level kernel phases (α, β, δ) on top of θ —
  unlocks the continuous rotation↔reflection morph (disconnected components
  in O(2), connected through U(2)) and per-level eigenphase control.
  `KronMixer::setRotation` already takes a kernel phase argument.
- **Per-level drift** depth/rate instead of the fixed two-LFO DRIFT macro.

## Phasor layer
- **SPREAD**: per-line shift detune (Ω_i = Ω·(1 + spread·s_i), balanced
  pattern s_i) — currently the phasor is uniform across lines.
- **Per-partition SHIFT**: independent Ω for the freezable (even) and live
  (odd) partitions — the "frozen cloud drifts while the live half holds"
  effect, and its inverse.
- **CONJ**: blend toward conjugating the signal (x* e^{iψ}) rather than
  mirroring the phasor. Antiunitary (norm-preserving) at the endpoint:
  period-2 spectral ping-pong / reflection about Ω/2 instead of the MIRROR
  knob's binomial diffusion. Needs the same freeze guard as MIRROR at
  intermediate blends.

## Nonlinearity layer
- **GRAIN**: morph the shimmer read-pointer resets from periodic sawtooth to
  randomized (paper 05 §3.5 granular time compression): randomize
  `ShimmerTap::phase` at wrap, blend amount 0→1. The energy normalization
  already in place covers granular's wider energy spread (their Fig. 8).
- **SDFD depth**: k in d_i = d₀ + k·|x| driving the Thiran fractional delay —
  amplitude-dependent dispersion, designed but not yet wired to a control.
- **Shimmer normalization exposure**: the compensation cap (currently ×4)
  and reference (currently the static main tap) — turning the cap up /
  reference off recovers the un-normalized runaway snarl as a flavor.
- **Per-line NL sends** g_i instead of the fixed odd-lines shimmer routing.

## I/O and quality
- **Hilbert input option**: analytic-signal input for exact SSB on the first
  pass (currently x = L + iR; stereo input under SHIFT barber-poles).
- **Oversampling** for WARP (Kerr) at high γ.
- **SIMD** the butterfly (float_4 across pairs) if CPU becomes a concern.
