# Attribution and licensing scope

The training implementation is adapted from **Self-Distilled Reasoner:
On-Policy Self-Distillation for Large Language Models**:

- Paper: https://arxiv.org/abs/2601.18734
- Upstream: https://github.com/siyan-zhao/OPSD
- Base commit: `7448751f307a9cdbcc1246dd1565a1a605b443df`

The frozen self-teacher is an upstream feature. Study additions include the
L1--L5 context interfaces, bridge controls, local data loading, and evaluation
code. Release changes connect the compact dataset to those implementations and
provide portable generation, training, and evaluation commands.

`opsd/opsd_trainer.py` retains its Hugging Face copyright and Apache-2.0 header.
The complete license text is included in `LICENSES/Apache-2.0.txt`. This notice
does not remove any original attribution or extend that file's license to
unrelated files or datasets.

The locally available upstream commit has no repository-wide LICENSE file.
No blanket MIT/Apache license is asserted over all adapted files in this public
research artifact. Applicable rights for files without an explicit license must
be confirmed from upstream terms or permission before an open-source release;
public hosting does not supply that permission. No such confirmation is claimed
here.

The training data derive from
https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd . See
`data/README.md` for transformations and the unverified license status of the
local data snapshot. Evaluation datasets are not bundled; the repository only
records their fingerprints and preparation code. All upstream dataset terms
continue to apply, and no blanket license is asserted over third-party data.
