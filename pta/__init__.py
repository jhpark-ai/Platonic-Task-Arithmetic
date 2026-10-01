"""Platonic Task Arithmetic (NeurIPS 2026).

A task's Universal Task Descriptor UTD[i, j] = <f_ft(x_i), g(p_j)> - <f_pre(x_i), g(p_j)>
is read on a source model, transferred to another pre-trained model by editing that target
until it reproduces the descriptor, and composed by arithmetic on the edits. Two
realizations of the one objective: a closed-form operator folded into the target's last
layer (`pta.solver`, `pta.operator`, `pta.head`) and a LoRA adapter (`pta.adapter`).
"""
__version__ = "1.0.0"
