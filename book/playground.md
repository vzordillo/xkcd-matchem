---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Interactive Live Playground

Every tutorial page in this Jupyter Book is live-enabled with **Thebe** (in-browser execution via MyBinder), **Binder**, and **Google Colab**.

Try editing the code snippet below after clicking the rocket icon -> **Live Code** in the top navigation bar, or launch one of the 10 step-by-step notebooks under **Tutorials**!

```{code-cell} ipython3
import xkcd_matchem as xm

print("xkcd-matchem version:", xm.__version__)
print("Bundled samples:", len(xm.list_samples()))
ax = xm.plot_crystal("nacl.cif", title="Rock-Salt NaCl (Fm-3m)")
```
