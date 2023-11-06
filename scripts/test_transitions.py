from fractals.transitions import FunctionForm
from fractals.transitions import transition
import pandas as pd

series = []
for f in FunctionForm:
    s = pd.Series(
        [
            transition(i, 100, value_from=1.0, value_to=0.0, method=f)
            for i in range(100)
        ],
        name=f,
    )
    series.append(s)

df = pd.DataFrame(series).T
df.plot.line()

import matplotlib.pyplot as plt

plt.show()
