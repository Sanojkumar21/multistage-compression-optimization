# Multistage Gas Compression Optimization

Class assignment for Process Optimization course.

Interactive dashboard to optimize intermediate pressures in a 3-stage gas compression system using mechanical energy balance.

## Demo

[Live App](https://multistage-compression-optimization-bfrdpql2bh4zvnb53adbq8.streamlit.app/)

## What it does

- Computes optimal intermediate pressures (p₂, p₃) that minimize total compression work
- Compares analytical solution (equal pressure ratios) with numerical optimization
- Shows how heat exchanger effectiveness affects total work

## Stack

- Python, Streamlit, NumPy, SciPy, Matplotlib

## Run locally
```bash
pip install streamlit numpy scipy matplotlib
streamlit run app.py
```

## License

MIT
