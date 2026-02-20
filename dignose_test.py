"""
diagnose_moesm.py
Investigates what MOESM8 and MOESM9 actually are before using them as test sets.
Checks C0 score distributions and sequence properties to determine if they are
truly random libraries or genomic/tiling libraries.
"""

import pandas as pd
import numpy as np

FILES = {
    'MOESM6 (training negative)' : r"D:\exprement_16\data\41586_2020_3052_MOESM6_ESM.txt",
    'MOESM8 (unknown)'           : r"D:\exprement_16\data\41586_2020_3052_MOESM8_ESM.txt",
    'MOESM9 (unknown)'           : r"D:\exprement_16\data\41586_2020_3052_MOESM9_ESM.txt",
}

for name, path in FILES.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        df = pd.read_csv(path, sep='\t')
        df.columns = df.columns.str.strip()
        print(f"  Rows          : {len(df):,}")
        print(f"  Columns       : {list(df.columns)}")

        # C0 distribution — random synthetic DNA has C0 near 0
        # Genomic/nucleosomal DNA has systematically higher C0
        if 'C0' in df.columns or ' C0' in df.columns:
            c0_col = 'C0' if 'C0' in df.columns else ' C0'
            c0 = df[c0_col].dropna()
            print(f"\n  C0 score statistics:")
            print(f"    Mean   : {c0.mean():.4f}")
            print(f"    Median : {c0.median():.4f}")
            print(f"    Std    : {c0.std():.4f}")
            print(f"    Min    : {c0.min():.4f}")
            print(f"    Max    : {c0.max():.4f}")
            print(f"    % > 0  : {(c0 > 0).mean()*100:.1f}%")

        # Sequence length distribution
        if 'Sequence' in df.columns:
            lengths = df['Sequence'].dropna().str.len()
            print(f"\n  Sequence lengths:")
            print(f"    Min/Max/Mean: {lengths.min()} / {lengths.max()} / {lengths.mean():.1f}")

        # GC content — random DNA ~50%, genomic varies
        if 'Sequence' in df.columns:
            sample = df['Sequence'].dropna().head(1000).str.upper()
            gc = sample.apply(lambda s: (s.count('G') + s.count('C')) / len(s) if len(s) > 0 else 0)
            print(f"\n  GC content (first 1000 seqs):")
            print(f"    Mean: {gc.mean():.3f}  Std: {gc.std():.3f}")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\nINTERPRETATION GUIDE:")
print("  Random synthetic DNA : C0 mean near 0, GC ~0.50, uniform distribution")
print("  Genomic/tiling DNA   : C0 mean elevated, GC reflects organism bias")
print("  If MOESM8/9 have similar C0 stats to MOESM6 -> valid generalization test")
print("  If MOESM8/9 have different C0 stats        -> different library type, wrong comparison")