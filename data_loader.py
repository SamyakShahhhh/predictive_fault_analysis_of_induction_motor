# =============================================================================
# data_loader.py — Stage 1 & Stage 2
#   load_files()        : reads all CSVs and reorders columns
#   combine_and_label() : merges into one df_final with fault_type column
# =============================================================================

import pandas as pd
from config import DATA_DIR, REFERENCE_COLS, FEATURE_COLS, FAULT_NAMES, FAULT_LABEL_MAP


def load_files(file_map):
    """
    Load each CSV in file_map and reorder its columns to REFERENCE_COLS.

    Parameters
    ----------
    file_map : dict  { csv_filename : internal_key }

    Returns
    -------
    dict  { internal_key : DataFrame }
    """
    loaded = {}

    for filename, key in file_map.items():
        filepath = f"{DATA_DIR}\\{filename}"
        try:
            df = pd.read_csv(filepath)

            # Reorder to fixed column order — fixes P_P_FAULT_claude.csv
            df = df[REFERENCE_COLS]

            print(f"\n--- {key} ({filename}) ---")
            print(f"    Shape : {df.shape}")
            print(df.head())

            loaded[key] = df

        except Exception as e:
            print(f"[ERROR] Could not load {filename}: {e}")

    return loaded


def combine_and_label(dataframes):
    """
    Merge all loaded DataFrames into one df_final with a fault_type column.

    Rules
    -----
    - Fault files : label==0 rows → fault class only. label==1 rows discarded.
    - Healthy file : all rows → healthy (0). This is the sole source for class 0.

    Returns
    -------
    DataFrame  df_final  (shuffled, index reset, label column dropped)
    """
    all_pieces = []

    print("\n[Discarding contaminated healthy-reference rows from fault files...]")
    for key, fault_type_id in FAULT_LABEL_MAP.items():
        df = dataframes[key].copy()

        # label==0 rows are the actual fault samples — keep only these
        fault_rows = df[df['label'] == 0].copy()
        fault_rows['fault_type'] = fault_type_id

        discarded = (df['label'] == 1).sum()
        print(f"    [{key}] fault rows kept: {len(fault_rows)}  |  healthy-ref rows discarded: {discarded}")

        all_pieces.append(fault_rows)

    # Dedicated healthy file is the only source for class 0
    healthy_df = dataframes['healthy'].copy()
    healthy_df['fault_type'] = 0
    all_pieces.append(healthy_df)

    # Drop the original binary label column — fault_type replaces it
    for i, piece in enumerate(all_pieces):
        all_pieces[i] = piece.drop(columns=['label'])

    # Concatenate and shuffle to remove ordering bias
    df_final = pd.concat(all_pieces, ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print class distribution
    print("\nClass distribution (fault_type counts):")
    dist = df_final['fault_type'].value_counts().sort_index()
    for ft, count in dist.items():
        print(f"    {ft} — {FAULT_NAMES[ft]:<15} : {count}")

    # Check and remove duplicate rows
    dup_count = df_final.duplicated().sum()
    print(f"\nDuplicate rows found : {dup_count}")
    if dup_count > 0:
        df_final = df_final.drop_duplicates().reset_index(drop=True)
        print(f"Duplicates removed. Rows remaining : {len(df_final)}")

    print(f"\nFinal df_final shape : {df_final.shape}")

    return df_final
