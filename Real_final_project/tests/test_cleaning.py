'''# Debugging: Check if the dataset is loaded correctly
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The input data must be a pandas DataFrame.")

    print(f"Dataset contains {len(data)} rows and the following columns: {data.columns.tolist()}")

    # Ensure the wavelength column exists
    if wavelength not in data.columns:
        raise ValueError(f"Wavelength '{wavelength}' not found in the dataset.")

    # Ensure the column contains numeric data
    try:
        data[wavelength] = pd.to_numeric(data[wavelength], errors='coerce')
    except Exception as e:
        raise ValueError(f"Error converting column '{wavelength}' to numeric: {e}")'''