# # """
# # INFERENCE PIPELINE - Local Development Version
# # ===============================================
# # Loads the trained XGBoost model and feature metadata from local MLflow artifacts.
# # Applies identical feature transformations as training for consistent predictions.
# # """

# # import os
# # import json
# # import pandas as pd
# # import mlflow

# # # === PROJECT ROOT ===
# # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# # # === MODEL LOADING ===
# # # Load from local MLflow artifacts
# # MODEL_PATH = os.path.join(
# #     PROJECT_ROOT,
# #     "mlruns/952049124698160865/4ecc9f7506374dffaaee8a1d4cc4c697/artifacts/model"
# # )

# # try:
# #     model = mlflow.pyfunc.load_model(MODEL_PATH)
# #     print(f"✅ Model loaded from {MODEL_PATH}")
# # except Exception as e:
# #     raise Exception(f"❌ Failed to load model: {e}")

# # # === FEATURE COLUMNS LOADING ===
# # # Load from artifacts/feature_columns.json (saved during training)
# # FEATURE_COLS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "feature_columns.json")

# # try:
# #     with open(FEATURE_COLS_PATH) as f:
# #         FEATURE_COLS = json.load(f)
# #     print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {FEATURE_COLS_PATH}")
# # except Exception as e:
# #     raise Exception(f"❌ Failed to load feature columns: {e}")

# # # === FEATURE TRANSFORMATION CONSTANTS ===
# # # Must exactly match training-time transformations

# # BINARY_MAP = {
# #     "gender":          {"Female": 0, "Male": 1},
# #     "Partner":         {"No": 0, "Yes": 1},
# #     "Dependents":      {"No": 0, "Yes": 1},
# #     "PhoneService":    {"No": 0, "Yes": 1},
# #     "PaperlessBilling":{"No": 0, "Yes": 1},
# # }

# # NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# # def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
# #     """
# #     Apply identical feature transformations as used during training.
# #     1. Clean column names
# #     2. Coerce numeric columns
# #     3. Apply binary encoding
# #     4. One-hot encode remaining categoricals (drop_first=True)
# #     5. Convert booleans to int
# #     6. Align to training feature schema
# #     """
# #     df = df.copy()

# #     # Clean column names
# #     df.columns = df.columns.str.strip()

# #     # Step 1: Numeric coercion
# #     for c in NUMERIC_COLS:
# #         if c in df.columns:
# #             df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# #     # Step 2: Binary encoding
# #     for c, mapping in BINARY_MAP.items():
# #         if c in df.columns:
# #             df[c] = (
# #                 df[c].astype(str).str.strip()
# #                 .map(mapping)
# #                 .astype("Int64")
# #                 .fillna(0)
# #                 .astype(int)
# #             )

# #     # Step 3: One-hot encode remaining categoricals
# #     obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
# #     if obj_cols:
# #         df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

# #     # Step 4: Boolean to int
# #     bool_cols = df.select_dtypes(include=["bool"]).columns
# #     if len(bool_cols) > 0:
# #         df[bool_cols] = df[bool_cols].astype(int)

# #     # Step 5: Align to training feature schema (fill missing with 0, drop extras)
# #     df = df.reindex(columns=FEATURE_COLS, fill_value=0)

# #     return df


# # def predict(input_dict: dict) -> str:
# #     """
# #     Predict customer churn from raw input dictionary.

# #     Args:
# #         input_dict: Dictionary with customer attributes (18 features)

# #     Returns:
# #         "Likely to churn" or "Not likely to churn"
# #     """
# #     # Convert to DataFrame
# #     df = pd.DataFrame([input_dict])

# #     # Apply feature transformations
# #     df_enc = _serve_transform(df)

# #     # Run inference
# #     try:
# #         preds = model.predict(df_enc)
# #         if hasattr(preds, "tolist"):
# #             preds = preds.tolist()
# #         result = preds[0] if isinstance(preds, (list, tuple)) else preds
# #     except Exception as e:
# #         raise Exception(f"Model prediction failed: {e}")

# #     return "Likely to churn" if result == 1 else "Not likely to churn"
















# # """
# # INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
# # =========================================================================

# # This module provides the core inference functionality for the Telco Churn prediction model.
# # It ensures that serving-time feature transformations exactly match training-time transformations,
# # which is CRITICAL for model accuracy in production.

# # Key Responsibilities:
# # 1. Load MLflow-logged model and feature metadata from training
# # 2. Apply identical feature transformations as used during training
# # 3. Ensure correct feature ordering for model input
# # 4. Convert model predictions to user-friendly output

# # CRITICAL PATTERN: Training/Serving Consistency
# # - Uses fixed BINARY_MAP for deterministic binary encoding
# # - Applies same one-hot encoding with drop_first=True
# # - Maintains exact feature column order from training
# # - Handles missing/new categorical values gracefully

# # Production Deployment:
# # - MODEL_DIR points to containerized model artifacts
# # - Feature schema loaded from training-time artifacts
# # - Optimized for single-row inference (real-time serving)
# # """

# import os
# import pandas as pd
# import mlflow

# # === MODEL LOADING CONFIGURATION ===
# # IMPORTANT: This path is set during Docker container build
# # In development: uses local MLflow artifacts
# # In production: uses model copied to container at build time
# MODEL_DIR = "/app/model"

# try:
#     # Load the trained XGBoost model in MLflow pyfunc format
#     # This ensures compatibility regardless of the underlying ML library
#     model = mlflow.pyfunc.load_model(MODEL_DIR)
#     print(f"✅ Model loaded successfully from {MODEL_DIR}")
# except Exception as e:
#     print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
#     # Fallback for local development (OPTIONAL)
#     try:
#         # Try loading from local MLflow tracking
#         import glob
#         local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
#         if local_model_paths:
#             latest_model = max(local_model_paths, key=os.path.getmtime)
#             model = mlflow.pyfunc.load_model(latest_model)
#             MODEL_DIR = latest_model
#             print(f"✅ Fallback: Loaded model from {latest_model}")
#         else:
#             raise Exception("No model found in local mlruns")
#     except Exception as fallback_error:
#         raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# # === FEATURE SCHEMA LOADING ===
# # CRITICAL: Load the exact feature column order used during training
# # This ensures the model receives features in the expected order
# try:
#     feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
#     with open(feature_file) as f:
#         FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
#     print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
# except Exception as e:
#     raise Exception(f"Failed to load feature columns: {e}")

# # === FEATURE TRANSFORMATION CONSTANTS ===
# # CRITICAL: These mappings must exactly match those used in training
# # Any changes here will cause train/serve skew and degrade model performance

# # Deterministic binary feature mappings (consistent with training)
# BINARY_MAP = {
#     "gender": {"Female": 0, "Male": 1},           # Demographics
#     "Partner": {"No": 0, "Yes": 1},               # Has partner
#     "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
#     "PhoneService": {"No": 0, "Yes": 1},          # Phone service
#     "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
# }

# # Numeric columns that need type coercion
# NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Apply identical feature transformations as used during model training.
    
#     This function is CRITICAL for production ML - it ensures that features are
#     transformed exactly as they were during training to prevent train/serve skew.
    
#     Transformation Pipeline:
#     1. Clean column names and handle data types
#     2. Apply deterministic binary encoding (using BINARY_MAP)
#     3. One-hot encode remaining categorical features  
#     4. Convert boolean columns to integers
#     5. Align features with training schema and order
    
#     Args:
#         df: Single-row DataFrame with raw customer data
        
#     Returns:
#         DataFrame with features transformed and ordered for model input
        
#     IMPORTANT: Any changes to this function must be reflected in training
#     feature engineering to maintain consistency.
#     """
#     df = df.copy()
    
#     # Clean column names (remove any whitespace)
#     df.columns = df.columns.str.strip()
    
#     # === STEP 1: Numeric Type Coercion ===
#     # Ensure numeric columns are properly typed (handle string inputs)
#     for c in NUMERIC_COLS:
#         if c in df.columns:
#             # Convert to numeric, replacing invalid values with NaN
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#             # Fill NaN with 0 (same as training preprocessing)
#             df[c] = df[c].fillna(0)
    
#     # === STEP 2: Binary Feature Encoding ===
#     # Apply deterministic mappings for binary features
#     # CRITICAL: Must use exact same mappings as training
#     for c, mapping in BINARY_MAP.items():
#         if c in df.columns:
#             df[c] = (
#                 df[c]
#                 .astype(str)                    # Convert to string
#                 .str.strip()                    # Remove whitespace
#                 .map(mapping)                   # Apply binary mapping
#                 .astype("Int64")                # Handle NaN values
#                 .fillna(0)                      # Fill unknown values with 0
#                 .astype(int)                    # Final integer conversion
#             )
    
#     # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
#     # Find remaining object/categorical columns (not in BINARY_MAP)
#     obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
#     if obj_cols:
#         # Apply one-hot encoding with drop_first=True (same as training)
#         # This prevents multicollinearity by dropping the first category
#         df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
#     # === STEP 4: Boolean to Integer Conversion ===
#     # Convert any boolean columns to integers (XGBoost compatibility)
#     bool_cols = df.select_dtypes(include=["bool"]).columns
#     if len(bool_cols) > 0:
#         df[bool_cols] = df[bool_cols].astype(int)
    
#     # === STEP 5: Feature Alignment with Training Schema ===
#     # CRITICAL: Ensure features are in exact same order as training
#     # Missing features get filled with 0, extra features are dropped
#     df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
#     return df

# def predict(input_dict: dict) -> str:
#     """
#     Main prediction function for customer churn inference.
    
#     This function provides the complete inference pipeline from raw customer data
#     to business-friendly prediction output. It's called by both the FastAPI endpoint
#     and the Gradio interface to ensure consistent predictions.
    
#     Pipeline:
#     1. Convert input dictionary to DataFrame
#     2. Apply feature transformations (identical to training)
#     3. Generate model prediction using loaded XGBoost model
#     4. Convert prediction to user-friendly string
    
#     Args:
#         input_dict: Dictionary containing raw customer data with keys matching
#                    the CustomerData schema (18 features total)
                   
#     Returns:
#         Human-readable prediction string:
#         - "Likely to churn" for high-risk customers (model prediction = 1)
#         - "Not likely to churn" for low-risk customers (model prediction = 0)
        
#     Example:
#         >>> customer_data = {
#         ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
#         ...     "MonthlyCharges": 85.0, ... # other features
#         ... }
#         >>> predict(customer_data)
#         "Likely to churn"
#     """
    
#     # === STEP 1: Convert Input to DataFrame ===
#     # Create single-row DataFrame for pandas transformations
#     df = pd.DataFrame([input_dict])
    
#     # === STEP 2: Apply Feature Transformations ===
#     # Use the same transformation pipeline as training
#     df_enc = _serve_transform(df)
    
#     # === STEP 3: Generate Model Prediction ===
#     # Call the loaded MLflow model for inference
#     # The model returns predictions in various formats depending on the ML library
#     try:
#         preds = model.predict(df_enc)
        
#         # Normalize prediction output to consistent format
#         if hasattr(preds, "tolist"):
#             preds = preds.tolist()  # Convert numpy array to list
            
#         # Extract single prediction value (for single-row input)
#         if isinstance(preds, (list, tuple)) and len(preds) == 1:
#             result = preds[0]
#         else:
#             result = preds
            
#     except Exception as e:
#         raise Exception(f"Model prediction failed: {e}")
    
#     # === STEP 4: Convert to Business-Friendly Output ===
#     # Convert binary prediction (0/1) to actionable business language
#     if result == 1:
#         return "Likely to churn"      # High risk - needs intervention
#     else:
#         return "Not likely to churn"  # Low risk - maintain normal service





## from here for the docker error runtime gui

"""
INFERENCE PIPELINE - Telco Churn Model Serving
===============================================
Loads trained XGBoost model and applies identical feature transformations
as training to ensure consistent predictions.
"""

import os
import json
import glob
import pandas as pd
import mlflow

# === MODEL LOADING ===
MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    try:
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE COLUMNS LOADING ===
try:
    json_path = os.path.join(MODEL_DIR, "feature_columns.json")
    txt_path  = os.path.join(MODEL_DIR, "feature_columns.txt")

    if os.path.exists(json_path):
        feature_file = json_path
    elif os.path.exists(txt_path):
        feature_file = txt_path
    else:
        raise FileNotFoundError(
            f"No feature_columns file found in {MODEL_DIR}. "
            f"Looked for: {json_path} and {txt_path}"
        )

    with open(feature_file) as f:
        content = f.read().strip()

    # Handle both formats:
    # - JSON array:  ["gender", "SeniorCitizen", ...]
    # - Plain text:  one feature name per line
    if content.startswith("["):
        FEATURE_COLS = json.loads(content)
    else:
        FEATURE_COLS = [ln.strip() for ln in content.splitlines() if ln.strip()]

    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")
    print(f"   First: {FEATURE_COLS[0]} | Last: {FEATURE_COLS[-1]}")

except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender":          {"Female": 0, "Male": 1},
    "Partner":         {"No": 0, "Yes": 1},
    "Dependents":      {"No": 0, "Yes": 1},
    "PhoneService":    {"No": 0, "Yes": 1},
    "PaperlessBilling":{"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during training.
    """
    df = df.copy()

    # Step 1: Clean column names
    df.columns = df.columns.str.strip()

    # Step 2: Numeric coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Step 3: Binary encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # Step 4: One-hot encode remaining categoricals
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Step 5: Boolean to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Step 6: Align to exact training feature schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> str:
    """
    Predict customer churn from raw input dictionary.

    Args:
        input_dict: Dictionary with 18 customer attributes

    Returns:
        "Likely to churn" or "Not likely to churn"
    """
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"