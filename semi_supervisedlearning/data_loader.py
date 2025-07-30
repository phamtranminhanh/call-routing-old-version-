import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE,ADASYN,KMeansSMOTE,SVMSMOTE,BorderlineSMOTE
FILE_PATH = r"Filtered_Call_Records.csv"  
target_col = "OverallCallQuality"

# Columns to ignore
cols_to_remove = [
     'StreamId', 'CallRecordId', 'Comment', 'SbcSessionId', 'CallerPhoneNumber',
        'CalleePhoneNumber', 'CallerIpAddress', 'CalleeIpAddress', 'CallerReflexiveIpAddress',
        'CalleeReflexiveIpAddress', 'CallerSubnet', 'CalleeSubnet',
        'SegmentFailedReason', 'SegmentFailureStage', 'CallerRelayIpAddress', 'CallerRelayPort',
        'CalleeRelayIpAddress', 'CalleeRelayPort', 'EstimatedGttCost', 'EstimatedSoftnetCost',
        'SbcEstimatedGttCost', 'SbcEstimatedSoftnetCost', 'CallStartTime', 'CallEndTime',
        'SbcSessionStartTime', 'SbcSessionEndTime',
        'SbcSessionStatus', 'Trunk', 'CallerPhoneNumberPrefix', 'CalleePhoneNumberPrefix', 'StreamQuality'
]

def load_call_quality_data(file_path=FILE_PATH, target_col=target_col, cols_to_remove=cols_to_remove):
    df = pd.read_csv(FILE_PATH)
    df = df.dropna(subset=[target_col])

    original_df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols_to_remove:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
        df = df.drop(columns=[col], errors='ignore')
    
    # Apply clipping only to numeric columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0.05, upper=0.95)
    
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    encoders = {}
    for col in categorical_cols + [target_col]:
        df = df.dropna(subset=[col])
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[numeric_cols + categorical_cols].values
    y = df[target_col].values
    feature_names = numeric_cols + categorical_cols
    return X, y, feature_names, original_df, scaler
def load_call_quality_data_using_Smote(file_path=FILE_PATH, target_col=target_col, cols_to_remove=cols_to_remove):
    X, y, feature_names, original_df, scaler = load_call_quality_data(file_path, target_col, cols_to_remove)
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    return X, y, feature_names, original_df, scaler
def load_call_quality_data_using_ADASYN(file_path=FILE_PATH, target_col=target_col, cols_to_remove=cols_to_remove):
    X, y, feature_names, original_df, scaler = load_call_quality_data(file_path, target_col, cols_to_remove)
    adasyn = ADASYN(random_state=42)
    X, y = adasyn.fit_resample(X, y)
    return X, y, feature_names, original_df, scaler
def load_call_quality_data_using_KMeansSMOTE(file_path=FILE_PATH, target_col=target_col, cols_to_remove=cols_to_remove):
    X, y, feature_names, original_df, scaler = load_call_quality_data(file_path, target_col, cols_to_remove)
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = np.min(counts)
    
    try:
        # Try with more lenient parameters
        kmeans_smote = KMeansSMOTE(
            kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0, n_clusters=min(8, min_class_count)), 
            random_state=42,
            cluster_balance_threshold=0.05,  # Very low threshold
            k_neighbors=min(3, min_class_count-1)  # Ensure k_neighbors < min_class_count
        )
        X, y = kmeans_smote.fit_resample(X, y)
    except Exception as e:
        print(f"Error with KMeansSMOTE: {str(e)}")
    return X, y, feature_names, original_df, scaler

def load_call_quality_data_using_SVMSMOTE(file_path=FILE_PATH, target_col=target_col, cols_to_remove=cols_to_remove):
    X, y, feature_names, original_df, scaler = load_call_quality_data(file_path, target_col, cols_to_remove)
    svm_smote = SVMSMOTE(random_state=42)
    X, y = svm_smote.fit_resample(X, y)
    return X, y, feature_names, original_df, scaler
def load_call_quality_data_using_BORDERSMOTE(file_path=FILE_PATH, target_col=target_col, cols_to_remove=cols_to_remove):
    X, y, feature_names, original_df, scaler = load_call_quality_data(file_path, target_col, cols_to_remove)
    bordersmote = BorderlineSMOTE(random_state=42)
    X, y = bordersmote.fit_resample(X, y)
    return X, y, feature_names, original_df, scaler