import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

class DataSampler:
    """
    A class for sampling large parquet datasets in batches and handling class imbalance.
    
    Attributes:
    -----------
    file_path : str
        Path to the parquet file.
    sample_fraction : float
        Fraction of majority class to sample.
    batch_size : int
        Number of rows to read per batch.
    """
    
    def __init__(self, file_path: str, sample_fraction: float = 0.05, batch_size: int = 100000):
        self.file_path = file_path
        self.sample_fraction = sample_fraction
        self.batch_size = batch_size
        self.parquet_file = pq.ParquetFile(self.file_path)
    
    def sample_data(self) -> pd.DataFrame:
        """
        Sample the data from the parquet file in batches, addressing class imbalance by undersampling
        the majority class ('Interrupted') and retaining all rows from the minority class ('Continue').
        
        Returns:
        --------
        pd.DataFrame
            A sampled pandas DataFrame.
        """
        sampled_dfs = []
        
        for batch in self.parquet_file.iter_batches(batch_size=self.batch_size):
            batch_df = batch.to_pandas()
            
            # separate the majority and minority classes
            interrupted_batch_df = batch_df[batch_df['label'] == 'Interrupted']
            continue_batch_df = batch_df[batch_df['label'] == 'Continue']
            
            # sample (self.sample_fraction)% of the 'Interrupted' class and keep all of 'Continue'
            continue_sample_df = continue_batch_df.sample(frac=self.sample_fraction, random_state=42)
            
            sampled_dfs.append(interrupted_batch_df)
            sampled_dfs.append(continue_sample_df)
        
        # concatenate sampled dataframes from the list into a single dataframe
        final_sample_df = pd.concat(sampled_dfs, ignore_index=True)
        return final_sample_df
