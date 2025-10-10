from src.configurations.enums import FileFormat
import pandas as pd


class DataFrameHandler:
    """
    A utility class for handling DataFrame operations.
    """

    @staticmethod
    def read_dataframe(file_path: str, file_format: FileFormat) -> pd.DataFrame:
        """
        Reads a DataFrame from a file based on the specified file format.
        """
        if file_format == FileFormat.CSV:
            return pd.read_csv(file_path)
        elif file_format == FileFormat.FEATHER:
            return pd.read_feather(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    @staticmethod
    def write_dataframe(df: pd.DataFrame, file_path: str, file_format: FileFormat):
        """
        Writes a DataFrame to a file based on the specified file format.
        """
        if file_format == FileFormat.PARQUET:
            df.to_parquet(file_path, index=False)
        elif file_format == FileFormat.CSV:
            df.to_csv(file_path, index=False)
        elif file_format == FileFormat.FEATHER:
            df.to_feather(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


