from evadb.functions.abstract.abstract_function import AbstractFunction
import pandas as pd
import numpy as np
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


class EmbeddingArrayConverter(AbstractFunction):
    @setup(cacheable=True, function_type="FeatureExtraction", batchable=True)
    def setup(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "EmbeddingArrayConverter"

    @staticmethod
    def convert_embedding_string_to_array(embedding_str):
        # Convert the string representation of the embedding into a NumPy array
        return np.fromstring(embedding_str, sep=',')

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["embedding_str"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None,)]  
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["features"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(1,1536)] 
            )
        ]
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        def _forward(row: pd.Series) -> np.ndarray:
            embedding_str =row.iloc[0]
            embedding_array = EmbeddingArrayConverter.convert_embedding_string_to_array(embedding_str)
            return embedding_array

        ret = pd.DataFrame()
        ret["features"] = df.apply(_forward, axis=1)
        return ret
