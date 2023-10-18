from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
import pandas as pd
import numpy as np

class STR2ARRAY(AbstractFunction):
    @setup(cacheable=True, function_type="FeatureExtraction", batchable=False)
    def setup(self):
        pass

    @property
    def name(self) -> str:
        return "STR2ARRAY"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["embedding"],           
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
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        df["features"] = df["embedding"].apply(lambda x: np.fromstring(x[2:-2], sep=','))
        return df[["features"]]  
