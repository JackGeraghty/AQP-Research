from .node import AQPNode
from qualitymetrics.visqol.nsim import nsim_map

class NSIMNode(AQPNode):
    
    def __init__(self, id_: str, output_key: str, ref_spect_key, test_spect_key, draw_options=None, **kwargs):
        super().__init__(id_, output_key=output_key, draw_options=draw_options)
        self.ref_spect_key = ref_spect_key
        self.test_spect_key = test_spect_key
        self.type_  = "NSIMNode"
        
    def execute(self, result: dict, **kwargs):
        super().execute(result, **kwargs)

        ref_spect = result[self.ref_spect_key]
        test_spect = result[self.test_spect_key]

        result[self.output_key] = nsim_map(test_spect, ref_spect, 1)
        return result