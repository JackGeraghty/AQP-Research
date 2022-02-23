# -*- coding: utf-8 -*-
import numpy as np
import pyvad
from .node import AQPNode

class VoiceMOSVADNode(AQPNode):
    
    def __init__(self, id_, output_key, target_key, draw_options=None, **kwargs):
        super().__init__(id_, output_key=output_key, draw_options=draw_options, **kwargs)
        self.target_key = target_key
        self.type_  = "VoiceMOSVADNode"
        
    def execute(self, result, **kwargs):
        super().execute(result, **kwargs)
        signal = result[self.target_key]
        sample_rate = result['sample_rate']
        try:
            activity = pyvad.split(signal, sample_rate, fs_vad=sample_rate, hop_length=30, vad_mode=3)
            x =  np.concatenate([signal[edge[0]:edge[1]] for edge in activity], axis=0)
            result[self.output_key] = x
        except (BaseException) as ex:
            print(ex)
        
        return result