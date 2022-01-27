import numpy as np
import pyvad
import multiprocessing as mp

from ..node import AQPNode

class MultiProcessVadNode(AQPNode):
    
    PATCH_SIZE = 30
    
    def __init__(self, id_: str, output_key: str, data_key: str, draw_options: dict = None, **kwargs):
        super().__init__(id_, output_key=output_key, draw_options=draw_options, **kwargs)
        self.data_key = data_key
        self.type_ ="MultiProcessVadNode"
        
    def vad( args):
        signal, sample_rate = args
        global PATCH_SIZE
        voice_activity = pyvad.vad(signal, sample_rate, fs_vad=sample_rate, hop_length=PATCH_SIZE, vad_mode=3)
        
    def execute(self, result: dict, **kwargs):
        super().execute(result, **kwargs)
        data = result[self.data_key]
        signals = result['Ref Wav']
        
        sample_rate = result['sr']
        pool = mp.Pool(processes = (mp.cpu_count()-1))
        pool.map(MultiProcessVadNode.vad, (signals, sample_rate))
        


        
        # keep_indexes = np.array([np.sum(voice_activity[reference_patch_indexes[i] : reference_patch_indexes[i] + PATCH_SIZE - 1]) < PATCH_SIZE * 0.8 for i in range(len(reference_patch_indexes))])