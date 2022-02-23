import librosa.feature

from .node import AQPNode

class SimpleMelNode(AQPNode):
    
    def __init__(self, id_, output_key, target_key, draw_options=None, **kwargs):
        super().__init__(id_, output_key=output_key, draw_options=draw_options, **kwargs)
        self.target_key = target_key
        self.type_ = "SimpleMelNode"
        
    def execute(self, result, **kwargs):
        super().execute(result, **kwargs)
        signal = result[self.target_key]
        result[self.output_key] = librosa.feature.melspectrogram(signal)
        return result