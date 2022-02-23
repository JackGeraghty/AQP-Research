import numpy as np
from .node import AQPNode

class GatherNode(AQPNode):
    
    def __init__(self, id_, output_key, keys_to_gather, draw_options=None, **kwargs ):
        super().__init__(id_, output_key=output_key, draw_options=draw_options, **kwargs)
        self.keys_to_gather= keys_to_gather
        self.type_ = "GatherNode"
        
    def execute(self, result, **kwargs):
        super().execute(result, **kwargs)
        values = []
        for gathered_value in self.recursive_items(result):
            values.append(gathered_value)
        result[self.output_key] = np.array(values).T
        return result
        
    def recursive_items(self, dictionary):
        for key, value in dictionary.items():
            if type(value) is dict:
                yield from self.recursive_items(value)
            else:
                if key in self.keys_to_gather:
                    yield value
