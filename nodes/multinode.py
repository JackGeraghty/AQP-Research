import graphutils
import logging
from .node import AQPNode
import multiprocessing as mp
import constants

LOGGER = logging.getLogger(constants.LOGGER_NAME)

class MultiNode(AQPNode):
    
    def __init__(self, id_: str, node_data: dict, root_nodes: dict,  output_key: str= None, draw_options: dict = None, **kwargs):
        super().__init__(id_, output_key=output_key, draw_options=draw_options, **kwargs)
        self.nodes = graphutils.build_graph(node_data)
        self.root_nodes = [self.nodes[root] for root in root_nodes]
        self.type_ = "MultiNode"
        
    def execute(self, result: dict, **kwargs):
        super().execute(result, **kwargs)
        num_cpus = mp.cpu_count()-1
        num_cpus_to_use = len(self.root_nodes) if len(self.root_nodes) < num_cpus else num_cpus
        # pool = mp.Pool(processes=num_cpus_to_use)
        LOGGER.info(f"num cpus {num_cpus_to_use}")
        LOGGER.info(f"Number of root node: {len(self.root_nodes)}")
        processes = []
        manager = mp.Manager()
        shared_result = manager.dict(result)

        for root in self.root_nodes:
            process = mp.Process(target=graphutils.run_node, args=(root, shared_result))
            process.start()
            processes.append(process)

        LOGGER.debug('Joining Processes')
        for process in processes:
            process.join()
        result = shared_result

        return result
